#include "VideoProcessor.h"
#include "Metrics.h"
#include "yolo/yolo.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
}

namespace fs = std::filesystem;

static void check_error(int result, const std::string &msg) {
  if (result < 0) {
    char errbuf[128];
    av_strerror(result, errbuf, sizeof(errbuf));
    throw std::runtime_error(msg + ": " + std::string(errbuf));
  }
}

class VideoDecoder {
public:
  explicit VideoDecoder(const std::string &inputPath) : inputPath(inputPath) {
    packet = av_packet_alloc();
    frame = av_frame_alloc();
    frameBGR = av_frame_alloc();
  }

  ~VideoDecoder() {
    if (swsCtx)
      sws_freeContext(swsCtx);
    if (codecCtx)
      avcodec_free_context(&codecCtx);
    if (fmtCtx)
      avformat_close_input(&fmtCtx);
    if (frameBGR)
      av_frame_free(&frameBGR);
    if (frame)
      av_frame_free(&frame);
    if (packet)
      av_packet_free(&packet);
  }

  bool open() {
    if (avformat_open_input(&fmtCtx, inputPath.c_str(), nullptr, nullptr) < 0)
      return false;
    if (avformat_find_stream_info(fmtCtx, nullptr) < 0)
      return false;

    for (unsigned int i = 0; i < fmtCtx->nb_streams; i++) {
      if (fmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
        videoStreamIdx = i;
        break;
      }
    }
    if (videoStreamIdx == -1)
      return false;

    AVCodecParameters *codecPar = fmtCtx->streams[videoStreamIdx]->codecpar;
    const AVCodec *decoder = avcodec_find_decoder(codecPar->codec_id);
    codecCtx = avcodec_alloc_context3(decoder);
    avcodec_parameters_to_context(codecCtx, codecPar);
    avcodec_open2(codecCtx, decoder, nullptr);

    frameBGR->format = AV_PIX_FMT_BGR24;
    frameBGR->width = codecCtx->width;
    frameBGR->height = codecCtx->height;
    av_frame_get_buffer(frameBGR, 0);

    swsCtx =
        sws_getContext(codecCtx->width, codecCtx->height, codecCtx->pix_fmt,
                       codecCtx->width, codecCtx->height, AV_PIX_FMT_BGR24,
                       SWS_BILINEAR, nullptr, nullptr, nullptr);
    return true;
  }

  bool readFrame(cv::Mat &outFrame, int64_t &outPts) {
    auto t0 = std::chrono::high_resolution_clock::now();
    while (av_read_frame(fmtCtx, packet) >= 0) {
      if (packet->stream_index == videoStreamIdx) {
        if (avcodec_send_packet(codecCtx, packet) == 0) {
          if (avcodec_receive_frame(codecCtx, frame) == 0) {
            auto t1 = std::chrono::high_resolution_clock::now();
            double read_time =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            Metrics::getInstance().addTimeToFrame(read_time);

            sws_scale(swsCtx, frame->data, frame->linesize, 0, frame->height,
                      frameBGR->data, frameBGR->linesize);

            outFrame = cv::Mat(frame->height, frame->width, CV_8UC3,
                               frameBGR->data[0], frameBGR->linesize[0])
                           .clone();
            outPts = frame->pts;
            av_packet_unref(packet);

            auto t2 = std::chrono::high_resolution_clock::now();
            double conv_time =
                std::chrono::duration<double, std::milli>(t2 - t1).count();
            Metrics::getInstance().addTimeToConversion(conv_time);
            Metrics::getInstance().incrementFramesDecoded();

            return true;
          }
        }
      }
      av_packet_unref(packet);
    }
    return false;
  }

  int getWidth() const { return codecCtx->width; }
  int getHeight() const { return codecCtx->height; }
  AVRational getTimeBase() const {
    return fmtCtx->streams[videoStreamIdx]->time_base;
  }
  AVStream *getStream() const { return fmtCtx->streams[videoStreamIdx]; }

private:
  std::string inputPath;
  AVFormatContext *fmtCtx = nullptr;
  AVCodecContext *codecCtx = nullptr;
  SwsContext *swsCtx = nullptr;
  int videoStreamIdx = -1;
  AVPacket *packet = nullptr;
  AVFrame *frame = nullptr;
  AVFrame *frameBGR = nullptr;
};

class VideoEncoder {
public:
  VideoEncoder(const std::string &outputPath, AVStream *inStream)
      : outputPath(outputPath), inStream(inStream) {
    packet = av_packet_alloc();
    encFrame = av_frame_alloc();
  }

  ~VideoEncoder() {
    if (swsCtx)
      sws_freeContext(swsCtx);
    if (codecCtx)
      avcodec_free_context(&codecCtx);
    if (fmtCtx) {
      if (!(fmtCtx->oformat->flags & AVFMT_NOFILE) && fmtCtx->pb)
        avio_closep(&fmtCtx->pb);
      avformat_free_context(fmtCtx);
    }
    if (encFrame)
      av_frame_free(&encFrame);
    if (packet)
      av_packet_free(&packet);
  }

  bool open() {
    avformat_alloc_output_context2(&fmtCtx, nullptr, "dash",
                                   outputPath.c_str());
    if (!fmtCtx)
      return false;

    outStream = avformat_new_stream(fmtCtx, nullptr);
    const AVCodec *encoder = avcodec_find_encoder(AV_CODEC_ID_H264);
    codecCtx = avcodec_alloc_context3(encoder);

    codecCtx->height = inStream->codecpar->height;
    codecCtx->width = inStream->codecpar->width;
    codecCtx->sample_aspect_ratio = inStream->codecpar->sample_aspect_ratio;
    codecCtx->pix_fmt = AV_PIX_FMT_YUV420P;
    codecCtx->time_base = inStream->time_base;
    codecCtx->profile = inStream->codecpar->profile;
    codecCtx->level = inStream->codecpar->level;

    if (fmtCtx->oformat->flags & AVFMT_GLOBALHEADER)
      codecCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    if (avcodec_open2(codecCtx, encoder, nullptr) < 0)
      return false;

    avcodec_parameters_from_context(outStream->codecpar, codecCtx);

    AVDictionary *opts = nullptr;
    av_dict_set(&opts, "window_size", "5", 0);
    av_dict_set(&opts, "extra_window_size", "5", 0);
    av_dict_set(&opts, "seg_duration", "2", 0);
    av_dict_set(&opts, "init_seg_name", "init.mp4", 0);
    av_dict_set(&opts, "media_seg_name", "chunk-$Number$.m4s", 0);

    if (!(fmtCtx->oformat->flags & AVFMT_NOFILE)) {
      if (avio_open(&fmtCtx->pb, outputPath.c_str(), AVIO_FLAG_WRITE) < 0)
        return false;
    }

    check_error(avformat_write_header(fmtCtx, &opts), "write header");

    encFrame->format = codecCtx->pix_fmt;
    encFrame->width = codecCtx->width;
    encFrame->height = codecCtx->height;
    av_frame_get_buffer(encFrame, 32);

    swsCtx =
        sws_getContext(codecCtx->width, codecCtx->height, AV_PIX_FMT_BGR24,
                       codecCtx->width, codecCtx->height, AV_PIX_FMT_YUV420P,
                       SWS_BILINEAR, nullptr, nullptr, nullptr);
    return true;
  }

  void writeFrame(const cv::Mat &frame, int64_t pts) {
    const uint8_t *data[1] = {frame.data};
    int linesize[1] = {static_cast<int>(frame.step)};

    sws_scale(swsCtx, data, linesize, 0, codecCtx->height, encFrame->data,
              encFrame->linesize);
    encFrame->pts = pts;

    if (avcodec_send_frame(codecCtx, encFrame) == 0) {
      receiveAndWritePackets();
    }
    Metrics::getInstance().incrementFramesEncoded();
  }

  void flush() {
    if (avcodec_send_frame(codecCtx, nullptr) == 0) {
      receiveAndWritePackets();
    }
    av_write_trailer(fmtCtx);
  }

private:
  void receiveAndWritePackets() {
    while (avcodec_receive_packet(codecCtx, packet) == 0) {
      packet->stream_index = 0;
      av_packet_rescale_ts(packet, codecCtx->time_base, outStream->time_base);
      av_interleaved_write_frame(fmtCtx, packet);
      av_packet_unref(packet);
    }
  }

  std::string outputPath;
  AVFormatContext *fmtCtx = nullptr;
  AVCodecContext *codecCtx = nullptr;
  AVStream *outStream = nullptr;
  AVStream *inStream = nullptr;
  SwsContext *swsCtx = nullptr;
  AVPacket *packet = nullptr;
  AVFrame *encFrame = nullptr;
};

// --- Video Processor Class ---

VideoProcessor::VideoProcessor(const std::string &modelPath)
    : modelPath(modelPath) {
  std::unique_ptr<YOLO> base_yolo = CreateFactory::instance().create(
      Backend_Type::ONNXRuntime, Task_Type::Segment);

  if (!base_yolo) {
    throw std::runtime_error("Failed to create YOLO model instance.");
  }

  yolo = std::unique_ptr<YOLO_Segment>(
      dynamic_cast<YOLO_Segment *>(base_yolo.release()));
  // Depending on system architecture, fallback to CPU might be necessary, but
  // defaulting to GPU for performance if available
  yolo->init(YOLOv8, CPU, FP32, modelPath);
}

VideoProcessor::~VideoProcessor() {}

bool VideoProcessor::processConfig(const std::string &initSegmentPath,
                                   const std::string &mediaSegmentPath,
                                   const std::string &outputDir) {
  Metrics::getInstance().startProcessing();
  std::string tempInput = "temp_full_input.mp4";

  {
    std::ofstream outfile(tempInput, std::ios::binary);

    // Check if init segment is valid and non-empty
    if (!initSegmentPath.empty() && fs::exists(initSegmentPath) &&
        fs::file_size(initSegmentPath) > 0) {
      std::ifstream initFile(initSegmentPath, std::ios::binary);
      outfile << initFile.rdbuf();
    }

    std::ifstream mediaFile(mediaSegmentPath, std::ios::binary);
    outfile << mediaFile.rdbuf();
  }

  VideoDecoder decoder(tempInput);
  if (!decoder.open()) {
    std::cerr << "Failed to open input video" << std::endl;
    return false;
  }

  Metrics::getInstance().setFrameSize(decoder.getWidth(), decoder.getHeight());

  std::string cleanOutputDir = outputDir;
  if (!cleanOutputDir.empty() && cleanOutputDir.back() == '/') {
    cleanOutputDir.pop_back();
  }
  std::string outputFileName = cleanOutputDir + "/manifest.mpd";
  VideoEncoder encoder(outputFileName, decoder.getStream());
  if (!encoder.open()) {
    std::cerr << "Failed to open output video" << std::endl;
    return false;
  }

  cv::Mat frame;
  int64_t pts;

  while (decoder.readFrame(frame, pts)) {
    processFrame(frame);
    encoder.writeFrame(frame, pts);
  }

  encoder.flush();
  fs::remove(tempInput);

  Metrics::getInstance().stopProcessing();
  Metrics::getInstance().printMetrics();

  return true;
}

void VideoProcessor::processFrame(cv::Mat &frame) {
  auto t0 = std::chrono::high_resolution_clock::now();

  yolo->infer_image(frame);
  const std::vector<OutputSeg> &output = yolo->getOutputSeg();

  static bool saved_debug_frame = false;

  for (const auto &det : output) {
    if (det.id == 0) { // Person
      std::cout << "Person Detected! Bounding box area: " << det.box.area()
                << std::endl;
      // intersection with frame
      cv::Rect bbox = det.box & cv::Rect(0, 0, frame.cols, frame.rows);

      if (bbox.area() > 0 && !det.mask.empty()) {
        std::cout << "Valid BBox and Non-Empty Mask! Mask cols: "
                  << det.mask.cols << ", rows: " << det.mask.rows << std::endl;

        // det.mask corresponds to det.box. We need to crop it to bbox.
        // The offset is the difference between bbox.tl() and det.box.tl()
        cv::Rect mask_roi(bbox.x - det.box.x, bbox.y - det.box.y, bbox.width,
                          bbox.height);

        // Ensure ROI is within mask bounds
        mask_roi = mask_roi & cv::Rect(0, 0, det.mask.cols, det.mask.rows);

        if (mask_roi.area() > 0 && mask_roi.width == bbox.width &&
            mask_roi.height == bbox.height) {
          cv::Mat valid_mask = det.mask(mask_roi).clone();
          if (!valid_mask.empty() && valid_mask.type() == CV_8UC1) {
            if (!saved_debug_frame) {
              cv::imwrite("pre_mask.jpg", frame);
            }
            frame(bbox).setTo(cv::Scalar(0, 0, 0), valid_mask);
            std::cout << "SUCCESS: Mask painted on frame!" << std::endl;
            if (!saved_debug_frame) {
              cv::imwrite("post_mask.jpg", frame);
              saved_debug_frame = true;
            }
          }
        } else {
          std::cout << "FAILED: mask_roi area zero or dimension mismatch."
                    << std::endl;
        }
      } else {
        std::cout << "FAILED: bbox area 0 or det.mask is empty!" << std::endl;
      }
    }
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  double inf_time = std::chrono::duration<double, std::milli>(t1 - t0).count();
  Metrics::getInstance().addTimeToInference(inf_time);
  Metrics::getInstance().incrementFramesInferred();
}
