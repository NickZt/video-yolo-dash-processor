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

  bool readFrame(cv::Mat &outFrame, AVFrame *&outYuvFrame, int64_t &outPts) {
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
            outYuvFrame = av_frame_clone(frame);
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

  void writeFrame(AVFrame *yuvFrame, int64_t pts) {
    yuvFrame->pts = pts;

    if (avcodec_send_frame(codecCtx, yuvFrame) == 0) {
      receiveAndWritePackets();
    }
    Metrics::getInstance().incrementFramesEncoded();
    av_frame_free(&yuvFrame);
  }

  void flush() {
    if (avcodec_send_frame(codecCtx, nullptr) == 0) {
      receiveAndWritePackets();
    }
    // Only write trailer if we actually encoded frames to avoid DASH manifest
    // divide-by-zero FPE
    if (Metrics::getInstance().getFramesEncoded() > 0) {
      av_write_trailer(fmtCtx);
    }
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

VideoProcessor::VideoProcessor(const std::map<std::string, std::string> &args)
    : args(args) {
  engineType = args.at("--engine");
  std::string modelPath = args.at("--model");

  if (engineType == "yolo") {
    numInferenceThreads = std::max(1u, std::thread::hardware_concurrency() /
                                           2); // default scaling
    int optimalYoloThreads =
        1; // YOLO optimally runs 1 IntraOp thread under scaling
    Metrics::getInstance().setThreadInfo(numInferenceThreads,
                                         std::thread::hardware_concurrency());
    Metrics::getInstance().setOptimizationInfo("ONNXRuntime CPU", "FP32", 640,
                                               640, 1, optimalYoloThreads);
    for (int i = 0; i < numInferenceThreads; ++i) {
      std::unique_ptr<YOLO> base_yolo = CreateFactory::instance().create(
          Backend_Type::ONNXRuntime, Task_Type::Segment);

      if (!base_yolo) {
        throw std::runtime_error("Failed to create YOLO model instance.");
      }

      auto yolo_instance = std::unique_ptr<YOLO_Segment>(
          dynamic_cast<YOLO_Segment *>(base_yolo.release()));

      // Defaulting to CPU FP32 for now
      yolo_instance->init(YOLOv8, CPU, FP32, modelPath);
      yoloPool.push_back(std::move(yolo_instance));
    }
  } else if (engineType == "dino") {
    // GroundingDINO relies on heavy self-attention mechanisms mapping
    // significantly better onto fewer individual concurrent queue dispatchers
    // paired with higher integrated thread limits.
    numInferenceThreads =
        std::max(1u, std::thread::hardware_concurrency() / 10);
    int intraOpThreads =
        std::max(1u, std::thread::hardware_concurrency() / numInferenceThreads);
    int optimalDinoThreads = 5; // Theoretical max bound per worker instance

    Metrics::getInstance().setThreadInfo(numInferenceThreads,
                                         std::thread::hardware_concurrency());

    // Instantiate the primary thread worker then copy its properties natively
    auto primary_dino = std::make_unique<GroundingDINO>(
        modelPath, 0.3f, "vocab.txt", 0.25f, intraOpThreads);

    std::string backend, precision;
    int t_width, t_height, optimal;
    primary_dino->get_model_info(backend, precision, t_width, t_height,
                                 optimal);

    Metrics::getInstance().setOptimizationInfo(
        backend, precision, t_width, t_height, intraOpThreads, optimal);

    dinoPool.push_back(std::move(primary_dino));

    for (int i = 1; i < numInferenceThreads; ++i) {
      dinoPool.push_back(std::make_unique<GroundingDINO>(
          modelPath, 0.3f, "vocab.txt", 0.25f, intraOpThreads));
    }
  }
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

  isDecodingFinished = false;

  std::thread decodeThread([&]() {
    cv::Mat frame;
    AVFrame *yuvFrame = nullptr;
    int64_t pts;

    int checkFramesLimit = -1;
    if (args.find("--checkframes") != args.end()) {
      checkFramesLimit = std::stoi(args.at("--checkframes"));
    }

    int frames_read = 0;
    while (decoder.readFrame(frame, yuvFrame, pts)) {
      if (checkFramesLimit > 0 && frames_read >= checkFramesLimit)
        break; // Dynamically bound benchmarking threshold
      frames_read++;
      FramePayload payload;
      payload.frameBGR = frame;
      payload.yuvFrame = yuvFrame;
      payload.pts = pts;
      payload.isValid = true;
      decodeQueue.push(payload);
    }
    isDecodingFinished = true;
    decodeQueue.close();
  });

  activeInferenceThreads = numInferenceThreads;
  std::vector<std::thread> inferenceThreads;
  for (int i = 0; i < numInferenceThreads; ++i) {
    inferenceThreads.emplace_back([this, i]() {
      while (true) {
        auto payloadOpt = decodeQueue.pop();
        if (!payloadOpt) {
          if (decodeQueue.is_closed() && decodeQueue.size() == 0)
            break;
          continue;
        }
        FramePayload payload = *payloadOpt;
        if (payload.isValid) {
          if (engineType == "yolo") {
            processFrame(payload.frameBGR, payload.yuvFrame, yoloPool[i].get());
          } else if (engineType == "dino") {
            processFrameDino(payload.frameBGR, payload.yuvFrame,
                             dinoPool[i].get(), args.at("--prompt"));
          }
        }
        inferenceQueue.push(payload);
      }
      if (--activeInferenceThreads == 0) {
        inferenceQueue.close();
      }
    });
  }

  // Mux / Encode on Main Thread
  std::map<int64_t, FramePayload> reorderBuffer;
  int64_t expected_pts = 0;

  while (true) {
    auto payloadOpt = inferenceQueue.pop();
    if (!payloadOpt) {
      if (inferenceQueue.is_closed() && inferenceQueue.size() == 0) {
        break;
      }
      continue;
    }
    FramePayload payload = *payloadOpt;
    reorderBuffer[payload.pts] = payload;

    // Output all consecutive frames
    while (!reorderBuffer.empty() &&
           reorderBuffer.begin()->first == expected_pts) {
      auto it = reorderBuffer.begin();
      if (it->second.isValid) {
        encoder.writeFrame(it->second.yuvFrame, it->second.pts);
      }
      reorderBuffer.erase(it);
      expected_pts++;
    }
  }

  // Flush any remaining frames in buffer just in case
  for (auto &pair : reorderBuffer) {
    if (pair.second.isValid) {
      encoder.writeFrame(pair.second.yuvFrame, pair.second.pts);
    }
  }

  decodeThread.join();
  for (auto &t : inferenceThreads) {
    t.join();
  }

  encoder.flush();
  fs::remove(tempInput);

  Metrics::getInstance().stopProcessing();
  Metrics::getInstance().printMetrics();

  return true;
}

void VideoProcessor::processFrame(cv::Mat &frame, AVFrame *yuvFrame,
                                  YOLO_Segment *yolo) {
  auto t0 = std::chrono::high_resolution_clock::now();

  yolo->infer_image(frame);
  const std::vector<OutputSeg> &output = yolo->getOutputSeg();

  // Create zero-copy cv::Mat wrapper around the hardware Y-plane (Luminance)
  cv::Mat y_plane(yuvFrame->height, yuvFrame->width, CV_8UC1, yuvFrame->data[0],
                  yuvFrame->linesize[0]);

  for (const auto &det : output) {
    if (det.id == 0) { // Person
      // intersection with frame
      cv::Rect bbox = det.box & cv::Rect(0, 0, frame.cols, frame.rows);

      if (bbox.area() > 0 && !det.mask.empty()) {
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
            // Apply mask to BGR frame (optional, for visual debugging)
            frame(bbox).setTo(cv::Scalar(0, 0, 0), valid_mask);
            // Apply mask directly to the Zero-Copy YUV hardware frame buffer
            // Sets luminance to 0 (black in YUV space) where the mask is active
            y_plane(bbox).setTo(0, valid_mask);
            std::cout << "Detected obj (person) mask painted\n";
          }
        }
      }
    }
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  double inf_time = std::chrono::duration<double, std::milli>(t1 - t0).count();
  Metrics::getInstance().addTimeToInference(inf_time);
  Metrics::getInstance().incrementFramesInferred();
}

void VideoProcessor::processFrameDino(cv::Mat &frame, AVFrame *yuvFrame,
                                      GroundingDINO *dino,
                                      const std::string &prompt) {
  auto t0 = std::chrono::high_resolution_clock::now();

  std::vector<DINOObject> output = dino->detect(frame, prompt);

  cv::Mat y_plane(yuvFrame->height, yuvFrame->width, CV_8UC1, yuvFrame->data[0],
                  yuvFrame->linesize[0]);

  for (const auto &det : output) {
    cv::Rect bbox = det.box & cv::Rect(0, 0, frame.cols, frame.rows);
    if (bbox.area() > 0) {
      // Draw a black bounding box around the detected text prompt objects onto
      // the Y-plane
      cv::rectangle(y_plane, bbox, cv::Scalar(0), 4);
    }
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  double inf_time = std::chrono::duration<double, std::milli>(t1 - t0).count();
  Metrics::getInstance().addTimeToInference(inf_time);
  Metrics::getInstance().incrementFramesInferred();
}
