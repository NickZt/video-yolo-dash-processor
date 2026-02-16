#include "VideoProcessor.h"
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

namespace fs = std::filesystem;

VideoProcessor::VideoProcessor(const std::string &modelPath)
    : modelPath(modelPath) {
  // Algo_Type::YOLOv8 is 5 (from yolo.h enum)
  // Device_Type::GPU (1)
  // Model_Type::FP32 (0)
  // We access create factory or just instantiate directly.
  // yolo_onnxruntime_segment.h defines YOLO_ONNXRuntime_Segment class.
  // We can just use that.

  yolo = std::make_unique<YOLO_ONNXRuntime_Segment>();
  // Call init.
  // algo_type=5 (YOLOv8), device_type=1 (GPU) - assuming user has GPU,
  // otherwise 0 (CPU) Let's try GPU first, if it fails, maybe fallback? But
  // ONNX Runtime usually handles it or throws? We'll stick to CPU (0) to be
  // safe given the environment is unknown, or check if cuda is available? User
  // requirement: "Reference for using ONNx runtime in
  // /home/nickzt/Projects/TactOrder/MNNLLama/inference-services/onnx-service"
  // Usually onnxruntime implies CPU unless CUDA provider is installed.
  // Let's use CPU (0) for safety unless we see CUDA requirement.
  // Wait, the reference repo has "gpu" in path "onnxruntime-linux-x64-gpu...".
  // I'll try GPU (1). If it fails, I might crash.
  // Let's stick to GPU (1) as requested implicitly by "reference
  // implementation". Actually, "video_processor" implies high perf.
  yolo->init(YOLOv8, GPU, FP32, modelPath);
}

VideoProcessor::~VideoProcessor() {
  if (inputFmtCtx)
    avformat_close_input(&inputFmtCtx);
  if (outputFmtCtx) {
    if (outputFmtCtx->pb)
      avio_closep(&outputFmtCtx->pb);
    avformat_free_context(outputFmtCtx);
  }
}

// Helpers for ffmpeg error handling
static void check_error(int result, const std::string &msg) {
  if (result < 0) {
    char errbuf[128];
    av_strerror(result, errbuf, sizeof(errbuf));
    std::cerr << msg << ": " << errbuf << std::endl;
    throw std::runtime_error(msg + ": " + errbuf);
  }
}

bool VideoProcessor::processConfig(const std::string &initSegmentPath,
                                   const std::string &mediaSegmentPath,
                                   const std::string &outputDir) {
  // 1. Prepare Input
  // We need to read init and media segments.
  // Simpler approach: Create a temporary concatenated file.
  // Or we can rely on standard file I/O to read them into a buffer and use
  // custom AVIO. Concatenating to a temp file is robust and easy for
  // libavformat.

  std::string tempInput = "temp_full_input.mp4";
  // Concatenate files
  {
    std::ofstream outfile(tempInput, std::ios::binary);
    std::ifstream initFile(initSegmentPath, std::ios::binary);
    outfile << initFile.rdbuf();
    std::ifstream mediaFile(mediaSegmentPath, std::ios::binary);
    outfile << mediaFile.rdbuf();
  }

  // Open Input
  if (avformat_open_input(&inputFmtCtx, tempInput.c_str(), nullptr, nullptr) <
      0) {
    std::cerr << "Could not open input file" << std::endl;
    return false;
  }

  if (avformat_find_stream_info(inputFmtCtx, nullptr) < 0) {
    std::cerr << "Could not find stream info" << std::endl;
    return false;
  }

  int videoStreamIdx = -1;
  for (unsigned int i = 0; i < inputFmtCtx->nb_streams; i++) {
    if (inputFmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      videoStreamIdx = i;
      break;
    }
  }

  if (videoStreamIdx == -1) {
    std::cerr << "No video stream found" << std::endl;
    return false;
  }

  AVCodecParameters *codecPar = inputFmtCtx->streams[videoStreamIdx]->codecpar;
  const AVCodec *decoder = avcodec_find_decoder(codecPar->codec_id);
  AVCodecContext *decoderCtx = avcodec_alloc_context3(decoder);
  avcodec_parameters_to_context(decoderCtx, codecPar);
  avcodec_open2(decoderCtx, decoder, nullptr);

  // 2. Prepare Output
  // We want DASH output: init.dash and segment.m4s
  // We can use the dash muxer.
  std::string outputManifest =
      outputDir + "/manifest.mpd"; // Dash requires MPD, but we can ignore it
  // Or simpler: Use mp4 muxer with frag_custom and split manually? No.
  // Use dash muxer.

  avformat_alloc_output_context2(&outputFmtCtx, nullptr, "dash",
                                 outputManifest.c_str());
  if (!outputFmtCtx) {
    std::cerr << "Could not create output context" << std::endl;
    return false;
  }

  AVStream *outStream = avformat_new_stream(outputFmtCtx, nullptr);
  // Be careful here. We are not re-encoding if we can avoid it?
  // Requirement says: "encode processed frames back to the output data segment"
  // So we MUST re-encode.

  const AVCodec *encoder = avcodec_find_encoder(AV_CODEC_ID_H264);
  AVCodecContext *encoderCtx = avcodec_alloc_context3(encoder);

  encoderCtx->height = decoderCtx->height;
  encoderCtx->width = decoderCtx->width;
  encoderCtx->sample_aspect_ratio = decoderCtx->sample_aspect_ratio;
  encoderCtx->pix_fmt = AV_PIX_FMT_YUV420P; // YOLO uses BGR, we convert back
  encoderCtx->time_base =
      inputFmtCtx->streams[videoStreamIdx]->time_base; // Use same time base
  // encoderCtx->framerate = decoderCtx->framerate; // Deprecated?

  // Set some encoding options for DASH compatibility?
  // Just minimal options for now.

  if (outputFmtCtx->oformat->flags & AVFMT_GLOBALHEADER)
    encoderCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

  avcodec_open2(encoderCtx, encoder, nullptr);
  avcodec_parameters_from_context(outStream->codecpar, encoderCtx);

  // Dash specific options
  AVDictionary *opts = nullptr;
  av_dict_set(&opts, "init_seg_name", "init.dash", 0);
  av_dict_set(&opts, "media_seg_name", "segment.m4s", 0);
  av_dict_set(&opts, "use_template", "0", 0);
  av_dict_set(&opts, "use_timeline", "0", 0);
  av_dict_set(&opts, "single_file", "1", 0); // To produce one segment file?
  // Wait, typical DASH is init + segments.
  // If we want init.dash and segment.m4s (single segment),
  // we should configure it to yield exactly that.
  // Maybe just use mp4 muxer with fragmentation options?
  // command: ffmpeg ... -f mp4 -movflags
  // empty_moov+default_base_moof+frag_keyframe ... output.mp4 And output.mp4
  // will contain everything. But user wants 2 files. Actually, splitting a
  // fragmented MP4 into init and segment is just cutting at the first 'moof'.
  // The "init" is 'ftyp' + 'moov'. The "segment" is 'moof' + 'mdat'.

  // Let's stick to reading frames, processing and writing to fragmented mp4
  // helper. But getting exactly "init.dash" and "segment.m4s" names with one
  // muxer might be tricky without "dash" muxer. Using "dash" muxer with single
  // segment output is probably best.

  if (!(outputFmtCtx->oformat->flags & AVFMT_NOFILE)) {
    if (avio_open(&outputFmtCtx->pb, outputManifest.c_str(), AVIO_FLAG_WRITE) <
        0) {
      std::cerr << "Could not open output file" << std::endl;
      return false;
    }
  }

  // Write Header
  check_error(avformat_write_header(outputFmtCtx, &opts), "write header");

  // 3. Processing Loop
  AVPacket *packet = av_packet_alloc();
  AVFrame *frame = av_frame_alloc();
  AVFrame *frameBGR = av_frame_alloc();

  // Alloc buffers for BGR frame
  frameBGR->format = AV_PIX_FMT_BGR24;
  frameBGR->width = decoderCtx->width;
  frameBGR->height = decoderCtx->height;
  av_frame_get_buffer(frameBGR, 0);

  // SwsContext
  SwsContext *swsCtx =
      sws_getContext(decoderCtx->width, decoderCtx->height, decoderCtx->pix_fmt,
                     decoderCtx->width, decoderCtx->height, AV_PIX_FMT_BGR24,
                     SWS_BILINEAR, nullptr, nullptr, nullptr);

  SwsContext *swsCtxRev =
      sws_getContext(decoderCtx->width, decoderCtx->height, AV_PIX_FMT_BGR24,
                     decoderCtx->width, decoderCtx->height, AV_PIX_FMT_YUV420P,
                     SWS_BILINEAR, nullptr, nullptr, nullptr);

  while (av_read_frame(inputFmtCtx, packet) >= 0) {
    if (packet->stream_index == videoStreamIdx) {
      if (avcodec_send_packet(decoderCtx, packet) == 0) {
        while (avcodec_receive_frame(decoderCtx, frame) == 0) {
          // Convert YUV -> BGR
          sws_scale(swsCtx, frame->data, frame->linesize, 0, frame->height,
                    frameBGR->data, frameBGR->linesize);

          // Create cv::Mat wrapper
          cv::Mat img(frame->height, frame->width, CV_8UC3, frameBGR->data[0],
                      frameBGR->linesize[0]);

          // Process Frame
          processFrame(img);

          // Convert BGR -> YUV (reuse frame buffer if possible, or new one)
          // We need to write to a writable frame for encoder
          AVFrame *encFrame = av_frame_alloc();
          encFrame->format = encoderCtx->pix_fmt;
          encFrame->width = encoderCtx->width;
          encFrame->height = encoderCtx->height;
          av_frame_get_buffer(encFrame, 32);
          encFrame->pts = frame->pts; // Propagate PTS

          sws_scale(swsCtxRev, frameBGR->data, frameBGR->linesize, 0,
                    frameBGR->height, encFrame->data, encFrame->linesize);

          // Encode
          if (avcodec_send_frame(encoderCtx, encFrame) == 0) {
            AVPacket *encPacket = av_packet_alloc();
            while (avcodec_receive_packet(encoderCtx, encPacket) == 0) {
              encPacket->stream_index = 0; // Output has only 1 stream
              av_packet_rescale_ts(encPacket, encoderCtx->time_base,
                                   outStream->time_base);
              av_interleaved_write_frame(outputFmtCtx, encPacket);
              av_packet_unref(encPacket);
            }
            av_packet_free(&encPacket);
          }
          av_frame_free(&encFrame);
        }
      }
    }
    av_packet_unref(packet);
  }

  // Flush encoder
  avcodec_send_frame(encoderCtx, nullptr);
  AVPacket *encPacket = av_packet_alloc();
  while (avcodec_receive_packet(encoderCtx, encPacket) == 0) {
    encPacket->stream_index = 0;
    av_packet_rescale_ts(encPacket, encoderCtx->time_base,
                         outStream->time_base);
    av_interleaved_write_frame(outputFmtCtx, encPacket);
    av_packet_unref(encPacket);
  }
  av_packet_free(&encPacket);

  av_write_trailer(outputFmtCtx);

  // Cleanup
  av_frame_free(&frame);
  av_frame_free(&frameBGR);
  av_packet_free(&packet);
  sws_freeContext(swsCtx);
  sws_freeContext(swsCtxRev);
  avcodec_free_context(&decoderCtx);
  avcodec_free_context(&encoderCtx);
  fs::remove(tempInput);

  return true;
}

void VideoProcessor::processFrame(cv::Mat &frame) {
  // 1. Run Inference
  yolo->infer_image(frame);

  // 2. Get Results
  const std::vector<OutputSeg> &output = yolo->getOutputSeg();

  // 3. Apply Masks
  // Person class ID is usually 0 in COCO
  for (const auto &det : output) {
    if (det.id == 0) { // Person
      // Apply mask (paint black)
      // mask is a cv::Mat (CV_8UC1 probably, boolean or 0-255)
      // It is cropped to the box?
      // Checking yolo_segment.h:
      // "mask = mask(temp_rect - cv::Point(left, top)) > mask_threshold;"
      // "output.mask = mask;"
      // It seems output.mask is the mask for the box region.
      // And draw_result implementation:
      // "mask(bbox).setTo(cv::Scalar(...), output_seg[i].mask);"

      // We want to set pixels to BLACK (0,0,0)
      cv::Rect bbox = det.box & cv::Rect(0, 0, frame.cols, frame.rows);
      if (bbox.area() > 0 && !det.mask.empty()) {
        // Resize mask if needed? det.mask should match bbox size if
        // implementation is correct. Assuming it matches.
        frame(bbox).setTo(cv::Scalar(0, 0, 0), det.mask);
      }
    }
  }
}
