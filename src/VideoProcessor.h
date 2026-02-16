#pragma once

#include <memory>
#include <string>
#include <vector>

// FFmpeg
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
}

// OpenCV
#include <opencv2/opencv.hpp>

// YOLO
#include "yolo/onnxruntime/yolo_onnxruntime.h"
#include "yolo/yolo_segment.h"

class VideoProcessor {
public:
  VideoProcessor(const std::string &modelPath);
  ~VideoProcessor();

  bool processConfig(const std::string &initSegmentPath,
                     const std::string &mediaSegmentPath,
                     const std::string &outputDir);

private:
  std::string modelPath;
  std::unique_ptr<YOLO_ONNXRuntime_Segment> yolo;

  // FFmpeg context
  AVFormatContext *inputFmtCtx = nullptr;
  AVFormatContext *outputFmtCtx = nullptr;

  // Process a single frame
  void processFrame(cv::Mat &frame);

  // Helpers
  int openInput(const std::string &filename);
  int setupOutput(const std::string &filename);
};
