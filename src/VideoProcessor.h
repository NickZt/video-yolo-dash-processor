#pragma once

#include <memory>
#include <string>

// OpenCV
#include <opencv2/opencv.hpp>

// YOLO
#include "yolo/yolo_segment.h"

class VideoProcessor {
public:
  explicit VideoProcessor(const std::string &modelPath);
  ~VideoProcessor();

  bool processConfig(const std::string &initSegmentPath,
                     const std::string &mediaSegmentPath,
                     const std::string &outputDir);

private:
  std::string modelPath;
  std::unique_ptr<YOLO_Segment> yolo;

  void processFrame(cv::Mat &frame);
};
