#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>

// OpenCV
#include <opencv2/opencv.hpp>

// YOLO
#include "ThreadSafeQueue.h"
#include "yolo/yolo_segment.h"

struct FramePayload {
  cv::Mat frameBGR; // For Inference
  int64_t pts;
  bool isValid = true;
};

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

  // Queues
  ThreadSafeQueue<FramePayload> decodeQueue{50};
  ThreadSafeQueue<FramePayload> inferenceQueue{50};

  // State
  std::atomic<bool> isDecodingFinished{false};
  std::atomic<bool> isInferenceFinished{false};

  void processFrame(cv::Mat &frame);
};
