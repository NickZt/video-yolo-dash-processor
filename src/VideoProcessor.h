#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

// OpenCV
#include <opencv2/opencv.hpp>

// YOLO and DINO
#include "ThreadSafeQueue.h"
#include "dino/grounding_dino.h"
#include "yolo/yolo_segment.h"

struct AVFrame;

struct FramePayload {
  cv::Mat frameBGR;            // For Inference
  AVFrame *yuvFrame = nullptr; // Original decoded frame from demuxer
  int64_t pts;
  bool isValid = true;
};

class VideoProcessor {
public:
  explicit VideoProcessor(const std::map<std::string, std::string> &args);
  ~VideoProcessor();

  bool processConfig(const std::string &initSegmentPath,
                     const std::string &mediaSegmentPath,
                     const std::string &outputDir);

private:
  std::map<std::string, std::string> args;
  int numInferenceThreads;

  std::string engineType;
  std::vector<std::unique_ptr<YOLO_Segment>> yoloPool;
  std::vector<std::unique_ptr<GroundingDINO>> dinoPool;

  // Queues
  ThreadSafeQueue<FramePayload> decodeQueue{50};
  ThreadSafeQueue<FramePayload> inferenceQueue{50};

  // State
  std::atomic<bool> isDecodingFinished{false};
  std::atomic<int> activeInferenceThreads{0};

  void processFrame(cv::Mat &frame, AVFrame *yuvFrame, YOLO_Segment *yolo);
  void processFrameDino(cv::Mat &frame, AVFrame *yuvFrame, GroundingDINO *dino,
                        const std::string &prompt);
};
