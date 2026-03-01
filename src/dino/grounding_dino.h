#pragma once

#include "Tokenizer.hpp"
#include <iostream>
#include <memory>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

struct DINOObject {
  cv::Rect box;
  std::string text;
  float prob;
};

class GroundingDINO {
public:
  GroundingDINO(std::string modelpath, float box_threshold,
                std::string vocab_path, float text_threshold,
                int num_threads = 1, bool use_optimization = false);
  std::vector<DINOObject> detect(cv::Mat srcimg, std::string text_prompt);
  void get_model_info(std::string &backend, std::string &precision, int &width,
                      int &height, int &optimal);

private:
  void preprocess(cv::Mat img);
  bool load_tokenizer(std::string vocab_path);
  static inline float sigmoid(float x) {
    return static_cast<float>(1.f / (1.f + exp(-x)));
  }

  const float mean[3] = {0.485, 0.456, 0.406};
  const float std[3] = {0.229, 0.224, 0.225};
  int size[2]; // (Width, Height)

  std::shared_ptr<TokenizerBase> tokenizer;

  std::vector<float> input_img;
  std::vector<std::vector<int64_t>> input_ids;
  std::vector<std::vector<int64_t>> attention_mask;
  std::vector<std::vector<int64_t>> token_type_ids;
  std::vector<std::vector<uint8_t>> text_self_attention_masks;
  std::vector<std::vector<int64_t>> position_ids;

  Ort::Env env;
  std::unique_ptr<Ort::Session> ort_session;
  Ort::SessionOptions sessionOptions;
  Ort::MemoryInfo memory_info_handler =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  const char *input_names[5] = {"pixel_values", "input_ids", "token_type_ids",
                                "attention_mask", "pixel_mask"};
  const char *output_names[2] = {"logits", "pred_boxes"};

  float box_threshold;
  float text_threshold;
  int optimal_threads = 5;
  std::string active_backend = "ONNXRuntime CPU";
  std::string active_precision = "FP32";
  const int max_text_len = 256;
  std::vector<int64_t> specical_tokens = {101, 102, 1012, 1029};
};
