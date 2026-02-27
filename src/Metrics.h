#pragma once

#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>

class Metrics {
public:
  static Metrics &getInstance() {
    static Metrics instance;
    return instance;
  }

  void startProcessing() { start_time = std::chrono::steady_clock::now(); }

  void stopProcessing() { end_time = std::chrono::steady_clock::now(); }

  void incrementFramesDecoded() { frames_decoded++; }
  void incrementFramesInferred() { frames_inferred++; }
  void incrementFramesEncoded() { frames_encoded++; }

  void addTimeToFrame(double ms) {
    std::lock_guard<std::mutex> lock(mtx);
    total_time_to_frame += ms;
  }

  void addTimeToConversion(double ms) {
    std::lock_guard<std::mutex> lock(mtx);
    total_time_to_conversion += ms;
  }

  void addTimeToInference(double ms) {
    std::lock_guard<std::mutex> lock(mtx);
    total_time_to_inference += ms;
  }

  void setFrameSize(int w, int h) {
    frame_width.store(w);
    frame_height.store(h);
  }

  void setThreadInfo(int workers, int concurrency) {
    num_workers.store(workers);
    hw_concurrency.store(concurrency);
  }

  void setOptimizationInfo(const std::string &backend,
                           const std::string &precision, int t_width,
                           int t_height, int intra_threads,
                           int optimal_threads) {
    std::lock_guard<std::mutex> lock(mtx);
    inference_backend = backend;
    model_precision = precision;
    tensor_width.store(t_width);
    tensor_height.store(t_height);
    intra_op_threads.store(intra_threads);
    optimal_intra_threads.store(optimal_threads);
  }

  void printMetrics() {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();
    double fps = frames_encoded.load() > 0 && duration > 0
                     ? (frames_encoded.load() * 1000.0) / duration
                     : 0.0;

    double avg_t2f = frames_decoded.load() > 0
                         ? total_time_to_frame / frames_decoded.load()
                         : 0.0;
    double avg_ttc = frames_decoded.load() > 0
                         ? total_time_to_conversion / frames_decoded.load()
                         : 0.0;
    double avg_tti = frames_inferred.load() > 0
                         ? total_time_to_inference / frames_inferred.load()
                         : 0.0;

    std::cout << "\n=== Video Processing Metrics ===\n";
    std::cout << "Hardware Concurrency: " << hw_concurrency.load()
              << " Cores\n";
    std::cout << "Inference Workers: " << num_workers.load() << " Threads\n";
    std::cout << "IntraOp Threads/Worker: " << intra_op_threads.load() << "\n";
    std::cout << "Optimal Threads/Worker: " << optimal_intra_threads.load()
              << "\n";
    std::cout << "Inference Backend: " << inference_backend << " ("
              << model_precision << ")\n";
    std::cout << "Frame Size: " << frame_width.load() << "x"
              << frame_height.load() << "\n";
    std::cout << "Tensor Resolution: " << tensor_width.load() << "x"
              << tensor_height.load() << "\n";
    std::cout << "Total Time: " << duration << " ms\n";
    std::cout << "Frames Decoded: " << frames_decoded.load() << "\n";
    std::cout << "Frames Inferred: " << frames_inferred.load() << "\n";
    std::cout << "Frames Encoded: " << frames_encoded.load() << "\n";
    std::cout << "Average FPS: " << fps << "\n";
    std::cout << "Average Time to Frame (T2F): " << avg_t2f << " ms\n";
    std::cout << "Average Time to Conversion (TTC): " << avg_ttc << " ms\n";
    std::cout << "Average Time to Inference (TTI): " << avg_tti << " ms\n";
    std::cout << "================================\n\n";
  }

private:
  Metrics() = default;
  ~Metrics() = default;

  std::atomic<int> frame_width{0};
  std::atomic<int> frame_height{0};

  std::atomic<int> num_workers{0};
  std::atomic<int> hw_concurrency{0};

  std::atomic<int> frames_decoded{0};
  std::atomic<int> frames_inferred{0};
  std::atomic<int> frames_encoded{0};

  double total_time_to_frame{0};
  double total_time_to_conversion{0};
  double total_time_to_inference{0};

  std::chrono::steady_clock::time_point start_time;
  std::chrono::steady_clock::time_point end_time;
  std::mutex mtx;

  std::string inference_backend{"CPU"};
  std::string model_precision{"FP32"};
  std::atomic<int> tensor_width{0};
  std::atomic<int> tensor_height{0};
  std::atomic<int> intra_op_threads{0};
  std::atomic<int> optimal_intra_threads{0};
};
