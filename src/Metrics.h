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

  std::atomic<int> frames_decoded{0};
  std::atomic<int> frames_inferred{0};
  std::atomic<int> frames_encoded{0};

  double total_time_to_frame{0};
  double total_time_to_conversion{0};
  double total_time_to_inference{0};

  std::chrono::steady_clock::time_point start_time;
  std::chrono::steady_clock::time_point end_time;
  std::mutex mtx;
};
