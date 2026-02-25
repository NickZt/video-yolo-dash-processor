#include "VideoProcessor.h"
#include <filesystem>
#include <iostream>
#include <opencv2/core.hpp>

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
  //  If threads == 1, OpenCV will disable threading optimizations and run its
  // functions sequentially.
 cv::setNumThreads(1);

  if (argc < 5) {
    std::cerr << "Usage: " << argv[0]
              << " <init_segment> <media_segment> <output_dir> <model_path>"
              << std::endl;
    return 1;
  }

  std::string initPath = argv[1];
  std::string mediaPath = argv[2];
  std::string outputDir = argv[3];
  std::string modelPath = argv[4];

  if (!fs::exists(outputDir)) {
    if (!fs::create_directories(outputDir)) {
      std::cerr << "Failed to create output directory: " << outputDir
                << std::endl;
      return 1;
    }
  }

  try {
    VideoProcessor vp(modelPath);
    if (vp.processConfig(initPath, mediaPath, outputDir)) {
      std::cout << "Processing completed successfully." << std::endl;
      return 0;
    } else {
      std::cerr << "Processing failed." << std::endl;
      return 1;
    }
  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }
}
