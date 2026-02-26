#include "VideoProcessor.h"
#include <filesystem>
#include <iostream>
#include <opencv2/core.hpp>

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
  //  If threads == 1, OpenCV will disable threading optimizations and run its
  // functions sequentially.
  cv::setNumThreads(1);

  std::map<std::string, std::string> args;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--", 0) == 0 && i + 1 < argc) {
      args[arg] = argv[++i];
    }
  }

  // Default to YOLO if engine is not explicitly specified
  if (args.find("--engine") == args.end()) {
    args["--engine"] = "yolo";
  }

  // Mandatory fields
  if (args.find("--media") == args.end() || args.find("--out") == args.end() ||
      args.find("--model") == args.end()) {
    std::cerr << "Usage: " << argv[0] << "\n"
              << "  --engine <yolo|dino> (default: yolo)\n"
              << "  --init <init_segment> (optional)\n"
              << "  --media <media_segment>\n"
              << "  --out <output_dir>\n"
              << "  --model <path_to_onnx_model>\n"
              << "  --prompt <\"text prompt\"> (required if engine is dino)\n"
              << std::endl;
    return 1;
  }

  if (args["--engine"] == "dino" && args.find("--prompt") == args.end()) {
    std::cerr
        << "Error: The Grounding DINO engine requires a --prompt parameter."
        << std::endl;
    return 1;
  }

  std::string initPath = args["--init"];
  std::string mediaPath = args["--media"];
  std::string outputDir = args["--out"];

  if (!fs::exists(outputDir)) {
    if (!fs::create_directories(outputDir)) {
      std::cerr << "Failed to create output directory: " << outputDir
                << std::endl;
      return 1;
    }
  }

  try {
    VideoProcessor vp(args);
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
