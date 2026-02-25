# Zero-Copy Video Inference Processor

A high-performance C++ video processing pipeline designed to strip out the latency of traditional Machine Learning abstractions. This repository demonstrates a highly concurrent, zero-copy architecture that natively binds YUV hardware decoders directly to ONNX Runtime computer vision models.

By disabling python abstractions and implementing custom Multi-Threaded pooling and synchronization strategies, this processing node drastically scales performance on edge/local devices, elevating throughput from ~10 FPS to **29 FPS** on a CPU.

## Key Features

- **Zero-Copy Memory Mapping**: Bypasses expensive RGB to YUV `sws_scale` pixel conversions by mapping the YOLO output masks directly onto the hardware Y-plane natively.
- **Asynchronous Worker Pool**: Spawns concurrent `std::thread` instances matched to your hardware concurrency count, allocating an independent `YOLO_Segment` ONNX Model to each worker.
- **Expected_PTS Encoding Synchronization**: Forces out-of-order asynchronous inference frames into a strictly monotonic H.264 Muxer buffer, guaranteeing flawless, glitch-free FFmpeg DASH streaming playback.
- **Defeating CPU Cache Thrashing**: Eliminates implicit context-switching latency by explicitly destroying internal thread pools native to OpenCV and the ONNX Runtime engine. 

## Performance Metrics

Here is a performance capture running the processor on a standard 20-Core (10 worker) CPU edge architecture over a complex H.264 stream mapping segmented `.m4s` DASH inputs against a `yolov8n-seg.onnx` network:

```
=== Video Processing Metrics ===
Hardware Concurrency: 20 Cores
Inference Workers: 10 Threads
Frame Size: 960x540
Total Time: 6532 ms
Frames Decoded: 189
Frames Inferred: 189
Frames Encoded: 189
Average FPS: 28.9345
Average Time to Frame (T2F): 3.08831 ms
Average Time to Conversion (TTC): 1.702 ms
Average Time to Inference (TTI): 329.949 ms
================================
```

## Dependencies

- **FFmpeg** (`libavcodec`, `libavformat`, `libswscale`, `libavutil`)
- **OpenCV** (Core & Imgproc modules)
- **ONNX Runtime** (Vanilla C++ Backend)

## Building the Project

Ensure you have CMake installed and the dependencies properly linked in your system paths.

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

## Quick Start (Model Download)

Before running the processor, you will need a compatible ONNX segmentation model. You can download the standard YOLOv8n-Seg model natively formatted for ONNX Runtime directly from popular repositories rather than exporting it via Python:

```bash
mkdir model && cd model
wget https://github.com/NickZt/video-yolo-dash-processor/releases/download/v1.0/yolov8n-seg.onnx
cd ..
```
*(Alternatively, you can export any YOLOv8 segmentation PyTorch model to `.onnx` if you have a local python environment).*

## Usage

The processor expects a segmented video sequence (like DASH formatting) and an ONNX model file path.

```bash
./video_processor <init_segment> <media_segment> <output_dir> <model_path>
```

**Example:**
```bash
./video_processor init.dash segment1.m4s output_dir/ yolov8n-seg.onnx
```

The output will automatically bundle and format into a playable, highly-optimized `.mpd` (Media Presentation Description) and `.m4s` DASH sequence.

## What's Next?

Now that the core zero-copy, multi-threaded C++ inference pipeline is stable at 29 FPS, the foundation is set for heavier, state-of-the-art workloads. 

**Coming next to the roster: Grounding DINO integration.** Stay tuned!
