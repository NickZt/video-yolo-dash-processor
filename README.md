# Zero-Copy Video Inference Processor

A high-performance C++ video processing pipeline designed to strip out the latency of traditional Machine Learning abstractions. This repository demonstrates a highly concurrent, zero-copy architecture that natively binds YUV hardware decoders directly to ONNX Runtime computer vision models.

By disabling python abstractions and implementing custom Multi-Threaded pooling and synchronization strategies, this processing node drastically scales performance on edge/local devices, elevating throughput from ~10 FPS to **29 FPS** on a CPU.

## Key Features

- **Zero-Copy Memory Mapping**: Bypasses expensive RGB to YUV `sws_scale` pixel conversions by mapping the YOLO output masks directly onto the hardware Y-plane natively.
- **Asynchronous Worker Pool**: Spawns concurrent `std::thread` instances matched to your hardware concurrency count, allocating an independent `YOLO_Segment` ONNX Model to each worker.
- **Expected_PTS Encoding Synchronization**: Forces out-of-order asynchronous inference frames into a strictly monotonic H.264 Muxer buffer, guaranteeing flawless, glitch-free FFmpeg DASH streaming playback.
- **Defeating CPU Cache Thrashing**: Eliminates implicit context-switching latency by explicitly destroying internal thread pools native to OpenCV and the ONNX Runtime engine. 
- **Vision Transformer (ViT) Support**: Native integration for Grounding DINO open-set object detection with BERT-style text tokenization.
- **Dynamic INT8 Quantization**: On-the-fly execution graph quantization natively yielding up to a 24% CPU latency speedup for Transformer models without memory arena corruption.

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

And running the massive Grounding DINO vision transformer model leveraging Dynamic INT8 Quantization:

```
=== Video Processing Metrics ===
Hardware Concurrency: 20 Cores
Inference Workers: 2 Threads
IntraOp Threads/Worker: 10
Optimal Threads/Worker: 5
Inference Backend: ONNXRuntime CPU (FP32)
Frame Size: 960x540
Tensor Resolution: 800x800
Total Time: 23245 ms
Frames Decoded: 11
Frames Inferred: 10
Frames Encoded: 10
Average FPS: 0.4302
Average Time to Frame (T2F): 2.34479 ms
Average Time to Conversion (TTC): 0.647118 ms
Average Time to Inference (TTI): 4628.88 ms
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

### Grounding DINO Model

To download and prepare the open-set Grounding DINO model natively for ONNX, you can use the HuggingFace CLI:

```bash
mkdir test_assets
python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='onnx-community/grounding-dino-tiny-ONNX', filename='onnx/model.onnx', local_dir='test_assets')"
mv test_assets/onnx/model.onnx test_assets/groundingdino.onnx && rm -rf test_assets/onnx
```

To optimize the Grounding DINO model for CPU execution, you can dynamically quantize it to INT8 using the provided script:

```bash
python3 export_gdino_1_5.py
```
This generates `test_assets/groundingdino_int8.onnx`, natively slicing inference latency by ~24%.

## Usage

The processor supports multiple execution engines (`yolo` or `dino`), initialized by explicit CLI parameters. It expects a segmented video sequence (like DASH formatting) and an ONNX model file path.

```bash
./video_processor --engine <yolo|dino> --media <media_segment> --out <output_dir> --model <path_to_onnx_model> [options]
```

**Options:**
- `--engine <yolo|dino>`: Specifies the inference engine (default is `yolo`).
- `--init <init_segment>`: Optional initialization segment for the DASH stream.
- `--prompt "<text>"`: The text prompt (Required if using the `dino` engine, format: `"person . bag ."`).
- `--checkframes <count>`: Optional bounding limit for testing/benchmarking to terminate the pipeline early.
- `--optimize <1|0>`: Optional aggressive graph layout optimization (Warning: may crash on some Transformer architectures).

**YOLO Example:**
```bash
./video_processor --engine yolo --init init.dash --media segment1.m4s --out output_dir/ --model yolov8n-seg.onnx
```

**Grounding DINO INT8 Example:**
```bash
./video_processor --engine dino --init init.dash --media segment1.m4s --out test_dino_output/ --model test_assets/groundingdino_int8.onnx --prompt "person . bag ."
```

The output will automatically bundle and format into a playable, highly-optimized `.mpd` (Media Presentation Description) and `.m4s` DASH sequence.

## The FogAI Ecosystem

This repository isn't a standalone toy—it is a dedicated testbed. It is actively used to rigorously stress-test specific computer vision models, engine builds, and optimization patterns before they are promoted to the FogAI core. If a strategy (like Zero-Copy hardware mapping) can't survive here at 29 FPS, it has no business being inside an industrial autonomous nervous system.

## Acknowledgments

Special thanks to the following open-source projects and developers whose foundational work made this optimized pipeline possible:

* **[taifyang/yolo-inference](https://github.com/taifyang/yolo-inference)**: For providing an excellent foundation and abstraction API for ONNX/TensorRT bounding-box and segmentation extraction in C++. It extensively influenced the YOLO tensor integrations used here.
* **[FFmpeg](https://ffmpeg.org/)**: For the unparalleled hardware decoding and H.264 DASH payload mapping backbone.
* **[OpenCV](https://opencv.org/)**: For matrix manipulations and masking layers.
* **[ONNX Runtime](https://onnxruntime.ai/)**: The primary Machine Learning inference engine driving the zero-copy node.

## License

This project and its original source logic are provided under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** License. See the `LICENSE` file for full disclosure.
