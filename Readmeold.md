
Download model to model folder
````
python3 -c "from ultralytics import YOLO; model = YOLO('yolov8n-seg.pt'); model.export(format='onnx')" && mv yolov8n-seg.onnx model/
````
python script to print ONNX model output shapes
````
python3 -c "import onnxruntime as ort; session = ort.InferenceSession('../model/yolov8n-seg.onnx'); print([o.shape for o in session.get_outputs()])"
````
Compile the code
````
cd build && cmake .. && make -j$(nproc)
````

````
ffmpeg -f lavfi -i testsrc=duration=1:size=640x480:rate=10 -c:v libx264 -y test.mp4 && touch empty_init.mp4
````
cd build && ./video_processor ../empty_init.mp4 ../test.mp4 ../out_dir ../model/yolov8n-seg.onnx

yolo export model=yolov8n-seg.pt format=onnx opset=18

video_processor ~/init.dash ~/segment1.m4s output_dir/ ../model/yolov8n-seg.onnx

````
cd build && make -j$(nproc) && mkdir -p output_dir7 && ./video_processor "~/init.dash" "~/segment1.m4s" output_dir7/ ../model/yolov8n-seg.onnx && ls -la output_dir7/

[  9%] Building CXX object CMakeFiles/video_processor.dir/src/VideoProcessor.cpp.o
[ 18%] Linking CXX executable video_processor
[100%] Built target video_processor
Detected obj (person) mask painted
Detected obj (person) mask painted
[dash @ 0x622f39f1f940] Opening '~/video_processor/output_dir/manifest.mpd.tmp' for writing
[libx264 @ 0x622f39eb66c0] frame I:1     Avg QP:21.23  size: 31741
[libx264 @ 0x622f39eb66c0] frame P:47    Avg QP:23.45  size:  1779
[libx264 @ 0x622f39eb66c0] frame B:141   Avg QP:32.03  size:   358
[libx264 @ 0x622f39eb66c0] consecutive B-frames:  0.5%  0.0%  0.0% 99.5%
[libx264 @ 0x622f39eb66c0] mb I  I16..4: 12.3% 65.2% 22.5%
[libx264 @ 0x622f39eb66c0] mb P  I16..4:  0.4%  0.9%  0.1%  P16..4:  9.9%  2.8%  2.3%  0.0%  0.0%    skip:83.7%
[libx264 @ 0x622f39eb66c0] mb B  I16..4:  0.0%  0.1%  0.0%  B16..8:  6.8%  0.9%  0.2%  direct: 0.2%  skip:91.7%  L0:50.0% L1:47.6% BI: 2.4%
[libx264 @ 0x622f39eb66c0] 8x8 transform intra:66.4% inter:49.0%
[libx264 @ 0x622f39eb66c0] coded y,uvDC,uvAC intra: 32.8% 37.9% 25.7% inter: 0.9% 1.1% 0.1%
[libx264 @ 0x622f39eb66c0] i16 v,h,dc,p: 49% 39%  6%  6%
[libx264 @ 0x622f39eb66c0] i8 v,h,dc,ddl,ddr,vr,hd,vl,hu: 35%  9% 42%  2%  2%  6%  1%  2%  2%
[libx264 @ 0x622f39eb66c0] i4 v,h,dc,ddl,ddr,vr,hd,vl,hu: 35% 17% 14%  5%  6% 11%  4%  5%  3%
[libx264 @ 0x622f39eb66c0] i8c dc,h,v,p: 52% 19% 26%  2%
[libx264 @ 0x622f39eb66c0] Weighted P-Frames: Y:0.0% UV:0.0%
[libx264 @ 0x622f39eb66c0] ref P L0: 71.2%  4.0% 14.7% 10.0%
[libx264 @ 0x622f39eb66c0] ref B L0: 78.7% 17.0%  4.2%
[libx264 @ 0x622f39eb66c0] ref B L1: 91.1%  8.9%
[libx264 @ 0x622f39eb66c0] kb/s:351.06
````

Before optimization start. 

````
=== Video Processing Metrics ===
Frame Size: 960x540
Total Time: 8258 ms
Frames Decoded: 189
Frames Inferred: 189
Frames Encoded: 189
Average FPS: 22.8869
Average Time to Frame (T2F): 1.07271 ms
Average Time to Conversion (TTC): 0.263274 ms
Average Time to Inference (TTI): 41.1597 ms
================================

Processing completed successfully.

Process finished with exit code 0

chunk-1.m4s  init.mp4  manifest.mpd
````
Zero-Copy YUV Masking

````

=== Video Processing Metrics ===
Frame Size: 960x540
Total Time: 8051 ms
Frames Decoded: 189
Frames Inferred: 189
Frames Encoded: 189
Average FPS: 23.4753
Average Time to Frame (T2F): 1.33232 ms
Average Time to Conversion (TTC): 0.468198 ms
Average Time to Inference (TTI): 42.1632 ms

````

Zero-Copy YUV Masking

````
=== Video Processing Metrics ===
Frame Size: 960x540
Total Time: 7992 ms
Frames Decoded: 189
Frames Inferred: 189
Frames Encoded: 189
Average FPS: 23.6486
Average Time to Frame (T2F): 1.49884 ms
Average Time to Conversion (TTC): 0.48923 ms
Average Time to Inference (TTI): 41.9236 ms

````

Multi-Inference Worker

in YOLO abstraction base class, it relies on shared mutable state blocks (e.g. m_image memory pools). This means we cannot simply share 1 YOLO instance across multiple threads natively.
Thread Pool: Spin up a pool of N inference threads (e.g., matching half your CPU core count)

````
=== Video Processing Metrics ===
Hardware Concurrency: 20 Cores
Inference Workers: 10 Threads
Frame Size: 960x540
Total Time: 17439 ms
Frames Decoded: 189
Frames Inferred: 189
Frames Encoded: 189
Average FPS: 10.8378
Average Time to Frame (T2F): 6.70182 ms
Average Time to Conversion (TTC): 4.53343 ms
Average Time to Inference (TTI): 890.229 ms
================================

````

This is because 10 independent ONNX sessions are currently defaulting to using all 10 CPU cores each, creating a 100-thread thrashing bottleneck.
session_options.SetIntraOpNumThreads(std::thread::hardware_concurrency() / 2);
in Ort::SessionOptions::SessionOptions

Then

added optimization session_options.SetIntraOpNumThreads(1);
session_options.SetInterOpNumThreads(1);
this limits every worker instance to use only 1 thread
Result is
````
=== Video Processing Metrics ===
Hardware Concurrency: 20 Cores
Inference Workers: 10 Threads
Frame Size: 960x540
Total Time: 6703 ms
Frames Decoded: 189
Frames Inferred: 189
Frames Encoded: 189
Average FPS: 28.1963
Average Time to Frame (T2F): 2.2161 ms
Average Time to Conversion (TTC): 1.19051 ms
Average Time to Inference (TTI): 336.523 ms
````
Better control for inference, inference used 10 threads, as in the previous example before using multiworker. Average frames per second: 23.6486, but now it is better controlled, and we have an average frames per second: 28.1963 on the same hardware

Disable implicit OpenCV multi-threading to prevent OS thread thrashing since we are already spawning N hardware-concurrent Inference Workers.
cv::setNumThreads(1);
If threads == 1, OpenCV will disable threading optimizations and run its
functions sequentially. This adds 1 frame per second )

````
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
````

Grounding DINO


python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='onnx-community/grounding-dino-tiny-ONNX', filename='onnx/model.onnx', local_dir='test_assets')"

mv test_assets/onnx/model.onnx test_assets/groundingdino.onnx && rm -rf test_assets/onnx && ./build/video_processor --engine dino --media test_assets/dash_720p/chunk-1.m4s --init test_assets/dash_720p/init.dash --out test_dino_output/ --model test_assets/groundingdino.onnx --prompt "white square ."

# Extract DASH chunks to simulate the live video stream ingest
cd test_assets/dash_4k && ffmpeg -re -i ../test_4k.mp4 -c:v copy -f dash -window_size 5 -extra_window_size 5 -seg_duration 2 -init_seg_name init.dash -media_seg_name chunk-\$Number\$.m4s manifest.mpd
cd ../dash_1080p && ffmpeg -re -i ../test_1080p.mp4 -c:v copy -f dash -window_size 5 -extra_window_size 5 -seg_duration 2 -init_seg_name init.dash -media_seg_name chunk-\$Number\$.m4s manifest.mpd
cd ../dash_720p && ffmpeg -re -i ../test_720p.mp4 -c:v copy -f dash -window_size 5 -extra_window_size 5 -seg_duration 2 -init_seg_name init.dash -media_seg_name chunk-\$Number\$.m4s manifest.mpd

ffmpeg -y -f lavfi -i testsrc=size=3840x2160:rate=30 -vcodec libx264 -preset ultrafast -t 10 -pix_fmt yuv420p test_assets/test_4k.mp4 && ffmpeg -y -f lavfi -i testsrc=size=1920x1080:rate=30 -vcodec libx264 -preset ultrafast -t 10 -pix_fmt yuv420p test_assets/test_1080p.mp4 && ffmpeg -y -f lavfi -i testsrc=size=1280x720:rate=30 -vcodec libx264 -preset ultrafast -t 10 -pix_fmt yuv420p test_assets/test_720p.mp4

# Extract DASH chunks to simulate the live video stream ingest
cd test_assets/dash_4k && ffmpeg -re -i ../test_4k.mp4 -c:v copy -f dash -window_size 5 -extra_window_size 5 -seg_duration 2 -init_seg_name init.dash -media_seg_name chunk-\$Number\$.m4s manifest.mpd
cd ../dash_1080p && ffmpeg -re -i ../test_1080p.mp4 -c:v copy -f dash -window_size 5 -extra_window_size 5 -seg_duration 2 -init_seg_name init.dash -media_seg_name chunk-\$Number\$.m4s manifest.mpd
cd ../dash_720p && ffmpeg -re -i ../test_720p.mp4 -c:v copy -f dash -window_size 5 -extra_window_size 5 -seg_duration 2 -init_seg_name init.dash -media_seg_name chunk-\$Number\$.m4s manifest.mpd

find test_assets -name "*.onnx"


# Run Dino Run


mv test_assets/onnx/model.onnx test_assets/groundingdino.onnx && rm -rf test_assets/onnx && ./build/video_processor --engine dino --media test_assets/dash_720p/chunk-1.m4s --init test_assets/dash_720p/init.dash --out test_dino_output/ --model test_assets/groundingdino.onnx --prompt "white square ."


./build/video_processor --engine dino --media ../segment1.m4s --init ../init.dash --out test_dino_output/ --model test_assets/groundingdino.onnx --prompt "person . tag ."

./build/video_processor --engine dino --media ../segment1.m4s --init ../init.dash --out test_dino_output/ --model test_assets/groundingdino.onnx --prompt "person . tag ."

video_processor$ cmake --build build -j$(nproc) && ./build/video_processor --engine dino --media ../segment1.m4s --init ../init.dash --out test_dino_output/ --model test_assets/groundingdino.onnx --prompt "person . bag ."


=== Video Processing Metrics ===
Hardware Concurrency: 20 Cores
Inference Workers: 10 Threads
Frame Size: 960x540
Total Time: 614572 ms
Frames Decoded: 189
Frames Inferred: 189
Frames Encoded: 189
Average FPS: 0.307531
Average Time to Frame (T2F): 4.03664 ms
Average Time to Conversion (TTC): 5.5615 ms
Average Time to Inference (TTI): 32095.4 ms
================================

# Run Dino Run 

New metrics data

=== Video Processing Metrics ===
Hardware Concurrency: 20 Cores
Inference Workers: 10 Threads
IntraOp Threads/Worker: 1
Inference Backend: ONNXRuntime CPU (FP32)
Frame Size: 960x540
Tensor Resolution: 800x800
Total Time: 526339 ms
Frames Decoded: 189
Frames Inferred: 189
Frames Encoded: 189
Average FPS: 0.359084
Average Time to Frame (T2F): 9.754 ms
Average Time to Conversion (TTC): 2.5109 ms
Average Time to Inference (TTI): 27442.9 ms
================================


=== Video Processing Metrics ===
Hardware Concurrency: 20 Cores
Inference Workers: 2 Threads
IntraOp Threads/Worker: 10
Optimal Threads/Worker: 5
Inference Backend: ONNXRuntime CPU (FP32)
Frame Size: 960x540
Tensor Resolution: 800x800
Total Time: 569683 ms
Frames Decoded: 189
Frames Inferred: 189
Frames Encoded: 189
Average FPS: 0.331763
Average Time to Frame (T2F): 5.81481 ms
Average Time to Conversion (TTC): 4.09431 ms
Average Time to Inference (TTI): 6010.53 ms

=== Video Processing Metrics ===
Hardware Concurrency: 20 Cores
Inference Workers: 2 Threads
IntraOp Threads/Worker: 10
Optimal Threads/Worker: 5
Inference Backend: ONNXRuntime CPU (FP32)
Frame Size: 960x540
Tensor Resolution: 800x800
Total Time: 451489 ms
Frames Decoded: 189
Frames Inferred: 189
Frames Encoded: 189
Average FPS: 0.418615
Average Time to Frame (T2F): 2.81651 ms
Average Time to Conversion (TTC): 2.2257 ms
Average Time to Inference (TTI): 4760.82 ms

