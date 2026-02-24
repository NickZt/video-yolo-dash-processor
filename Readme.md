
Download model to model folder
python3 -c "from ultralytics import YOLO; model = YOLO('yolov8n-seg.pt'); model.export(format='onnx')" && mv yolov8n-seg.onnx model/

Compile the code

cd build && cmake .. && make -j$(nproc)


ffmpeg -f lavfi -i testsrc=duration=1:size=640x480:rate=10 -c:v libx264 -y test.mp4 && touch empty_init.mp4

cd build && ./video_processor ../empty_init.mp4 ../test.mp4 ../out_dir ../model/yolov8n-seg.onnx

yolo export model=yolov8n-seg.pt format=onnx opset=18

video_processor ~/init.dash ~/segment1.m4s output_dir/ ../model/yolov8n-seg.onnx

cd build && make -j$(nproc) && mkdir -p output_dir7 && ./video_processor "~/init.dash" "~/segment1.m4s" output_dir7/ ../model/yolov8n-seg.onnx && ls -la output_dir7/
[  9%] Building CXX object CMakeFiles/video_processor.dir/src/VideoProcessor.cpp.o
[ 18%] Linking CXX executable video_processor
[100%] Built target video_processor
[libx264 @ 0x60315c483380] using SAR=1/1
[libx264 @ 0x60315c483380] MB rate (102000000) > level limit (108000)
[libx264 @ 0x60315c483380] using cpu capabilities: MMX2 SSE2Fast SSSE3 SSE4.2 AVX FMA3 BMI2 AVX2
[libx264 @ 0x60315c483380] profile High, level 3.1, 4:2:0, 8-bit
[libx264 @ 0x60315c483380] 264 - core 163 r3060 5db6aa6 - H.264/MPEG-4 AVC codec - Copyleft 2003-2021 - http://www.videolan.org/x264.html - options: cabac=1 ref=3 deblock=1:0:0 analyse=0x3:0x113 me=hex subme=7 psy=1 psy_rd=1.00:0.00 mixed_ref=1 me_range=16 chroma_me=1 trellis=1 8x8dct=1 cqm=0 deadzone=21,11 fast_pskip=1 chroma_qp_offset=-2 threads=17 lookahead_threads=2 sliced_threads=0 nr=0 decimate=1 interlaced=0 bluray_compat=0 constrained_intra=0 bframes=3 b_pyramid=2 b_adapt=1 b_bias=0 direct=1 weightb=1 open_gop=0 weightp=2 keyint=250 keyint_min=25 scenecut=40 intra_refresh=0 rc_lookahead=40 rc=crf mbtree=1 crf=23.0 qcomp=0.60 qpmin=0 qpmax=69 qpstep=4 ip_ratio=1.40 aq=1:1.00
[dash @ 0x60315ba04440] No bit rate set for stream 0
[dash @ 0x60315ba04440] Opening 'output_dir7/init.mp4' for writing
Person Detected! Bounding box area: 209139
Valid BBox and Non-Empty Mask! Mask cols: 483, rows: 433
SUCCESS: Mask painted on frame!

Person Detected! Bounding box area: 210438
Valid BBox and Non-Empty Mask! Mask cols: 486, rows: 433
SUCCESS: Mask painted on frame!
[dash @ 0x60315ba04440] Opening 'output_dir7/manifest.mpd.tmp' for writing

=== Video Processing Metrics ===
Frame Size: 960x540
Total Time: 7538 ms
Frames Decoded: 189
Frames Inferred: 189
Frames Encoded: 189
Average FPS: 25.073
Average Time to Frame (T2F): 1.00056 ms
Average Time to Conversion (TTC): 0.255791 ms
Average Time to Inference (TTI): 37.4914 ms
================================

[libx264 @ 0x60315c483380] frame I:1     Avg QP:21.23  size: 31741
[libx264 @ 0x60315c483380] frame P:47    Avg QP:23.45  size:  1779
[libx264 @ 0x60315c483380] frame B:141   Avg QP:32.03  size:   358
[libx264 @ 0x60315c483380] consecutive B-frames:  0.5%  0.0%  0.0% 99.5%
[libx264 @ 0x60315c483380] mb I  I16..4: 12.3% 65.2% 22.5%
[libx264 @ 0x60315c483380] mb P  I16..4:  0.4%  0.9%  0.1%  P16..4:  9.9%  2.8%  2.3%  0.0%  0.0%    skip:83.7%
[libx264 @ 0x60315c483380] mb B  I16..4:  0.0%  0.1%  0.0%  B16..8:  6.8%  0.9%  0.2%  direct: 0.2%  skip:91.7%  L0:50.0% L1:47.6% BI: 2.4%
[libx264 @ 0x60315c483380] 8x8 transform intra:66.4% inter:49.0%
[libx264 @ 0x60315c483380] coded y,uvDC,uvAC intra: 32.8% 37.9% 25.7% inter: 0.9% 1.1% 0.1%
[libx264 @ 0x60315c483380] i16 v,h,dc,p: 49% 39%  6%  6%
[libx264 @ 0x60315c483380] i8 v,h,dc,ddl,ddr,vr,hd,vl,hu: 35%  9% 42%  2%  2%  6%  1%  2%  2%
[libx264 @ 0x60315c483380] i4 v,h,dc,ddl,ddr,vr,hd,vl,hu: 35% 17% 14%  5%  6% 11%  4%  5%  3%
[libx264 @ 0x60315c483380] i8c dc,h,v,p: 52% 19% 26%  2%
[libx264 @ 0x60315c483380] Weighted P-Frames: Y:0.0% UV:0.0%
[libx264 @ 0x60315c483380] ref P L0: 71.2%  4.0% 14.7% 10.0%
[libx264 @ 0x60315c483380] ref B L0: 78.7% 17.0%  4.2%
[libx264 @ 0x60315c483380] ref B L1: 91.1%  8.9%
[libx264 @ 0x60315c483380] kb/s:351.06
Processing completed successfully.
chunk-1.m4s  init.mp4  manifest.mpd

Next step
Demux
↓
Decode (libavcodec)
↓
Frame (YUV420p AVFrame)
↓
Preprocess Worker
↓
ONNX Runtime
↓
Mask Apply (in Y plane)
↓
Encode
↓
Mux (fragmented mp4)


python script to print ONNX model output shapes

python3 -c "import onnxruntime as ort; session = ort.InferenceSession('../model/yolov8n-seg.onnx'); print([o.shape for o in session.get_outputs()])"