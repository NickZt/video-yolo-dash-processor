
Download model to model folder
python3 -c "from ultralytics import YOLO; model = YOLO('yolov8n-seg.pt'); model.export(format='onnx')" && mv yolov8n-seg.onnx model/

Compile the code

cd build && cmake .. && make -j$(nproc)


ffmpeg -f lavfi -i testsrc=duration=1:size=640x480:rate=10 -c:v libx264 -y test.mp4 && touch empty_init.mp4

cd build && ./video_processor ../empty_init.mp4 ../test.mp4 ../out_dir ../model/yolov8n-seg.onnx

yolo export model=yolov8n-seg.pt format=onnx opset=18

video_processor ~/init.dash ~/segment1.m4s output_dir/ ../model/yolov8n-seg.onnx

/video_processor$ cd build && make -j$(nproc) && ./video_processor "~/init.dash" "~/segment1.m4s" output_dir/ ../model/yolov8n-seg.onnx
[100%] Built target video_processor
[libx264 @ 0x5d67c45b4a40] using SAR=1/1
[libx264 @ 0x5d67c45b4a40] MB rate (102000000) > level limit (108000)
[libx264 @ 0x5d67c45b4a40] using cpu capabilities: MMX2 SSE2Fast SSSE3 SSE4.2 AVX FMA3 BMI2 AVX2
[libx264 @ 0x5d67c45b4a40] profile High, level 3.1, 4:2:0, 8-bit
[libx264 @ 0x5d67c45b4a40] 264 - core 163 r3060 5db6aa6 - H.264/MPEG-4 AVC codec - Copyleft 2003-2021 - http://www.videolan.org/x264.html - options: cabac=1 ref=3 deblock=1:0:0 analyse=0x3:0x113 me=hex subme=7 psy=1 psy_rd=1.00:0.00 mixed_ref=1 me_range=16 chroma_me=1 trellis=1 8x8dct=1 cqm=0 deadzone=21,11 fast_pskip=1 chroma_qp_offset=-2 threads=17 lookahead_threads=2 sliced_threads=0 nr=0 decimate=1 interlaced=0 bluray_compat=0 constrained_intra=0 bframes=3 b_pyramid=2 b_adapt=1 b_bias=0 direct=1 weightb=1 open_gop=0 weightp=2 keyint=250 keyint_min=25 scenecut=40 intra_refresh=0 rc_lookahead=40 rc=crf mbtree=1 crf=23.0 qcomp=0.60 qpmin=0 qpmax=69 qpstep=4 ip_ratio=1.40 aq=1:1.00
[mp4 @ 0x5d67c45b3fc0] Estimating the duration of the last packet in a fragment, consider setting the duration field in AVPacket instead.

=== Video Processing Metrics ===
Total Time: 9058 ms
Frames Decoded: 189
Frames Inferred: 189
Frames Encoded: 189
Average FPS: 20.8655
Average Time to Frame (T2F): 1.03262 ms
Average Time to Conversion (TTC): 0.281381 ms
================================

[libx264 @ 0x5d67c45b4a40] frame I:1     Avg QP:20.96  size: 35702
[libx264 @ 0x5d67c45b4a40] frame P:49    Avg QP:22.19  size:  2044
[libx264 @ 0x5d67c45b4a40] frame B:139   Avg QP:28.83  size:   253
[libx264 @ 0x5d67c45b4a40] consecutive B-frames:  0.5%  4.2%  0.0% 95.2%
[libx264 @ 0x5d67c45b4a40] mb I  I16..4: 10.0% 64.8% 25.1%
[libx264 @ 0x5d67c45b4a40] mb P  I16..4:  0.3%  0.7%  0.0%  P16..4: 15.4%  4.0%  2.8%  0.0%  0.0%    skip:76.9%
[libx264 @ 0x5d67c45b4a40] mb B  I16..4:  0.0%  0.0%  0.0%  B16..8: 10.7%  0.2%  0.0%  direct: 0.0%  skip:89.1%  L0:52.7% L1:46.0% BI: 1.3%
[libx264 @ 0x5d67c45b4a40] 8x8 transform intra:64.9% inter:77.6%
[libx264 @ 0x5d67c45b4a40] coded y,uvDC,uvAC intra: 48.9% 52.0% 34.9% inter: 1.4% 1.9% 0.1%
[libx264 @ 0x5d67c45b4a40] i16 v,h,dc,p: 48% 27%  9% 15%
[libx264 @ 0x5d67c45b4a40] i8 v,h,dc,ddl,ddr,vr,hd,vl,hu: 35% 11% 26%  3%  4% 10%  3%  4%  3%
[libx264 @ 0x5d67c45b4a40] i4 v,h,dc,ddl,ddr,vr,hd,vl,hu: 35% 14%  9%  5%  8% 15%  5%  6%  4%
[libx264 @ 0x5d67c45b4a40] i8c dc,h,v,p: 55% 13% 29%  3%
[libx264 @ 0x5d67c45b4a40] Weighted P-Frames: Y:0.0% UV:0.0%
[libx264 @ 0x5d67c45b4a40] ref P L0: 68.9% 10.1% 14.7%  6.3%
[libx264 @ 0x5d67c45b4a40] ref B L0: 91.4%  7.1%  1.5%
[libx264 @ 0x5d67c45b4a40] ref B L1: 97.1%  2.9%
[libx264 @ 0x5d67c45b4a40] kb/s:361.98
Processing completed successfully.
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