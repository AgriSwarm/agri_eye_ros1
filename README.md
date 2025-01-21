# Agri Eye ROS1

### TensorRT Optimization for YOLOv8


```bash
python3 trt_tools/gen_yolo_engine.py --precision fp32 # or fp16, int8
```

### TensorRT Optimization for 6DRepNet

pytorch to onnx
```bash
python3 trt_tools/gen_sixedrepnet_onnx.py
```
onnx to tensorrt engine
```bash
/usr/src/tensorrt/bin/trtexec --onnx=sixdrepnet.onnx --saveEngine=sixdrepnet_fp32.engine
```
fp16
```bash
/usr/src/tensorrt/bin/trtexec --onnx=sixdrepnet.onnx --saveEngine=sixdrepnet_fp16.engine --fp16
```
int8
```bash
/usr/src/tensorrt/bin/trtexec --onnx=sixdrepnet.onnx --saveEngine=sixdrepnet_int8.engine --int8
```