import onnxruntime as ort
import numpy as np

onnx_model_path = "/home/initial/catkin_ws/src/agri_eye_ros1/models/sixdrepnet.onnx"

ort_session = ort.InferenceSession(onnx_model_path)
dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
outputs = ort_session.run(None, {'input': dummy_input})
print("ONNX Runtime 出力:", outputs)
