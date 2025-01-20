import torch
from scripts.module import SixDRepNetModule

# モデルのロード
checkpoint_path = '/home/tomoking/catkin_ws/src/agri_eye_ros1/models/sixdrepnet.ckpt'  # 適宜パスを変更してください
model = SixDRepNetModule.load_from_checkpoint(checkpoint_path)
model.eval().cuda()

# ダミー入力を作成
dummy_input = torch.randn(1, 3, 224, 224).cuda()

# ONNX 形式でエクスポート
onnx_model_path = "/home/tomoking/catkin_ws/src/agri_eye_ros1/models/sixdrepnet.onnx"

torch.onnx.export(
    model, 
    dummy_input, 
    onnx_model_path,
    export_params=True,
    opset_version=12,  # バージョン変更
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"ONNXモデルをエクスポートしました: {onnx_model_path}")

import onnx
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
print("ONNXモデルは正しい形式です。")