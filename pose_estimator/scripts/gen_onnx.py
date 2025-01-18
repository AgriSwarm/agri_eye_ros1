import torch
from scripts.module import SixDRepNetModule

# モデルのロード
checkpoint_path = '/home/initial/catkin_ws/src/agri_eye_ros1/models/HPE.ckpt'  # 適宜パスを変更してください
model = SixDRepNetModule.load_from_checkpoint(checkpoint_path)
model.eval().cuda()

# ダミー入力を作成
dummy_input = torch.randn(1, 3, 224, 224).cuda()

# ONNX 形式でエクスポート
onnx_model_path = "sixdrepnet.onnx"
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_model_path,
    export_params=True,             # モデルパラメータも含める
    opset_version=11,               # 適切な ONNX opset バージョンを指定
    do_constant_folding=True,       # 定数畳み込みを有効化
    input_names=['input'],          # 入力ノードの名前
    output_names=['output'],        # 出力ノードの名前
    dynamic_axes={'input': {0: 'batch_size'},    # バッチサイズを動的に対応
                  'output': {0: 'batch_size'}}
)

print(f"ONNXモデルをエクスポートしました: {onnx_model_path}")