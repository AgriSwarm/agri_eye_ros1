import torch
from torch2trt import torch2trt
from scripts.module import SixDRepNetModule

# モデルのロード
checkpoint_path = '/home/initial/catkin_ws/src/agri_eye_ros1/models/HPE.ckpt'  # 適宜パスを変更してください
model = SixDRepNetModule.load_from_checkpoint(checkpoint_path)
model.eval().cuda()

# ダミー入力を作成
dummy_input = torch.randn(1, 3, 224, 224).cuda()

# TensorRTエンジンに変換
model_trt = torch2trt(model, [dummy_input])

# エンジンの保存
torch.save(model_trt.state_dict(), 'sixdrepnet_trt.engine')
print("TensorRTエンジンを保存しました。")