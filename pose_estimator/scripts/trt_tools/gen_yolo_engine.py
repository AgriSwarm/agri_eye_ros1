import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="YOLOモデルのエクスポートスクリプト")
    parser.add_argument(
        '--precision',
        type=str,
        default='fp16',
        choices=['fp16', 'int', 'fp32'],
        help='エクスポートするモデルの精度タイプ。fp16 または int を指定。'
    )
    args = parser.parse_args()

    # モデルのチェックポイントパスを指定
    yolo_checkpoint = '/home/initial/catkin_ws/src/agri_eye_ros1/models/YOLOv8.pt'
    yolo_model = YOLO(yolo_checkpoint)

    # 精度に応じたエクスポート設定
    export_kwargs = {"format": "engine"}

    if args.precision == 'fp32':
        # FP32精度でエクスポートする場合
        export_kwargs["half"] = False
        export_kwargs["int8"] = False
        print("FP32精度でエクスポートします。")
    elif args.precision == 'fp16':
        # FP16精度でエクスポートする場合
        export_kwargs["half"] = True
        export_kwargs["int8"] = False
        print("FP16精度でエクスポートします。")
    elif args.precision == 'int':
        export_kwargs["half"] = False
        export_kwargs["int8"] = True
        print("INT8精度でエクスポートします。")
    else:
        print("精度指定が不正です。")

    # モデルをエクスポート
    yolo_model.export(**export_kwargs)

if __name__ == "__main__":
    main()
