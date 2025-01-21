import os
import json
import math
import cv2
import numpy as np
import scripts.utils as utils

from scipy.spatial.transform import Rotation

def main():
    data_dir = "/home/tomoking/Downloads/mito_datasets"
    json_path = os.path.join(data_dir, "dataset.json")
    with open(json_path, "r") as f:
        dataset = json.load(f)

    # 例: 3つだけ処理
    output_num = 30
    dataset = dataset[20:20+output_num]

    for entry in dataset:
        image_path = os.path.join(data_dir, entry["image_path"])
        if not os.path.exists(image_path):
            print(f"画像が見つかりません: {image_path}")
            continue

        # 画像読み込み
        img = cv2.imread(image_path)
        if img is None:
            continue
        h, w, _ = img.shape

        euler = entry["euler"]
        normal_vec = entry["normal_vector"]

        print(f"normal_vec: {normal_vec}")

        ex = float(euler["x"]) # pitch
        ey = float(euler["y"]) # yaw
        ez = float(euler["z"]) # roll

        print(f"ex: {ex}, ey: {ey}, ez: {ez}")

        r = Rotation.from_euler("xyz", [ex, ey, ez], degrees=True)
        R = r.as_matrix()
        print(f"R: {R}")
        # angles = r.as_euler("xyz", degrees=False)

        r = Rotation.from_matrix(R)
        angles = r.as_euler("xyz", degrees=True)
        z_axis = r.apply([0, 0, 1])
        print(f"angles: {angles}")
        print(f"z_axis: {z_axis}")

        # 法線ベクトル
        nx = -float(normal_vec["x"])
        ny = float(normal_vec["y"])
        nz = float(normal_vec["z"])

        # 画像中心
        center_x = w // 2
        center_y = h // 2

        # ============== 法線ベクトル 2D描画 ==============
        norm_xy = math.sqrt(nx*nx + ny*ny)

        # print(f"norm_xy: {norm_xy}")
        # print(f"nx2d: {nx/norm_xy}, ny2d: {ny/norm_xy}")
        if norm_xy > 1e-6:
            nx2d = nx
            ny2d = ny
        else:
            nx2d = 0.0
            ny2d = 0.0

        arrow_length = 100
        end_x = int(center_x + arrow_length * nx2d)
        end_y = int(center_y + arrow_length * ny2d)

        # 赤い矢印
        # print(f"nx2d: {nx2d}, ny2d: {ny2d}")
        cv2.arrowedLine(img, (center_x, center_y), (end_x, end_y),
                        (0, 0, 255), 3, tipLength=0.3)
        # ファイル名を表示
        cv2.putText(img, os.path.basename(image_path), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # ============== Euler角を用いた軸描画 ==============
        # ex = math.radians(ex) # pitch
        # ey = math.radians(ey) # yaw
        # ez = math.radians(ez) # roll
        utils.draw_axis(img, ex, ey, ez, center_x, center_y, size=80)
        # utils.draw_axis_from_R(img, R, center_x, center_y, size=80)

        # ウィンドウ表示
        cv2.imshow("result", img)

        # *重要* ウィンドウが閉じられたか / キーが押されたかをループで監視
        while True:
            # 10msごとにキー入力を待つ (0xFF でマスクして下位8bitを取得)
            key = cv2.waitKey(10) & 0xFF

            # [Esc] (ASCII=27) または 'q' で強制的に終了したい場合
            if key == 27 or key == ord('q'):
                # もし「ここで全体を終了」したいなら:
                # cv2.destroyAllWindows()
                # return  # main()を抜ける
                # 「続けて次の画像を表示」なら breakだけ
                break

            # ウィンドウが閉じられたかをチェック
            # getWindowProperty は閉じられると -1 が返る
            if cv2.getWindowProperty("result", cv2.WND_PROP_VISIBLE) < 1:
                # ユーザーがウィンドウを閉じた場合は break
                break
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
