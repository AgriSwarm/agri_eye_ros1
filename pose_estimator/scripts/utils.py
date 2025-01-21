import os
import math
from math import cos, sin

import numpy as np
import torch
#from torch.utils.serialization import load_lua
import scipy.io as sio
import cv2
from scipy.spatial.transform import Rotation


def plot_pose_cube(img, roll, pitch, yaw, tdx=None, tdy=None, size=150.):
    # Input is a cv2 image
    # pose_params: (pitch, yaw, roll, tdx, tdy)
    # Where (tdx, tdy) is the translation of the face.
    # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]

    # p = pitch * np.pi / 180
    # y = -(yaw * np.pi / 180)
    # r = roll * np.pi / 180
    p = pitch
    y = yaw
    r = roll
    
    if tdx != None and tdy != None:
        face_x = tdx - 0.50 * size 
        face_y = tdy - 0.50 * size

    else:
        height, width = img.shape[:2]
        face_x = width / 2 - 0.5 * size
        face_y = height / 2 - 0.5 * size

    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y 
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
    x3 = size * (sin(y)) + face_x
    y3 = size * (-cos(y) * sin(p)) + face_y

    # Draw base in red
    cv2.line(img, (int(face_x), int(face_y)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(face_x), int(face_y)), (int(x2),int(y2)),(0,0,255),3)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x1-face_x),int(y2+y1-face_y)),(0,0,255),3)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x2-face_x),int(y1+y2-face_y)),(0,0,255),3)
    # Draw pillars in blue
    cv2.line(img, (int(face_x), int(face_y)), (int(x3),int(y3)),(255,0,0),2)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x3-face_x),int(y1+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x3-face_x),int(y2+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2+x1-face_x),int(y2+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(255,0,0),2)
    # Draw top in green
    cv2.line(img, (int(x3+x1-face_x),int(y3+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x2+x3-face_x),int(y2+y3-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x1-face_x),int(y3+y1-face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x2-face_x),int(y3+y2-face_y)),(0,255,0),2)

    return img

def draw_axis_from_rotation_matrix(img, R, tdx=None, tdy=None, size=100):
    """
    回転行列から三軸を描画する関数
    
    Parameters:
    img: 描画対象の画像
    R: 3x3の回転行列
    tdx: x座標の中心点（Noneの場合は画像の中心）
    tdy: y座標の中心点（Noneの場合は画像の中心）
    size: 軸の長さ
    
    Returns:
    軸が描画された画像
    """
    
    # 画像の中心を取得
    if tdx is None or tdy is None:
        height, width = img.shape[:2]
        tdx = width // 2
        tdy = height // 2

    # 座標軸の方向ベクトル
    axis = np.array([
        [size, 0, 0],  # X軸
        [0, size, 0],  # Y軸
        [0, 0, size]   # Z軸
    ])

    # 回転行列を適用して2D座標に変換
    points_2d = []
    for p in axis:
        # 回転を適用
        p_rotated = R.dot(p)
        
        # 単純な射影（Z成分は無視）
        x = p_rotated[0] + tdx
        y = p_rotated[1] + tdy
        
        points_2d.append((int(x), int(y)))

    # 軸の描画
    # X軸 (赤)
    cv2.line(img, (int(tdx), int(tdy)), points_2d[0], (0,0,255), 4)
    # Y軸 (緑)
    cv2.line(img, (int(tdx), int(tdy)), points_2d[1], (0,255,0), 4)
    # Z軸 (青)
    cv2.line(img, (int(tdx), int(tdy)), points_2d[2], (255,0,0), 4)

    return img

def draw_axis(img, pitch, yaw, roll, tdx=None, tdy=None, size=100):
    """
    左手系, ワールド座標系における回転行列から三軸を描画する関数
    """
    r = Rotation.from_euler("xyz", [pitch, yaw, roll], degrees=True)
    R = r.as_matrix()

    # print(f"draw_axis R: {R}")
    x_axis_3d = np.array([ size,   0.0,   0.0])
    y_axis_3d = np.array([  0.0,  size,   0.0])
    z_axis_3d = np.array([  0.0,   0.0,  size])
    x_rot = R @ x_axis_3d
    y_rot = R @ y_axis_3d
    z_rot = R @ z_axis_3d

    if tdx is None:
        tdx = img.shape[1] / 2.0
    if tdy is None:
        tdy = img.shape[0] / 2.0

    x_x = int(tdx - x_rot[0])
    x_y = int(tdy + x_rot[1])
    y_x = int(tdx - y_rot[0])
    y_y = int(tdy + y_rot[1])
    z_x = int(tdx - z_rot[0])
    z_y = int(tdy + z_rot[1])
    cx = int(tdx)
    cy = int(tdy)
    cv2.line(img, (cx, cy), (x_x, x_y), (0, 0, 255), 2)
    cv2.line(img, (cx, cy), (y_x, y_y), (0, 255, 0), 2)
    cv2.line(img, (cx, cy), (z_x, z_y), (255, 0, 0), 2)

    return img

def draw_axis_from_R(img, R, tdx=None, tdy=None, size=100):
    """
    回転行列から三軸を描画する関数
    
    Parameters:
    img: 描画対象の画像
    R: 3x3の回転行列
    tdx: x座標の中心点（Noneの場合は画像の中心）
    tdy: y座標の中心点（Noneの場合は画像の中心）
    size: 軸の長さ
    
    Returns:
    軸が描画された画像
    """

    x_axis_3d = np.array([ size,   0.0,   0.0])
    y_axis_3d = np.array([  0.0,  size,   0.0])
    z_axis_3d = np.array([  0.0,   0.0,  size])
    x_rot = R @ x_axis_3d
    y_rot = R @ y_axis_3d
    z_rot = R @ z_axis_3d

    if tdx is None:
        tdx = img.shape[1] / 2.0
    if tdy is None:
        tdy = img.shape[0] / 2.0

    x_x = int(tdx - x_rot[0])
    x_y = int(tdy + x_rot[1])
    y_x = int(tdx - y_rot[0])
    y_y = int(tdy + y_rot[1])
    z_x = int(tdx - z_rot[0])
    z_y = int(tdy + z_rot[1])
    cx = int(tdx)
    cy = int(tdy)
    cv2.line(img, (cx, cy), (x_x, x_y), (0, 0, 255), 2)
    cv2.line(img, (cx, cy), (y_x, y_y), (0, 255, 0), 2)
    cv2.line(img, (cx, cy), (z_x, z_y), (255, 0, 0), 2)

    return img

def get_R(pitch, yaw, roll):
    # roll  = np.deg2rad(roll)
    # pitch = np.deg2rad(pitch)
    # yaw   = np.deg2rad(yaw)
    Rx = np.array([
        [1,              0,              0],
        [0,  np.cos(pitch), -np.sin(pitch)],
        [0,  np.sin(pitch),  np.cos(pitch)]
    ], dtype=np.float32)
    Ry = np.array([
        [ np.cos(yaw), 0, np.sin(yaw)],
        [           0, 1,           0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ], dtype=np.float32)
    Rz = np.array([
        [ np.cos(roll), -np.sin(roll), 0],
        [ np.sin(roll),  np.cos(roll), 0],
        [            0,             0, 1]
    ], dtype=np.float32)

    R = Rz @ Ry @ Rx
    return R

def get_R_legacy(pitch, yaw, roll):
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    Rx = [[1, 0, 0], [0, cp, -sp], [0, sp, cp]]
    Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
    Rz = [[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]

    def matmul(A, B):
        return [[sum(a*b for a, b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]

    R = matmul(Rz, matmul(Ry, Rx))
    return R

def get_euler(R):
    """
    単一の3x3回転行列 R から [x, y, z] の3つのオイラー角を返す関数
    Z-Y-X の順番(あるいは X-Y-Z 等)で回転している想定であれば、
    計算に応じて適切に置き換えてください。
    """

    # R[0,0]^2 + R[1,0]^2 の平方根
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
    # シンギュラリティ判定
    singular = (sy < 1e-6)

    if not singular:
        # 通常ケース
        x = np.arctan2(R[2, 1], R[2, 2])   # roll
        y = np.arctan2(-R[2, 0], sy)       # pitch
        z = np.arctan2(R[1, 0], R[0, 0])   # yaw
    else:
        # 特異姿勢（gimbal lock）場合の近似
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0

    # ラジアンから度に変換
    x = np.rad2deg(x)
    y = np.rad2deg(y)
    z = np.rad2deg(z)

    return np.array([x, y, z])


def compute_euler_angles_from_rotation_matrix(R):
    """
    単一の3x3回転行列 R から [x, y, z] の3つのオイラー角を返す関数
    Z-Y-X の順番(あるいは X-Y-Z 等)で回転している想定であれば、
    計算に応じて適切に置き換えてください。
    """

    # R[0,0]^2 + R[1,0]^2 の平方根
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
    # シンギュラリティ判定
    singular = (sy < 1e-6)

    if not singular:
        # 通常ケース
        x = np.arctan2(R[2, 1], R[2, 2])   # roll
        y = np.arctan2(-R[2, 0], sy)       # pitch
        z = np.arctan2(R[1, 0], R[0, 0])   # yaw
    else:
        # 特異姿勢（gimbal lock）場合の近似
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0

    return np.array([x, y, z])


def get_pose_params_from_mat(mat_path):
    # This functions gets the pose parameters from the .mat
    # Annotations that come with the Pose_300W_LP dataset.
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll, tdx, tdy]
    pose_params = pre_pose_params[:5]
    return pose_params

def get_ypr_from_mat(mat_path):
    # Get yaw, pitch, roll from .mat annotation.
    # They are in radians
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]
    return pose_params

def get_pt2d_from_mat(mat_path):
    # Get 2D landmarks
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt2d']
    return pt2d

# batch*n
def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    gpu = v_mag.get_device()
    if gpu < 0:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device('cpu'))
    else:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device('cuda:%d' % gpu))
    v_mag = torch.max(v_mag, eps)
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    return v
    
# u, v batch*n
def cross_product(u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1) #batch*3
        
    return out
        
    
#poses batch*6
#poses
def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:,0:3] #batch*3
    y_raw = poses[:,3:6] #batch*3

    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z) #batch*3
    y = cross_product(z,x) #batch*3
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix


#input batch*4*4 or batch*3*3
#output torch batch*3 x, y, z in radiant
#the rotation is in the sequence of x,y,z
def compute_euler_angles_from_rotation_matrices(rotation_matrices):
    batch = rotation_matrices.shape[0]
    R = rotation_matrices
    sy = torch.sqrt(R[:,0,0]*R[:,0,0]+R[:,1,0]*R[:,1,0])
    singular = sy<1e-6
    singular = singular.float()
        
    x = torch.atan2(R[:,2,1], R[:,2,2])
    y = torch.atan2(-R[:,2,0], sy)
    z = torch.atan2(R[:,1,0],R[:,0,0])
    
    xs = torch.atan2(-R[:,1,2], R[:,1,1])
    ys = torch.atan2(-R[:,2,0], sy)
    zs = R[:,1,0]*0
        
    gpu = rotation_matrices.get_device()
    if gpu < 0:
        out_euler = torch.autograd.Variable(torch.zeros(batch,3)).to(torch.device('cpu'))
    else:
        out_euler = torch.autograd.Variable(torch.zeros(batch,3)).to(torch.device('cuda:%d' % gpu))
    out_euler[:,0] = x*(1-singular)+xs*singular
    out_euler[:,1] = y*(1-singular)+ys*singular
    out_euler[:,2] = z*(1-singular)+zs*singular
        
    return out_euler


def get_R(x,y,z):
    ''' Get rotation matrix from three rotation angles (radians). right-handed.
    Args:
        angles: [3,]. x, y, z angles
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    # x
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    # y
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    # z
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])

    R = Rz.dot(Ry.dot(Rx))
    return R