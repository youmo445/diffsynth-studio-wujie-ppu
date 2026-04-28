# utils_codex.py 逐行讲解导航

源文件：`/mnt/workspace/zsq/DiffSynth-Studio/diffsynth/trainers/utils_codex.py`

说明：下面按连续代码块解释，每个条目都对应明确行号；空行、括号行、列表元素行会解释其结构作用。


## 第 1-17 行

总体说明：导入依赖和处理可选 cv2：json/os 用于路径和配置；h5py 读机器人状态；imageio/decord 读写视频；matplotlib colormap 生成夹爪颜色；numpy/torch 做数组和 Dataset；PIL 作为 cv2 不存在时的绘图兜底。

- `0001` `import json`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0002` `import os`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0003` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0004` `import h5py`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0005` `import imageio`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0006` `import imageio.v3 as iio`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0007` `import matplotlib.cm as cm`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0008` `import numpy as np`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0009` `import torch`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0010` `from decord import VideoReader, cpu`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0011` `from PIL import Image, ImageDraw`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0012` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0013` `try:`

  解释：开始异常保护块，用于处理可选依赖或资源释放。

- `0014` `    import cv2`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0015` `except ModuleNotFoundError:`

  解释：异常处理分支，当前主要用于 cv2 缺失时降级。

- `0016` `    cv2 = None`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0017` ``

  解释：空行，用于分隔逻辑块，提高可读性。


## 第 19-42 行

总体说明：定义轨迹可视化常量：左右手颜色、末端执行器四个关键点、夹爪坐标到 EEF 坐标的 z 轴偏移、以及 EVAC 中对左右手姿态额外旋转的 EEF2CamLeft/Right。

- `0019` `ColorMapLeft = cm.Greens`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0020` `ColorMapRight = cm.Reds`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0021` `ColorListLeft = [(0, 0, 255), (255, 255, 0), (0, 255, 255)]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0022` `ColorListRight = [(255, 0, 255), (255, 0, 0), (0, 255, 0)]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0023` `EndEffectorPts = np.array(`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0024` `    [`

  解释：结构行，用于闭合或组织上一行开始的多行表达式。

- `0025` `        [0, 0, 0, 1],`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0026` `        [0.1, 0, 0, 1],`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0027` `        [0, 0.1, 0, 1],`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0028` `        [0, 0, 0.1, 1],`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0029` `    ],`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0030` `    dtype=np.float32,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0031` `)`

  解释：结构行，用于闭合或组织上一行开始的多行表达式。

- `0032` `Gripper2EEFCvt = np.array(`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0033` `    [`

  解释：结构行，用于闭合或组织上一行开始的多行表达式。

- `0034` `        [1, 0, 0, 0],`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0035` `        [0, 1, 0, 0],`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0036` `        [0, 0, 1, 0.23],`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0037` `        [0, 0, 0, 1],`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0038` `    ],`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0039` `    dtype=np.float32,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0040` `)`

  解释：结构行，用于闭合或组织上一行开始的多行表达式。

- `0041` `EEF2CamLeft = np.array([0, 0, -0.5236], dtype=np.float32)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0042` `EEF2CamRight = np.array([0, 0, 0.5236], dtype=np.float32)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。


## 第 45-53 行

总体说明：draw_circle：优先用 cv2 画实心圆；如果服务器没有 cv2，就把 numpy 图转成 PIL 图，用 ImageDraw 画圆，再转回 numpy。

- `0045` `def draw_circle(img, center, radius, color):`

  解释：定义函数，后续缩进代码实现这个功能。

- `0046` `    if cv2 is not None:`

  解释：条件分支，根据当前状态选择不同处理路径。

- `0047` `        cv2.circle(img, tuple(center), radius, color, -1)`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0048` `        return img`

  解释：返回当前函数的计算结果。

- `0049` `    pil_img = Image.fromarray(img)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0050` `    draw = ImageDraw.Draw(pil_img)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0051` `    x, y = int(center[0]), int(center[1])`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0052` `    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=tuple(color))`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0053` `    return np.array(pil_img)`

  解释：返回当前函数的计算结果。


## 第 56-63 行

总体说明：draw_line：优先用 cv2 画线；没有 cv2 时使用 PIL ImageDraw 画线，保证数据集可视化不依赖 opencv。

- `0056` `def draw_line(img, p0, p1, color, width):`

  解释：定义函数，后续缩进代码实现这个功能。

- `0057` `    if cv2 is not None:`

  解释：条件分支，根据当前状态选择不同处理路径。

- `0058` `        cv2.line(img, tuple(p0), tuple(p1), color, width)`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0059` `        return img`

  解释：返回当前函数的计算结果。

- `0060` `    pil_img = Image.fromarray(img)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0061` `    draw = ImageDraw.Draw(pil_img)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0062` `    draw.line([tuple(map(int, p0)), tuple(map(int, p1))], fill=tuple(color), width=width)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0063` `    return np.array(pil_img)`

  解释：返回当前函数的计算结果。


## 第 66-85 行

总体说明：quaternion_xyzw_to_matrix：把 xyzw 顺序四元数转换成 3x3 旋转矩阵，支持批量输入，最后统一转 float32。

- `0066` `def quaternion_xyzw_to_matrix(quat):`

  解释：定义函数，后续缩进代码实现这个功能。

- `0067` `    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0068` `    tx, ty, tz = 2.0 * x, 2.0 * y, 2.0 * z`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0069` `    twx, twy, twz = tx * w, ty * w, tz * w`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0070` `    txx, txy, txz = tx * x, ty * x, tz * x`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0071` `    tyy, tyz, tzz = ty * y, tz * y, tz * z`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0072` `    return np.stack(`

  解释：返回当前函数的计算结果。

- `0073` `        [`

  解释：结构行，用于闭合或组织上一行开始的多行表达式。

- `0074` `            1.0 - (tyy + tzz),`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0075` `            txy - twz,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0076` `            txz + twy,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0077` `            txy + twz,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0078` `            1.0 - (txx + tzz),`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0079` `            tyz - twx,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0080` `            txz - twy,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0081` `            tyz + twx,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0082` `            1.0 - (txx + tyy),`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0083` `        ],`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0084` `        axis=-1,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0085` `    ).reshape(quat.shape[:-1] + (3, 3)).astype(np.float32)`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。


## 第 88-101 行

总体说明：euler_xyz_to_quaternion_xyzw：把 roll/pitch/yaw 欧拉角转成 xyzw 四元数，用来把 EEF2Cam 的 z 轴旋转变成可右乘的四元数。

- `0088` `def euler_xyz_to_quaternion_xyzw(euler):`

  解释：定义函数，后续缩进代码实现这个功能。

- `0089` `    roll, pitch, yaw = euler`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0090` `    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0091` `    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0092` `    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0093` `    return np.array(`

  解释：返回当前函数的计算结果。

- `0094` `        [`

  解释：结构行，用于闭合或组织上一行开始的多行表达式。

- `0095` `            sr * cp * cy - cr * sp * sy,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0096` `            cr * sp * cy + sr * cp * sy,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0097` `            cr * cp * sy - sr * sp * cy,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0098` `            cr * cp * cy + sr * sp * sy,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0099` `        ],`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0100` `        dtype=np.float32,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0101` `    )`

  解释：结构行，用于闭合或组织上一行开始的多行表达式。


## 第 104-115 行

总体说明：quaternion_multiply_xyzw：实现 xyzw 四元数乘法 q1*q2，用于把原始 EEF 姿态叠加 EVAC 的视觉坐标修正。

- `0104` `def quaternion_multiply_xyzw(q1, q2):`

  解释：定义函数，后续缩进代码实现这个功能。

- `0105` `    x1, y1, z1, w1 = np.moveaxis(q1, -1, 0)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0106` `    x2, y2, z2, w2 = np.moveaxis(q2, -1, 0)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0107` `    return np.stack(`

  解释：返回当前函数的计算结果。

- `0108` `        [`

  解释：结构行，用于闭合或组织上一行开始的多行表达式。

- `0109` `            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0110` `            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0111` `            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0112` `            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0113` `        ],`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0114` `        axis=-1,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0115` `    ).astype(np.float32)`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。


## 第 118-126 行

总体说明：apply_eef2cam_visual_rotation：复现 EVAC get_actions 的 rot_vis = quat * cvt_vis，把左右夹爪的视觉姿态修正提前烘焙进 abs_actions。

- `0118` `def apply_eef2cam_visual_rotation(quat, side):`

  解释：定义函数，后续缩进代码实现这个功能。

- `0119` `    """`

  解释：文档字符串边界或内容，用于说明函数/类的用途和输入输出。

- `0120` `    Match EVAC get_actions(): rot_vis = Rotation.from_quat(quat) * cvt_vis.`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0121` `    The extra rotation is baked into absolute actions before get_traj uses them.`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0122` `    """`

  解释：文档字符串边界或内容，用于说明函数/类的用途和输入输出。

- `0123` `    offset = EEF2CamLeft if side == "left" else EEF2CamRight`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0124` `    offset_quat = euler_xyz_to_quaternion_xyzw(offset)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0125` `    rotated = quaternion_multiply_xyzw(quat.astype(np.float32), offset_quat)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0126` `    return rotated / (np.linalg.norm(rotated, axis=-1, keepdims=True) + 1e-8)`

  解释：返回当前函数的计算结果。


## 第 129-137 行

总体说明：get_transformation_matrix_from_quat：把 [xyz, quat_xyzw] 批量转成 4x4 位姿矩阵，供投影轨迹时从 EEF 坐标变到世界/相机坐标。

- `0129` `def get_transformation_matrix_from_quat(pose):`

  解释：定义函数，后续缩进代码实现这个功能。

- `0130` `    """`

  解释：文档字符串边界或内容，用于说明函数/类的用途和输入输出。

- `0131` `    EVAC-compatible conversion from [x, y, z, qx, qy, qz, qw] to T x 4 x 4.`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0132` `    """`

  解释：文档字符串边界或内容，用于说明函数/类的用途和输入输出。

- `0133` `    pose = np.asarray(pose, dtype=np.float32)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0134` `    mat = np.tile(np.eye(4, dtype=np.float32), (pose.shape[0], 1, 1))`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0135` `    mat[:, :3, :3] = quaternion_xyzw_to_matrix(pose[:, 3:7])`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0136` `    mat[:, :3, 3] = pose[:, :3]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0137` `    return mat`

  解释：返回当前函数的计算结果。


## 第 140-161 行

总体说明：compute_depth_radius：根据深度决定圆半径；constant 固定半径，perspective 近大远小，depth_norm 将深度线性映射到 min/max 半径。

- `0140` `def compute_depth_radius(`

  解释：定义函数，后续缩进代码实现这个功能。

- `0141` `    depth,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0142` `    radius,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0143` `    radius_mode="constant",`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0144` `    min_radius=20,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0145` `    max_radius=60,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0146` `    ref_depth=0.30,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0147` `    near_depth=0.19,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0148` `    far_depth=0.69,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0149` `):`

  解释：结构行，用于闭合或组织上一行开始的多行表达式。

- `0150` `    if radius_mode == "constant":`

  解释：条件分支，根据当前状态选择不同处理路径。

- `0151` `        return int(radius)`

  解释：返回当前函数的计算结果。

- `0152` `    safe_depth = max(float(depth), 1e-4)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0153` `    if radius_mode == "perspective":`

  解释：条件分支，根据当前状态选择不同处理路径。

- `0154` `        dynamic_radius = float(radius) * float(ref_depth) / safe_depth`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0155` `    elif radius_mode == "depth_norm":`

  解释：条件分支的另一种情况。

- `0156` `        alpha = (float(far_depth) - safe_depth) / (float(far_depth) - float(near_depth) + 1e-8)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0157` `        alpha = float(np.clip(alpha, 0.0, 1.0))`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0158` `        dynamic_radius = float(min_radius) + alpha * (float(max_radius) - float(min_radius))`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0159` `    else:`

  解释：条件分支的兜底情况。

- `0160` `        raise ValueError(f"Unsupported radius_mode: {radius_mode}")`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0161` `    return int(round(np.clip(dynamic_radius, min_radius, max_radius)))`

  解释：返回当前函数的计算结果。


## 第 164-255 行

总体说明：generate_traj_map：核心投影函数；输入每帧 abs_actions、相机 w2c 和内参，把左右夹爪 3D 关键点投到图像平面，画圆和三轴方向线，输出每帧控制图。

- `0164` `def generate_traj_map(`

  解释：定义函数，后续缩进代码实现这个功能。

- `0165` `    abs_actions,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0166` `    w2c,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0167` `    intrinsic,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0168` `    h,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0169` `    w,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0170` `    radius=50,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0171` `    radius_mode="constant",`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0172` `    min_radius=15,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0173` `    max_radius=70,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0174` `    ref_depth=1.0,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0175` `    near_depth=0.3,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0176` `    far_depth=2.0,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0177` `):`

  解释：结构行，用于闭合或组织上一行开始的多行表达式。

- `0178` `    """`

  解释：文档字符串边界或内容，用于说明函数/类的用途和输入输出。

- `0179` `    Generate trajectory maps following EVAC's get_traj logic as closely as possible.`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0180` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0181` `    abs_actions: T x 16, [left xyz, left quat xyzw, left grip,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0182` `                          right xyz, right quat xyzw, right grip]`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0183` `    w2c: T x 4 x 4 or 4 x 4 world-to-camera matrices.`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0184` `    intrinsic: 3 x 3 camera intrinsic matrix scaled to output size.`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0185` `    radius_mode:`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0186` `        constant    - EVAC-compatible fixed radius.`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0187` `        perspective - radius * ref_depth / camera_depth, clipped.`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0188` `        depth_norm  - linearly maps [near_depth, far_depth] to [max_radius, min_radius].`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0189` `    returns: list of T uint8 RGB arrays with shape H x W x 3.`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0190` `    """`

  解释：文档字符串边界或内容，用于说明函数/类的用途和输入输出。

- `0191` `    abs_actions = np.asarray(abs_actions, dtype=np.float32)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0192` `    intrinsic = np.asarray(intrinsic, dtype=np.float32)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0193` `    w2c = np.asarray(w2c, dtype=np.float32)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0194` `    if w2c.ndim == 2:`

  解释：条件分支，根据当前状态选择不同处理路径。

- `0195` `        w2c = np.repeat(w2c[None], abs_actions.shape[0], axis=0)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0196` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0197` `    ee_key_pts = EndEffectorPts.reshape(1, 4, 4).transpose(0, 2, 1)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0198` `    cvt_matrix = Gripper2EEFCvt.reshape(1, 4, 4)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0199` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0200` `    pose_l_mat = get_transformation_matrix_from_quat(abs_actions[:, 0:7])`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0201` `    pose_r_mat = get_transformation_matrix_from_quat(abs_actions[:, 8:15])`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0202` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0203` `    ee2cam_l = np.matmul(w2c, pose_l_mat)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0204` `    ee2cam_r = np.matmul(w2c, pose_r_mat)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0205` `    ee2cam_l = np.matmul(ee2cam_l, cvt_matrix)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0206` `    ee2cam_r = np.matmul(ee2cam_r, cvt_matrix)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0207` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0208` `    pts_l = np.matmul(ee2cam_l, ee_key_pts)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0209` `    pts_r = np.matmul(ee2cam_r, ee_key_pts)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0210` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0211` `    intr = intrinsic.reshape(1, 3, 3)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0212` `    uvs_l = np.matmul(intr, pts_l[:, :3, :])`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0213` `    uvs_l = (uvs_l / (pts_l[:, 2:3, :] + 1e-8))[:, :2, :].transpose(0, 2, 1).astype(np.int64)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0214` `    uvs_r = np.matmul(intr, pts_r[:, :3, :])`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0215` `    uvs_r = (uvs_r / (pts_r[:, 2:3, :] + 1e-8))[:, :2, :].transpose(0, 2, 1).astype(np.int64)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0216` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0217` `    img_list = []`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0218` `    for i in range(abs_actions.shape[0]):`

  解释：循环遍历集合或时间帧，逐项处理。

- `0219` `        img = np.zeros((h, w, 3), dtype=np.uint8) + 50`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0220` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0221` `        normalized_value_l = abs_actions[i, 7].item() / 120.0`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0222` `        normalized_value_r = abs_actions[i, 15].item() / 120.0`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0223` `        color_l = tuple(int(c * 255) for c in ColorMapLeft(normalized_value_l)[:3])`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0224` `        color_r = tuple(int(c * 255) for c in ColorMapRight(normalized_value_r)[:3])`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0225` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0226` `        for points, pts, color in zip([uvs_l[i], uvs_r[i]], [pts_l[i], pts_r[i]], [color_l, color_r]):`

  解释：循环遍历集合或时间帧，逐项处理。

- `0227` `            base = np.array(points[0])`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0228` `            if base[0] < 0 or base[0] >= w or base[1] < 0 or base[1] >= h:`

  解释：条件分支，根据当前状态选择不同处理路径。

- `0229` `                continue`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0230` `            point = np.array(points[0][:2])`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0231` `            radius_i = compute_depth_radius(`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0232` `                pts[2, 0],`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0233` `                radius,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0234` `                radius_mode=radius_mode,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0235` `                min_radius=min_radius,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0236` `                max_radius=max_radius,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0237` `                ref_depth=ref_depth,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0238` `                near_depth=near_depth,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0239` `                far_depth=far_depth,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0240` `            )`

  解释：结构行，用于闭合或组织上一行开始的多行表达式。

- `0241` `            img = draw_circle(img, point, radius_i, color)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0242` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0243` `        for points, colors in zip([uvs_l[i], uvs_r[i]], [ColorListLeft, ColorListRight]):`

  解释：循环遍历集合或时间帧，逐项处理。

- `0244` `            base = np.array(points[0])`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0245` `            if base[0] < 0 or base[0] >= w or base[1] < 0 or base[1] >= h:`

  解释：条件分支，根据当前状态选择不同处理路径。

- `0246` `                continue`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0247` `            for point_idx, point in enumerate(points):`

  解释：循环遍历集合或时间帧，逐项处理。

- `0248` `                point = np.array(point[:2])`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0249` `                if point_idx == 0:`

  解释：条件分支，根据当前状态选择不同处理路径。

- `0250` `                    continue`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0251` `                img = draw_line(img, base, point, colors[point_idx - 1], 8)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0252` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0253` `        img_list.append(img)`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0254` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0255` `    return img_list`

  解释：返回当前函数的计算结果。


## 第 258-352 行

总体说明：AgiBotWCDataset4Wancontrolmultiview.__init__：初始化多视角 Dataset，保存参数、计算降采样步长、发现 episode、预构建每个 episode 的动作/相机信息和可采样窗口。

- `0258` `class AgiBotWCDataset4Wancontrolmultiview(torch.utils.data.Dataset):`

  解释：定义 Dataset 类，继承 torch Dataset 以接入 Wan 训练 DataLoader。

- `0259` `    """`

  解释：文档字符串边界或内容，用于说明函数/类的用途和输入输出。

- `0260` `    Multi-view AgiBotWorld dataset for Wan VACE training.`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0261` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0262` `    Each returned frame vertically concatenates views in camera_names order. The`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0263` `    control video uses EVAC-style trajectory maps generated per camera frame.`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0264` `    """`

  解释：文档字符串边界或内容，用于说明函数/类的用途和输入输出。

- `0265` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0266` `    def __init__(`

  解释：定义函数，后续缩进代码实现这个功能。

- `0267` `        self,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0268` `        base_path="/mnt/workspace/zsq/Agi2024subset_split/train",`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0269` `        num_frames=9,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0270` `        repeat=1,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0271` `        episode_stride=1,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0272` `        episode_limit=None,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0273` `        original_hz=30,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0274` `        target_hz=5,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0275` `        stride=1,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0276` `        first_round_prob=0.05,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0277` `        context_length=1,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0278` `        traj_radius=50,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0279` `        traj_radius_mode="constant",`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0280` `        traj_min_radius=20,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0281` `        traj_max_radius=60,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0282` `        traj_ref_depth=0.30,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0283` `        traj_near_depth=0.19,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0284` `        traj_far_depth=0.69,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0285` `        output_raymap=False,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0286` `        ray_o_vmin=-1.5,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0287` `        ray_o_vmax=1.5,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0288` `        ray_d_vmin=-1.0,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0289` `        ray_d_vmax=1.0,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0290` `        resize_to=(320, 512),`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0291` `        dataset_type="vace",`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0292` `        camera_names=None,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0293` `    ):`

  解释：结构行，用于闭合或组织上一行开始的多行表达式。

- `0294` `        super().__init__()`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0295` `        if original_hz % target_hz != 0:`

  解释：条件分支，根据当前状态选择不同处理路径。

- `0296` `            raise ValueError(f"Cannot downsample from {original_hz}Hz to {target_hz}Hz")`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0297` `        if camera_names is None:`

  解释：条件分支，根据当前状态选择不同处理路径。

- `0298` `            camera_names = ["head", "hand_left", "hand_right"]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0299` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0300` `        self.base_path = base_path`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0301` `        self.num_frames = num_frames`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0302` `        self.repeat = repeat`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0303` `        self.stride = stride`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0304` `        self.first_round_prob = first_round_prob`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0305` `        self.context_length = context_length`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0306` `        self.traj_radius = traj_radius`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0307` `        self.traj_radius_mode = traj_radius_mode`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0308` `        self.traj_min_radius = traj_min_radius`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0309` `        self.traj_max_radius = traj_max_radius`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0310` `        self.traj_ref_depth = traj_ref_depth`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0311` `        self.traj_near_depth = traj_near_depth`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0312` `        self.traj_far_depth = traj_far_depth`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0313` `        self.output_raymap = output_raymap`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0314` `        self.ray_o_vmin = ray_o_vmin`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0315` `        self.ray_o_vmax = ray_o_vmax`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0316` `        self.ray_d_vmin = ray_d_vmin`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0317` `        self.ray_d_vmax = ray_d_vmax`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0318` `        self.resize_to = resize_to`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0319` `        self.load_from_cache = False`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0320` `        self.type = dataset_type`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0321` `        self.camera_names = list(camera_names)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0322` `        self.downsample_step = original_hz // target_hz`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0323` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0324` `        print(`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0325` `            f"[AgiBotWCDataset4Wancontrolmultiview] Downsample: "`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0326` `            f"{original_hz}Hz -> {target_hz}Hz (step={self.downsample_step})"`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0327` `        )`

  解释：结构行，用于闭合或组织上一行开始的多行表达式。

- `0328` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0329` `        episodes = self._discover_episodes(base_path)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0330` `        if episode_limit is not None:`

  解释：条件分支，根据当前状态选择不同处理路径。

- `0331` `            episodes = episodes[:episode_limit]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0332` `        self.episodes = episodes[::episode_stride]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0333` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0334` `        self.episode_info = []`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0335` `        self.sample_indices = []`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0336` `        for ep_idx, ep in enumerate(self.episodes):`

  解释：循环遍历集合或时间帧，逐项处理。

- `0337` `            info = self._build_episode_info(ep)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0338` `            if info is None:`

  解释：条件分支，根据当前状态选择不同处理路径。

- `0339` `                continue`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0340` `            valid_ep_idx = len(self.episode_info)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0341` `            self.episode_info.append(info)`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0342` `            max_start = info["T_ds"] - self.num_frames + 1`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0343` `            for start in range(0, max_start, self.stride):`

  解释：循环遍历集合或时间帧，逐项处理。

- `0344` `                self.sample_indices.append((valid_ep_idx, start))`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0345` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0346` `        self.total_samples = len(self.sample_indices)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0347` `        self.length = self.total_samples * repeat`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0348` `        print(`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0349` `            f"[AgiBotWCDataset4Wancontrolmultiview] {len(self.episode_info)} episodes, "`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0350` `            f"{self.total_samples} samples, repeat={repeat}, length={self.length}, "`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0351` `            f"cameras={self.camera_names}"`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0352` `        )`

  解释：结构行，用于闭合或组织上一行开始的多行表达式。


## 第 354-382 行

总体说明：_discover_episodes：扫描数据集目录；优先使用 split 布局 proprio_stats/observations/parameters，找不到时兼容旧的 chunk 分块布局。

- `0354` `    def _discover_episodes(self, base_path):`

  解释：定义函数，后续缩进代码实现这个功能。

- `0355` `        proprio_base = os.path.join(base_path, "proprio_stats")`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0356` `        if os.path.isdir(proprio_base):`

  解释：条件分支，根据当前状态选择不同处理路径。

- `0357` `            episodes = []`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0358` `            for task in sorted(os.listdir(proprio_base)):`

  解释：循环遍历集合或时间帧，逐项处理。

- `0359` `                task_path = os.path.join(proprio_base, task)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0360` `                if not os.path.isdir(task_path):`

  解释：条件分支，根据当前状态选择不同处理路径。

- `0361` `                    continue`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0362` `                for ep in sorted(os.listdir(task_path)):`

  解释：循环遍历集合或时间帧，逐项处理。

- `0363` `                    h5_path = os.path.join(task_path, ep, "proprio_stats.h5")`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0364` `                    if not os.path.exists(h5_path):`

  解释：条件分支，根据当前状态选择不同处理路径。

- `0365` `                        continue`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0366` `                    episodes.append(`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0367` `                        {`

  解释：结构行，用于闭合或组织上一行开始的多行表达式。

- `0368` `                            "h5_path": h5_path,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0369` `                            "video_dir": os.path.join(base_path, "observations", task, ep, "videos"),`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0370` `                            "camera_dir": os.path.join(base_path, "parameters", task, ep, "parameters", "camera"),`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0371` `                        }`

  解释：结构行，用于闭合或组织上一行开始的多行表达式。

- `0372` `                    )`

  解释：结构行，用于闭合或组织上一行开始的多行表达式。

- `0373` `            return episodes`

  解释：返回当前函数的计算结果。

- `0374` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0375` `        # Fallback for the chunk-grouped layout used by the older single-view dataset.`

  解释：注释行，说明下面代码的兼容目的或设计意图。

- `0376` `        episode_dict = {}`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0377` `        for name in sorted(os.listdir(base_path)):`

  解释：循环遍历集合或时间帧，逐项处理。

- `0378` `            path = os.path.join(base_path, name)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0379` `            if os.path.isdir(path):`

  解释：条件分支，根据当前状态选择不同处理路径。

- `0380` `                key = "-".join(name.split("-")[:2])`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0381` `                episode_dict.setdefault(key, []).append(path)`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0382` `        return [{"chunk_paths": sorted(episode_dict[k])} for k in sorted(episode_dict.keys())]`

  解释：返回当前函数的计算结果。


## 第 384-439 行

总体说明：_build_episode_info：读取一个 episode 的 proprio、夹爪状态和相机参数；降采样到目标 Hz；生成 EVAC 风格 abs_actions；返回后续 __getitem__ 可直接索引的缓存信息。

- `0384` `    def _build_episode_info(self, ep):`

  解释：定义函数，后续缩进代码实现这个功能。

- `0385` `        if "h5_path" in ep:`

  解释：条件分支，根据当前状态选择不同处理路径。

- `0386` `            with h5py.File(ep["h5_path"], "r") as f:`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0387` `                pos = f["state/end/position"][:]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0388` `                quat = f["state/end/orientation"][:]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0389` `                grip = f["state/effector/position"][:]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0390` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0391` `            total_raw = pos.shape[0]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0392` `            grip = grip.reshape(total_raw, 2, -1)[..., 0] if grip.ndim == 3 else grip`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0393` `            ds_idx = np.arange(0, total_raw, self.downsample_step)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0394` `            abs_actions = self._compute_abs_action(pos[ds_idx], quat[ds_idx], grip[ds_idx])`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0395` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0396` `            cameras = {`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0397` `                cam: self._load_camera_params(ep["camera_dir"], cam, ds_idx)`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0398` `                for cam in self.camera_names`

  解释：循环遍历集合或时间帧，逐项处理。

- `0399` `            }`

  解释：结构行，用于闭合或组织上一行开始的多行表达式。

- `0400` `            return {`

  解释：返回当前函数的计算结果。

- `0401` `                "layout": "split",`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0402` `                "video_dir": ep["video_dir"],`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0403` `                "ds_indices": ds_idx,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0404` `                "T_ds": len(ds_idx),`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0405` `                "abs_actions": abs_actions,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0406` `                "cameras": cameras,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0407` `            }`

  解释：结构行，用于闭合或组织上一行开始的多行表达式。

- `0408` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0409` `        chunk_paths = ep["chunk_paths"]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0410` `        pos_list, quat_list, grip_list, chunk_frame_counts = [], [], [], []`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0411` `        for chunk_path in chunk_paths:`

  解释：循环遍历集合或时间帧，逐项处理。

- `0412` `            h5_path = os.path.join(chunk_path, "proprio_stats.h5")`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0413` `            with h5py.File(h5_path, "r") as f:`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0414` `                pos = f["state/end/position"][:]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0415` `                quat = f["state/end/orientation"][:]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0416` `                grip = f["state/effector/position"][:]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0417` `            chunk_frame_counts.append(pos.shape[0])`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0418` `            pos_list.append(pos)`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0419` `            quat_list.append(quat)`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0420` `            grip_list.append(grip.reshape(pos.shape[0], 2, -1)[..., 0] if grip.ndim == 3 else grip)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0421` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0422` `        pos_all = np.concatenate(pos_list, axis=0)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0423` `        quat_all = np.concatenate(quat_list, axis=0)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0424` `        grip_all = np.concatenate(grip_list, axis=0)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0425` `        ds_idx = np.arange(0, pos_all.shape[0], self.downsample_step)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0426` `        abs_actions = self._compute_abs_action(pos_all[ds_idx], quat_all[ds_idx], grip_all[ds_idx])`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0427` `        cameras = {`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0428` `            cam: self._load_camera_params(chunk_paths[0], cam, ds_idx, chunk_layout=True)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0429` `            for cam in self.camera_names`

  解释：循环遍历集合或时间帧，逐项处理。

- `0430` `        }`

  解释：结构行，用于闭合或组织上一行开始的多行表达式。

- `0431` `        return {`

  解释：返回当前函数的计算结果。

- `0432` `            "layout": "chunk",`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0433` `            "chunk_paths": chunk_paths,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0434` `            "chunk_frame_counts": chunk_frame_counts,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0435` `            "ds_indices": ds_idx,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0436` `            "T_ds": len(ds_idx),`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0437` `            "abs_actions": abs_actions,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0438` `            "cameras": cameras,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0439` `        }`

  解释：结构行，用于闭合或组织上一行开始的多行表达式。


## 第 441-449 行

总体说明：_compute_abs_action：把左右手 position/quaternion/gripper 拼成 T x 16，并对左右手 quaternion 应用 EEF2Cam 修正。

- `0441` `    def _compute_abs_action(self, pos, quat, grip):`

  解释：定义函数，后续缩进代码实现这个功能。

- `0442` `        out = np.zeros((pos.shape[0], 16), dtype=np.float32)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0443` `        out[:, 0:3] = pos[:, 0]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0444` `        out[:, 3:7] = apply_eef2cam_visual_rotation(quat[:, 0], "left")`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0445` `        out[:, 7] = grip[:, 0]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0446` `        out[:, 8:11] = pos[:, 1]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0447` `        out[:, 11:15] = apply_eef2cam_visual_rotation(quat[:, 1], "right")`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0448` `        out[:, 15] = grip[:, 1]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0449` `        return out`

  解释：返回当前函数的计算结果。


## 第 451-479 行

总体说明：_load_camera_params：读取相机内参 JSON 和每帧 c2w 外参 JSON，按降采样索引对齐，额外计算 w2c 供投影使用。

- `0451` `    def _load_camera_params(self, camera_dir, cam_name, ds_idx, chunk_layout=False):`

  解释：定义函数，后续缩进代码实现这个功能。

- `0452` `        if chunk_layout:`

  解释：条件分支，根据当前状态选择不同处理路径。

- `0453` `            intrinsic_path = os.path.join(camera_dir, f"{cam_name}_intrinsic_params.json")`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0454` `            extrinsic_path = os.path.join(camera_dir, f"{cam_name}_extrinsic_params_aligned.json")`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0455` `        else:`

  解释：条件分支的兜底情况。

- `0456` `            intrinsic_path = os.path.join(camera_dir, f"{cam_name}_intrinsic_params.json")`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0457` `            extrinsic_path = os.path.join(camera_dir, f"{cam_name}_extrinsic_params_aligned.json")`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0458` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0459` `        with open(intrinsic_path, "r") as f:`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0460` `            info = json.load(f)["intrinsic"]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0461` `        intrinsic = np.eye(3, dtype=np.float32)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0462` `        intrinsic[0, 0] = info["fx"]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0463` `        intrinsic[1, 1] = info["fy"]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0464` `        intrinsic[0, 2] = info["ppx"]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0465` `        intrinsic[1, 2] = info["ppy"]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0466` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0467` `        with open(extrinsic_path, "r") as f:`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0468` `            extr_list = json.load(f)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0469` `        c2w_all = []`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0470` `        for item in extr_list:`

  解释：循环遍历集合或时间帧，逐项处理。

- `0471` `            mat = np.eye(4, dtype=np.float32)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0472` `            mat[:3, :3] = np.array(item["extrinsic"]["rotation_matrix"], dtype=np.float32)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0473` `            mat[:3, 3] = np.array(item["extrinsic"]["translation_vector"], dtype=np.float32)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0474` `            c2w_all.append(mat)`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0475` `        c2w_all = np.stack(c2w_all, axis=0)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0476` `        valid_idx = np.clip(ds_idx, 0, c2w_all.shape[0] - 1)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0477` `        c2w_seq = c2w_all[valid_idx]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0478` `        w2c_seq = np.linalg.inv(c2w_seq).astype(np.float32)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0479` `        return {"intrinsic": intrinsic, "w2c": w2c_seq, "c2w": c2w_seq}`

  解释：返回当前函数的计算结果。


## 第 481-514 行

总体说明：_load_frames_by_ids：按 frame_ids 读取多视角 RGB 帧；split 布局用 imageio 逐帧读 mp4，chunk 布局用 decord 读对应 chunk。

- `0481` `    def _load_frames_by_ids(self, ep_info, frame_ids):`

  解释：定义函数，后续缩进代码实现这个功能。

- `0482` `        result = {}`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0483` `        if ep_info["layout"] == "split":`

  解释：条件分支，根据当前状态选择不同处理路径。

- `0484` `            for fid in frame_ids:`

  解释：循环遍历集合或时间帧，逐项处理。

- `0485` `                orig_idx = int(ep_info["ds_indices"][fid])`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0486` `                frames_cam = {}`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0487` `                for cam in self.camera_names:`

  解释：循环遍历集合或时间帧，逐项处理。

- `0488` `                    video_path = os.path.join(ep_info["video_dir"], f"{cam}_color.mp4")`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0489` `                    frame = iio.imread(video_path, index=orig_idx)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0490` `                    if frame.shape[-1] == 4:`

  解释：条件分支，根据当前状态选择不同处理路径。

- `0491` `                        frame = frame[..., :3]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0492` `                    frames_cam[cam] = frame`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0493` `                result[fid] = frames_cam`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0494` `            return result`

  解释：返回当前函数的计算结果。

- `0495` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0496` `        chunk_cum = np.cumsum([0] + ep_info["chunk_frame_counts"])`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0497` `        readers = {}`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0498` `        try:`

  解释：开始异常保护块，用于处理可选依赖或资源释放。

- `0499` `            for fid in frame_ids:`

  解释：循环遍历集合或时间帧，逐项处理。

- `0500` `                orig_idx = int(ep_info["ds_indices"][fid])`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0501` `                cid = int(np.searchsorted(chunk_cum[1:], orig_idx, side="right"))`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0502` `                local_idx = orig_idx - int(chunk_cum[cid])`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0503` `                frames_cam = {}`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0504` `                for cam in self.camera_names:`

  解释：循环遍历集合或时间帧，逐项处理。

- `0505` `                    key = (cid, cam)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0506` `                    if key not in readers:`

  解释：条件分支，根据当前状态选择不同处理路径。

- `0507` `                        video_path = os.path.join(ep_info["chunk_paths"][cid], f"{cam}_color.mp4")`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0508` `                        readers[key] = VideoReader(video_path, ctx=cpu(0))`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0509` `                    vr = readers[key]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0510` `                    frames_cam[cam] = vr[min(local_idx, len(vr) - 1)].asnumpy()`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0511` `                result[fid] = frames_cam`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0512` `        finally:`

  解释：无论是否异常都会执行的清理逻辑。

- `0513` `            readers.clear()`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0514` `        return result`

  解释：返回当前函数的计算结果。


## 第 516-540 行

总体说明：_generate_raymap：根据内参和 c2w 为每个像素生成 ray origin 和 ray direction；origin 是相机光心，direction 是该像素射线在世界坐标系下的方向。

- `0516` `    def _generate_raymap(self, intrinsic, c2w, height, width):`

  解释：定义函数，后续缩进代码实现这个功能。

- `0517` `        fx = intrinsic[0, 0]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0518` `        fy = intrinsic[1, 1]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0519` `        cx = intrinsic[0, 2]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0520` `        cy = intrinsic[1, 2]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0521` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0522` `        xs = np.arange(width, dtype=np.float32) + 0.5`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0523` `        ys = np.arange(height, dtype=np.float32) + 0.5`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0524` `        xx, yy = np.meshgrid(xs, ys)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0525` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0526` `        dirs = np.stack(`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0527` `            [`

  解释：结构行，用于闭合或组织上一行开始的多行表达式。

- `0528` `                (xx - cx) / fx,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0529` `                (yy - cy) / fy,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0530` `                np.ones_like(xx),`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0531` `            ],`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0532` `            axis=-1,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0533` `        )`

  解释：结构行，用于闭合或组织上一行开始的多行表达式。

- `0534` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0535` `        rotation = c2w[:3, :3]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0536` `        translation = c2w[:3, 3]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0537` `        rays_d = dirs @ rotation.T`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0538` `        rays_d = rays_d / (np.linalg.norm(rays_d, axis=-1, keepdims=True) + 1e-8)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0539` `        rays_o = np.broadcast_to(translation, rays_d.shape).copy()`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0540` `        return rays_o.astype(np.float32), rays_d.astype(np.float32)`

  解释：返回当前函数的计算结果。


## 第 542-545 行

总体说明：_to_uint8_img：把 float raymap 线性归一化到 0-255，转成 PIL 可保存/可送 VAE 的 RGB 图。

- `0542` `    def _to_uint8_img(self, value, vmin, vmax):`

  解释：定义函数，后续缩进代码实现这个功能。

- `0543` `        value = (value - vmin) / (vmax - vmin + 1e-8)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0544` `        value = np.clip(value, 0.0, 1.0)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0545` `        return (value * 255).astype(np.uint8)`

  解释：返回当前函数的计算结果。


## 第 547-548 行

总体说明：__len__：返回 Dataset 对外暴露的长度，也就是样本窗口数乘 repeat。

- `0547` `    def __len__(self):`

  解释：定义函数，后续缩进代码实现这个功能。

- `0548` `        return self.length`

  解释：返回当前函数的计算结果。


## 第 550-557 行

总体说明：_make_frame_ids：根据 start_idx 构造一个训练样本包含的帧序列，并兼容 context_length>1 时的首帧上下文策略。

- `0550` `    def _make_frame_ids(self, start_idx):`

  解释：定义函数，后续缩进代码实现这个功能。

- `0551` `        if self.context_length > 1:`

  解释：条件分支，根据当前状态选择不同处理路径。

- `0552` `            if np.random.rand() < self.first_round_prob:`

  解释：条件分支，根据当前状态选择不同处理路径。

- `0553` `                horizon = self.num_frames - self.context_length`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0554` `                return np.array([0] * self.context_length + list(range(1, horizon + 1)))`

  解释：返回当前函数的计算结果。

- `0555` `            consecutive_ids = np.arange(start_idx, start_idx + self.num_frames - 1)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0556` `            return np.concatenate([[0], consecutive_ids])`

  解释：返回当前函数的计算结果。

- `0557` `        return np.arange(start_idx, start_idx + self.num_frames)`

  解释：返回当前函数的计算结果。


## 第 559-655 行

总体说明：__getitem__：核心取样逻辑；读取视频帧、多视角投影图、可选 raymap，按 camera_names 竖向拼接三个视角，最后返回 Wan VACE 或 control 训练需要的字段。

- `0559` `    def __getitem__(self, idx):`

  解释：定义函数，后续缩进代码实现这个功能。

- `0560` `        sample_idx = idx % self.total_samples`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0561` `        ep_idx, start_idx = self.sample_indices[sample_idx]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0562` `        info = self.episode_info[ep_idx]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0563` `        frame_ids = self._make_frame_ids(start_idx)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0564` `        frame_map = self._load_frames_by_ids(info, frame_ids.tolist())`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0565` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0566` `        cam_frames = {cam: [] for cam in self.camera_names}`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0567` `        cam_traj_maps = {cam: [] for cam in self.camera_names}`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0568` `        cam_ray_o = {cam: [] for cam in self.camera_names}`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0569` `        cam_ray_d = {cam: [] for cam in self.camera_names}`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0570` `        actions_for_traj = info["abs_actions"][frame_ids]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0571` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0572` `        for cam in self.camera_names:`

  解释：循环遍历集合或时间帧，逐项处理。

- `0573` `            cam_info = info["cameras"][cam]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0574` `            sample_frame = frame_map[frame_ids[0]][cam]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0575` `            ori_h, ori_w = sample_frame.shape[:2]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0576` `            if self.resize_to is not None:`

  解释：条件分支，根据当前状态选择不同处理路径。

- `0577` `                traj_h, traj_w = self.resize_to`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0578` `            else:`

  解释：条件分支的兜底情况。

- `0579` `                traj_h, traj_w = ori_h, ori_w`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0580` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0581` `            scaled_intrinsic = cam_info["intrinsic"].copy()`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0582` `            scaled_intrinsic[0, 0] *= traj_w / ori_w`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0583` `            scaled_intrinsic[0, 2] *= traj_w / ori_w`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0584` `            scaled_intrinsic[1, 1] *= traj_h / ori_h`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0585` `            scaled_intrinsic[1, 2] *= traj_h / ori_h`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0586` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0587` `            traj_maps = generate_traj_map(`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0588` `                actions_for_traj,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0589` `                cam_info["w2c"][frame_ids],`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0590` `                scaled_intrinsic,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0591` `                traj_h,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0592` `                traj_w,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0593` `                radius=self.traj_radius,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0594` `                radius_mode=self.traj_radius_mode,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0595` `                min_radius=self.traj_min_radius,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0596` `                max_radius=self.traj_max_radius,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0597` `                ref_depth=self.traj_ref_depth,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0598` `                near_depth=self.traj_near_depth,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0599` `                far_depth=self.traj_far_depth,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0600` `            )`

  解释：结构行，用于闭合或组织上一行开始的多行表达式。

- `0601` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0602` `            for i, fid in enumerate(frame_ids):`

  解释：循环遍历集合或时间帧，逐项处理。

- `0603` `                img = frame_map[fid][cam]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0604` `                if img.max() <= 1.0:`

  解释：条件分支，根据当前状态选择不同处理路径。

- `0605` `                    img = (img * 255).clip(0, 255)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0606` `                pil_img = Image.fromarray(img.astype(np.uint8))`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0607` `                if self.resize_to is not None:`

  解释：条件分支，根据当前状态选择不同处理路径。

- `0608` `                    pil_img = pil_img.resize((traj_w, traj_h), Image.BILINEAR)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0609` `                cam_frames[cam].append(pil_img)`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0610` `                cam_traj_maps[cam].append(Image.fromarray(traj_maps[i]))`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0611` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0612` `                if self.output_raymap:`

  解释：条件分支，根据当前状态选择不同处理路径。

- `0613` `                    c2w = cam_info["c2w"][fid]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0614` `                    ray_o, ray_d = self._generate_raymap(scaled_intrinsic, c2w, traj_h, traj_w)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0615` `                    cam_ray_o[cam].append(Image.fromarray(self._to_uint8_img(ray_o, self.ray_o_vmin, self.ray_o_vmax)))`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0616` `                    cam_ray_d[cam].append(Image.fromarray(self._to_uint8_img(ray_d, self.ray_d_vmin, self.ray_d_vmax)))`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0617` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0618` `        video_list = []`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0619` `        control_list = []`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0620` `        ray_o_list = []`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0621` `        ray_d_list = []`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0622` `        for i in range(len(frame_ids)):`

  解释：循环遍历集合或时间帧，逐项处理。

- `0623` `            concat_img = np.concatenate([np.array(cam_frames[cam][i]) for cam in self.camera_names], axis=0)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0624` `            concat_control = np.concatenate([np.array(cam_traj_maps[cam][i]) for cam in self.camera_names], axis=0)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0625` `            video_list.append(Image.fromarray(concat_img))`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0626` `            control_list.append(Image.fromarray(concat_control))`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0627` `            if self.output_raymap:`

  解释：条件分支，根据当前状态选择不同处理路径。

- `0628` `                concat_ray_o = np.concatenate([np.array(cam_ray_o[cam][i]) for cam in self.camera_names], axis=0)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0629` `                concat_ray_d = np.concatenate([np.array(cam_ray_d[cam][i]) for cam in self.camera_names], axis=0)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0630` `                ray_o_list.append(Image.fromarray(concat_ray_o))`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0631` `                ray_d_list.append(Image.fromarray(concat_ray_d))`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0632` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0633` `        if self.type == "vace":`

  解释：条件分支，根据当前状态选择不同处理路径。

- `0634` `            sample = {`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0635` `                "video": video_list,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0636` `                "vace_reference_image": [video_list[0]],`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0637` `                "vace_video": control_list,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0638` `                "prompt": "机械臂按照要求移动夹爪执行任务",`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0639` `            }`

  解释：结构行，用于闭合或组织上一行开始的多行表达式。

- `0640` `            if self.output_raymap:`

  解释：条件分支，根据当前状态选择不同处理路径。

- `0641` `                sample["ray_map_o"] = ray_o_list`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0642` `                sample["ray_map_d"] = ray_d_list`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0643` `            return sample`

  解释：返回当前函数的计算结果。

- `0644` `        if self.type == "control":`

  解释：条件分支，根据当前状态选择不同处理路径。

- `0645` `            sample = {`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0646` `                "video": video_list,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0647` `                "reference_image": [video_list[0]],`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0648` `                "control_video": control_list,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0649` `                "prompt": "机械臂按照要求移动夹爪执行任务",`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0650` `            }`

  解释：结构行，用于闭合或组织上一行开始的多行表达式。

- `0651` `            if self.output_raymap:`

  解释：条件分支，根据当前状态选择不同处理路径。

- `0652` `                sample["ray_map_o"] = ray_o_list`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0653` `                sample["ray_map_d"] = ray_d_list`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0654` `            return sample`

  解释：返回当前函数的计算结果。

- `0655` `        raise ValueError(f"Unsupported dataset_type: {self.type}")`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。


## 第 658-659 行

总体说明：类别名：提供首字母大写的 AgiBotWCDataset4WanControlmultiview，方便 import 时不受 control/Control 命名差异影响。

- `0658` `# Also expose the conventional capitalization for easier importing.`

  解释：注释行，说明下面代码的兼容目的或设计意图。

- `0659` `AgiBotWCDataset4WanControlmultiview = AgiBotWCDataset4Wancontrolmultiview`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。


## 第 662-697 行

总体说明：save_multiview_dataset_visualization：抽一个样本，把 video、traj，以及可选 raymap 横向拼接成 mp4，便于人工检查 dataset 输出。

- `0662` `def save_multiview_dataset_visualization(`

  解释：定义函数，后续缩进代码实现这个功能。

- `0663` `    out_dir,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0664` `    base_path="/mnt/workspace/zsq/Agi2024subset_split/train",`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0665` `    sample_index=200,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0666` `    num_frames=9,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0667` `    episode_limit=2,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0668` `    radius_mode="constant",`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0669` `    output_raymap=False,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0670` `):`

  解释：结构行，用于闭合或组织上一行开始的多行表达式。

- `0671` `    os.makedirs(out_dir, exist_ok=True)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0672` `    dataset = AgiBotWCDataset4Wancontrolmultiview(`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0673` `        base_path=base_path,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0674` `        num_frames=num_frames,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0675` `        episode_limit=episode_limit,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0676` `        repeat=1,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0677` `        dataset_type="vace",`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0678` `        traj_radius_mode=radius_mode,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0679` `        output_raymap=output_raymap,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0680` `    )`

  解释：结构行，用于闭合或组织上一行开始的多行表达式。

- `0681` `    sample = dataset[sample_index]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0682` `    video = sample["video"]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0683` `    control = sample["vace_video"]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0684` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0685` `    out_path = os.path.join(out_dir, f"sample_multiview_video_traj_{radius_mode}.mp4")`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0686` `    ray_o = sample.get("ray_map_o")`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0687` `    ray_d = sample.get("ray_map_d")`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0688` ``

  解释：空行，用于分隔逻辑块，提高可读性。

- `0689` `    writer = imageio.get_writer(out_path, fps=5)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0690` `    for i, (img, traj) in enumerate(zip(video, control)):`

  解释：循环遍历集合或时间帧，逐项处理。

- `0691` `        parts = [np.array(img), np.array(traj)]`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0692` `        if ray_o is not None and ray_d is not None:`

  解释：条件分支，根据当前状态选择不同处理路径。

- `0693` `            parts.extend([np.array(ray_o[i]), np.array(ray_d[i])])`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0694` `        concat = np.concatenate(parts, axis=1)`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0695` `        writer.append_data(concat)`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0696` `    writer.close()`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0697` `    return out_path`

  解释：返回当前函数的计算结果。


## 第 700-721 行

总体说明：save_radius_mode_comparison_visualizations：分别用 constant/perspective/depth_norm 三种半径策略生成对比视频。

- `0700` `def save_radius_mode_comparison_visualizations(`

  解释：定义函数，后续缩进代码实现这个功能。

- `0701` `    out_dir,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0702` `    base_path="/mnt/workspace/zsq/Agi2024subset_split/train",`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0703` `    sample_index=200,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0704` `    num_frames=9,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0705` `    episode_limit=2,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0706` `    output_raymap=False,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0707` `):`

  解释：结构行，用于闭合或组织上一行开始的多行表达式。

- `0708` `    paths = []`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0709` `    for radius_mode in ["constant", "perspective", "depth_norm"]:`

  解释：循环遍历集合或时间帧，逐项处理。

- `0710` `        paths.append(`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0711` `            save_multiview_dataset_visualization(`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0712` `                out_dir=out_dir,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0713` `                base_path=base_path,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0714` `                sample_index=sample_index,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0715` `                num_frames=num_frames,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0716` `                episode_limit=episode_limit,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0717` `                radius_mode=radius_mode,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0718` `                output_raymap=output_raymap,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0719` `            )`

  解释：结构行，用于闭合或组织上一行开始的多行表达式。

- `0720` `        )`

  解释：结构行，用于闭合或组织上一行开始的多行表达式。

- `0721` `    return paths`

  解释：返回当前函数的计算结果。


## 第 724-740 行

总体说明：save_raymap_visualization：对 save_multiview_dataset_visualization 的便捷封装，固定 output_raymap=True，生成 video|traj|ray_o|ray_d 拼接视频。

- `0724` `def save_raymap_visualization(`

  解释：定义函数，后续缩进代码实现这个功能。

- `0725` `    out_dir,`

  解释：表达式或结构续行，参与上一行/当前代码块的计算。

- `0726` `    base_path="/mnt/workspace/zsq/Agi2024subset_split/train",`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0727` `    sample_index=200,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0728` `    num_frames=9,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0729` `    episode_limit=2,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0730` `    radius_mode="constant",`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0731` `):`

  解释：结构行，用于闭合或组织上一行开始的多行表达式。

- `0732` `    return save_multiview_dataset_visualization(`

  解释：返回当前函数的计算结果。

- `0733` `        out_dir=out_dir,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0734` `        base_path=base_path,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0735` `        sample_index=sample_index,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0736` `        num_frames=num_frames,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0737` `        episode_limit=episode_limit,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0738` `        radius_mode=radius_mode,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0739` `        output_raymap=True,`

  解释：赋值或参数配置，把右侧计算结果保存到左侧变量/字段。

- `0740` `    )`

  解释：结构行，用于闭合或组织上一行开始的多行表达式。
