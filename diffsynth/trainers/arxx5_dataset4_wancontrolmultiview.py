#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None

try:
    import pyarrow.parquet as pq
except Exception as exc:
    raise ImportError("This dataset needs pyarrow: pip install pyarrow") from exc

try:
    import av
except Exception:
    av = None

try:
    from decord import VideoReader, cpu
except Exception:
    VideoReader = None
    cpu = None

try:
    from .arxx5_urdf_fk import configure as configure_fk
    from .arxx5_urdf_fk import left_fk, right_fk
except ImportError:
    from arxx5_urdf_fk import configure as configure_fk
    from arxx5_urdf_fk import left_fk, right_fk

CAMERAS = ("head", "left_wrist", "right_wrist")
ColorListLeft = [(0, 0, 255), (255, 255, 0), (0, 255, 255)]
ColorListRight = [(255, 0, 255), (255, 0, 0), (0, 255, 0)]
EndEffectorPts = np.array([[0,0,0,1],[0.1,0,0,1],[0,0.1,0,1],[0,0,0.1,1]], dtype=np.float32)
Gripper2EEFCvt = np.array([[1,0,0,0.2],[0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float32)

FIXED_AXIS_MAPS: Dict[str, np.ndarray] = {
    "head": np.array([[0.0,-1.0,0.0],[0.0,0.0,-1.0],[1.0,0.0,0.0]], dtype=np.float64),
    "left_wrist": np.array([[0.0,-1.0,0.0],[-1.0,0.0,0.0],[0.0,0.0,-1.0]], dtype=np.float64),
    "right_wrist": np.array([[0.0,-1.0,0.0],[-1.0,0.0,0.0],[0.0,0.0,-1.0]], dtype=np.float64),
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def invert_transform(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def quaternion_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    tx, ty, tz = 2.0*x, 2.0*y, 2.0*z
    twx, twy, twz = tx*w, ty*w, tz*w
    txx, txy, txz = tx*x, ty*x, tz*x
    tyy, tyz, tzz = ty*y, tz*y, tz*z
    return np.stack([
        1.0-(tyy+tzz), txy-twz, txz+twy,
        txy+twz, 1.0-(txx+tzz), tyz-twx,
        txz-twy, tyz+twx, 1.0-(txx+tyy),
    ], axis=-1).reshape(quat.shape[:-1] + (3, 3)).astype(np.float32)


def get_transformation_matrix_from_quat(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    mat = np.tile(np.eye(4, dtype=np.float32), (pose.shape[0], 1, 1))
    mat[:, :3, :3] = quaternion_xyzw_to_matrix(pose[:, 3:7])
    mat[:, :3, 3] = pose[:, :3]
    return mat


def pose_qwxyz_to_transform(xyz_qwxyz: np.ndarray) -> np.ndarray:
    x, y, z, qw, qx, qy, qz = np.asarray(xyz_qwxyz, dtype=np.float64)
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    w, xq, yq, zq = q
    R = np.array([
        [1-2*(yq*yq+zq*zq), 2*(xq*yq-zq*w), 2*(xq*zq+yq*w)],
        [2*(xq*yq+zq*w), 1-2*(xq*xq+zq*zq), 2*(yq*zq-xq*w)],
        [2*(xq*zq-yq*w), 2*(yq*zq+xq*w), 1-2*(xq*xq+yq*yq)],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    tr = float(np.trace(R))
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return q


def transform_to_pose_qwxyz(T: np.ndarray) -> np.ndarray:
    out = np.zeros(7, dtype=np.float64)
    out[:3] = T[:3, 3]
    out[3:] = rotmat_to_quat_wxyz(T[:3, :3])
    return out


# def normalize_to_0_120(v: np.ndarray) -> np.ndarray:
#     v = np.asarray(v, dtype=np.float32)
#     lo, hi = float(np.min(v)), float(np.max(v))
#     if hi - lo < 1e-6:
#         return np.full_like(v, 60.0, dtype=np.float32)
#     return (v - lo) / (hi - lo) * 120.0
def normalize_to_0_120(v: np.ndarray) -> np.ndarray:
    """
    Normalize gripper value from fixed physical/statistical range [0, 3] to [0, 120].

    This makes gripper colors comparable across episodes.
    """
    v = np.asarray(v, dtype=np.float32)
    lo, hi = 0.0, 0.082
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0) * 120.0

def make_abs_actions_from_poses(left_pose_qwxyz: np.ndarray, right_pose_qwxyz: np.ndarray, state: np.ndarray) -> np.ndarray:
    T = left_pose_qwxyz.shape[0]
    out = np.zeros((T, 16), dtype=np.float32)
    out[:, 0:3] = left_pose_qwxyz[:, 0:3]
    out[:, 3:7] = np.stack([left_pose_qwxyz[:, 4], left_pose_qwxyz[:, 5], left_pose_qwxyz[:, 6], left_pose_qwxyz[:, 3]], axis=1)
    out[:, 8:11] = right_pose_qwxyz[:, 0:3]
    out[:, 11:15] = np.stack([right_pose_qwxyz[:, 4], right_pose_qwxyz[:, 5], right_pose_qwxyz[:, 6], right_pose_qwxyz[:, 3]], axis=1)
    out[:, 7] = normalize_to_0_120(state[:, 6])
    out[:, 15] = normalize_to_0_120(state[:, 13])
    return out


def make_ee_pose_from_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if action.ndim != 2 or action.shape[1] < 14:
        raise ValueError(f"Expected action shape [T, >=14], got {action.shape}")
    left_pose_qwxyz = left_fk(action[:, 0:6])
    right_pose_qwxyz = right_fk(action[:, 7:13])
    return np.concatenate([left_pose_qwxyz, right_pose_qwxyz], axis=1).astype(np.float32)


def grip_to_left_color(v: float) -> Tuple[int, int, int]:
    v = float(np.clip(v, 0.0, 1.0))
    return (int(40 + 140 * v), int(120 + 120 * v), int(40 + 40 * (1 - v)))


def grip_to_right_color(v: float) -> Tuple[int, int, int]:
    v = float(np.clip(v, 0.0, 1.0))
    return (int(120 + 135 * v), int(30 + 80 * v), int(30 + 80 * v))


def compute_depth_radius(depth: float, radius: int, radius_mode: str = "constant", min_radius: int = 20, max_radius: int = 60, ref_depth: float = 0.30, near_depth: float = 0.19, far_depth: float = 0.69) -> int:
    if radius_mode == "constant":
        return int(radius)
    safe_depth = max(float(depth), 1e-4)
    if radius_mode == "perspective":
        dynamic_radius = float(radius) * float(ref_depth) / safe_depth
    elif radius_mode == "depth_norm":
        alpha = (float(far_depth) - safe_depth) / (float(far_depth) - float(near_depth) + 1e-8)
        dynamic_radius = float(min_radius) + float(np.clip(alpha, 0.0, 1.0)) * (float(max_radius) - float(min_radius))
    else:
        raise ValueError(f"Unsupported radius_mode: {radius_mode}")
    return int(round(np.clip(dynamic_radius, min_radius, max_radius)))


def generate_traj_map(abs_actions: np.ndarray, w2c: np.ndarray, intrinsic: np.ndarray, h: int, w: int, radius: int = 40, radius_mode: str = "perspective", min_radius: int = 14, max_radius: int = 58, ref_depth: float = 0.30, near_depth: float = 0.19, far_depth: float = 0.69) -> List[np.ndarray]:
    """Original-style projection: abs_actions and w2c must be in the same per-camera world frame."""
    abs_actions = np.asarray(abs_actions, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    w2c = np.asarray(w2c, dtype=np.float32)
    if w2c.ndim == 2:
        w2c = np.repeat(w2c[None], abs_actions.shape[0], axis=0)

    ee_key_pts = EndEffectorPts.reshape(1, 4, 4).transpose(0, 2, 1)
    cvt_matrix = Gripper2EEFCvt.reshape(1, 4, 4)
    pose_l_mat = get_transformation_matrix_from_quat(abs_actions[:, 0:7])
    pose_r_mat = get_transformation_matrix_from_quat(abs_actions[:, 8:15])
    ee2cam_l = np.matmul(np.matmul(w2c, pose_l_mat), cvt_matrix)
    ee2cam_r = np.matmul(np.matmul(w2c, pose_r_mat), cvt_matrix)
    pts_l = np.matmul(ee2cam_l, ee_key_pts)
    pts_r = np.matmul(ee2cam_r, ee_key_pts)

    intr = intrinsic.reshape(1, 3, 3)
    uvs_l = np.matmul(intr, pts_l[:, :3, :])
    uvs_l = (uvs_l / (pts_l[:, 2:3, :] + 1e-8))[:, :2, :].transpose(0, 2, 1).astype(np.int64)
    uvs_r = np.matmul(intr, pts_r[:, :3, :])
    uvs_r = (uvs_r / (pts_r[:, 2:3, :] + 1e-8))[:, :2, :].transpose(0, 2, 1).astype(np.int64)

    img_list: List[np.ndarray] = []
    for i in range(abs_actions.shape[0]):
        img = np.zeros((h, w, 3), dtype=np.uint8) + 50
        color_l = grip_to_left_color(abs_actions[i, 7].item() / 120.0)
        color_r = grip_to_right_color(abs_actions[i, 15].item() / 120.0)

        for points, pts, color in zip([uvs_l[i], uvs_r[i]], [pts_l[i], pts_r[i]], [color_l, color_r]):
            if float(pts[2, 0]) <= 1e-6:
                continue
            base = np.asarray(points[0])
            if base[0] < 0 or base[0] >= w or base[1] < 0 or base[1] >= h:
                continue
            radius_i = compute_depth_radius(float(pts[2, 0]), radius, radius_mode, min_radius, max_radius, ref_depth, near_depth, far_depth)
            if cv2 is not None:
                cv2.circle(img, tuple(base), radius_i, color, -1)
            else:
                from PIL import ImageDraw
                pil = Image.fromarray(img)
                d = ImageDraw.Draw(pil)
                x, y = int(base[0]), int(base[1])
                d.ellipse([x-radius_i, y-radius_i, x+radius_i, y+radius_i], fill=tuple(color))
                img = np.asarray(pil)

        for points, pts, colors in zip([uvs_l[i], uvs_r[i]], [pts_l[i], pts_r[i]], [ColorListLeft, ColorListRight]):
            if float(pts[2, 0]) <= 1e-6:
                continue
            base = np.asarray(points[0])
            if base[0] < 0 or base[0] >= w or base[1] < 0 or base[1] >= h:
                continue
            for point_idx, point in enumerate(points):
                if point_idx == 0 or float(pts[2, point_idx]) <= 1e-6:
                    continue
                if cv2 is not None:
                    cv2.line(img, tuple(base), tuple(point[:2]), colors[point_idx - 1], 8)
                else:
                    from PIL import ImageDraw
                    pil = Image.fromarray(img)
                    ImageDraw.Draw(pil).line([tuple(map(int, base)), tuple(map(int, point[:2]))], fill=tuple(colors[point_idx - 1]), width=8)
                    img = np.asarray(pil)
        img_list.append(img)
    return img_list


def left_multiply_camera_axis(w2c: np.ndarray, R_axis: np.ndarray) -> np.ndarray:
    A = np.eye(4, dtype=np.float64)
    A[:3, :3] = R_axis
    w2c = np.asarray(w2c, dtype=np.float64)
    return (A[None] @ w2c).astype(np.float32) if w2c.ndim == 3 else (A @ w2c).astype(np.float32)


def apply_fixed_camera_axes(w2c_by_cam: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {cam: left_multiply_camera_axis(w2c_by_cam[cam], FIXED_AXIS_MAPS[cam]) for cam in w2c_by_cam}


def get_color_intrinsics() -> Dict[str, np.ndarray]:
    return {
        "head": np.array([[390.48712158203125,0.0,315.5419616699219],[0.0,389.9453430175781,236.520263671875],[0.0,0.0,1.0]], dtype=np.float32),
        "left_wrist": np.array([[390.5675354003906,0.0,318.5806884765625],[0.0,390.024169921875,236.2669677734375],[0.0,0.0,1.0]], dtype=np.float32),
        "right_wrist": np.array([[392.8573913574219,0.0,320.2360534667969],[0.0,392.26654052734375,238.5270233154297],[0.0,0.0,1.0]], dtype=np.float32),
    }


def load_calib_extrinsics(head_left_calib: Path, head_right_calib: Path, left_calib: Path, right_calib: Path) -> Dict[str, np.ndarray]:
    hl = load_json(head_left_calib)
    hr = load_json(head_right_calib)
    l = load_json(left_calib)
    r = load_json(right_calib)
    T_bl2cam = np.asarray(hl["T_base2cam"], dtype=np.float64)
    T_br2cam = np.asarray(hr["T_base2cam"], dtype=np.float64)
    T_cam2bl = invert_transform(T_bl2cam)
    T_cam2br = invert_transform(T_br2cam)
    T_br2bl = T_cam2bl @ T_br2cam
    T_bl2br = T_cam2br @ T_bl2cam
    return {
        "head_T_bl2cam": T_bl2cam,
        "head_T_br2cam": T_br2cam,
        "T_br2bl": T_br2bl,
        "T_bl2br": T_bl2br,
        "left_T_gripper2cam": np.asarray(l["T_gripper2cam"], dtype=np.float64),
        "right_T_gripper2cam": np.asarray(r["T_gripper2cam"], dtype=np.float64),
    }


def make_center_base_transforms(extr: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    T_br2bl = extr["T_br2bl"]
    center_origin_in_left = 0.5 * T_br2bl[:3, 3]
    T_center2bl = np.eye(4, dtype=np.float64)
    T_center2bl[:3, 3] = center_origin_in_left
    T_bl2center = invert_transform(T_center2bl)
    T_center2br = extr["T_bl2br"] @ T_center2bl
    T_br2center = invert_transform(T_center2br)
    return {"T_center2bl": T_center2bl, "T_bl2center": T_bl2center, "T_center2br": T_center2br, "T_br2center": T_br2center}


def make_per_camera_sequences(ee_pose: np.ndarray, state: np.ndarray, extr: Dict[str, np.ndarray], camera_axis_mode: str = "fixed") -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Original traj_map flow: per-camera actions + per-camera w2c."""
    T = int(ee_pose.shape[0])
    T_gL2bL = np.zeros((T, 4, 4), dtype=np.float64)
    T_gR2bR = np.zeros((T, 4, 4), dtype=np.float64)
    T_gR2bL = np.zeros((T, 4, 4), dtype=np.float64)
    T_gL2bR = np.zeros((T, 4, 4), dtype=np.float64)
    w2c_left = np.zeros((T, 4, 4), dtype=np.float64)
    w2c_right = np.zeros((T, 4, 4), dtype=np.float64)

    for i in range(T):
        T_g2b_left = pose_qwxyz_to_transform(ee_pose[i, 0:7])
        T_g2b_right = pose_qwxyz_to_transform(ee_pose[i, 7:14])
        T_gL2bL[i] = T_g2b_left
        T_gR2bR[i] = T_g2b_right
        T_gR2bL[i] = extr["T_br2bl"] @ T_g2b_right
        T_gL2bR[i] = extr["T_bl2br"] @ T_g2b_left
        w2c_left[i] = extr["left_T_gripper2cam"] @ invert_transform(T_g2b_left)
        w2c_right[i] = extr["right_T_gripper2cam"] @ invert_transform(T_g2b_right)

    w2c_head = np.repeat(extr["head_T_br2cam"][None, :, :], T, axis=0)

    left_pose_bl = np.stack([transform_to_pose_qwxyz(Tm) for Tm in T_gL2bL], axis=0)
    right_pose_bl = np.stack([transform_to_pose_qwxyz(Tm) for Tm in T_gR2bL], axis=0)
    left_pose_br = np.stack([transform_to_pose_qwxyz(Tm) for Tm in T_gL2bR], axis=0)
    right_pose_br = np.stack([transform_to_pose_qwxyz(Tm) for Tm in T_gR2bR], axis=0)

    actions_by_cam = {
        "head": make_abs_actions_from_poses(left_pose_br, right_pose_br, state),
        "left_wrist": make_abs_actions_from_poses(left_pose_bl, right_pose_bl, state),
        "right_wrist": make_abs_actions_from_poses(left_pose_br, right_pose_br, state),
    }
    w2c_by_cam = {"head": w2c_head.astype(np.float32), "left_wrist": w2c_left.astype(np.float32), "right_wrist": w2c_right.astype(np.float32)}
    if camera_axis_mode == "fixed":
        w2c_by_cam = apply_fixed_camera_axes(w2c_by_cam)
    elif camera_axis_mode in {"identity", "frames3d"}:
        pass
    else:
        raise ValueError(f"Unsupported camera_axis_mode: {camera_axis_mode}")
    return actions_by_cam, w2c_by_cam


def make_center_ray_w2c_sequences(ee_pose: np.ndarray, extr: Dict[str, np.ndarray], camera_axis_mode: str = "fixed") -> Dict[str, np.ndarray]:
    """Raymap flow: center-base -> camera w2c for each camera."""
    T = int(ee_pose.shape[0])
    center = make_center_base_transforms(extr)
    w2c_left = np.zeros((T, 4, 4), dtype=np.float64)
    w2c_right = np.zeros((T, 4, 4), dtype=np.float64)
    for i in range(T):
        T_gL2bl = pose_qwxyz_to_transform(ee_pose[i, 0:7])
        T_gR2br = pose_qwxyz_to_transform(ee_pose[i, 7:14])
        w2c_left[i] = extr["left_T_gripper2cam"] @ invert_transform(T_gL2bl) @ center["T_center2bl"]
        w2c_right[i] = extr["right_T_gripper2cam"] @ invert_transform(T_gR2br) @ center["T_center2br"]
    w2c_head = np.repeat((extr["head_T_bl2cam"] @ center["T_center2bl"])[None], T, axis=0)
    w2c_by_cam = {"head": w2c_head.astype(np.float32), "left_wrist": w2c_left.astype(np.float32), "right_wrist": w2c_right.astype(np.float32)}
    if camera_axis_mode == "fixed":
        w2c_by_cam = apply_fixed_camera_axes(w2c_by_cam)
    elif camera_axis_mode in {"identity", "frames3d"}:
        pass
    else:
        raise ValueError(f"Unsupported camera_axis_mode: {camera_axis_mode}")
    return w2c_by_cam


def scale_intrinsic(intrinsic: np.ndarray, in_h: int, in_w: int, out_h: int, out_w: int) -> np.ndarray:
    K = np.asarray(intrinsic, dtype=np.float32).copy()
    K[0, 0] *= out_w / float(in_w)
    K[0, 2] *= out_w / float(in_w)
    K[1, 1] *= out_h / float(in_h)
    K[1, 2] *= out_h / float(in_h)
    return K


def generate_raymap(intrinsic: np.ndarray, c2w: np.ndarray, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy = intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]
    xs = np.arange(width, dtype=np.float32) + 0.5
    ys = np.arange(height, dtype=np.float32) + 0.5
    xx, yy = np.meshgrid(xs, ys)
    dirs = np.stack([(xx - cx) / fx, (yy - cy) / fy, np.ones_like(xx)], axis=-1)
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    rays_d = dirs @ R.T
    rays_d = rays_d / (np.linalg.norm(rays_d, axis=-1, keepdims=True) + 1e-8)
    rays_o = np.broadcast_to(t, rays_d.shape).copy()
    return rays_o.astype(np.float32), rays_d.astype(np.float32)


def to_uint8_img(value: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    value = (value - vmin) / (vmax - vmin + 1e-8)
    return (np.clip(value, 0.0, 1.0) * 255).astype(np.uint8)


def _read_video_rgb_by_indices_pyav(path: str, indices: Sequence[int]) -> Optional[np.ndarray]:
    if av is None:
        return None
    indices = [int(i) for i in indices]
    wanted = set(indices)
    max_idx = max(indices) if indices else -1
    frames: Dict[int, np.ndarray] = {}
    try:
        container = av.open(path)
        try:
            stream = next((s for s in container.streams if s.type == "video"), None)
            if stream is None:
                return None
            for frame_idx, frame in enumerate(container.decode(stream)):
                if frame_idx in wanted:
                    frames[frame_idx] = frame.to_ndarray(format="rgb24")
                    if len(frames) == len(wanted):
                        break
                if frame_idx > max_idx:
                    break
        finally:
            container.close()
    except Exception:
        return None
    if not frames:
        return None
    available = sorted(frames.keys())
    out = []
    for idx in indices:
        if idx in frames:
            out.append(frames[idx])
        else:
            earlier = [k for k in available if k <= idx]
            out.append(frames[max(earlier) if earlier else available[0]])
    return np.stack(out, axis=0).astype(np.uint8)


def _read_video_rgb_by_indices_cv2(path: str, indices: Sequence[int]) -> Optional[np.ndarray]:
    if cv2 is None:
        return None
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            return None
        out = []
        try:
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame_bgr = cap.read()
                if not ok:
                    return None
                out.append(frame_bgr[:, :, ::-1].copy())
        finally:
            cap.release()
        return np.stack(out, axis=0).astype(np.uint8)
    except Exception:
        return None


def _read_video_rgb_by_indices_decord(path: str, indices: Sequence[int]) -> Optional[np.ndarray]:
    if VideoReader is None:
        return None
    try:
        vr = VideoReader(path, ctx=cpu(0))
        safe = [min(int(i), len(vr) - 1) for i in indices]
        frames = vr.get_batch(safe).asnumpy()
        del vr
        if frames.shape[-1] == 4:
            frames = frames[..., :3]
        return frames.astype(np.uint8)
    except Exception:
        return None


def read_video_rgb_by_indices(path: Path, indices: Sequence[int]) -> np.ndarray:
    path_str = str(path)
    indices = [int(i) for i in indices]
    if not indices:
        return np.zeros((0, 480, 640, 3), dtype=np.uint8)
    for reader in (_read_video_rgb_by_indices_pyav, _read_video_rgb_by_indices_cv2, _read_video_rgb_by_indices_decord):
        frames = reader(path_str, indices)
        if frames is not None:
            return frames
    raise RuntimeError(f"Cannot read video frames: {path_str}\nindices={indices[:10]}...\nInstall PyAV for AV1 LeRobot videos: pip install av")


def get_video_len(path: Path) -> int:
    path_str = str(path)
    if av is not None:
        try:
            container = av.open(path_str)
            try:
                stream = next((s for s in container.streams if s.type == "video"), None)
                if stream is not None:
                    if stream.frames is not None and int(stream.frames) > 0:
                        return int(stream.frames)
                    count = 0
                    for _ in container.decode(stream):
                        count += 1
                    if count > 0:
                        return count
            finally:
                container.close()
        except Exception as e:
            pyav_err = repr(e)
    else:
        pyav_err = "PyAV is not installed"
    if cv2 is not None:
        try:
            cap = cv2.VideoCapture(path_str)
            try:
                if cap.isOpened():
                    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if n > 0:
                        return n
                cv2_err = "OpenCV opened no stream or returned zero frames"
            finally:
                cap.release()
        except Exception as e:
            cv2_err = repr(e)
    else:
        cv2_err = "OpenCV is not installed"
    if VideoReader is not None:
        try:
            vr = VideoReader(path_str, ctx=cpu(0))
            n = len(vr)
            del vr
            if n > 0:
                return n
        except Exception as e:
            decord_err = repr(e)
    else:
        decord_err = "decord is not installed"
    raise RuntimeError(f"Cannot determine video length: {path_str}\nPyAV error: {pyav_err}\nOpenCV error: {cv2_err}\ndecord error: {decord_err}")


def write_video_rgb(frames_rgb: Sequence[np.ndarray], output_path: Path, fps: float = 5.0) -> None:
    import imageio.v2 as imageio
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output_path), fps=fps)
    try:
        for f in frames_rgb:
            writer.append_data(np.asarray(f, dtype=np.uint8))
    finally:
        writer.close()


class Arxx5Dataset4Wancontrolmultiview(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path: str = "/mnt/data/zsq/agx",
        num_frames: int = 9,
        repeat: int = 1,
        episode_stride: int = 1,
        episode_limit: Optional[int] = None,
        episode_indices: Optional[Sequence[int]] = None,
        exclude_episode_indices: Optional[Sequence[int]] = None,
        original_hz: Optional[float] = 30,
        target_hz: Optional[float] = 30,
        stride: int = 1,
        first_round_prob: float = 0.05,
        context_length: int = 1,
        traj_radius: int = 40,
        traj_radius_mode: str = "perspective",
        traj_min_radius: int = 14,
        traj_max_radius: int = 58,
        traj_ref_depth: float = 0.30,
        traj_near_depth: float = 0.19,
        traj_far_depth: float = 0.69,
        output_raymap: bool = False,
        ray_o_vmin: float = -1.5,
        ray_o_vmax: float = 1.5,
        ray_d_vmin: float = -1.0,
        ray_d_vmax: float = 1.0,
        raymap_mode: str = "image",
        resize_to: Optional[Tuple[int, int]] = (160, 224),
        dataset_type: str = "vace",
        camera_names: Optional[Sequence[str]] = None,
        camera_sample_mode: str = "all",
        camera_axis_mode: str = "identity",
        head_left_calib: str = "/mnt/data/zsq/outputs/calib_eye_to_hand_head_left/result_eye_to_hand.json",
        head_right_calib: str = "/mnt/data/zsq/outputs/calib_eye_to_hand_head_right/result_eye_to_hand.json",
        left_calib: str = "/mnt/data/zsq/outputs/calib_eye_in_hand_left/result_eye_in_hand.json",
        right_calib: str = "/mnt/data/zsq/outputs/calib_eye_in_hand_right/result_eye_in_hand.json",
        fk_urdf_path: Optional[str] = None,
        fk_offset_path: Optional[str] = None,
        fk_state_unit: str = "rad",
        prompt: str = "机械臂按照要求移动夹爪执行任务",
    ):
        super().__init__()
        if camera_names is None:
            camera_names = CAMERAS
        if camera_sample_mode not in ("all", "random_one", "cycle_one"):
            raise ValueError(f"Unsupported camera_sample_mode: {camera_sample_mode}")
        if raymap_mode not in ("image", "latent"):
            raise ValueError(f"Unsupported raymap_mode: {raymap_mode}")
        if dataset_type not in ("vace", "control"):
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")

        self.base_path = Path(base_path)
        self.num_frames = int(num_frames)
        self.repeat = int(repeat)
        self.stride = int(stride)
        self.first_round_prob = float(first_round_prob)
        self.context_length = int(context_length)
        self.traj_radius = int(traj_radius)
        self.traj_radius_mode = traj_radius_mode
        self.traj_min_radius = int(traj_min_radius)
        self.traj_max_radius = int(traj_max_radius)
        self.traj_ref_depth = float(traj_ref_depth)
        self.traj_near_depth = float(traj_near_depth)
        self.traj_far_depth = float(traj_far_depth)
        self.output_raymap = bool(output_raymap)
        self.ray_o_vmin = float(ray_o_vmin)
        self.ray_o_vmax = float(ray_o_vmax)
        self.ray_d_vmin = float(ray_d_vmin)
        self.ray_d_vmax = float(ray_d_vmax)
        self.raymap_mode = raymap_mode
        self.vae_spatial_downsample = 8
        self.vae_temporal_downsample = 4
        self.resize_to = resize_to
        self.type = dataset_type
        self.camera_names = list(camera_names)
        self.camera_sample_mode = camera_sample_mode
        self.camera_axis_mode = camera_axis_mode
        self.prompt = prompt
        self.intrinsics = get_color_intrinsics()
        self.extr = load_calib_extrinsics(Path(head_left_calib), Path(head_right_calib), Path(left_calib), Path(right_calib))
        configure_fk(urdf_path=fk_urdf_path, offset_path=fk_offset_path, state_unit=fk_state_unit)
        self.load_from_cache = False
        self.info_json = load_json(self.base_path / "meta" / "info.json")
        fps = float(original_hz if original_hz is not None else self.info_json.get("fps", target_hz or 5))
        if target_hz is None or target_hz <= 0:
            self.downsample_step = 1
            self.output_fps = fps
        else:
            self.downsample_step = max(1, int(round(fps / float(target_hz))))
            self.output_fps = float(target_hz)

        episodes = self._discover_episodes()
        if episode_indices is not None:
            include_indices = {int(idx) for idx in episode_indices}
            episodes = [ep for ep in episodes if ep["episode_index"] in include_indices]
        if exclude_episode_indices is not None:
            exclude_indices = {int(idx) for idx in exclude_episode_indices}
            episodes = [ep for ep in episodes if ep["episode_index"] not in exclude_indices]
        if episode_limit is not None:
            episodes = episodes[: int(episode_limit)]
        self.episodes = episodes[:: int(episode_stride)]

        self.episode_info: List[dict] = []
        self.sample_indices: List[Tuple[int, int]] = []
        for ep in self.episodes:
            info = self._build_episode_info(ep)
            if info is None:
                continue
            valid_ep_idx = len(self.episode_info)
            self.episode_info.append(info)
            max_start = info["T_ds"] - self.num_frames + 1
            for start in range(0, max_start, self.stride):
                self.sample_indices.append((valid_ep_idx, start))

        self.total_samples = len(self.sample_indices)
        self.length = self.total_samples * self.repeat
        print(f"[Arxx5Dataset] base={self.base_path}, fps={fps}, target_hz={target_hz}, step={self.downsample_step}, episodes={len(self.episode_info)}, samples={self.total_samples}, traj=fk_action_per_camera, raymap=center_base")

    def _discover_episodes(self) -> List[dict]:
        data_dir = self.base_path / "data"
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        out = []
        for parquet_path in sorted(data_dir.glob("chunk-*/episode_*.parquet")):
            m = re.search(r"episode_(\d+)\.parquet$", parquet_path.name)
            if m is None:
                continue
            episode_index = int(m.group(1))
            chunk = parquet_path.parent.name
            video_paths = {cam: self.base_path / "videos" / chunk / f"observation.images.{cam}" / f"episode_{episode_index:06d}.mp4" for cam in CAMERAS}
            if all(video_paths[cam].exists() for cam in self.camera_names):
                out.append({"episode_index": episode_index, "parquet_path": parquet_path, "video_paths": video_paths})
        return out

    def _load_episode_action(self, parquet_path: Path) -> np.ndarray:
        table = pq.read_table(parquet_path, columns=["action"])
        return np.asarray(table["action"].to_pylist(), dtype=np.float32)

    def _build_episode_info(self, ep: dict) -> Optional[dict]:
        action = self._load_episode_action(ep["parquet_path"])
        n = action.shape[0]
        for cam in self.camera_names:
            n = min(n, get_video_len(ep["video_paths"][cam]))
        if n < self.num_frames:
            return None
        raw_idx = np.arange(0, n, self.downsample_step, dtype=np.int64)
        action_ds = action[raw_idx]
        fk_ee_pose_ds = make_ee_pose_from_action(action_ds)
        traj_actions_by_cam, traj_w2c_by_cam = make_per_camera_sequences(fk_ee_pose_ds, action_ds, self.extr, self.camera_axis_mode)
        ray_w2c_by_cam = make_center_ray_w2c_sequences(fk_ee_pose_ds, self.extr, self.camera_axis_mode)
        return {
            "episode_index": ep["episode_index"],
            "parquet_path": ep["parquet_path"],
            "video_paths": ep["video_paths"],
            "ds_indices": raw_idx,
            "T_ds": len(raw_idx),
            "traj_actions_by_cam": traj_actions_by_cam,
            "traj_w2c_by_cam": traj_w2c_by_cam,
            "ray_w2c_by_cam": ray_w2c_by_cam,
            "traj_source": "fk_action",
        }

    def __len__(self) -> int:
        return self.length

    def _make_frame_ids(self, start_idx: int) -> np.ndarray:
        if self.context_length > 1:
            if np.random.rand() < self.first_round_prob:
                horizon = self.num_frames - self.context_length
                return np.array([0] * self.context_length + list(range(1, horizon + 1)), dtype=np.int64)
            consecutive_ids = np.arange(start_idx, start_idx + self.num_frames - 1, dtype=np.int64)
            return np.concatenate([[0], consecutive_ids]).astype(np.int64)
        return np.arange(start_idx, start_idx + self.num_frames, dtype=np.int64)

    def _select_camera_names(self, idx: int, sample_idx: int) -> List[str]:
        if self.camera_sample_mode == "all" or len(self.camera_names) <= 1:
            return self.camera_names
        if self.camera_sample_mode == "cycle_one":
            return [self.camera_names[idx % len(self.camera_names)]]
        return [self.camera_names[np.random.randint(len(self.camera_names))]]

    def _load_frames_by_ids(self, info: dict, frame_ids: Sequence[int], camera_names: Sequence[str]) -> Dict[int, Dict[str, np.ndarray]]:
        result = {int(fid): {} for fid in frame_ids}
        raw_indices = [int(info["ds_indices"][int(fid)]) for fid in frame_ids]
        for cam in camera_names:
            frames = read_video_rgb_by_indices(info["video_paths"][cam], raw_indices)
            for fid, frame in zip(frame_ids, frames):
                result[int(fid)][cam] = frame.astype(np.uint8)
        return result

    def _latent_frame_groups(self, num_frames: int) -> List[List[int]]:
        latent_t = (num_frames - 1) // self.vae_temporal_downsample + 1
        groups = [[0] * self.vae_temporal_downsample]
        for i in range(1, latent_t):
            start = 1 + (i - 1) * self.vae_temporal_downsample
            groups.append([min(start + j, num_frames - 1) for j in range(self.vae_temporal_downsample)])
        return groups

    def _generate_raymap_latents_for_camera(self, ray_w2c_seq: np.ndarray, frame_ids: Sequence[int], image_intrinsic: np.ndarray, image_height: int, image_width: int):
        latent_h = image_height // self.vae_spatial_downsample
        latent_w = image_width // self.vae_spatial_downsample
        K = image_intrinsic.copy()
        K[0, 0] *= latent_w / image_width
        K[0, 2] *= latent_w / image_width
        K[1, 1] *= latent_h / image_height
        K[1, 2] *= latent_h / image_height
        ray_o_list, ray_d_list = [], []
        for group in self._latent_frame_groups(len(frame_ids)):
            ro_group, rd_group = [], []
            for pos in group:
                fid = int(frame_ids[pos])
                c2w = np.linalg.inv(ray_w2c_seq[fid]).astype(np.float32)
                ro, rd = generate_raymap(K, c2w, latent_h, latent_w)
                ro_group.append(ro)
                rd_group.append(rd)
            ray_o_list.append(np.concatenate(ro_group, axis=-1))
            ray_d_list.append(np.concatenate(rd_group, axis=-1))
        return ray_o_list, ray_d_list

    def __getitem__(self, idx: int) -> dict:
        if self.total_samples == 0:
            raise IndexError("Dataset has no valid samples")
        sample_idx = idx % self.total_samples
        ep_idx, start_idx = self.sample_indices[sample_idx]
        info = self.episode_info[ep_idx]
        frame_ids = self._make_frame_ids(start_idx)
        active_cams = self._select_camera_names(idx, sample_idx)
        frame_map = self._load_frames_by_ids(info, frame_ids.tolist(), active_cams)

        cam_frames, cam_traj_maps = {cam: [] for cam in active_cams}, {cam: [] for cam in active_cams}
        cam_ray_o, cam_ray_d = {cam: [] for cam in active_cams}, {cam: [] for cam in active_cams}

        for cam in active_cams:
            sample_frame = frame_map[int(frame_ids[0])][cam]
            ori_h, ori_w = sample_frame.shape[:2]
            if self.resize_to is not None:
                out_h, out_w = self.resize_to
            else:
                out_h, out_w = ori_h, ori_w
            K = scale_intrinsic(self.intrinsics[cam], ori_h, ori_w, out_h, out_w)

            traj_actions = info["traj_actions_by_cam"][cam][frame_ids]
            traj_w2c_seq = info["traj_w2c_by_cam"][cam]
            traj_maps = generate_traj_map(
                traj_actions,
                traj_w2c_seq[frame_ids],
                K,
                out_h,
                out_w,
                radius=self.traj_radius,
                radius_mode=self.traj_radius_mode,
                min_radius=self.traj_min_radius,
                max_radius=self.traj_max_radius,
                ref_depth=self.traj_ref_depth,
                near_depth=self.traj_near_depth,
                far_depth=self.traj_far_depth,
            )
            ray_w2c_seq = info["ray_w2c_by_cam"][cam]

            for i, fid in enumerate(frame_ids):
                img = Image.fromarray(frame_map[int(fid)][cam])
                if self.resize_to is not None:
                    img = img.resize((out_w, out_h), Image.BILINEAR)
                cam_frames[cam].append(img)
                cam_traj_maps[cam].append(Image.fromarray(traj_maps[i]))
                if self.output_raymap and self.raymap_mode == "image":
                    c2w = np.linalg.inv(ray_w2c_seq[int(fid)]).astype(np.float32)
                    ro, rd = generate_raymap(K, c2w, out_h, out_w)
                    cam_ray_o[cam].append(Image.fromarray(to_uint8_img(ro, self.ray_o_vmin, self.ray_o_vmax)))
                    cam_ray_d[cam].append(Image.fromarray(to_uint8_img(rd, self.ray_d_vmin, self.ray_d_vmax)))

            if self.output_raymap and self.raymap_mode == "latent":
                ro_lat, rd_lat = self._generate_raymap_latents_for_camera(ray_w2c_seq, frame_ids, K, out_h, out_w)
                cam_ray_o[cam].extend(ro_lat)
                cam_ray_d[cam].extend(rd_lat)

        video_list, control_list, ray_o_list, ray_d_list = [], [], [], []
        for i in range(len(frame_ids)):
            video_list.append(Image.fromarray(np.concatenate([np.array(cam_frames[cam][i]) for cam in active_cams], axis=0)))
            control_list.append(Image.fromarray(np.concatenate([np.array(cam_traj_maps[cam][i]) for cam in active_cams], axis=0)))
            if self.output_raymap and self.raymap_mode == "image":
                ray_o_list.append(Image.fromarray(np.concatenate([np.array(cam_ray_o[cam][i]) for cam in active_cams], axis=0)))
                ray_d_list.append(Image.fromarray(np.concatenate([np.array(cam_ray_d[cam][i]) for cam in active_cams], axis=0)))

        if self.output_raymap and self.raymap_mode == "latent":
            ro_latents, rd_latents = [], []
            for i in range(len(cam_ray_o[active_cams[0]])):
                ro_latents.append(np.concatenate([cam_ray_o[cam][i] for cam in active_cams], axis=0))
                rd_latents.append(np.concatenate([cam_ray_d[cam][i] for cam in active_cams], axis=0))
            ray_o_list = torch.from_numpy(np.stack(ro_latents, axis=0)).permute(3, 0, 1, 2).float()
            ray_d_list = torch.from_numpy(np.stack(rd_latents, axis=0)).permute(3, 0, 1, 2).float()

        if self.type == "vace":
            sample = {"video": video_list, "vace_reference_image": [video_list[0]], "vace_video": control_list, "prompt": self.prompt}
        else:
            sample = {"video": video_list, "reference_image": [video_list[0]], "control_video": control_list, "prompt": self.prompt}
        if self.output_raymap:
            sample["ray_map_o"] = ray_o_list
            sample["ray_map_d"] = ray_d_list
        sample["meta"] = {
            "episode_index": info["episode_index"],
            "frame_ids_ds": frame_ids.tolist(),
            "frame_ids_raw": [int(info["ds_indices"][int(fid)]) for fid in frame_ids],
            "camera_names": active_cams,
            "traj_map_frame": "fk_action_per_camera_actions_and_fk_action_per_camera_w2c",
            "raymap_frame": "center_base_midpoint_between_left_and_right_bases",
            "traj_source": info.get("traj_source", "fk_action"),
        }
        return sample
