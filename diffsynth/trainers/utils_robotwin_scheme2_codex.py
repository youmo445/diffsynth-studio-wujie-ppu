import io
import os
from pathlib import Path

import h5py
import imageio
import matplotlib.cm as cm
import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


ROBOTWIN_CAMERA_NAMES = ["head_camera", "left_camera", "right_camera"]
ROBOTWIN_DEFAULT_DATA_DIR = "/data/zsq/RoboTwin_dataset/dataset/blocks_ranking_rgb/arx-x5_randomized_500/data"
ROBOTWIN_ORIGINAL_HZ = 30
ROBOTWIN_TARGET_HZ = 5
ROBOTWIN_GRIPPER_BIAS = 0.19680682
ROBOTWIN_GRIPPER_BIAS_AXIS = "center_ray_fit"
ROBOTWIN_GRIPPER_VISUAL_OFFSET = np.array([0.19680682, -0.00000105, 0.00911938], dtype=np.float32)
ROBOTWIN_SCHEME_NAME = "scheme2_shared_center_ray_offset"

ColorMapLeft = cm.Greens
ColorMapRight = cm.Reds
ColorListLeft = [(0, 0, 255), (255, 255, 0), (0, 255, 255)]
ColorListRight = [(255, 0, 255), (255, 0, 0), (0, 255, 0)]
EndEffectorPts = np.array(
    [
        [0, 0, 0, 1],
        [0.08, 0, 0, 1],
        [0, 0.08, 0, 1],
        [0, 0, 0.08, 1],
    ],
    dtype=np.float32,
)


def draw_circle(img, center, radius, color):
    if cv2 is not None:
        cv2.circle(img, tuple(map(int, center)), int(radius), tuple(map(int, color)), -1)
        return img
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    x, y = int(center[0]), int(center[1])
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=tuple(map(int, color)))
    return np.array(pil_img)


def draw_line(img, p0, p1, color, width):
    if cv2 is not None:
        cv2.line(img, tuple(map(int, p0)), tuple(map(int, p1)), tuple(map(int, color)), int(width))
        return img
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    draw.line([tuple(map(int, p0)), tuple(map(int, p1))], fill=tuple(map(int, color)), width=int(width))
    return np.array(pil_img)


def discover_robotwin_episodes(data_dir=ROBOTWIN_DEFAULT_DATA_DIR):
    data_dir = Path(data_dir)
    episodes = sorted(data_dir.glob("episode*.hdf5"), key=lambda p: int(p.stem.replace("episode", "")))
    if not episodes:
        raise RuntimeError(f"No episode*.hdf5 found under {data_dir}")
    return episodes


def episode_path(data_dir, episode):
    if isinstance(episode, (str, os.PathLike)) and str(episode).endswith(".hdf5"):
        return Path(episode)
    if isinstance(episode, str) and episode.startswith("episode"):
        name = episode if episode.endswith(".hdf5") else f"{episode}.hdf5"
    else:
        name = f"episode{int(episode)}.hdf5"
    path = Path(data_dir) / name
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def normalize_quat_xyzw(quat):
    quat = np.asarray(quat, dtype=np.float32)
    return quat / (np.linalg.norm(quat, axis=-1, keepdims=True) + 1e-8)


def quat_wxyz_to_xyzw(quat):
    quat = np.asarray(quat, dtype=np.float32)
    return normalize_quat_xyzw(np.stack([quat[..., 1], quat[..., 2], quat[..., 3], quat[..., 0]], axis=-1))


def gripper_bias_vector(bias=ROBOTWIN_GRIPPER_BIAS, axis=ROBOTWIN_GRIPPER_BIAS_AXIS):
    axis = str(axis).lower()
    if axis in ("center_ray_fit", "shared_visual_offset", "visual", "vector"):
        return ROBOTWIN_GRIPPER_VISUAL_OFFSET.astype(np.float32).copy()
    bias_arr = np.asarray(bias, dtype=np.float32)
    if bias_arr.shape == (3,):
        return bias_arr.astype(np.float32).copy()
    vec = np.zeros(3, dtype=np.float32)
    sign = -1.0 if axis.startswith("-") else 1.0
    name = axis[1:] if axis.startswith("-") else axis
    axis_to_idx = {"x": 0, "y": 1, "z": 2}
    if name not in axis_to_idx:
        raise ValueError(f"Unsupported gripper bias axis: {axis}")
    vec[axis_to_idx[name]] = sign * float(bias_arr)
    return vec


def apply_gripper_center_bias_to_pose(pose, bias=ROBOTWIN_GRIPPER_BIAS, axis=ROBOTWIN_GRIPPER_BIAS_AXIS):
    pose = np.asarray(pose, dtype=np.float32).copy()
    rot = quaternion_xyzw_to_matrix(pose[:, 3:7])
    offset = gripper_bias_vector(bias=bias, axis=axis)
    pose[:, :3] = pose[:, :3] + np.einsum("tij,j->ti", rot, offset)
    return pose


def quaternion_xyzw_to_matrix(quat):
    quat = normalize_quat_xyzw(quat)
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    tx, ty, tz = 2.0 * x, 2.0 * y, 2.0 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z
    return np.stack(
        [
            1.0 - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            1.0 - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            1.0 - (txx + tyy),
        ],
        axis=-1,
    ).reshape(quat.shape[:-1] + (3, 3)).astype(np.float32)


def pose7_to_matrix(pose):
    pose = np.asarray(pose, dtype=np.float32)
    mat = np.tile(np.eye(4, dtype=np.float32), (pose.shape[0], 1, 1))
    mat[:, :3, :3] = quaternion_xyzw_to_matrix(pose[:, 3:7])
    mat[:, :3, 3] = pose[:, :3]
    return mat


def extrinsic_cv_to_w2c(extrinsic):
    extrinsic = np.asarray(extrinsic, dtype=np.float32)
    if extrinsic.shape[-2:] == (3, 4):
        bottom_shape = extrinsic.shape[:-2] + (1, 4)
        bottom = np.zeros(bottom_shape, dtype=np.float32)
        bottom[..., 0, 3] = 1.0
        return np.concatenate([extrinsic, bottom], axis=-2)
    if extrinsic.shape[-2:] == (4, 4):
        return extrinsic
    raise ValueError(f"Unsupported extrinsic shape: {extrinsic.shape}")


def w2c_to_c2w(w2c):
    w2c = np.asarray(w2c, dtype=np.float32)
    if w2c.ndim == 2:
        return np.linalg.inv(w2c).astype(np.float32)
    return np.linalg.inv(w2c).astype(np.float32)


def decode_robotwin_rgb_bytes(encoded):
    return np.array(Image.open(io.BytesIO(bytes(encoded))).convert("RGB"))


def read_robotwin_rgb(h5_file, camera_name, frame_idx):
    return decode_robotwin_rgb_bytes(h5_file[f"observation/{camera_name}/rgb"][int(frame_idx)])


def get_episode_length(h5_file):
    return int(h5_file["endpose/left_endpose"].shape[0])


def get_downsample_stride(original_hz=ROBOTWIN_ORIGINAL_HZ, target_hz=ROBOTWIN_TARGET_HZ):
    original_hz = int(original_hz)
    target_hz = int(target_hz)
    if original_hz <= 0 or target_hz <= 0:
        raise ValueError("original_hz and target_hz must be positive.")
    if original_hz % target_hz != 0:
        raise ValueError(f"original_hz={original_hz} must be divisible by target_hz={target_hz}.")
    return original_hz // target_hz


def make_frame_indices(
    length,
    stride=None,
    max_frames=None,
    start=0,
    original_hz=ROBOTWIN_ORIGINAL_HZ,
    target_hz=ROBOTWIN_TARGET_HZ,
):
    if stride is None or int(stride) <= 0:
        stride = get_downsample_stride(original_hz=original_hz, target_hz=target_hz)
    indices = np.arange(int(start), int(length), int(stride), dtype=np.int64)
    if max_frames is not None and max_frames > 0:
        indices = indices[: int(max_frames)]
    return indices


def resize_intrinsic(intrinsic, src_height, src_width, dst_height, dst_width):
    intrinsic = np.asarray(intrinsic, dtype=np.float32).copy()
    intrinsic[0, 0] *= float(dst_width) / float(src_width)
    intrinsic[0, 2] *= float(dst_width) / float(src_width)
    intrinsic[1, 1] *= float(dst_height) / float(src_height)
    intrinsic[1, 2] *= float(dst_height) / float(src_height)
    return intrinsic


def build_robotwin_abs_actions(
    h5_file,
    frame_indices,
    apply_gripper_bias=True,
    gripper_bias=ROBOTWIN_GRIPPER_BIAS,
    gripper_bias_axis=ROBOTWIN_GRIPPER_BIAS_AXIS,
    wrist_center_control=False,
    wrist_center_radius=34,
    wrist_center_axis_len=70,
    wrist_other_pixels_per_meter=220,
    wrist_other_projection_scale=0.45,
    wrist_other_max_offset_ratio=0.38,
):
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    left_raw = np.asarray(h5_file["endpose/left_endpose"][frame_indices], dtype=np.float32)
    right_raw = np.asarray(h5_file["endpose/right_endpose"][frame_indices], dtype=np.float32)
    left_grip = np.asarray(h5_file["endpose/left_gripper"][frame_indices], dtype=np.float32)
    right_grip = np.asarray(h5_file["endpose/right_gripper"][frame_indices], dtype=np.float32)

    left_pose = np.zeros_like(left_raw)
    right_pose = np.zeros_like(right_raw)
    left_pose[:, :3] = left_raw[:, :3]
    right_pose[:, :3] = right_raw[:, :3]
    left_pose[:, 3:7] = quat_wxyz_to_xyzw(left_raw[:, 3:7])
    right_pose[:, 3:7] = quat_wxyz_to_xyzw(right_raw[:, 3:7])

    if apply_gripper_bias:
        left_pose = apply_gripper_center_bias_to_pose(left_pose, bias=gripper_bias, axis=gripper_bias_axis)
        right_pose = apply_gripper_center_bias_to_pose(right_pose, bias=gripper_bias, axis=gripper_bias_axis)

    actions = np.zeros((len(frame_indices), 16), dtype=np.float32)
    actions[:, 0:7] = left_pose
    actions[:, 7] = np.clip(left_grip, 0.0, 1.0)
    actions[:, 8:15] = right_pose
    actions[:, 15] = np.clip(right_grip, 0.0, 1.0)
    return actions


def compute_depth_radius(
    depth,
    radius,
    radius_mode="perspective",
    min_radius=2,
    max_radius=8,
    ref_depth=1.0,
    near_depth=0.3,
    far_depth=2.0,
):
    if radius_mode == "constant":
        return int(radius)
    safe_depth = max(float(depth), 1e-4)
    if radius_mode == "perspective":
        dynamic_radius = float(radius) * float(ref_depth) / safe_depth
    elif radius_mode == "depth_norm":
        alpha = (float(far_depth) - safe_depth) / (float(far_depth) - float(near_depth) + 1e-8)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        dynamic_radius = float(min_radius) + alpha * (float(max_radius) - float(min_radius))
    else:
        raise ValueError(f"Unsupported radius_mode: {radius_mode}")
    return int(round(np.clip(dynamic_radius, min_radius, max_radius)))


def draw_centered_wrist_marker(
    img,
    color,
    side="left",
    center=None,
    radius=28,
    axis_len=60,
    axis_width=6,
    rotation_cam=None,
):
    h, w = img.shape[:2]
    if center is None:
        center = np.array([w // 2, h // 2], dtype=np.float32)
    else:
        center = np.asarray(center, dtype=np.float32)
    radius = int(radius)
    center_i = center.astype(np.int64)
    img = draw_circle(img, center_i, radius, color)

    colors = ColorListLeft if side == "left" else ColorListRight
    if rotation_cam is None:
        default = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, -1.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]
        if side == "right":
            default[0] = np.array([-1.0, 0.0], dtype=np.float32)
        axes = default
    else:
        rot = np.asarray(rotation_cam, dtype=np.float32)
        axes = []
        for axis_idx in range(3):
            d = rot[:2, axis_idx].astype(np.float32)
            n = float(np.linalg.norm(d))
            if n < 1e-6:
                d = np.array([1.0, 0.0], dtype=np.float32) if axis_idx == 0 else np.array([0.0, 1.0], dtype=np.float32)
                n = float(np.linalg.norm(d))
            axes.append(d / n)

    for direction, axis_color in zip(axes, colors):
        delta = direction * float(axis_len)
        img = draw_line(img, center_i, (center + delta).astype(np.int64), axis_color, axis_width)
    return img


def generate_robotwin_action_map(
    abs_actions,
    w2c,
    intrinsic,
    height,
    width,
    radius=22,
    radius_mode="perspective",
    min_radius=12,
    max_radius=40,
    ref_depth=1.0,
    near_depth=0.3,
    far_depth=2.0,
    axis_width=3,
    background=50,
    wrist_center_side=None,
    wrist_center_radius=None,
    wrist_center_axis_len=60,
    wrist_other_pixels_per_meter=220,
    wrist_other_projection_scale=0.45,
    wrist_other_max_offset_ratio=0.38,
):
    abs_actions = np.asarray(abs_actions, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    w2c = np.asarray(w2c, dtype=np.float32)
    if w2c.ndim == 2:
        w2c = np.repeat(w2c[None], abs_actions.shape[0], axis=0)

    ee_key_pts = EndEffectorPts.reshape(1, 4, 4).transpose(0, 2, 1)
    pose_l_mat = pose7_to_matrix(abs_actions[:, 0:7])
    pose_r_mat = pose7_to_matrix(abs_actions[:, 8:15])
    ee2cam_l = np.matmul(w2c, pose_l_mat)
    ee2cam_r = np.matmul(w2c, pose_r_mat)
    pts_l = np.matmul(ee2cam_l, ee_key_pts)
    pts_r = np.matmul(ee2cam_r, ee_key_pts)

    intr = intrinsic.reshape(1, 3, 3)
    uvs_l = np.matmul(intr, pts_l[:, :3, :])
    uvs_l = (uvs_l / (pts_l[:, 2:3, :] + 1e-8))[:, :2, :].transpose(0, 2, 1).astype(np.int64)
    uvs_r = np.matmul(intr, pts_r[:, :3, :])
    uvs_r = (uvs_r / (pts_r[:, 2:3, :] + 1e-8))[:, :2, :].transpose(0, 2, 1).astype(np.int64)

    frames = []
    for i in range(abs_actions.shape[0]):
        img = np.zeros((height, width, 3), dtype=np.uint8) + int(background)
        color_l = tuple(int(c * 255) for c in ColorMapLeft(float(np.clip(abs_actions[i, 7], 0, 1)))[:3])
        color_r = tuple(int(c * 255) for c in ColorMapRight(float(np.clip(abs_actions[i, 15], 0, 1)))[:3])
        if wrist_center_side in ("left", "right"):
            center_radius = wrist_center_radius if wrist_center_radius is not None else max_radius
            center = np.array([width // 2, height // 2], dtype=np.float32)
            own_idx = 0 if wrist_center_side == "left" else 1
            own_rot = ee2cam_l[i, :3, :3] if own_idx == 0 else ee2cam_r[i, :3, :3]

            if wrist_center_side == "left":
                img = draw_centered_wrist_marker(
                    img,
                    color_l,
                    side="left",
                    center=center,
                    radius=center_radius,
                    axis_len=wrist_center_axis_len,
                    axis_width=max(axis_width + 2, 5),
                    rotation_cam=own_rot,
                )
                other_points, other_pts, other_color, other_colors = uvs_r[i], pts_r[i], color_r, ColorListRight
            else:
                img = draw_centered_wrist_marker(
                    img,
                    color_r,
                    side="right",
                    center=center,
                    radius=center_radius,
                    axis_len=wrist_center_axis_len,
                    axis_width=max(axis_width + 2, 5),
                    rotation_cam=own_rot,
                )
                other_points, other_pts, other_color, other_colors = uvs_l[i], pts_l[i], color_l, ColorListLeft

            # Keep the other gripper geometrically honest: project it with the
            # resized camera intrinsics and the current wrist-camera w2c. If it
            # is outside the view, do not clamp or rescale it into the image.
            base = np.array(other_points[0])
            if 0 <= base[0] < width and 0 <= base[1] < height and other_pts[2, 0] > 1e-4:
                radius_i = compute_depth_radius(
                    other_pts[2, 0],
                    radius,
                    radius_mode=radius_mode,
                    min_radius=min_radius,
                    max_radius=max_radius,
                    ref_depth=ref_depth,
                    near_depth=near_depth,
                    far_depth=far_depth,
                )
                img = draw_circle(img, base, radius_i, other_color)
                for point_idx, point in enumerate(other_points):
                    if point_idx == 0:
                        continue
                    point = np.array(point[:2])
                    if point[0] < -width or point[0] > 2 * width or point[1] < -height or point[1] > 2 * height:
                        continue
                    img = draw_line(img, base, point, other_colors[point_idx - 1], axis_width)
            frames.append(img)
            continue
        for points, pts, color in zip([uvs_l[i], uvs_r[i]], [pts_l[i], pts_r[i]], [color_l, color_r]):
            base = np.array(points[0])
            if base[0] < 0 or base[0] >= width or base[1] < 0 or base[1] >= height or pts[2, 0] <= 1e-4:
                continue
            radius_i = compute_depth_radius(
                pts[2, 0],
                radius,
                radius_mode=radius_mode,
                min_radius=min_radius,
                max_radius=max_radius,
                ref_depth=ref_depth,
                near_depth=near_depth,
                far_depth=far_depth,
            )
            img = draw_circle(img, base, radius_i, color)
        for points, pts, colors in zip([uvs_l[i], uvs_r[i]], [pts_l[i], pts_r[i]], [ColorListLeft, ColorListRight]):
            base = np.array(points[0])
            if base[0] < 0 or base[0] >= width or base[1] < 0 or base[1] >= height or pts[2, 0] <= 1e-4:
                continue
            for point_idx, point in enumerate(points):
                if point_idx == 0:
                    continue
                point = np.array(point[:2])
                if point[0] < -width or point[0] > 2 * width or point[1] < -height or point[1] > 2 * height:
                    continue
                img = draw_line(img, base, point, colors[point_idx - 1], axis_width)
        frames.append(img)
    return frames


def generate_robotwin_raymap_image(intrinsic, c2w, height, width, ray_o_vmin, ray_o_vmax, ray_d_vmin, ray_d_vmax):
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    c2w = np.asarray(c2w, dtype=np.float32)
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    xs = np.arange(width, dtype=np.float32) + 0.5
    ys = np.arange(height, dtype=np.float32) + 0.5
    xx, yy = np.meshgrid(xs, ys)
    dirs = np.stack([(xx - cx) / fx, (yy - cy) / fy, np.ones_like(xx)], axis=-1)
    rotation = c2w[:3, :3]
    translation = c2w[:3, 3]
    rays_d = dirs @ rotation.T
    rays_d = rays_d / (np.linalg.norm(rays_d, axis=-1, keepdims=True) + 1e-8)
    rays_o = np.broadcast_to(translation, rays_d.shape).copy()
    ray_o = np.clip((rays_o - ray_o_vmin) / (ray_o_vmax - ray_o_vmin + 1e-8), 0.0, 1.0)
    ray_d = np.clip((rays_d - ray_d_vmin) / (ray_d_vmax - ray_d_vmin + 1e-8), 0.0, 1.0)
    return (ray_o * 255).astype(np.uint8), (ray_d * 255).astype(np.uint8)


def build_robotwin_multiview_rgb(h5_file, frame_indices, camera_names, view_height, view_width):
    frames = []
    for idx in frame_indices:
        parts = []
        for cam in camera_names:
            frame = read_robotwin_rgb(h5_file, cam, int(idx))
            parts.append(np.array(Image.fromarray(frame).resize((view_width, view_height), Image.BILINEAR)))
        frames.append(np.concatenate(parts, axis=0))
    return frames


def build_robotwin_multiview_condition_videos(
    h5_file,
    frame_indices,
    camera_names=ROBOTWIN_CAMERA_NAMES,
    view_height=320,
    view_width=512,
    ray_o_vmin=np.array([-1.5, -1.5, -1.5], dtype=np.float32),
    ray_o_vmax=np.array([1.5, 1.5, 1.5], dtype=np.float32),
    ray_d_vmin=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
    ray_d_vmax=np.array([1.0, 1.0, 1.0], dtype=np.float32),
    traj_radius=22,
    traj_radius_mode="perspective",
    traj_min_radius=12,
    traj_max_radius=40,
    apply_gripper_bias=True,
    gripper_bias=ROBOTWIN_GRIPPER_BIAS,
    gripper_bias_axis=ROBOTWIN_GRIPPER_BIAS_AXIS,
    wrist_center_control=False,
    wrist_center_radius=34,
    wrist_center_axis_len=70,
    wrist_other_pixels_per_meter=220,
    wrist_other_projection_scale=0.45,
    wrist_other_max_offset_ratio=0.38,
):
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    abs_actions = build_robotwin_abs_actions(
        h5_file,
        frame_indices,
        apply_gripper_bias=apply_gripper_bias,
        gripper_bias=gripper_bias,
        gripper_bias_axis=gripper_bias_axis,
    )
    action_per_cam = {}
    ray_o_per_cam = {}
    ray_d_per_cam = {}
    for cam in camera_names:
        sample = read_robotwin_rgb(h5_file, cam, int(frame_indices[0]))
        src_h, src_w = sample.shape[:2]
        intrinsic = resize_intrinsic(
            h5_file[f"observation/{cam}/intrinsic_cv"][int(frame_indices[0])],
            src_h,
            src_w,
            view_height,
            view_width,
        )
        w2c = extrinsic_cv_to_w2c(h5_file[f"observation/{cam}/extrinsic_cv"][frame_indices])
        c2w = w2c_to_c2w(w2c)
        # Scheme 2 keeps one shared 3D visual point per gripper for every
        # camera, so the wrist views use the same true projection path as head.
        wrist_center_side = None
        action_per_cam[cam] = generate_robotwin_action_map(
            abs_actions,
            w2c,
            intrinsic,
            view_height,
            view_width,
            radius=traj_radius,
            radius_mode=traj_radius_mode,
            min_radius=traj_min_radius,
            max_radius=traj_max_radius,
            wrist_center_side=wrist_center_side,
            wrist_center_radius=wrist_center_radius,
            wrist_center_axis_len=wrist_center_axis_len,
            wrist_other_pixels_per_meter=wrist_other_pixels_per_meter,
            wrist_other_projection_scale=wrist_other_projection_scale,
            wrist_other_max_offset_ratio=wrist_other_max_offset_ratio,
        )
        ray_o_frames = []
        ray_d_frames = []
        for t in range(len(frame_indices)):
            ray_o, ray_d = generate_robotwin_raymap_image(
                intrinsic,
                c2w[t],
                view_height,
                view_width,
                np.asarray(ray_o_vmin, dtype=np.float32),
                np.asarray(ray_o_vmax, dtype=np.float32),
                np.asarray(ray_d_vmin, dtype=np.float32),
                np.asarray(ray_d_vmax, dtype=np.float32),
            )
            ray_o_frames.append(ray_o)
            ray_d_frames.append(ray_d)
        ray_o_per_cam[cam] = ray_o_frames
        ray_d_per_cam[cam] = ray_d_frames

    action_video = []
    ray_o_video = []
    ray_d_video = []
    for t in range(len(frame_indices)):
        action_video.append(np.concatenate([action_per_cam[cam][t] for cam in camera_names], axis=0))
        ray_o_video.append(np.concatenate([ray_o_per_cam[cam][t] for cam in camera_names], axis=0))
        ray_d_video.append(np.concatenate([ray_d_per_cam[cam][t] for cam in camera_names], axis=0))
    return action_video, ray_o_video, ray_d_video, abs_actions


def save_video_uint8(frames, path, fps=15):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps)
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    finally:
        writer.close()
