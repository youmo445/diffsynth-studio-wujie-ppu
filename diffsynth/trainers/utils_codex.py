import json
import os

import h5py
import imageio
import imageio.v3 as iio
import matplotlib.cm as cm
import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image, ImageDraw

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


ColorMapLeft = cm.Greens
ColorMapRight = cm.Reds
ColorListLeft = [(0, 0, 255), (255, 255, 0), (0, 255, 255)]
ColorListRight = [(255, 0, 255), (255, 0, 0), (0, 255, 0)]
EndEffectorPts = np.array(
    [
        [0, 0, 0, 1],
        [0.1, 0, 0, 1],
        [0, 0.1, 0, 1],
        [0, 0, 0.1, 1],
    ],
    dtype=np.float32,
)
Gripper2EEFCvt = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0.23],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)
EEF2CamLeft = np.array([0, 0, -0.5236], dtype=np.float32)
EEF2CamRight = np.array([0, 0, 0.5236], dtype=np.float32)


def draw_circle(img, center, radius, color):
    if cv2 is not None:
        cv2.circle(img, tuple(center), radius, color, -1)
        return img
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    x, y = int(center[0]), int(center[1])
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=tuple(color))
    return np.array(pil_img)


def draw_line(img, p0, p1, color, width):
    if cv2 is not None:
        cv2.line(img, tuple(p0), tuple(p1), color, width)
        return img
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    draw.line([tuple(map(int, p0)), tuple(map(int, p1))], fill=tuple(color), width=width)
    return np.array(pil_img)


def quaternion_xyzw_to_matrix(quat):
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


def euler_xyz_to_quaternion_xyzw(euler):
    roll, pitch, yaw = euler
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float32,
    )


def quaternion_multiply_xyzw(q1, q2):
    x1, y1, z1, w1 = np.moveaxis(q1, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(q2, -1, 0)
    return np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    ).astype(np.float32)


def apply_eef2cam_visual_rotation(quat, side):
    """
    Match EVAC get_actions(): rot_vis = Rotation.from_quat(quat) * cvt_vis.
    The extra rotation is baked into absolute actions before get_traj uses them.
    """
    offset = EEF2CamLeft if side == "left" else EEF2CamRight
    offset_quat = euler_xyz_to_quaternion_xyzw(offset)
    rotated = quaternion_multiply_xyzw(quat.astype(np.float32), offset_quat)
    return rotated / (np.linalg.norm(rotated, axis=-1, keepdims=True) + 1e-8)


def get_transformation_matrix_from_quat(pose):
    """
    EVAC-compatible conversion from [x, y, z, qx, qy, qz, qw] to T x 4 x 4.
    """
    pose = np.asarray(pose, dtype=np.float32)
    mat = np.tile(np.eye(4, dtype=np.float32), (pose.shape[0], 1, 1))
    mat[:, :3, :3] = quaternion_xyzw_to_matrix(pose[:, 3:7])
    mat[:, :3, 3] = pose[:, :3]
    return mat


def compute_depth_radius(
    depth,
    radius,
    radius_mode="constant",
    min_radius=20,
    max_radius=60,
    ref_depth=0.30,
    near_depth=0.19,
    far_depth=0.69,
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


def generate_traj_map(
    abs_actions,
    w2c,
    intrinsic,
    h,
    w,
    radius=50,
    radius_mode="constant",
    min_radius=15,
    max_radius=70,
    ref_depth=1.0,
    near_depth=0.3,
    far_depth=2.0,
):
    """
    Generate trajectory maps following EVAC's get_traj logic as closely as possible.

    abs_actions: T x 16, [left xyz, left quat xyzw, left grip,
                          right xyz, right quat xyzw, right grip]
    w2c: T x 4 x 4 or 4 x 4 world-to-camera matrices.
    intrinsic: 3 x 3 camera intrinsic matrix scaled to output size.
    radius_mode:
        constant    - EVAC-compatible fixed radius.
        perspective - radius * ref_depth / camera_depth, clipped.
        depth_norm  - linearly maps [near_depth, far_depth] to [max_radius, min_radius].
    returns: list of T uint8 RGB arrays with shape H x W x 3.
    """
    abs_actions = np.asarray(abs_actions, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    w2c = np.asarray(w2c, dtype=np.float32)
    if w2c.ndim == 2:
        w2c = np.repeat(w2c[None], abs_actions.shape[0], axis=0)

    ee_key_pts = EndEffectorPts.reshape(1, 4, 4).transpose(0, 2, 1)
    cvt_matrix = Gripper2EEFCvt.reshape(1, 4, 4)

    pose_l_mat = get_transformation_matrix_from_quat(abs_actions[:, 0:7])
    pose_r_mat = get_transformation_matrix_from_quat(abs_actions[:, 8:15])

    ee2cam_l = np.matmul(w2c, pose_l_mat)
    ee2cam_r = np.matmul(w2c, pose_r_mat)
    ee2cam_l = np.matmul(ee2cam_l, cvt_matrix)
    ee2cam_r = np.matmul(ee2cam_r, cvt_matrix)

    pts_l = np.matmul(ee2cam_l, ee_key_pts)
    pts_r = np.matmul(ee2cam_r, ee_key_pts)

    intr = intrinsic.reshape(1, 3, 3)
    uvs_l = np.matmul(intr, pts_l[:, :3, :])
    uvs_l = (uvs_l / (pts_l[:, 2:3, :] + 1e-8))[:, :2, :].transpose(0, 2, 1).astype(np.int64)
    uvs_r = np.matmul(intr, pts_r[:, :3, :])
    uvs_r = (uvs_r / (pts_r[:, 2:3, :] + 1e-8))[:, :2, :].transpose(0, 2, 1).astype(np.int64)

    img_list = []
    for i in range(abs_actions.shape[0]):
        img = np.zeros((h, w, 3), dtype=np.uint8) + 50

        normalized_value_l = abs_actions[i, 7].item() / 120.0
        normalized_value_r = abs_actions[i, 15].item() / 120.0
        color_l = tuple(int(c * 255) for c in ColorMapLeft(normalized_value_l)[:3])
        color_r = tuple(int(c * 255) for c in ColorMapRight(normalized_value_r)[:3])

        for points, pts, color in zip([uvs_l[i], uvs_r[i]], [pts_l[i], pts_r[i]], [color_l, color_r]):
            base = np.array(points[0])
            if base[0] < 0 or base[0] >= w or base[1] < 0 or base[1] >= h:
                continue
            point = np.array(points[0][:2])
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
            img = draw_circle(img, point, radius_i, color)

        for points, colors in zip([uvs_l[i], uvs_r[i]], [ColorListLeft, ColorListRight]):
            base = np.array(points[0])
            if base[0] < 0 or base[0] >= w or base[1] < 0 or base[1] >= h:
                continue
            for point_idx, point in enumerate(points):
                point = np.array(point[:2])
                if point_idx == 0:
                    continue
                img = draw_line(img, base, point, colors[point_idx - 1], 8)

        img_list.append(img)

    return img_list


class AgiBotWCDataset4Wancontrolmultiview(torch.utils.data.Dataset):
    """
    Multi-view AgiBotWorld dataset for Wan VACE training.

    Each returned frame vertically concatenates views in camera_names order. The
    control video uses EVAC-style trajectory maps generated per camera frame.
    """

    def __init__(
        self,
        base_path="/mnt/data/zsq/Agi2024subset_split/train",
        num_frames=9,
        repeat=1,
        episode_stride=1,
        episode_limit=None,
        original_hz=30,
        target_hz=5,
        stride=1,
        first_round_prob=0.05,
        context_length=1,
        traj_radius=50,
        traj_radius_mode="constant",
        traj_min_radius=20,
        traj_max_radius=60,
        traj_ref_depth=0.30,
        traj_near_depth=0.19,
        traj_far_depth=0.69,
        output_raymap=False,
        ray_o_vmin=-1.5,
        ray_o_vmax=1.5,
        ray_d_vmin=-1.0,
        ray_d_vmax=1.0,
        raymap_mode="image",
        resize_to=(320, 512),
        dataset_type="vace",
        camera_names=None,
        camera_sample_mode="all",
        output_global_reference=False,
    ):
        super().__init__()
        if original_hz % target_hz != 0:
            raise ValueError(f"Cannot downsample from {original_hz}Hz to {target_hz}Hz")
        if camera_names is None:
            camera_names = ["head", "hand_left", "hand_right"]
        if camera_sample_mode not in ("all", "random_one", "cycle_one"):
            raise ValueError(f"Unsupported camera_sample_mode: {camera_sample_mode}")
        if raymap_mode not in ("image", "latent"):
            raise ValueError(f"Unsupported raymap_mode: {raymap_mode}")

        self.base_path = base_path
        if os.path.normpath(base_path).startswith(os.path.normpath("/mnt/data/zsq/Agi2024subset_split")):
            self.video_base_path = base_path.replace(
                "/mnt/data/zsq/Agi2024subset_split",
                "/mnt/data/zsq/Agi2024subset_split_h264",
                1,
            )
        else:
            raise NotImplementedError('avi2h264 conversion is only supported for paths starting with "/mnt/data/zsq/Agi2024subset_split"')
            self.video_base_path = base_path
        self.num_frames = num_frames
        self.repeat = repeat
        self.stride = stride
        self.first_round_prob = first_round_prob
        self.context_length = context_length
        self.traj_radius = traj_radius
        self.traj_radius_mode = traj_radius_mode
        self.traj_min_radius = traj_min_radius
        self.traj_max_radius = traj_max_radius
        self.traj_ref_depth = traj_ref_depth
        self.traj_near_depth = traj_near_depth
        self.traj_far_depth = traj_far_depth
        self.output_raymap = output_raymap
        self.ray_o_vmin = ray_o_vmin
        self.ray_o_vmax = ray_o_vmax
        self.ray_d_vmin = ray_d_vmin
        self.ray_d_vmax = ray_d_vmax
        self.raymap_mode = raymap_mode
        self.vae_spatial_downsample = 8
        self.vae_temporal_downsample = 4
        self.resize_to = resize_to
        self.load_from_cache = False
        self.type = dataset_type
        self.camera_names = list(camera_names)
        self.camera_sample_mode = camera_sample_mode
        self.output_global_reference = output_global_reference
        self.downsample_step = original_hz // target_hz

        print(
            f"[AgiBotWCDataset4Wancontrolmultiview] Downsample: "
            f"{original_hz}Hz -> {target_hz}Hz (step={self.downsample_step})"
        )

        episodes = self._discover_episodes(base_path)
        if episode_limit is not None:
            episodes = episodes[:episode_limit]
        self.episodes = episodes[::episode_stride]

        self.episode_info = []
        self.sample_indices = []
        for ep_idx, ep in enumerate(self.episodes):
            info = self._build_episode_info(ep)
            if info is None:
                continue
            valid_ep_idx = len(self.episode_info)
            self.episode_info.append(info)
            max_start = info["T_ds"] - self.num_frames + 1
            for start in range(0, max_start, self.stride):
                self.sample_indices.append((valid_ep_idx, start))

        self.total_samples = len(self.sample_indices)
        self.length = self.total_samples * repeat
        print(
            f"[AgiBotWCDataset4Wancontrolmultiview] {len(self.episode_info)} episodes, "
            f"{self.total_samples} samples, repeat={repeat}, length={self.length}, "
            f"cameras={self.camera_names}, camera_sample_mode={self.camera_sample_mode}"
        )

    def _discover_episodes(self, base_path):
        proprio_base = os.path.join(base_path, "proprio_stats")
        if os.path.isdir(proprio_base):
            episodes = []
            for task in sorted(os.listdir(proprio_base)):
                task_path = os.path.join(proprio_base, task)
                if not os.path.isdir(task_path):
                    continue
                for ep in sorted(os.listdir(task_path)):
                    h5_path = os.path.join(task_path, ep, "proprio_stats.h5")
                    if not os.path.exists(h5_path):
                        continue
                    episodes.append(
                        {
                            "h5_path": h5_path,
                            "video_dir": os.path.join(self.video_base_path, "observations", task, ep, "videos"),
                            "camera_dir": os.path.join(base_path, "parameters", task, ep, "parameters", "camera"),
                        }
                    )
            return episodes

        # Fallback for the chunk-grouped layout used by the older single-view dataset.
        episode_dict = {}
        for name in sorted(os.listdir(base_path)):
            path = os.path.join(base_path, name)
            if os.path.isdir(path):
                key = "-".join(name.split("-")[:2])
                episode_dict.setdefault(key, []).append(path)
        return [{"chunk_paths": sorted(episode_dict[k])} for k in sorted(episode_dict.keys())]

    def _build_episode_info(self, ep):
        if "h5_path" in ep:
            with h5py.File(ep["h5_path"], "r") as f:
                pos = f["state/end/position"][:]
                quat = f["state/end/orientation"][:]
                grip = f["state/effector/position"][:]

            total_raw = pos.shape[0]
            grip = grip.reshape(total_raw, 2, -1)[..., 0] if grip.ndim == 3 else grip
            ds_idx = np.arange(0, total_raw, self.downsample_step)
            abs_actions = self._compute_abs_action(pos[ds_idx], quat[ds_idx], grip[ds_idx])

            cameras = {
                cam: self._load_camera_params(ep["camera_dir"], cam, ds_idx)
                for cam in self.camera_names
            }
            return {
                "layout": "split",
                "video_dir": ep["video_dir"],
                "ds_indices": ds_idx,
                "T_ds": len(ds_idx),
                "abs_actions": abs_actions,
                "cameras": cameras,
            }

        chunk_paths = ep["chunk_paths"]
        pos_list, quat_list, grip_list, chunk_frame_counts = [], [], [], []
        for chunk_path in chunk_paths:
            h5_path = os.path.join(chunk_path, "proprio_stats.h5")
            with h5py.File(h5_path, "r") as f:
                pos = f["state/end/position"][:]
                quat = f["state/end/orientation"][:]
                grip = f["state/effector/position"][:]
            chunk_frame_counts.append(pos.shape[0])
            pos_list.append(pos)
            quat_list.append(quat)
            grip_list.append(grip.reshape(pos.shape[0], 2, -1)[..., 0] if grip.ndim == 3 else grip)

        pos_all = np.concatenate(pos_list, axis=0)
        quat_all = np.concatenate(quat_list, axis=0)
        grip_all = np.concatenate(grip_list, axis=0)
        ds_idx = np.arange(0, pos_all.shape[0], self.downsample_step)
        abs_actions = self._compute_abs_action(pos_all[ds_idx], quat_all[ds_idx], grip_all[ds_idx])
        cameras = {
            cam: self._load_camera_params(chunk_paths[0], cam, ds_idx, chunk_layout=True)
            for cam in self.camera_names
        }
        return {
            "layout": "chunk",
            "chunk_paths": chunk_paths,
            "chunk_frame_counts": chunk_frame_counts,
            "ds_indices": ds_idx,
            "T_ds": len(ds_idx),
            "abs_actions": abs_actions,
            "cameras": cameras,
        }

    def _compute_abs_action(self, pos, quat, grip):
        out = np.zeros((pos.shape[0], 16), dtype=np.float32)
        out[:, 0:3] = pos[:, 0]
        out[:, 3:7] = apply_eef2cam_visual_rotation(quat[:, 0], "left")
        out[:, 7] = grip[:, 0]
        out[:, 8:11] = pos[:, 1]
        out[:, 11:15] = apply_eef2cam_visual_rotation(quat[:, 1], "right")
        out[:, 15] = grip[:, 1]
        return out

    def _load_camera_params(self, camera_dir, cam_name, ds_idx, chunk_layout=False):
        if chunk_layout:
            intrinsic_path = os.path.join(camera_dir, f"{cam_name}_intrinsic_params.json")
            extrinsic_path = os.path.join(camera_dir, f"{cam_name}_extrinsic_params_aligned.json")
        else:
            intrinsic_path = os.path.join(camera_dir, f"{cam_name}_intrinsic_params.json")
            extrinsic_path = os.path.join(camera_dir, f"{cam_name}_extrinsic_params_aligned.json")

        with open(intrinsic_path, "r") as f:
            info = json.load(f)["intrinsic"]
        intrinsic = np.eye(3, dtype=np.float32)
        intrinsic[0, 0] = info["fx"]
        intrinsic[1, 1] = info["fy"]
        intrinsic[0, 2] = info["ppx"]
        intrinsic[1, 2] = info["ppy"]

        with open(extrinsic_path, "r") as f:
            extr_list = json.load(f)
        c2w_all = []
        for item in extr_list:
            mat = np.eye(4, dtype=np.float32)
            mat[:3, :3] = np.array(item["extrinsic"]["rotation_matrix"], dtype=np.float32)
            mat[:3, 3] = np.array(item["extrinsic"]["translation_vector"], dtype=np.float32)
            c2w_all.append(mat)
        c2w_all = np.stack(c2w_all, axis=0)
        valid_idx = np.clip(ds_idx, 0, c2w_all.shape[0] - 1)
        c2w_seq = c2w_all[valid_idx]
        w2c_seq = np.linalg.inv(c2w_seq).astype(np.float32)
        return {"intrinsic": intrinsic, "w2c": w2c_seq, "c2w": c2w_seq}

    def _select_camera_names(self, idx, sample_idx):
        if self.camera_sample_mode == "all" or len(self.camera_names) <= 1:
            return self.camera_names
        if self.camera_sample_mode == "cycle_one":
            return [self.camera_names[idx % len(self.camera_names)]]
        return [self.camera_names[np.random.randint(len(self.camera_names))]]

    def _load_frames_by_ids(self, ep_info, frame_ids, camera_names=None):
        if camera_names is None:
            camera_names = self.camera_names
        result = {}
        if ep_info["layout"] == "split":
            raw_indices = [int(ep_info["ds_indices"][fid]) for fid in frame_ids]
            for fid in frame_ids:
                result[fid] = {}
            for cam in camera_names:
                video_path = os.path.join(ep_info["video_dir"], f"{cam}_color.mp4")
                try:
                    vr = VideoReader(video_path, ctx=cpu(0))
                    safe_indices = [min(idx, len(vr) - 1) for idx in raw_indices]
                    frames = vr.get_batch(safe_indices).asnumpy()
                    del vr
                except Exception as e:
                    raise RuntimeError(
                        f"Decord failed for {video_path}\n"
                        f"Original error: {repr(e)}"
                    )
                    frames = []
                    for orig_idx in raw_indices:
                        frame = iio.imread(video_path, index=orig_idx)
                        if frame.shape[-1] == 4:
                            frame = frame[..., :3]
                        frames.append(frame)
                    frames = np.stack(frames, axis=0)

                if frames.shape[-1] == 4:
                    frames = frames[..., :3]
                for fid, frame in zip(frame_ids, frames):
                    result[fid][cam] = frame
            return result

        chunk_cum = np.cumsum([0] + ep_info["chunk_frame_counts"])
        readers = {}
        try:
            for fid in frame_ids:
                orig_idx = int(ep_info["ds_indices"][fid])
                cid = int(np.searchsorted(chunk_cum[1:], orig_idx, side="right"))
                local_idx = orig_idx - int(chunk_cum[cid])
                frames_cam = {}
                for cam in camera_names:
                    key = (cid, cam)
                    if key not in readers:
                        video_path = os.path.join(ep_info["chunk_paths"][cid], f"{cam}_color.mp4")
                        readers[key] = VideoReader(video_path, ctx=cpu(0))
                    vr = readers[key]
                    frames_cam[cam] = vr[min(local_idx, len(vr) - 1)].asnumpy()
                result[fid] = frames_cam
        finally:
            readers.clear()
        return result

    def _generate_raymap(self, intrinsic, c2w, height, width):
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]

        xs = np.arange(width, dtype=np.float32) + 0.5
        ys = np.arange(height, dtype=np.float32) + 0.5
        xx, yy = np.meshgrid(xs, ys)

        dirs = np.stack(
            [
                (xx - cx) / fx,
                (yy - cy) / fy,
                np.ones_like(xx),
            ],
            axis=-1,
        )

        rotation = c2w[:3, :3]
        translation = c2w[:3, 3]
        rays_d = dirs @ rotation.T
        rays_d = rays_d / (np.linalg.norm(rays_d, axis=-1, keepdims=True) + 1e-8)
        rays_o = np.broadcast_to(translation, rays_d.shape).copy()
        return rays_o.astype(np.float32), rays_d.astype(np.float32)

    def _latent_frame_groups(self, num_frames):
        latent_t = (num_frames - 1) // self.vae_temporal_downsample + 1
        groups = [[0] * self.vae_temporal_downsample]
        for i in range(1, latent_t):
            start = 1 + (i - 1) * self.vae_temporal_downsample
            group = [min(start + j, num_frames - 1) for j in range(self.vae_temporal_downsample)]
            groups.append(group)
        return groups

    def _generate_raymap_latents_for_camera(self, cam_info, frame_ids, image_intrinsic, image_height, image_width):
        latent_h = image_height // self.vae_spatial_downsample
        latent_w = image_width // self.vae_spatial_downsample
        latent_intrinsic = image_intrinsic.copy()
        latent_intrinsic[0, 0] *= latent_w / image_width
        latent_intrinsic[0, 2] *= latent_w / image_width
        latent_intrinsic[1, 1] *= latent_h / image_height
        latent_intrinsic[1, 2] *= latent_h / image_height

        ray_o_list = []
        ray_d_list = []
        for group in self._latent_frame_groups(len(frame_ids)):
            ray_o_group = []
            ray_d_group = []
            for pos in group:
                fid = int(frame_ids[pos])
                ray_o, ray_d = self._generate_raymap(latent_intrinsic, cam_info["c2w"][fid], latent_h, latent_w)
                ray_o_group.append(ray_o)
                ray_d_group.append(ray_d)
            ray_o_list.append(np.concatenate(ray_o_group, axis=-1))
            ray_d_list.append(np.concatenate(ray_d_group, axis=-1))
        return ray_o_list, ray_d_list

    def _to_uint8_img(self, value, vmin, vmax):
        value = (value - vmin) / (vmax - vmin + 1e-8)
        value = np.clip(value, 0.0, 1.0)
        return (value * 255).astype(np.uint8)

    def __len__(self):
        return self.length

    def _make_frame_ids(self, start_idx):
        if self.context_length > 1:
            if np.random.rand() < self.first_round_prob:
                horizon = self.num_frames - self.context_length
                return np.array([0] * self.context_length + list(range(1, horizon + 1)))
            consecutive_ids = np.arange(start_idx, start_idx + self.num_frames - 1)
            return np.concatenate([[0], consecutive_ids])
        return np.arange(start_idx, start_idx + self.num_frames)

    def __getitem__(self, idx):
        sample_idx = idx % self.total_samples
        ep_idx, start_idx = self.sample_indices[sample_idx]
        info = self.episode_info[ep_idx]
        frame_ids = self._make_frame_ids(start_idx)
        active_camera_names = self._select_camera_names(idx, sample_idx)
        frame_map = self._load_frames_by_ids(info, frame_ids.tolist(), active_camera_names)
        global_frame_map = None
        if self.output_global_reference:
            global_frame_map = self._load_frames_by_ids(info, [int(frame_ids[0])], self.camera_names)

        cam_frames = {cam: [] for cam in active_camera_names}
        cam_traj_maps = {cam: [] for cam in active_camera_names}
        cam_ray_o = {cam: [] for cam in active_camera_names}
        cam_ray_d = {cam: [] for cam in active_camera_names}
        actions_for_traj = info["abs_actions"][frame_ids]

        for cam in active_camera_names:
            cam_info = info["cameras"][cam]
            sample_frame = frame_map[frame_ids[0]][cam]
            ori_h, ori_w = sample_frame.shape[:2]
            if self.resize_to is not None:
                traj_h, traj_w = self.resize_to
            else:
                traj_h, traj_w = ori_h, ori_w

            scaled_intrinsic = cam_info["intrinsic"].copy()
            scaled_intrinsic[0, 0] *= traj_w / ori_w
            scaled_intrinsic[0, 2] *= traj_w / ori_w
            scaled_intrinsic[1, 1] *= traj_h / ori_h
            scaled_intrinsic[1, 2] *= traj_h / ori_h

            traj_maps = generate_traj_map(
                actions_for_traj,
                cam_info["w2c"][frame_ids],
                scaled_intrinsic,
                traj_h,
                traj_w,
                radius=self.traj_radius,
                radius_mode=self.traj_radius_mode,
                min_radius=self.traj_min_radius,
                max_radius=self.traj_max_radius,
                ref_depth=self.traj_ref_depth,
                near_depth=self.traj_near_depth,
                far_depth=self.traj_far_depth,
            )

            for i, fid in enumerate(frame_ids):
                img = frame_map[fid][cam]
                if img.max() <= 1.0:
                    img = (img * 255).clip(0, 255)
                pil_img = Image.fromarray(img.astype(np.uint8))
                if self.resize_to is not None:
                    pil_img = pil_img.resize((traj_w, traj_h), Image.BILINEAR)
                cam_frames[cam].append(pil_img)
                cam_traj_maps[cam].append(Image.fromarray(traj_maps[i]))

                if self.output_raymap and self.raymap_mode == "image":
                    c2w = cam_info["c2w"][fid]
                    ray_o, ray_d = self._generate_raymap(scaled_intrinsic, c2w, traj_h, traj_w)
                    cam_ray_o[cam].append(Image.fromarray(self._to_uint8_img(ray_o, self.ray_o_vmin, self.ray_o_vmax)))
                    cam_ray_d[cam].append(Image.fromarray(self._to_uint8_img(ray_d, self.ray_d_vmin, self.ray_d_vmax)))

            if self.output_raymap and self.raymap_mode == "latent":
                ray_o_latents, ray_d_latents = self._generate_raymap_latents_for_camera(
                    cam_info, frame_ids, scaled_intrinsic, traj_h, traj_w
                )
                cam_ray_o[cam].extend(ray_o_latents)
                cam_ray_d[cam].extend(ray_d_latents)

        video_list = []
        control_list = []
        ray_o_list = []
        ray_d_list = []
        for i in range(len(frame_ids)):
            concat_img = np.concatenate([np.array(cam_frames[cam][i]) for cam in active_camera_names], axis=0)
            concat_control = np.concatenate([np.array(cam_traj_maps[cam][i]) for cam in active_camera_names], axis=0)
            video_list.append(Image.fromarray(concat_img))
            control_list.append(Image.fromarray(concat_control))
            if self.output_raymap and self.raymap_mode == "image":
                concat_ray_o = np.concatenate([np.array(cam_ray_o[cam][i]) for cam in active_camera_names], axis=0)
                concat_ray_d = np.concatenate([np.array(cam_ray_d[cam][i]) for cam in active_camera_names], axis=0)
                ray_o_list.append(Image.fromarray(concat_ray_o))
                ray_d_list.append(Image.fromarray(concat_ray_d))

        if self.output_raymap and self.raymap_mode == "latent":
            ray_o_latents = []
            ray_d_latents = []
            for i in range(len(cam_ray_o[active_camera_names[0]])):
                concat_ray_o = np.concatenate([cam_ray_o[cam][i] for cam in active_camera_names], axis=0)
                concat_ray_d = np.concatenate([cam_ray_d[cam][i] for cam in active_camera_names], axis=0)
                ray_o_latents.append(concat_ray_o)
                ray_d_latents.append(concat_ray_d)
            ray_o_list = torch.from_numpy(np.stack(ray_o_latents, axis=0)).permute(3, 0, 1, 2).float()
            ray_d_list = torch.from_numpy(np.stack(ray_d_latents, axis=0)).permute(3, 0, 1, 2).float()

        global_reference_images = None
        if self.output_global_reference:
            global_reference_images = []
            for cam in self.camera_names:
                img = global_frame_map[int(frame_ids[0])][cam]
                if img.max() <= 1.0:
                    img = (img * 255).clip(0, 255)
                pil_img = Image.fromarray(img.astype(np.uint8))
                if self.resize_to is not None:
                    pil_img = pil_img.resize((self.resize_to[1], self.resize_to[0]), Image.BILINEAR)
                global_reference_images.append(pil_img)

        if self.type == "vace":
            sample = {
                "video": video_list,
                "vace_reference_image": [video_list[0]],
                "vace_video": control_list,
                "prompt": "机械臂按照要求移动夹爪执行任务",
            }
            if self.output_raymap:
                sample["ray_map_o"] = ray_o_list
                sample["ray_map_d"] = ray_d_list
            if self.output_global_reference:
                sample["vace_global_reference_images"] = global_reference_images
            return sample
        if self.type == "control":
            sample = {
                "video": video_list,
                "reference_image": [video_list[0]],
                "control_video": control_list,
                "prompt": "机械臂按照要求移动夹爪执行任务",
            }
            if self.output_raymap:
                sample["ray_map_o"] = ray_o_list
                sample["ray_map_d"] = ray_d_list
            return sample
        raise ValueError(f"Unsupported dataset_type: {self.type}")


# Also expose the conventional capitalization for easier importing.
AgiBotWCDataset4WanControlmultiview = AgiBotWCDataset4Wancontrolmultiview


def save_multiview_dataset_visualization(
    out_dir,
    base_path="/mnt/data/zsq/Agi2024subset_split/train",
    sample_index=200,
    num_frames=9,
    episode_limit=2,
    radius_mode="constant",
    output_raymap=False,
):
    os.makedirs(out_dir, exist_ok=True)
    dataset = AgiBotWCDataset4Wancontrolmultiview(
        base_path=base_path,
        num_frames=num_frames,
        episode_limit=episode_limit,
        repeat=1,
        dataset_type="vace",
        traj_radius_mode=radius_mode,
        output_raymap=output_raymap,
    )
    sample = dataset[sample_index]
    video = sample["video"]
    control = sample["vace_video"]

    out_path = os.path.join(out_dir, f"sample_multiview_video_traj_{radius_mode}.mp4")
    ray_o = sample.get("ray_map_o")
    ray_d = sample.get("ray_map_d")

    writer = imageio.get_writer(out_path, fps=5)
    for i, (img, traj) in enumerate(zip(video, control)):
        parts = [np.array(img), np.array(traj)]
        if ray_o is not None and ray_d is not None:
            parts.extend([np.array(ray_o[i]), np.array(ray_d[i])])
        concat = np.concatenate(parts, axis=1)
        writer.append_data(concat)
    writer.close()
    return out_path


def save_radius_mode_comparison_visualizations(
    out_dir,
    base_path="/mnt/data/zsq/Agi2024subset_split/train",
    sample_index=200,
    num_frames=9,
    episode_limit=2,
    output_raymap=False,
):
    paths = []
    for radius_mode in ["constant", "perspective", "depth_norm"]:
        paths.append(
            save_multiview_dataset_visualization(
                out_dir=out_dir,
                base_path=base_path,
                sample_index=sample_index,
                num_frames=num_frames,
                episode_limit=episode_limit,
                radius_mode=radius_mode,
                output_raymap=output_raymap,
            )
        )
    return paths


def save_raymap_visualization(
    out_dir,
    base_path="/mnt/data/zsq/Agi2024subset_split/train",
    sample_index=200,
    num_frames=9,
    episode_limit=2,
    radius_mode="constant",
):
    return save_multiview_dataset_visualization(
        out_dir=out_dir,
        base_path=base_path,
        sample_index=sample_index,
        num_frames=num_frames,
        episode_limit=episode_limit,
        radius_mode=radius_mode,
        output_raymap=True,
    )
