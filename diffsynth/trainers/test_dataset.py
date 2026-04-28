import imageio, os, torch, warnings, torchvision, argparse, json
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation
import h5py
from decord import VideoReader, cpu
import matplotlib.cm as cm
import imageio.v3 as iio

_EVAC_ColorMapLeft = cm.Greens
_EVAC_ColorMapRight = cm.Reds
_EVAC_ColorListLeft = [(0, 0, 255), (255, 255, 0), (0, 255, 255)]
_EVAC_ColorListRight = [(255, 0, 255), (255, 0, 0), (0, 255, 0)]

_EVAC_EndEffectorPts = np.array([
    [0, 0, 0, 1],
    [0.1, 0, 0, 1],
    [0, 0.1, 0, 1],
    [0, 0, 0.1, 1],
], dtype=np.float32)

_EVAC_Gripper2EEFCvt = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0.23],
    [0, 0, 0, 1],
], dtype=np.float32)

_EVAC_EEF2CamLeft = [0, 0, -0.5236]
_EVAC_EEF2CamRight = [0, 0, 0.5236]


def _quaternion_to_matrix_np(wxyz):
    w, x, y, z = wxyz[..., 0], wxyz[..., 1], wxyz[..., 2], wxyz[..., 3]

    tx, ty, tz = 2.0 * x, 2.0 * y, 2.0 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z

    mat = np.stack([
        1.0 - (tyy + tzz), txy - twz, txz + twy,
        txy + twz, 1.0 - (txx + tzz), tyz - twx,
        txz - twy, tyz + twx, 1.0 - (txx + tyy),
    ], axis=-1).reshape(wxyz.shape[:-1] + (3, 3))

    return mat


def _get_transformation_matrix_np(xyz_quat):
    rot_quat = xyz_quat[:, 3:]
    wxyz = rot_quat[:, [3, 0, 1, 2]]
    rot = _quaternion_to_matrix_np(wxyz)

    trans = xyz_quat[:, :3]
    T = xyz_quat.shape[0]

    mat = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
    mat[:, :3, :3] = rot
    mat[:, :3, 3] = trans
    return mat


def generate_traj_map(abs_actions, w2c_seq, intrinsic, h, w, radius=50):
    """
    支持逐帧外参:
        abs_actions : (T,16)
        w2c_seq     : (T,4,4) 或 (4,4)
    """
    T = abs_actions.shape[0]

    if w2c_seq.ndim == 2:
        w2c_seq = np.repeat(w2c_seq[None], T, axis=0)

    ee_key_pts = _EVAC_EndEffectorPts.T
    cvt = _EVAC_Gripper2EEFCvt

    cvt_vis_l = Rotation.from_euler("xyz", _EVAC_EEF2CamLeft)
    cvt_vis_r = Rotation.from_euler("xyz", _EVAC_EEF2CamRight)

    traj_maps = []

    for t in range(T):
        w2c_mat = w2c_seq[t].astype(np.float32)

        pos_l = abs_actions[t, 0:3]
        quat_l = abs_actions[t, 3:7]
        rot_l = Rotation.from_quat(quat_l)
        rot_vis_l = rot_l * cvt_vis_l
        vis_quat_l = np.concatenate([pos_l, rot_vis_l.as_quat()])

        pos_r = abs_actions[t, 8:11]
        quat_r = abs_actions[t, 11:15]
        rot_r = Rotation.from_quat(quat_r)
        rot_vis_r = rot_r * cvt_vis_r
        vis_quat_r = np.concatenate([pos_r, rot_vis_r.as_quat()])

        pose_l_mat = _get_transformation_matrix_np(vis_quat_l[None])[0]
        pose_r_mat = _get_transformation_matrix_np(vis_quat_r[None])[0]

        ee2cam_l = w2c_mat @ pose_l_mat @ cvt
        ee2cam_r = w2c_mat @ pose_r_mat @ cvt

        pts_l = (ee2cam_l @ ee_key_pts)[:3, :]
        pts_r = (ee2cam_r @ ee_key_pts)[:3, :]

        uvs_l = (intrinsic @ pts_l)
        uvs_l = (uvs_l[:2] / uvs_l[2:3]).T.astype(np.int64)

        uvs_r = (intrinsic @ pts_r)
        uvs_r = (uvs_r[:2] / uvs_r[2:3]).T.astype(np.int64)

        from PIL import ImageDraw
        pil_img = Image.new("RGB", (w, h), (50, 50, 50))
        draw = ImageDraw.Draw(pil_img)

        grip_l = abs_actions[t, 7]
        grip_r = abs_actions[t, 15]

        norm_l = np.clip(grip_l / 120.0, 0.0, 1.0)
        norm_r = np.clip(grip_r / 120.0, 0.0, 1.0)

        color_l = tuple(int(c * 255) for c in _EVAC_ColorMapLeft(norm_l)[:3])
        color_r = tuple(int(c * 255) for c in _EVAC_ColorMapRight(norm_r)[:3])

        for uvs, color, color_list in [
            (uvs_l, color_l, _EVAC_ColorListLeft),
            (uvs_r, color_r, _EVAC_ColorListRight),
        ]:
            bx, by = int(uvs[0][0]), int(uvs[0][1])

            if 0 <= bx < w and 0 <= by < h:
                draw.ellipse(
                    [bx - radius, by - radius, bx + radius, by + radius],
                    fill=color
                )

                for k in range(1, len(uvs)):
                    pt = uvs[k]
                    draw.line(
                        [(bx, by), (int(pt[0]), int(pt[1]))],
                        fill=color_list[k - 1],
                        width=8
                    )

        traj_maps.append(np.array(pil_img))

    return traj_maps


class AgiBotWCDataset4WanControlmultiview(torch.utils.data.Dataset):

    def __init__(
        self,
        base_path,
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
        resize_to=(320, 512),   # (H, W), 每个视角的 resize 尺寸，拼接后总高度为 3*H
        dataset_type='vace',
        camera_names=None,      # ['head', 'hand_left', 'hand_right']
    ):
        super().__init__()
        self.first_round_prob = first_round_prob
        self.traj_radius = traj_radius
        if camera_names is None:
            camera_names = ["head", "hand_left", "hand_right"]

        self.camera_names = camera_names
        self.base_path = base_path
        self.num_frames = num_frames
        self.repeat = repeat
        self.stride = stride
        self.context_length = context_length
        self.resize_to = resize_to
        self.type = dataset_type

        self.downsample_step = original_hz // target_hz

        proprio_base = os.path.join(base_path, "proprio_stats")
        episodes = []

        for task in sorted(os.listdir(proprio_base)):
            task_path = os.path.join(proprio_base, task)
            if not os.path.isdir(task_path):
                continue

            for ep in sorted(os.listdir(task_path)):
                ep_path = os.path.join(task_path, ep)
                h5_path = os.path.join(ep_path, "proprio_stats.h5")

                if not os.path.exists(h5_path):
                    continue

                episodes.append({
                    "h5_path": h5_path,
                    "video_dir": os.path.join(base_path, "observations", task, ep, "videos"),
                    "camera_dir": os.path.join(base_path, "parameters", task, ep, "parameters", "camera"),
                })

        if episode_limit is not None:
            episodes = episodes[:episode_limit]

        self.episode_info = []
        self.sample_indices = []

        for ep_idx, ep in enumerate(episodes):

            with h5py.File(ep["h5_path"], "r") as f:
                pos = f["state/end/position"][:]
                quat = f["state/end/orientation"][:]
                grip = f["state/effector/position"][:]

            T = pos.shape[0]
            grip = grip.reshape(T, 2, -1)[..., 0]

            ds_idx = np.arange(0, T, self.downsample_step)
            T_ds = len(ds_idx)

            abs_actions = self._compute_abs_action(
                pos[ds_idx], quat[ds_idx], grip[ds_idx]
            )

            cameras = {}
            for cam in self.camera_names:
                intrinsic, w2c_seq, c2w_seq = self._load_camera_params(
                    ep["camera_dir"], cam, ds_idx
                )
                cameras[cam] = {
                    "intrinsic": intrinsic,
                    "w2c": w2c_seq,
                    "c2w": c2w_seq,
                }

            self.episode_info.append({
                "video_dir": ep["video_dir"],
                "ds_indices": ds_idx,
                "abs_actions": abs_actions,
                "cameras": cameras,
                "T_ds": T_ds,
            })

            for s in range(0, T_ds - num_frames + 1, stride):
                self.sample_indices.append((ep_idx, s))

        self.total_samples = len(self.sample_indices)
        self.length = self.total_samples * repeat

    def __len__(self):
        return self.length

    def _compute_abs_action(self, pos, quat, grip):
        T = pos.shape[0]
        out = np.zeros((T, 16), dtype=np.float32)

        for t in range(T):
            out[t, 0:3] = pos[t, 0]
            out[t, 3:7] = quat[t, 0]
            out[t, 7] = grip[t, 0]

            out[t, 8:11] = pos[t, 1]
            out[t, 11:15] = quat[t, 1]
            out[t, 15] = grip[t, 1]

        return out

    def _load_camera_params(self, camera_dir, cam_name, ds_idx):
        intrinsic_path = os.path.join(
            camera_dir, f"{cam_name}_intrinsic_params.json"
        )
        extrinsic_path = os.path.join(
            camera_dir, f"{cam_name}_extrinsic_params_aligned.json"
        )

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
            ex = item["extrinsic"]

            mat = np.eye(4, dtype=np.float32)
            mat[:3, :3] = np.array(ex["rotation_matrix"], dtype=np.float32)
            mat[:3, 3] = np.array(ex["translation_vector"], dtype=np.float32)
            c2w_all.append(mat)

        c2w_all = np.stack(c2w_all, axis=0)

        c2w_seq = c2w_all[ds_idx]
        w2c_seq = np.linalg.inv(c2w_seq).astype(np.float32)

        return intrinsic, w2c_seq, c2w_seq

    def _load_frames_by_ids(self, ep_info, frame_ids):
        result = {}

        for fid in frame_ids:
            orig_idx = int(ep_info["ds_indices"][fid])
            frames_cam = {}

            for cam in self.camera_names:
                video_path = os.path.join(
                    ep_info["video_dir"], f"{cam}_color.mp4"
                )

                frame = iio.imread(video_path, index=orig_idx)
                if frame.shape[-1] == 4:
                    frame = frame[..., :3]

                frames_cam[cam] = frame

            result[fid] = frames_cam

        return result
    def _generate_raymap(self, intrinsic, c2w, H, W):
        """
        输入:
            intrinsic : (3,3)
            c2w       : (4,4) 当前帧外参（camera-to-world）
            H,W

        输出:
            ray_o : (H,W,3)
            ray_d : (H,W,3)

        ray_o = 相机中心（每像素相同）
        ray_d = 每像素射线方向（世界坐标系）
        """

        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]

        xs = np.arange(W, dtype=np.float32) + 0.5
        ys = np.arange(H, dtype=np.float32) + 0.5
        xx, yy = np.meshgrid(xs, ys)

        # camera space rays
        dirs = np.stack([
            (xx - cx) / fx,
            (yy - cy) / fy,
            np.ones_like(xx)
        ], axis=-1)   # (H,W,3)

        R = c2w[:3, :3]
        t = c2w[:3, 3]

        # 转到 world space
        rays_d = dirs @ R.T
        rays_d = rays_d / (
            np.linalg.norm(rays_d, axis=-1, keepdims=True) + 1e-8
        )

        # 相机中心 broadcast
        rays_o = np.broadcast_to(t, rays_d.shape).copy()

        return rays_o.astype(np.float32), rays_d.astype(np.float32)


    def _to_uint8_img(self, x, vmin=None, vmax=None):
        """
        float -> uint8 RGB
        支持固定范围，避免逐帧归一化导致画面看起来不动
        """

        if vmin is None:
            vmin = x.min()

        if vmax is None:
            vmax = x.max()

        x = (x - vmin) / (vmax - vmin + 1e-8)
        x = np.clip(x, 0.0, 1.0)

        return (x * 255).astype(np.uint8)
    def __getitem__(self, idx):
        sample_idx = idx % self.total_samples
        ep_idx, start_idx = self.sample_indices[sample_idx]

        info = self.episode_info[ep_idx]
        abs_actions = info['abs_actions']

        # ------------------------------------------------------------
        # frame ids
        # ------------------------------------------------------------
        if self.context_length > 1:
            if np.random.rand() < self.first_round_prob:
                horizon = self.num_frames - self.context_length
                frame_ids = np.array(
                    [0] * self.context_length +
                    list(range(1, horizon + 1))
                )
            else:
                consecutive_ids = np.arange(
                    start_idx,
                    start_idx + self.num_frames - 1
                )
                frame_ids = np.concatenate([[0], consecutive_ids])
        else:
            frame_ids = np.arange(start_idx, start_idx + self.num_frames)

        # ------------------------------------------------------------
        # load video frames
        # ------------------------------------------------------------
        frame_map = self._load_frames_by_ids(info, frame_ids.tolist())

        cam_frames = {cam: [] for cam in self.camera_names}
        cam_traj_maps = {cam: [] for cam in self.camera_names}
        cam_ray_o = {cam: [] for cam in self.camera_names}
        cam_ray_d = {cam: [] for cam in self.camera_names}

        # ============================================================
        # per camera
        # ============================================================
        for cam in self.camera_names:

            cam_info = info['cameras'][cam]

            intrinsic_all = cam_info['intrinsic']
            w2c_all = cam_info['w2c']      # (T,4,4)
            c2w_all = cam_info['c2w']      # (T,4,4)

            sample_frame = frame_map[frame_ids[0]][cam]
            ori_h, ori_w = sample_frame.shape[:2]

            if self.resize_to is not None:
                traj_h, traj_w = self.resize_to
            else:
                traj_h, traj_w = ori_h, ori_w

            h_scale = traj_h / ori_h
            w_scale = traj_w / ori_w

            # resize intrinsic
            scaled_intrinsic = intrinsic_all.copy()
            scaled_intrinsic[0, 0] *= w_scale
            scaled_intrinsic[0, 2] *= w_scale
            scaled_intrinsic[1, 1] *= h_scale
            scaled_intrinsic[1, 2] *= h_scale

            # --------------------------------------------------------
            # traj map（逐帧外参）
            # --------------------------------------------------------
            w2c_seq = w2c_all[frame_ids]

            actions_for_traj = abs_actions[frame_ids]

            traj_maps = generate_traj_map(
                actions_for_traj,
                w2c_seq,
                scaled_intrinsic,
                traj_h,
                traj_w,
                radius=self.traj_radius
            )

            # --------------------------------------------------------
            # 每帧处理
            # --------------------------------------------------------
            for i, fid in enumerate(frame_ids):

                # ---------------- image ----------------
                img = frame_map[fid][cam]

                if img.max() <= 1.0:
                    img = (img * 255).clip(0, 255)

                pil_img = Image.fromarray(
                    img.astype(np.uint8)
                )

                if self.resize_to is not None:
                    pil_img = pil_img.resize(
                        (traj_w, traj_h),
                        Image.BILINEAR
                    )

                cam_frames[cam].append(pil_img)
                cam_traj_maps[cam].append(traj_maps[i])

                # ---------------- raymap ----------------
                c2w = c2w_all[fid]   # 当前帧外参（关键）

                ray_o, ray_d = self._generate_raymap(
                    scaled_intrinsic,
                    c2w,
                    traj_h,
                    traj_w
                )

                # 固定范围，确保视觉变化明显
                ray_o_img = self._to_uint8_img(
                    ray_o,
                    vmin=-1.5,
                    vmax=1.5
                )

                ray_d_img = self._to_uint8_img(
                    ray_d,
                    vmin=-1.0,
                    vmax=1.0
                )

                cam_ray_o[cam].append(ray_o_img)
                cam_ray_d[cam].append(ray_d_img)

        # ============================================================
        # concat multiview
        # ============================================================
        video_list = []
        control_list = []
        ray_o_list = []
        ray_d_list = []

        for i in range(len(frame_ids)):

            concat_img = np.concatenate([
                np.array(cam_frames[cam][i])
                for cam in self.camera_names
            ], axis=0)

            concat_control = np.concatenate([
                cam_traj_maps[cam][i]
                for cam in self.camera_names
            ], axis=0)

            concat_ray_o = np.concatenate([
                cam_ray_o[cam][i]
                for cam in self.camera_names
            ], axis=0)

            concat_ray_d = np.concatenate([
                cam_ray_d[cam][i]
                for cam in self.camera_names
            ], axis=0)

            video_list.append(Image.fromarray(concat_img))
            control_list.append(Image.fromarray(concat_control))
            ray_o_list.append(Image.fromarray(concat_ray_o))
            ray_d_list.append(Image.fromarray(concat_ray_d))

        # ============================================================
        # return
        # ============================================================
        if self.type == 'vace':
            return {
                "video": video_list,
                "vace_reference_image": [video_list[0]],
                "vace_video": control_list,
                "raymap_o": ray_o_list,
                "raymap_d": ray_d_list,
                "prompt": "机械臂按照要求移动夹爪执行任务",
            }

        if self.type == 'control':
            return {
                "video": video_list,
                "reference_image": [video_list[0]],
                "control_video": control_list,
                "prompt": "机械臂按照要求移动夹爪执行任务",
            }

if __name__ == "__main__":

    BASE_PATH = "/mnt/workspace/zsq/Agi2024subset_split/train"

    dataset = AgiBotWCDataset4WanControlmultiview(
        base_path=BASE_PATH,
        num_frames=9,
        episode_limit=2,
    )

    sample = dataset[10]

    out_dir = "./dataset_multiview_vis"
    os.makedirs(out_dir, exist_ok=True)

    writer = imageio.get_writer(
        os.path.join(out_dir, "all_vis.mp4"),
        fps=5
    )

    video = sample["video"]
    control = sample["vace_video"]
    ray_o = sample["raymap_o"]
    ray_d = sample["raymap_d"]

    for i in range(len(video)):
        arr = np.concatenate([
            np.array(control[i]),
            np.array(video[i]),
            np.array(ray_o[i]),
            np.array(ray_d[i]),
        ], axis=1)

        writer.append_data(arr)

    writer.close()

    print("saved:", os.path.join(out_dir, "all_vis.mp4"))