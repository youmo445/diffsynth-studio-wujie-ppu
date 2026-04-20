import imageio, os, torch, warnings, torchvision, argparse, json
from ..utils import ModelConfig
from ..models.utils import load_state_dict
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from scipy.spatial.transform import Rotation
import h5py
from decord import VideoReader, cpu
import matplotlib.cm as cm

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
    """Convert wxyz quaternion(s) to 3x3 rotation matrix. Input shape: (..., 4)."""
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
    """
    Convert [x,y,z, qx,qy,qz,qw] to 4x4 transformation matrix.
    Input: (T, 7)  Output: (T, 4, 4)
    """
    rot_quat = xyz_quat[:, 3:]                     # (T, 4) xyzw
    wxyz = rot_quat[:, [3, 0, 1, 2]]               # convert to wxyz for _quaternion_to_matrix_np
    rot = _quaternion_to_matrix_np(wxyz)            # (T, 3, 3)
    trans = xyz_quat[:, :3]                         # (T, 3)
    T = xyz_quat.shape[0]
    mat = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))  # (T, 4, 4)
    mat[:, :3, :3] = rot
    mat[:, :3, 3] = trans
    return mat


def generate_traj_map(abs_actions, w2c, intrinsic, h, w, radius=50):
    """
    Generate EVAC-style trajectory map from absolute actions and camera params.

    Args:
        abs_actions: np.ndarray (T, 16) — [xyz_l(3), quat_xyzw_l(4), grip_l(1),
                                            xyz_r(3), quat_xyzw_r(4), grip_r(1)]
        w2c:        np.ndarray (4, 4) — world-to-camera matrix
        intrinsic:  np.ndarray (3, 3) — camera intrinsic matrix
        h, w:       int — output image size
        radius:     int — circle radius for end-effector base point

    Returns:
        List[np.ndarray] of length T, each (h, w, 3) uint8 trajectory map.
    """
    T = abs_actions.shape[0]
    ee_key_pts = _EVAC_EndEffectorPts.T                     # (4, 4)
    cvt = _EVAC_Gripper2EEFCvt                              # (4, 4)
    w2c_mat = w2c.astype(np.float32)                        # (4, 4)

    # EEF2Cam rotations
    cvt_vis_l = Rotation.from_euler("xyz", _EVAC_EEF2CamLeft)
    cvt_vis_r = Rotation.from_euler("xyz", _EVAC_EEF2CamRight)

    traj_maps = []
    for t in range(T):
        # Build left arm pose with visual rotation
        pos_l = abs_actions[t, 0:3]
        quat_l = abs_actions[t, 3:7]             # xyzw
        rot_l = Rotation.from_quat(quat_l)
        rot_vis_l = rot_l * cvt_vis_l
        vis_quat_l = np.concatenate([pos_l, rot_vis_l.as_quat()])  # (7,) xyzw

        # Build right arm pose with visual rotation
        pos_r = abs_actions[t, 8:11]
        quat_r = abs_actions[t, 11:15]            # xyzw
        rot_r = Rotation.from_quat(quat_r)
        rot_vis_r = rot_r * cvt_vis_r
        vis_quat_r = np.concatenate([pos_r, rot_vis_r.as_quat()])  # (7,) xyzw

        # Transformation matrices (1, 4, 4)
        pose_l_mat = _get_transformation_matrix_np(vis_quat_l[None])[0]  # (4,4)
        pose_r_mat = _get_transformation_matrix_np(vis_quat_r[None])[0]  # (4,4)

        # world -> camera -> eef
        ee2cam_l = w2c_mat @ pose_l_mat @ cvt  # (4,4)
        ee2cam_r = w2c_mat @ pose_r_mat @ cvt  # (4,4)

        # Project key points
        pts_l = (ee2cam_l @ ee_key_pts)[:3, :]  # (3, 4)
        pts_r = (ee2cam_r @ ee_key_pts)[:3, :]  # (3, 4)

        uvs_l = (intrinsic @ pts_l)             # (3, 4)
        uvs_l = (uvs_l[:2, :] / uvs_l[2:3, :]).T.astype(np.int64)  # (4, 2)
        uvs_r = (intrinsic @ pts_r)
        uvs_r = (uvs_r[:2, :] / uvs_r[2:3, :]).T.astype(np.int64)  # (4, 2)

        # Draw trajectory map using PIL
        from PIL import ImageDraw
        pil_img = Image.new("RGB", (w, h), (50, 50, 50))  # dark grey background
        draw = ImageDraw.Draw(pil_img)

        # Gripper colors based on opening
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
            base = uvs[0]
            bx, by = int(base[0]), int(base[1])
            if 0 <= bx < w and 0 <= by < h:
                draw.ellipse(
                    [bx - radius, by - radius, bx + radius, by + radius],
                    fill=color,
                )
                for k in range(1, len(uvs)):
                    pt = uvs[k]
                    draw.line(
                        [(bx, by), (int(pt[0]), int(pt[1]))],
                        fill=color_list[k - 1], width=8,
                    )

        traj_maps.append(np.array(pil_img))

    return traj_maps


class AgiBotWCDataset4WanControl(torch.utils.data.Dataset):
    """
    AgiBotWC dataset for Wan2.2-A14B-Control training.

    输出 control_video (EVAC trajectory map) 代替 action tensor.

    输出格式:
      {
        "video":           List[PIL.Image], 长度 = num_frames,
        "reference_image": [PIL.Image],     第一帧,
        "control_video":   List[PIL.Image], 长度 = num_frames (trajectory map),
      }

    帧结构 (以 num_frames=13, context_length=5 为例):
      正常采样 (95%):
        frame_ids = [0, start, start+1, ..., start+11]
      首轮模拟 (5%):
        frame_ids = [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    """

    def __init__(
        self,
        base_path='/opt/zsq/AgiBotWorldChallenge2026/WorldModel/mnt/public/chenshengcong/dataset/iros_challenge_2025_acwm/train',
        num_frames=9,
        repeat=1,
        episode_stride=1,
        episode_limit=254,
        original_hz=30,
        target_hz=5,
        stride=1,
        first_round_prob=0.05,
        context_length=5,
        traj_radius=50,
        resize_to=(320, 512),   # (H, W), 默认与 EVAC sample_size 一致
        dataset_type = 'vace',
    ):
        super().__init__()

        self.base_path = base_path
        self.num_frames = num_frames
        self.repeat = repeat
        self.load_from_cache = False
        self.stride = stride
        self.resize_to = resize_to
        self.first_round_prob = first_round_prob
        self.context_length = context_length
        self.traj_radius = traj_radius
        self.type = dataset_type
        if original_hz % target_hz != 0:
            raise ValueError(f"Cannot downsample from {original_hz}Hz to {target_hz}Hz")
        self.downsample_step = original_hz // target_hz
        print(f"[AgiBotWCDataset4WanControl] Downsample: {original_hz}Hz -> {target_hz}Hz (step={self.downsample_step})")

        # ================================================================
        # 发现并分组 episodes
        # ================================================================
        episode_dict = {}
        for name in sorted(os.listdir(base_path)):
            path = os.path.join(base_path, name)
            if not os.path.isdir(path):
                continue
            key = "-".join(name.split("-")[:2])
            episode_dict.setdefault(key, []).append(path)

        episodes = []
        for k in sorted(episode_dict.keys()):
            episodes.append(sorted(episode_dict[k]))

        original_count = len(episodes)
        if episode_limit is not None:
            episodes = episodes[:episode_limit]
        self.episodes = episodes[::episode_stride]
        print(f"[AgiBotWCDataset4WanControl] {original_count} episodes found, using {len(self.episodes)} after limit/stride")

        # ================================================================
        # 预加载 h5 + 相机参数, 计算绝对动作, 构建滑动窗口索引
        # ================================================================
        self.episode_info = []
        self.sample_indices = []

        for ep_idx, ep_paths in enumerate(self.episodes):
            # 加载所有 chunk 的 h5 数据
            pos_list, quat_list, grip_list = [], [], []
            chunk_frame_counts = []

            for sub in ep_paths:
                h5_path = os.path.join(sub, "proprio_stats.h5")
                with h5py.File(h5_path, 'r') as f:
                    pos = f['state/end/position'][:]
                    quat = f['state/end/orientation'][:]
                    grip = f['state/effector/position'][:]

                T = pos.shape[0]
                chunk_frame_counts.append(T)
                pos_list.append(pos)
                quat_list.append(quat)
                grip_list.append(grip.reshape(T, 2, -1)[..., 0] if grip.ndim == 3 else grip)

            pos_all = np.concatenate(pos_list, axis=0)
            quat_all = np.concatenate(quat_list, axis=0)
            grip_all = np.concatenate(grip_list, axis=0)

            # 下采样
            T_raw = pos_all.shape[0]
            ds_idx = np.arange(0, T_raw, self.downsample_step)
            T_ds = len(ds_idx)

            # 计算绝对动作 (用于 get_traj): [T_ds, 16]
            abs_actions = self._compute_abs_action(
                pos_all[ds_idx], quat_all[ds_idx], grip_all[ds_idx]
            )

            # 加载相机参数 (从第一个 chunk 读取)
            intrinsic, w2c = self._load_camera_params(ep_paths[0])

            self.episode_info.append({
                'ep_paths': ep_paths,
                'chunk_frame_counts': chunk_frame_counts,
                'ds_indices': ds_idx,
                'T_ds': T_ds,
                'abs_actions': abs_actions,      # [T_ds, 16]
                'intrinsic': intrinsic,          # (3, 3)
                'w2c': w2c,                      # (T_ds, 4, 4) or (1, 4, 4)
            })

            max_start = T_ds - (self.num_frames - 1)
            for start in range(0, max_start, self.stride):
                self.sample_indices.append((ep_idx, start))

        self.total_samples = len(self.sample_indices)
        self.length = self.total_samples * repeat
        print(
            f"[AgiBotWCDataset4WanControl] {len(self.episodes)} episodes, "
            f"{self.total_samples} samples, repeat={repeat}, length={self.length}"
        )

    # ====================================================================
    # 绝对动作计算
    # ====================================================================

    def _compute_abs_action(self, pos, quat, grip):
        """
        构造 EVAC 格式的绝对动作: [T, 16]
        [xyz_l(3), quat_xyzw_l(4), grip_l(1), xyz_r(3), quat_xyzw_r(4), grip_r(1)]
        """
        T = pos.shape[0]
        abs_actions = np.zeros((T, 16), dtype=np.float32)
        for t in range(T):
            abs_actions[t, 0:3] = pos[t, 0]          # left arm xyz
            abs_actions[t, 3:7] = quat[t, 0]         # left arm quat xyzw
            abs_actions[t, 7] = grip[t, 0]            # left gripper
            abs_actions[t, 8:11] = pos[t, 1]          # right arm xyz
            abs_actions[t, 11:15] = quat[t, 1]        # right arm quat xyzw
            abs_actions[t, 15] = grip[t, 1]           # right gripper
        return abs_actions

    # ====================================================================
    # 相机参数加载
    # ====================================================================

    def _load_camera_params(self, chunk_path):
        """
        从 episode chunk 目录加载 intrinsic 和 extrinsic.
        Returns:
            intrinsic: np.ndarray (3, 3)
            w2c:       np.ndarray (4, 4) — 使用第一帧的 extrinsic
        """
        # Intrinsic
        intr_path = os.path.join(chunk_path, "head_intrinsic_params.json")
        with open(intr_path, "r") as f:
            info = json.load(f)["intrinsic"]
        intrinsic = np.eye(3, dtype=np.float32)
        intrinsic[0, 0] = info["fx"]
        intrinsic[1, 1] = info["fy"]
        intrinsic[0, 2] = info["ppx"]
        intrinsic[1, 2] = info["ppy"]

        # Extrinsic (use first frame)
        extr_path = os.path.join(chunk_path, "head_extrinsic_params_aligned.json")
        with open(extr_path, "r") as f:
            extr_info = json.load(f)[0]
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = np.array(extr_info["extrinsic"]["rotation_matrix"], dtype=np.float32)
        c2w[:3, 3] = np.array(extr_info["extrinsic"]["translation_vector"], dtype=np.float32)
        w2c = np.linalg.inv(c2w).astype(np.float32)

        return intrinsic, w2c

    # ====================================================================
    # 视频帧加载 (与 AgiBotWCDatasetV2 相同)
    # ====================================================================

    def _load_frames_by_ids(self, ep_info, frame_ids):
        chunk_frame_counts = ep_info['chunk_frame_counts']
        ep_paths = ep_info['ep_paths']
        ds_indices = ep_info['ds_indices']

        unique_ids = sorted(set(frame_ids))
        orig_needed = {fid: int(ds_indices[fid]) for fid in unique_ids}

        chunk_cum = np.cumsum([0] + chunk_frame_counts)
        chunk_loads = {}
        for fid, orig in orig_needed.items():
            cid = int(np.searchsorted(chunk_cum[1:], orig, side='right'))
            local = orig - int(chunk_cum[cid])
            chunk_loads.setdefault(cid, {})[local] = fid

        result = {}
        for cid, local_map in chunk_loads.items():
            video_path = os.path.join(ep_paths[cid], "head_color.mp4")
            vr = VideoReader(video_path, ctx=cpu(0))
            for local_idx, fid in local_map.items():
                idx_to_read = min(local_idx, len(vr) - 1)
                frame = vr[idx_to_read].asnumpy()  # (H, W, 3) RGB
                result[fid] = frame
            del vr

        return result

    # ====================================================================
    # Dataset 接口
    # ====================================================================

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample_idx = idx % self.total_samples
        ep_idx, start_idx = self.sample_indices[sample_idx]
        info = self.episode_info[ep_idx]
        abs_actions = info['abs_actions']
        intrinsic = info['intrinsic']
        w2c = info['w2c']

        if self.context_length>1:
            # 构造 frame_ids
            if np.random.rand() < self.first_round_prob:
                horizon = self.num_frames - self.context_length
                frame_ids = np.array(
                    [0] * self.context_length
                    + list(range(1, horizon + 1))
                )
            else:
                consecutive_ids = np.arange(start_idx, start_idx + self.num_frames - 1)
                frame_ids = np.concatenate([[0], consecutive_ids])
        else:
            frame_ids = np.arange(start_idx, start_idx + self.num_frames)
        # 加载视频帧
        print(f'frame_ids:{frame_ids}')
        frame_map = self._load_frames_by_ids(info, frame_ids.tolist())

        # 获取原始视频帧尺寸
        sample_frame = frame_map[frame_ids[0]]
        ori_h, ori_w = sample_frame.shape[:2]

        # 确定 traj map 的生成尺寸, 并按 EVAC 方式缩放 intrinsic
        if self.resize_to is not None:
            traj_h, traj_w = self.resize_to
            # EVAC: intrinsic 必须按 resize 比例缩放, 否则投影坐标会错位
            h_scale = traj_h / ori_h
            w_scale = traj_w / ori_w
            scaled_intrinsic = intrinsic.copy()
            scaled_intrinsic[0, 0] *= w_scale   # fx
            scaled_intrinsic[0, 2] *= w_scale   # cx
            scaled_intrinsic[1, 1] *= h_scale   # fy
            scaled_intrinsic[1, 2] *= h_scale   # cy
        else:
            traj_h, traj_w = ori_h, ori_w
            scaled_intrinsic = intrinsic

        # 生成 trajectory map (control_video)
        actions_for_traj = abs_actions[frame_ids]
        traj_maps = generate_traj_map(
            actions_for_traj, w2c, scaled_intrinsic,
            traj_h, traj_w, radius=self.traj_radius,
        )

        # 组装输出
        video_list = []
        control_list = []
        for i, fid in enumerate(frame_ids):
            img = frame_map[fid]
            if img.max() <= 1.0:
                img = (img * 255).clip(0, 255)
            pil_img = Image.fromarray(img.astype(np.uint8))
            if self.resize_to is not None:
                pil_img = pil_img.resize((traj_w, traj_h), Image.BILINEAR)  # PIL.resize takes (w, h)
            video_list.append(pil_img)
            control_list.append(Image.fromarray(traj_maps[i]))
        print(f'length:{len(video_list)}')
        if self.type == 'vace':
            return {
                "video": video_list,
                "vace_reference_image": [video_list[0]],
                "vace_video": control_list,
                "prompt": "机械臂按照要求移动夹爪执行任务",
            }
        if self.type == 'control':
            return {
                "video": video_list,
                "reference_image": [video_list[0]],
                "control_video": control_list,
                "prompt": "机械臂按照要求移动夹爪执行任务",
            }  


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("image",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
            
        self.base_path = base_path
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.repeat = repeat

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in tqdm(f):
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]


    def generate_metadata(self, folder):
        image_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            image_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["image"] = image_list
        metadata["prompt"] = prompt_list
        return metadata
    
    
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        return image
    
    
    def load_data(self, file_path):
        return self.load_image(file_path)


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                if isinstance(data[key], list):
                    path = [os.path.join(self.base_path, p) for p in data[key]]
                    data[key] = [self.load_data(p) for p in path]
                else:
                    path = os.path.join(self.base_path, data[key])
                    data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat



class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm", "gif"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
        
        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat
        
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
            
    
    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata
        
        
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
    
    def _load_gif(self, file_path):
        gif_img = Image.open(file_path)
        frame_count = 0
        delays, frames = [], []
        while True:
            delay = gif_img.info.get('duration', 100) # ms
            delays.append(delay)
            rgb_frame = gif_img.convert("RGB")   
            croped_frame = self.crop_and_resize(rgb_frame, *self.get_height_width(rgb_frame))
            frames.append(croped_frame)             
            frame_count += 1
            try:
                gif_img.seek(frame_count)
            except:
                break
        # delays canbe used to calculate framerates
        # i guess it is better to sample images with stable interval,
        # and using minimal_interval as the interval, 
        # and framerate = 1000 / minimal_interval
        if any((delays[0] != i) for i in delays):
            minimal_interval = min([i for i in delays if i > 0])
            # make a ((start,end),frameid) struct
            start_end_idx_map = [((sum(delays[:i]), sum(delays[:i+1])), i) for i in range(len(delays))]
            _frames = []
            # according gemini-code-assist, make it more efficient to locate
            # where to sample the frame
            last_match = 0
            for i in range(sum(delays) // minimal_interval):
                current_time = minimal_interval * i
                for idx, ((start, end), frame_idx) in enumerate(start_end_idx_map[last_match:]):
                    if start <= current_time < end:
                        _frames.append(frames[frame_idx])
                        last_match = idx + last_match
                        break
            frames = _frames
        num_frames = len(frames)
        if num_frames > self.num_frames:
            num_frames = self.num_frames
        else:
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        frames = frames[:num_frames]
        return frames
    
    def load_video(self, file_path):
        if file_path.lower().endswith(".gif"):
            return self._load_gif(file_path)
        reader = imageio.get_reader(file_path)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        reader.close()
        return frames
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        frames = [image]
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension
    
    
    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension
    
    
    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            return None


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                path = os.path.join(self.base_path, data[key])
                data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat



class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None, upcast_dtype=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        if upcast_dtype is not None:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.to(upcast_dtype)
        return model


    def mapping_lora_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                new_state_dict[new_key] = value
            elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                new_state_dict[key] = value
        return new_state_dict


    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict
    
    
    def transfer_data_to_device(self, data, device, torch_float_dtype=None):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
                if torch_float_dtype is not None and data[key].dtype in [torch.float, torch.float16, torch.bfloat16]:
                    data[key] = data[key].to(torch_float_dtype)
        return data
    
    
    def parse_model_configs(self, model_paths, model_id_with_origin_paths, enable_fp8_training=False):
        offload_dtype = torch.float8_e4m3fn if enable_fp8_training else None
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path, offload_dtype=offload_dtype) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1], offload_dtype=offload_dtype) for i in model_id_with_origin_paths]
        return model_configs
    
    
    def switch_pipe_to_training_mode(
        self,
        pipe,
        trainable_models,
        lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=None,
        enable_fp8_training=False,
    ):
        # Scheduler
        pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze untrainable models
        pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
        # Enable FP8 if pipeline supports
        if enable_fp8_training and hasattr(pipe, "_enable_fp8_lora_training"):
            pipe._enable_fp8_lora_training(torch.float8_e4m3fn)
        
        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank,
                upcast_dtype=pipe.torch_dtype,
            )
            if lora_checkpoint is not None:
                state_dict = load_state_dict(lora_checkpoint)
                state_dict = self.mapping_lora_state_dict(state_dict)
                load_result = model.load_state_dict(state_dict, strict=False)
                print(f"LoRA checkpoint loaded: {lora_checkpoint}, total {len(state_dict)} keys")
                if len(load_result[1]) > 0:
                    print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")
            setattr(pipe, lora_base_model, model)


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0


    def on_step_end(self, accelerator, model, save_steps=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator, model, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator, model, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)


def launch_training_task(
    dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 8,
    save_steps: int = None,
    save_epochs: int = 1,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        gradient_accumulation_steps = args.gradient_accumulation_steps
        find_unused_parameters = args.find_unused_parameters
        ###验证集
        val_interval = args.val_interval
    writer = SummaryWriter(log_dir=os.path.join(args.output_path, "tensorboard") if args is not None else "./tensorboard")
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    global_step = 0
    epoch_loss = 0.0
    epoch_steps = 0

    for epoch_id in range(num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                print(f'Epoch {epoch_id} Step {global_step} Loss: {loss.item()}')

                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()
                epoch_loss += loss.item()
                epoch_steps += 1
                global_step += 1
                # to avoid the too long epoch 
                if epoch_steps > 500:
                    break
        if accelerator.is_main_process and epoch_steps > 0:
            writer.add_scalar("Loss/epoch", epoch_loss / epoch_steps, epoch_id)
        if val_dataloader is not None and (epoch_id + 1) % val_interval == 0:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            if accelerator.is_main_process:
                print(f"\nRunning validation for epoch {epoch_id}...")

            for data in tqdm(val_dataloader, desc="Validation", disable=not accelerator.is_local_main_process):
                with torch.no_grad():
                    if getattr(val_dataset, 'load_from_cache', False):
                        loss = model({}, inputs=data)
                    else:
                        loss = model(data)
                    

                    avg_loss = accelerator.gather(loss).mean().item()
                    val_loss += avg_loss
                    val_steps += 1

                    if val_steps > 2001:
                        break
            
            if val_steps > 0:
                avg_val_loss = val_loss / val_steps
                if accelerator.is_main_process:
                    writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch_id)
                    print(f"Epoch {epoch_id} Validation Loss: {avg_val_loss}")
            
            model.train() 
            # --------------------
        if save_steps is None and (epoch_id + 1) % save_epochs == 0:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in tqdm(enumerate(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data, return_inputs=True)
                torch.save(data, save_path)



def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1280*720, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--audio_processor_config", type=str, default=None, help="Model ID with origin paths to the audio processor config, e.g., Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--save_epochs", type=int, default=1, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    return parser



def flux_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--align_to_opensource_format", default=False, action="store_true", help="Whether to align the lora format to opensource format. Only for DiT's LoRA.")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    return parser



def qwen_image_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Paths to tokenizer.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--processor_path", type=str, default=None, help="Path to the processor. If provided, the processor will be used for image editing.")
    parser.add_argument("--enable_fp8_training", default=False, action="store_true", help="Whether to enable FP8 training. Only available for LoRA training on a single GPU.")
    parser.add_argument("--task", type=str, default="sft", required=False, help="Task type.")
    return parser
