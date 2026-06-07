#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

try:
    import yaml
except ImportError:
    yaml = None

try:
    from .arxx5_dataset4_wancontrolmultiview import (
        generate_raymap,
        generate_traj_map,
        to_uint8_img,
        write_video_rgb,
    )
except ImportError:
    from arxx5_dataset4_wancontrolmultiview import (
        generate_raymap,
        generate_traj_map,
        to_uint8_img,
        write_video_rgb,
    )


ROBOTWIN_THREE_VIEW_CAMERAS = ("head_camera", "left_camera", "right_camera")
ROBOTWIN_ALL_CAMERAS = ("front_camera", "head_camera", "left_camera", "right_camera")
DEFAULT_FK_CONFIG_PATH = "/data/zsq/RoboTwin/assets/embodiments/aloha-agilex/config.yml"


def _episode_sort_key(path: Path) -> Tuple[int, str]:
    match = re.search(r"episode(\d+)", path.stem)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def _decode_rgb_bytes(raw) -> np.ndarray:
    if isinstance(raw, np.bytes_):
        raw = raw.tobytes()
    elif isinstance(raw, np.ndarray):
        raw = raw.tobytes()
    elif not isinstance(raw, (bytes, bytearray)):
        raw = bytes(raw)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)


def _to_4x4_w2c(extrinsic_cv: np.ndarray) -> np.ndarray:
    extrinsic_cv = np.asarray(extrinsic_cv, dtype=np.float32)
    if extrinsic_cv.shape == (4, 4):
        return extrinsic_cv
    if extrinsic_cv.shape != (3, 4):
        raise ValueError(f"Expected extrinsic_cv shape (3, 4) or (4, 4), got {extrinsic_cv.shape}")
    out = np.eye(4, dtype=np.float32)
    out[:3, :] = extrinsic_cv
    return out


def _scale_intrinsic(intrinsic: np.ndarray, in_h: int, in_w: int, out_h: int, out_w: int) -> np.ndarray:
    K = np.asarray(intrinsic, dtype=np.float32).copy()
    K[0, 0] *= out_w / float(in_w)
    K[0, 2] *= out_w / float(in_w)
    K[1, 1] *= out_h / float(in_h)
    K[1, 2] *= out_h / float(in_h)
    return K


def _make_abs_actions_from_endpose(
    left_endpose: np.ndarray,
    right_endpose: np.ndarray,
    left_gripper: np.ndarray,
    right_gripper: np.ndarray,
    quat_order: str,
) -> np.ndarray:
    left_endpose = np.asarray(left_endpose, dtype=np.float32)
    right_endpose = np.asarray(right_endpose, dtype=np.float32)
    out = np.zeros((left_endpose.shape[0], 16), dtype=np.float32)
    out[:, 0:3] = left_endpose[:, 0:3]
    out[:, 8:11] = right_endpose[:, 0:3]
    if quat_order == "xyzw":
        out[:, 3:7] = left_endpose[:, 3:7]
        out[:, 11:15] = right_endpose[:, 3:7]
    elif quat_order == "wxyz":
        out[:, 3:7] = left_endpose[:, [4, 5, 6, 3]]
        out[:, 11:15] = right_endpose[:, [4, 5, 6, 3]]
    else:
        raise ValueError(f"Unsupported quat_order: {quat_order}")
    out[:, 7] = np.clip(np.asarray(left_gripper, dtype=np.float32), 0.0, 1.0) * 120.0
    out[:, 15] = np.clip(np.asarray(right_gripper, dtype=np.float32), 0.0, 1.0) * 120.0
    return out


def _parse_base_paths(base_path: str) -> List[Path]:
    text = str(base_path).strip()
    if not text:
        raise ValueError("base_path is empty")
    if text.startswith("["):
        items = json.loads(text)
    else:
        items = [item.strip() for item in text.split(",") if item.strip()]
    if not items:
        raise ValueError(f"No valid base paths parsed from {base_path!r}")
    return [Path(item) for item in items]


def _load_yaml(path: Path) -> dict:
    if yaml is None:
        raise ImportError("PyYAML is required for RoboTwin action-FK dataset.")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _gripper_joint_names(config: dict) -> Tuple[str, str]:
    if "gripper_joint_names" in config:
        return tuple(config["gripper_joint_names"])
    return config["gripper_name"][0]["base"], config["gripper_name"][1]["base"]


def _transformed_eef_pose(config: dict, joint_global_transform: np.ndarray) -> np.ndarray:
    global_trans_matrix = np.asarray(config["global_trans_matrix"], dtype=np.float64)
    delta_matrix = np.asarray(config["delta_matrix"], dtype=np.float64)
    dis = float(config.get("gripper_bias", 0.0)) - 0.12
    rot = np.asarray(joint_global_transform[:3, :3], dtype=np.float64) @ global_trans_matrix @ delta_matrix
    pos = np.asarray(joint_global_transform[:3, 3], dtype=np.float64) + rot @ np.asarray([dis, 0.0, 0.0])
    quat_xyzw = Rotation.from_matrix(rot).as_quat()
    quat_wxyz = np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)
    return np.concatenate([pos, quat_wxyz]).astype(np.float64)


def _make_abs_actions_from_fk_wxyz(
    left_pose_wxyz: np.ndarray,
    right_pose_wxyz: np.ndarray,
    left_gripper: np.ndarray,
    right_gripper: np.ndarray,
) -> np.ndarray:
    left_pose_wxyz = np.asarray(left_pose_wxyz, dtype=np.float32)
    right_pose_wxyz = np.asarray(right_pose_wxyz, dtype=np.float32)
    out = np.zeros((left_pose_wxyz.shape[0], 16), dtype=np.float32)
    out[:, 0:3] = left_pose_wxyz[:, 0:3]
    out[:, 3:7] = left_pose_wxyz[:, [4, 5, 6, 3]]
    out[:, 7] = np.clip(np.asarray(left_gripper, dtype=np.float32), 0.0, 1.0) * 120.0
    out[:, 8:11] = right_pose_wxyz[:, 0:3]
    out[:, 11:15] = right_pose_wxyz[:, [4, 5, 6, 3]]
    out[:, 15] = np.clip(np.asarray(right_gripper, dtype=np.float32), 0.0, 1.0) * 120.0
    return out


def _parse_xyz(text: Optional[str], default=(0.0, 0.0, 0.0)) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(x) for x in text.split()], dtype=np.float64)


def _make_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    out[:3, 3] = xyz
    return out


def _joint_motion(joint_type: str, axis: np.ndarray, value: float) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    if joint_type in {"revolute", "continuous"}:
        norm = float(np.linalg.norm(axis))
        if norm > 1e-12:
            out[:3, :3] = Rotation.from_rotvec(axis / norm * float(value)).as_matrix()
    elif joint_type == "prismatic":
        out[:3, 3] = axis * float(value)
    return out


class _UrdfSerialFK:
    def __init__(self, urdf_path: Path, root_pose: Sequence[float]):
        self.children_by_parent: Dict[str, List[dict]] = {}
        self.child_links = set()
        robot = ET.parse(urdf_path).getroot()
        for elem in robot.findall("joint"):
            name = elem.attrib["name"]
            joint_type = elem.attrib.get("type", "fixed")
            parent = elem.find("parent").attrib["link"]
            child = elem.find("child").attrib["link"]
            origin = elem.find("origin")
            axis = elem.find("axis")
            joint = {
                "name": name,
                "type": joint_type,
                "parent": parent,
                "child": child,
                "origin": _make_transform(
                    _parse_xyz(origin.attrib.get("xyz") if origin is not None else None),
                    _parse_xyz(origin.attrib.get("rpy") if origin is not None else None),
                ),
                "axis": _parse_xyz(axis.attrib.get("xyz") if axis is not None else None, default=(0.0, 0.0, 1.0)),
            }
            self.children_by_parent.setdefault(parent, []).append(joint)
            self.child_links.add(child)

        roots = sorted(set(self.children_by_parent) - self.child_links)
        if not roots:
            raise ValueError(f"Cannot infer URDF root link from {urdf_path}")
        self.root_link = roots[0]

        root_pose = np.asarray(root_pose, dtype=np.float64)
        self.root_transform = np.eye(4, dtype=np.float64)
        self.root_transform[:3, 3] = root_pose[:3]
        quat_wxyz = root_pose[3:7]
        self.root_transform[:3, :3] = Rotation.from_quat(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        ).as_matrix()

    def forward(self, joint_values: Dict[str, float]) -> Dict[str, np.ndarray]:
        link_transforms = {self.root_link: self.root_transform}
        joint_transforms: Dict[str, np.ndarray] = {}
        stack = [self.root_link]
        while stack:
            parent = stack.pop()
            parent_tf = link_transforms[parent]
            for joint in self.children_by_parent.get(parent, []):
                value = float(joint_values.get(joint["name"], 0.0))
                child_tf = parent_tf @ joint["origin"] @ _joint_motion(joint["type"], joint["axis"], value)
                joint_transforms[joint["name"]] = child_tf
                link_transforms[joint["child"]] = child_tf
                stack.append(joint["child"])
        return joint_transforms


class RoboTwinActionFKDatasetWancontrolmultiview(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path: str = "/mnt/data/zsq/RoboTwin/dataset",
        num_frames: int = 13,
        repeat: int = 1,
        split: str = "train",
        train_episodes_per_task: int = 40,
        val_episodes_per_task: int = 10,
        stride: int = 1,
        first_round_prob: float = 0.05,
        context_length: int = 5,
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
        resize_to: Optional[Tuple[int, int]] = None,
        dataset_type: str = "vace",
        view_mode: str = "three",
        single_camera: str = "head_camera",
        camera_names: Optional[Sequence[str]] = None,
        endpose_quat_order: str = "wxyz",
        prompt: str = "机械臂按照要求移动夹爪执行任务",
        task_limit: Optional[int] = None,
        fk_config_path: str = DEFAULT_FK_CONFIG_PATH,
    ):
        super().__init__()
        if split not in {"train", "val", "all"}:
            raise ValueError(f"Unsupported split: {split}")
        if dataset_type not in {"vace", "control"}:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")
        if raymap_mode != "image":
            raise ValueError("RoboTwinDatasetWancontrolmultiview currently supports raymap_mode='image'")
        if view_mode not in {"three", "single"}:
            raise ValueError(f"Unsupported view_mode: {view_mode}")
        if single_camera not in ROBOTWIN_ALL_CAMERAS:
            raise ValueError(f"Unsupported single_camera: {single_camera}")
        if endpose_quat_order not in {"wxyz", "xyzw"}:
            raise ValueError(f"Unsupported endpose_quat_order: {endpose_quat_order}")

        self.base_paths = _parse_base_paths(base_path)
        self.base_path = self.base_paths[0]
        self.num_frames = int(num_frames)
        self.repeat = int(repeat)
        self.split = split
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
        if resize_to is None and view_mode == "single":
            resize_to = (256, 320)
        self.resize_to = resize_to
        self.type = dataset_type
        self.view_mode = view_mode
        self.single_camera = single_camera
        self.endpose_quat_order = endpose_quat_order
        self.prompt = prompt
        self.output_fps = 30.0
        self.load_from_cache = False
        self.fk_config_path = Path(fk_config_path)
        self._fk_config = None
        self._fk_engine = None
        self._fk_scene = None
        self._fk_robot = None
        self._fk_name_to_idx = None
        self._fk_left_ee = None
        self._fk_right_ee = None

        if camera_names is not None:
            self.camera_names = list(camera_names)
        elif view_mode == "single":
            self.camera_names = [single_camera]
        else:
            self.camera_names = list(ROBOTWIN_THREE_VIEW_CAMERAS)
        for cam in self.camera_names:
            if cam not in ROBOTWIN_ALL_CAMERAS:
                raise ValueError(f"Unsupported camera name: {cam}")

        self.episodes = self._discover_episodes(
            train_episodes_per_task=int(train_episodes_per_task),
            val_episodes_per_task=int(val_episodes_per_task),
            task_limit=task_limit,
        )
        self.episode_info: List[dict] = []
        self.sample_indices: List[Tuple[int, int]] = []
        for ep in self.episodes:
            info = self._build_episode_info(ep)
            if info is None:
                continue
            ep_idx = len(self.episode_info)
            self.episode_info.append(info)
            max_start = info["T"] - self.num_frames + 1
            for start in range(0, max_start, self.stride):
                self.sample_indices.append((ep_idx, start))
        self.total_samples = len(self.sample_indices)
        self.length = self.total_samples * self.repeat
        print(
            "[RoboTwinActionFKDataset] "
            f"bases={len(self.base_paths)} first_base={self.base_path} split={self.split} episodes={len(self.episode_info)} "
            f"samples={self.total_samples} view_mode={self.view_mode} cameras={self.camera_names} "
            f"fps=30 resize_to={self.resize_to} traj_source=joint_action_fk fk_config={self.fk_config_path}"
        )

    def _discover_episodes(self, train_episodes_per_task: int, val_episodes_per_task: int, task_limit: Optional[int]) -> List[dict]:
        episodes: List[dict] = []
        roots = self.base_paths[: int(task_limit)] if task_limit is not None else self.base_paths
        for root in roots:
            if not root.exists():
                raise FileNotFoundError(f"RoboTwin dataset root not found: {root}")
            data_dirs: List[Path] = []
            if root.is_dir() and any(root.glob("*.hdf5")):
                data_dirs.append(root)
            elif root.is_dir() and (root / "data").is_dir():
                data_dirs.append(root / "data")
            else:
                task_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
                for task_dir in task_dirs:
                    run_dirs = [p for p in sorted(task_dir.iterdir()) if p.is_dir() and (p / "data").is_dir()]
                    data_dirs.extend([run_dir / "data" for run_dir in run_dirs])

            for data_dir in data_dirs:
                files = sorted(data_dir.glob("*.hdf5"), key=_episode_sort_key)
                if self.split == "train":
                    selected = files[:train_episodes_per_task]
                elif self.split == "val":
                    selected = files[train_episodes_per_task : train_episodes_per_task + val_episodes_per_task]
                else:
                    selected = files
                if data_dir.name == "data":
                    run_name = data_dir.parent.name
                    task_name = data_dir.parent.parent.name
                else:
                    run_name = "direct"
                    task_name = data_dir.name
                for path in selected:
                    ep_id, _ = _episode_sort_key(path)
                    episodes.append(
                        {
                            "task": task_name,
                            "run": run_name,
                            "episode_index": int(ep_id) if isinstance(ep_id, int) else -1,
                            "path": path,
                        }
                    )
        return episodes

    def _build_episode_info(self, ep: dict) -> Optional[dict]:
        try:
            with h5py.File(ep["path"], "r") as f:
                n = int(f["joint_action/left_arm"].shape[0])
                for cam in self.camera_names:
                    n = min(n, int(f[f"observation/{cam}/rgb"].shape[0]))
        except Exception as exc:
            print(f"[RoboTwinActionFKDataset] skip {ep['path']}: {exc}")
            return None
        if n < self.num_frames:
            return None
        out = dict(ep)
        out["T"] = n
        return out

    def get_constant_head_ray_maps(self) -> Tuple[List[Image.Image], List[Image.Image]]:
        if "head_camera" not in self.camera_names:
            raise ValueError("head_camera is not active in this dataset instance.")
        if not self.episode_info:
            raise RuntimeError("Dataset has no valid episodes for building head ray maps.")
        info = self.episode_info[0]
        with h5py.File(info["path"], "r") as f:
            frame = _decode_rgb_bytes(f["observation/head_camera/rgb"][0])
            ori_h, ori_w = frame.shape[:2]
            out_h, out_w = self.resize_to if self.resize_to is not None else (ori_h, ori_w)
            K = _scale_intrinsic(np.asarray(f["observation/head_camera/intrinsic_cv"][0], dtype=np.float32), ori_h, ori_w, out_h, out_w)
            c2w = np.linalg.inv(_to_4x4_w2c(f["observation/head_camera/extrinsic_cv"][0])).astype(np.float32)
        ro, rd = generate_raymap(K, c2w, out_h, out_w)
        ray_o = Image.fromarray(to_uint8_img(ro, self.ray_o_vmin, self.ray_o_vmax))
        ray_d = Image.fromarray(to_uint8_img(rd, self.ray_d_vmin, self.ray_d_vmax))
        return [ray_o.copy() for _ in range(self.num_frames)], [ray_d.copy() for _ in range(self.num_frames)]

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

    def _ensure_fk(self) -> None:
        if self._fk_robot is not None:
            return
        if not self.fk_config_path.exists():
            raise FileNotFoundError(f"RoboTwin FK config not found: {self.fk_config_path}")
        self._fk_config = _load_yaml(self.fk_config_path)
        urdf_path = Path(self._fk_config["urdf_path"])
        if not urdf_path.is_absolute():
            urdf_path = self.fk_config_path.parent / urdf_path
        root_pose = self._fk_config.get("robot_pose", [[0, 0, 0, 1, 0, 0, 0]])[0]
        self._fk_robot = _UrdfSerialFK(urdf_path, root_pose)

    def _fk_abs_actions_from_joint_action(
        self,
        left_arm: np.ndarray,
        left_gripper: np.ndarray,
        right_arm: np.ndarray,
        right_gripper: np.ndarray,
    ) -> np.ndarray:
        self._ensure_fk()
        n = min(len(left_arm), len(left_gripper), len(right_arm), len(right_gripper))
        left_pose = np.zeros((n, 7), dtype=np.float64)
        right_pose = np.zeros((n, 7), dtype=np.float64)
        left_joint_names, right_joint_names = self._fk_config["arm_joints_name"]
        left_ee_name, right_ee_name = self._fk_config["ee_joints"]
        for i in range(n):
            joint_values = {}
            for name, value in zip(left_joint_names, np.asarray(left_arm[i], dtype=np.float64)):
                joint_values[name] = float(value)
            for name, value in zip(right_joint_names, np.asarray(right_arm[i], dtype=np.float64)):
                joint_values[name] = float(value)
            joint_transforms = self._fk_robot.forward(joint_values)
            left_pose[i] = _transformed_eef_pose(self._fk_config, joint_transforms[left_ee_name])
            right_pose[i] = _transformed_eef_pose(self._fk_config, joint_transforms[right_ee_name])
        return _make_abs_actions_from_fk_wxyz(left_pose, right_pose, left_gripper[:n], right_gripper[:n])

    def _read_episode_arrays(self, path: Path, frame_ids: np.ndarray):
        unique_ids = sorted({int(i) for i in frame_ids.tolist()})
        unique_to_pos = {fid: pos for pos, fid in enumerate(unique_ids)}
        with h5py.File(path, "r") as f:
            left_arm = np.asarray(f["joint_action/left_arm"][unique_ids], dtype=np.float32)
            right_arm = np.asarray(f["joint_action/right_arm"][unique_ids], dtype=np.float32)
            left_gripper = np.asarray(f["joint_action/left_gripper"][unique_ids], dtype=np.float32)
            right_gripper = np.asarray(f["joint_action/right_gripper"][unique_ids], dtype=np.float32)
            abs_unique = self._fk_abs_actions_from_joint_action(
                left_arm,
                left_gripper,
                right_arm,
                right_gripper,
            )
            abs_actions = np.stack([abs_unique[unique_to_pos[int(fid)]] for fid in frame_ids], axis=0)

            frames_by_cam: Dict[str, List[np.ndarray]] = {}
            w2c_by_cam: Dict[str, np.ndarray] = {}
            K_by_cam: Dict[str, np.ndarray] = {}
            for cam in self.camera_names:
                rgb_ds = f[f"observation/{cam}/rgb"]
                frame_cache = {fid: _decode_rgb_bytes(rgb_ds[fid]) for fid in unique_ids}
                frames_by_cam[cam] = [frame_cache[int(fid)] for fid in frame_ids]
                w2c_unique = np.stack([_to_4x4_w2c(f[f"observation/{cam}/extrinsic_cv"][fid]) for fid in unique_ids], axis=0)
                w2c_by_cam[cam] = np.stack([w2c_unique[unique_to_pos[int(fid)]] for fid in frame_ids], axis=0)
                K_unique = np.stack([np.asarray(f[f"observation/{cam}/intrinsic_cv"][fid], dtype=np.float32) for fid in unique_ids], axis=0)
                K_by_cam[cam] = np.stack([K_unique[unique_to_pos[int(fid)]] for fid in frame_ids], axis=0)
        return abs_actions, frames_by_cam, w2c_by_cam, K_by_cam

    def __getitem__(self, idx: int) -> dict:
        if self.total_samples == 0:
            raise IndexError("Dataset has no valid samples")
        sample_idx = idx % self.total_samples
        ep_idx, start_idx = self.sample_indices[sample_idx]
        info = self.episode_info[ep_idx]
        frame_ids = self._make_frame_ids(start_idx)
        abs_actions, frames_by_cam, w2c_by_cam, K_by_cam = self._read_episode_arrays(info["path"], frame_ids)

        cam_frames: Dict[str, List[Image.Image]] = {cam: [] for cam in self.camera_names}
        cam_traj_maps: Dict[str, List[Image.Image]] = {cam: [] for cam in self.camera_names}
        cam_ray_o: Dict[str, List[Image.Image]] = {cam: [] for cam in self.camera_names}
        cam_ray_d: Dict[str, List[Image.Image]] = {cam: [] for cam in self.camera_names}

        for cam in self.camera_names:
            first_frame = frames_by_cam[cam][0]
            ori_h, ori_w = first_frame.shape[:2]
            out_h, out_w = self.resize_to if self.resize_to is not None else (ori_h, ori_w)
            traj_maps = []
            for i in range(len(frame_ids)):
                K = _scale_intrinsic(K_by_cam[cam][i], ori_h, ori_w, out_h, out_w)
                traj_maps.append(
                    generate_traj_map(
                        abs_actions[i : i + 1],
                        w2c_by_cam[cam][i : i + 1],
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
                    )[0]
                )
            for i, fid in enumerate(frame_ids):
                img = Image.fromarray(frames_by_cam[cam][i])
                if self.resize_to is not None:
                    img = img.resize((out_w, out_h), Image.BILINEAR)
                cam_frames[cam].append(img)
                cam_traj_maps[cam].append(Image.fromarray(traj_maps[i]))
                if self.output_raymap:
                    K = _scale_intrinsic(K_by_cam[cam][i], ori_h, ori_w, out_h, out_w)
                    c2w = np.linalg.inv(w2c_by_cam[cam][i]).astype(np.float32)
                    ro, rd = generate_raymap(K, c2w, out_h, out_w)
                    cam_ray_o[cam].append(Image.fromarray(to_uint8_img(ro, self.ray_o_vmin, self.ray_o_vmax)))
                    cam_ray_d[cam].append(Image.fromarray(to_uint8_img(rd, self.ray_d_vmin, self.ray_d_vmax)))

        video_list, control_list, ray_o_list, ray_d_list = [], [], [], []
        for i in range(len(frame_ids)):
            video_list.append(Image.fromarray(np.concatenate([np.asarray(cam_frames[cam][i]) for cam in self.camera_names], axis=0)))
            control_list.append(Image.fromarray(np.concatenate([np.asarray(cam_traj_maps[cam][i]) for cam in self.camera_names], axis=0)))
            if self.output_raymap:
                ray_o_list.append(Image.fromarray(np.concatenate([np.asarray(cam_ray_o[cam][i]) for cam in self.camera_names], axis=0)))
                ray_d_list.append(Image.fromarray(np.concatenate([np.asarray(cam_ray_d[cam][i]) for cam in self.camera_names], axis=0)))

        if self.type == "vace":
            sample = {"video": video_list, "vace_reference_image": [video_list[0]], "vace_video": control_list, "prompt": self.prompt}
        else:
            sample = {"video": video_list, "reference_image": [video_list[0]], "control_video": control_list, "prompt": self.prompt}
        if self.output_raymap:
            sample["ray_map_o"] = ray_o_list
            sample["ray_map_d"] = ray_d_list
        sample["meta"] = {
            "task": info["task"],
            "run": info["run"],
            "episode_index": info["episode_index"],
            "path": str(info["path"]),
            "frame_ids": frame_ids.tolist(),
            "camera_names": list(self.camera_names),
            "view_mode": self.view_mode,
            "traj_source": "robotwin_joint_action_fk",
            "fk_config_path": str(self.fk_config_path),
            "raymap_frame": "robotwin_world_cv_extrinsic",
        }
        return sample


def _as_rgb_array(image_like) -> np.ndarray:
    return np.asarray(image_like.convert("RGB") if isinstance(image_like, Image.Image) else image_like, dtype=np.uint8)


def make_debug_grid_video(sample: dict, output_path: Path, fps: float = 30.0) -> None:
    frames = []
    for i in range(len(sample["video"])):
        cols = [
            _as_rgb_array(sample["video"][i]),
            _as_rgb_array(sample["vace_video"][i]),
            _as_rgb_array(sample["ray_map_o"][i]),
            _as_rgb_array(sample["ray_map_d"][i]),
        ]
        frames.append(np.concatenate(cols, axis=1))
    write_video_rgb(frames, Path(output_path), fps=fps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="/mnt/data/zsq/RoboTwin/dataset")
    parser.add_argument("--split", type=str, default="train", choices=("train", "val", "all"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=str, default="/tmp/robotwin_dataset_item_debug.mp4")
    parser.add_argument("--view_mode", type=str, default="three", choices=("three", "single"))
    parser.add_argument("--single_camera", type=str, default="head_camera", choices=ROBOTWIN_ALL_CAMERAS)
    parser.add_argument("--num_frames", type=int, default=13)
    parser.add_argument("--context_length", type=int, default=5)
    parser.add_argument("--first_round_prob", type=float, default=0.05)
    parser.add_argument("--endpose_quat_order", type=str, default="wxyz", choices=("wxyz", "xyzw"))
    parser.add_argument("--task_limit", type=int, default=None)
    parser.add_argument("--fk_config_path", type=str, default=DEFAULT_FK_CONFIG_PATH)
    args = parser.parse_args()

    np.random.seed(0)
    dataset = RoboTwinActionFKDatasetWancontrolmultiview(
        base_path=args.base_path,
        split=args.split,
        num_frames=args.num_frames,
        context_length=args.context_length,
        first_round_prob=args.first_round_prob,
        output_raymap=True,
        view_mode=args.view_mode,
        single_camera=args.single_camera,
        endpose_quat_order=args.endpose_quat_order,
        task_limit=args.task_limit,
        fk_config_path=args.fk_config_path,
    )
    sample = dataset[int(args.index)]
    make_debug_grid_video(sample, Path(args.output), fps=dataset.output_fps)
    print(f"saved={args.output}")
    print(f"video_size={sample['video'][0].size} frames={len(sample['video'])}")
    print(f"meta={sample['meta']}")


if __name__ == "__main__":
    main()
