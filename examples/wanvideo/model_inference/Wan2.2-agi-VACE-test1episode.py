# compare_gt_pred_proj.py
# ------------------------------------------------------------
# 三层拼接视频：
# Top    : GT真实视频
# Middle : Wan 自回归生成视频
# Bottom : Projection Control Video
#
# 使用原设置:
# predict_frames = 8
# num_inference_steps = 5
# ------------------------------------------------------------

import os
import json
import math
import h5py
import numpy as np
import torch

from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
from decord import VideoReader, cpu
import matplotlib.cm as cm

from diffsynth import save_video, load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig


# ============================================================
# CONFIG
# ============================================================

VAL_PATH = "/mnt/workspace/zsq/Agibotsubset/val"
OUT_PATH = "./compare_gt_pred_proj.mp4"
EPISODE_ID = 5

H = 320
W = 512

ORIGINAL_HZ = 30
TARGET_HZ = 5
DOWNSAMPLE = ORIGINAL_HZ // TARGET_HZ

FPS = 15

PREDICT_FRAMES = 8
NUM_STEPS = 5
CFG = 1.0
SEED = 42
DEVICE = "cuda:0"

BASE_MODEL_PATHS = [
    "/mnt/workspace/zsq/Wan_model/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors",
    "/mnt/workspace/zsq/Wan_model/Wan2.2-Fun-A14B-Control/models_t5_umt5-xxl-enc-bf16.pth",
    "/mnt/workspace/zsq/Wan_model/Wan2.2-Fun-A14B-Control/Wan2.1_VAE.pth",
]

VACE_PATH = "/mnt/workspace/zsq/DiffSynth-Studio/outputs/Wan2.1-VACE-1.3B/epoch-49.safetensors"


# ============================================================
# DRAW CONFIG
# ============================================================

ColorMapLeft = cm.Greens
ColorMapRight = cm.Reds

ColorListLeft = [(0,0,255),(255,255,0),(0,255,255)]
ColorListRight = [(255,0,255),(255,0,0),(0,255,0)]

EndPts = np.array([
    [0,0,0,1],
    [0.1,0,0,1],
    [0,0.1,0,1],
    [0,0,0.1,1],
],dtype=np.float32).T

Grip2EEF = np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0.23],
    [0,0,0,1],
],dtype=np.float32)


# ============================================================
# FIND FIRST EPISODE
# ============================================================

def first_episode(ep_id=0):

    names = sorted(os.listdir(VAL_PATH))
    groups = {}

    for n in names:
        p = os.path.join(VAL_PATH,n)
        if not os.path.isdir(p):
            continue

        key = "-".join(n.split("-")[:2])
        groups.setdefault(key, []).append(p)

    keys = sorted(groups.keys())
    key = keys[ep_id]
    return sorted(groups[key])


EP_PATHS = first_episode(EPISODE_ID)


# ============================================================
# LOAD DATA
# ============================================================

def load_episode():

    pos_list = []
    quat_list = []
    grip_list = []
    rgb_frames = []

    for sub in EP_PATHS:

        with h5py.File(os.path.join(sub,"proprio_stats.h5"),"r") as f:
            pos = f["state/end/position"][:]
            quat = f["state/end/orientation"][:]
            grip = f["state/effector/position"][:]

        if grip.ndim == 3:
            grip = grip[...,0]

        pos_list.append(pos)
        quat_list.append(quat)
        grip_list.append(grip)

        vr = VideoReader(
            os.path.join(sub,"head_color.mp4"),
            ctx=cpu(0)
        )

        for i in range(len(vr)):
            rgb_frames.append(vr[i].asnumpy())

    pos = np.concatenate(pos_list,axis=0)
    quat = np.concatenate(quat_list,axis=0)
    grip = np.concatenate(grip_list,axis=0)

    return pos, quat, grip, rgb_frames


# ============================================================
# CAMERA
# ============================================================

def load_camera():

    intr = json.load(open(
        os.path.join(EP_PATHS[0],"head_intrinsic_params.json")
    ))["intrinsic"]

    K = np.eye(3,dtype=np.float32)
    K[0,0]=intr["fx"]
    K[1,1]=intr["fy"]
    K[0,2]=intr["ppx"]
    K[1,2]=intr["ppy"]

    ext = json.load(open(
        os.path.join(EP_PATHS[0],"head_extrinsic_params_aligned.json")
    ))[0]

    c2w = np.eye(4,dtype=np.float32)
    c2w[:3,:3] = np.array(ext["extrinsic"]["rotation_matrix"])
    c2w[:3,3] = np.array(ext["extrinsic"]["translation_vector"])

    w2c = np.linalg.inv(c2w)

    vr = VideoReader(
        os.path.join(EP_PATHS[0],"head_color.mp4"),
        ctx=cpu(0)
    )

    raw = vr[0].asnumpy()
    oh, ow = raw.shape[:2]

    sx = W / ow
    sy = H / oh

    K2 = K.copy()
    K2[0,0] *= sx
    K2[0,2] *= sx
    K2[1,1] *= sy
    K2[1,2] *= sy

    return K2, w2c


# ============================================================
# HELPERS
# ============================================================

def transform(xyz, quat):

    T = np.eye(4,dtype=np.float32)
    T[:3,:3] = Rotation.from_quat(quat).as_matrix()
    T[:3,3] = xyz
    return T


def draw_projection(pos, quat, grip, K, W2C):

    img = Image.new("RGB",(W,H),(40,40,40))
    draw = ImageDraw.Draw(img)

    for side in [0,1]:

        xyz = pos[side]
        q = quat[side]
        g = grip[side]

        T = W2C @ transform(xyz,q) @ Grip2EEF
        pts = (T @ EndPts)[:3]

        uv = K @ pts
        uv = (uv[:2]/uv[2:3]).T.astype(int)

        cmap = ColorMapLeft if side==0 else ColorMapRight
        clst = ColorListLeft if side==0 else ColorListRight

        color = tuple(
            int(v*255) for v in cmap(np.clip(g/120,0,1))[:3]
        )

        x,y = uv[0]
        draw.ellipse([x-40,y-40,x+40,y+40],fill=color)

        for k in range(1,4):
            draw.line(
                [(x,y),(uv[k][0],uv[k][1])],
                fill=clst[k-1],
                width=6
            )

    return img


def build_control_chunk(all_maps, start):

    padded = [all_maps[0]] + all_maps
    chunk = padded[start:start+9]

    if len(chunk) < 9:
        chunk += [padded[-1]]*(9-len(chunk))

    return chunk


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model_configs = [ModelConfig(path=p) for p in BASE_MODEL_PATHS]

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device=DEVICE,
    model_configs=model_configs,
)

pipe.vace.load_state_dict(load_state_dict(VACE_PATH))

print("Model loaded.")


# ============================================================
# MAIN
# ============================================================

def main():

    pos, quat, grip, rgb_raw = load_episode()
    K, W2C = load_camera()

    total_raw = len(rgb_raw)
    ds_idx = np.arange(0,total_raw,DOWNSAMPLE)

    gt_frames = []
    maps = []

    ##############
    freeze_start = 10
    freeze_end   = 100

    freeze_pos = pos[ds_idx[freeze_start], 1].copy()
    for t, idx in enumerate(ds_idx):

        gt = Image.fromarray(rgb_raw[idx]).resize((W,H))
        gt_frames.append(gt)

        cur_pos = pos[idx].copy()
        cur_quat = quat[idx].copy()
        cur_grip = grip[idx].copy()

        # 冻结右臂位置
        if freeze_start <= t <= freeze_end:
            cur_pos[0,0] -= 0.05
            cur_pos[0,1] += 0.05
            cur_pos[0,2] -= 0.05
            # cur_pos = pos[3].copy()
            cur_grip[0] = 30
        proj = draw_projection(
            cur_pos,
            cur_quat,
            cur_grip,
            K,
            W2C
        )

        maps.append(proj)
    ###############
    T = len(gt_frames)

    # autoregressive inference
    pred = [gt_frames[0]]
    context = gt_frames[0]

    total_pred = T - 1
    num_chunks = math.ceil(total_pred / PREDICT_FRAMES)

    for chunk in range(num_chunks):

        start = chunk * PREDICT_FRAMES
        valid = min(PREDICT_FRAMES, total_pred - start)

        control = build_control_chunk(maps,start)

        gen = pipe(
            prompt="机械臂按照要求移动夹爪执行任务",
            vace_video=control,
            vace_reference_image=context,
            height=H,
            width=W,
            num_frames=9,
            num_inference_steps=NUM_STEPS,
            cfg_scale=CFG,
            seed=SEED,
            tiled=True,
        )

        new_frames = gen[1:1+valid]

        for f in new_frames:
            pred.append(f)

        context = pred[-1]

        print(f"chunk {chunk+1}/{num_chunks}")

    # concat video
    out_frames = []

    for t in range(T):

        canvas = Image.new("RGB",(W,H*3))
        canvas.paste(gt_frames[t],(0,0))
        canvas.paste(pred[t],(0,H))
        canvas.paste(maps[t],(0,H*2))

        out_frames.append(canvas)

    save_video(out_frames, OUT_PATH, fps=FPS, quality=5)

    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()