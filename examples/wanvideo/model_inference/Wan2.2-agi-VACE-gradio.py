# gradio_agibot_demo.py
# ------------------------------------------------------------
# AgiBotWC Interactive Demo (Scheme C)
# - sliders initialized from first frame state
# - every click predicts next 8 frames
# - input first-frame action, repeated 8 times
# - realtime show generated video + control video + last frame
# ------------------------------------------------------------

import os
import json
import math
import tempfile
import numpy as np
import torch
import gradio as gr
import h5py

from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
from decord import VideoReader, cpu
import matplotlib.cm as cm

from diffsynth import save_video, load_state_dict

# ============================================================
# CONFIG
# ============================================================

VAL_PATH = "/mnt/workspace/zsq/Agibotsubset/val"

BASE_MODEL_PATHS = [
    [
        "/mnt/workspace/zsq/Wan_model/Wan2.1-VACE-14B/diffusion_pytorch_model-00001-of-00007.safetensors",
        "/mnt/workspace/zsq/Wan_model/Wan2.1-VACE-14B/diffusion_pytorch_model-00002-of-00007.safetensors",
        "/mnt/workspace/zsq/Wan_model/Wan2.1-VACE-14B/diffusion_pytorch_model-00003-of-00007.safetensors",
        "/mnt/workspace/zsq/Wan_model/Wan2.1-VACE-14B/diffusion_pytorch_model-00004-of-00007.safetensors",
        "/mnt/workspace/zsq/Wan_model/Wan2.1-VACE-14B/diffusion_pytorch_model-00005-of-00007.safetensors",
        "/mnt/workspace/zsq/Wan_model/Wan2.1-VACE-14B/diffusion_pytorch_model-00006-of-00007.safetensors",
        "/mnt/workspace/zsq/Wan_model/Wan2.1-VACE-14B/diffusion_pytorch_model-00007-of-00007.safetensors",
    ],
    "/mnt/workspace/zsq/Wan_model/Wan2.2-Fun-A14B-Control/models_t5_umt5-xxl-enc-bf16.pth",
    "/mnt/workspace/zsq/Wan_model/Wan2.2-Fun-A14B-Control/Wan2.1_VAE.pth",
]

VACE_PATH = "/mnt/workspace/zsq/DiffSynth-Studio/outputs/Wan2.1-VACE-14B/epoch-49.safetensors"

DEVICE = "cuda:0"
H = 320
W = 512
PREDICT_FRAMES = 8
NUM_STEPS = 5
CFG = 1.0
SEED = 42

# ============================================================
# DRAW CONFIG
# ============================================================

ColorMapLeft = cm.Greens
ColorMapRight = cm.Reds

EndPts = np.array([
    [0,0,0,1],
    [0.1,0,0,1],
    [0,0.1,0,1],
    [0,0,0.1,1]
], dtype=np.float32).T

Grip2EEF = np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0.23],
    [0,0,0,1]
], dtype=np.float32)

# ============================================================
# LOAD PIPELINE
# ============================================================

print("Loading pipeline...")

from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

model_configs = [ModelConfig(path=p) for p in BASE_MODEL_PATHS]

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device=DEVICE,
    model_configs=model_configs,
)

state_dict = load_state_dict(VACE_PATH)
pipe.vace.load_state_dict(state_dict)

print("Pipeline loaded.")

# ============================================================
# DATASET
# ============================================================

def first_episode():
    subs = sorted(os.listdir(VAL_PATH))
    groups = {}
    for s in subs:
        key = "-".join(s.split("-")[:2])
        groups.setdefault(key, []).append(os.path.join(VAL_PATH, s))
    k = sorted(groups.keys())[0]
    return sorted(groups[k])

EP_PATHS = first_episode()

# ============================================================
# LOAD INIT DATA
# ============================================================

def load_init():

    h5 = os.path.join(EP_PATHS[0], "proprio_stats.h5")
    with h5py.File(h5, "r") as f:
        pos = f["state/end/position"][0]
        quat = f["state/end/orientation"][0]
        grip = f["state/effector/position"][0]

    intr = json.load(open(
        os.path.join(EP_PATHS[0], "head_intrinsic_params.json")
    ))["intrinsic"]

    K = np.eye(3, dtype=np.float32)
    K[0,0]=intr["fx"]
    K[1,1]=intr["fy"]
    K[0,2]=intr["ppx"]
    K[1,2]=intr["ppy"]

    ext = json.load(open(
        os.path.join(EP_PATHS[0], "head_extrinsic_params_aligned.json")
    ))[0]

    c2w = np.eye(4,dtype=np.float32)
    c2w[:3,:3] = np.array(ext["extrinsic"]["rotation_matrix"])
    c2w[:3,3] = np.array(ext["extrinsic"]["translation_vector"])
    w2c = np.linalg.inv(c2w)

    vr = VideoReader(
        os.path.join(EP_PATHS[0], "head_color.mp4"),
        ctx=cpu(0)
    )

    raw = vr[0].asnumpy()

    ori_h, ori_w = raw.shape[:2]

    img = Image.fromarray(raw).resize((W,H))

    # -------- resize intrinsic --------
    sx = W / ori_w
    sy = H / ori_h

    K2 = K.copy()
    K2[0,0] *= sx
    K2[0,2] *= sx
    K2[1,1] *= sy
    K2[1,2] *= sy

    return img, pos, quat, grip, K2, w2c

# ============================================================
# INIT GLOBAL STATE
# ============================================================

init_img, init_pos, init_quat, init_grip, K, W2C = load_init()
current_context = init_img

# ============================================================
# HELPERS
# ============================================================

def pose_to_euler(q):
    return Rotation.from_quat(q).as_euler("xyz", degrees=True)

def euler_to_quat(r,p,y):
    return Rotation.from_euler(
        "xyz",[r,p,y],degrees=True
    ).as_quat()

def build_action(vals):

    lx,ly,lz,lr,lp,lyaw,lg,\
    rx,ry,rz,rr,rp,ryaw,rg = vals

    lq = euler_to_quat(lr,lp,lyaw)
    rq = euler_to_quat(rr,rp,ryaw)

    a = np.zeros(16,dtype=np.float32)

    a[0:3]=[lx,ly,lz]
    a[3:7]=lq
    a[7]=lg

    a[8:11]=[rx,ry,rz]
    a[11:15]=rq
    a[15]=rg

    return a

def transform(xyz, quat):

    T = np.eye(4,dtype=np.float32)
    T[:3,:3]=Rotation.from_quat(quat).as_matrix()
    T[:3,3]=xyz
    return T

def draw_map(action):

    img = Image.new("RGB",(W,H),(40,40,40))
    draw = ImageDraw.Draw(img)

    for side in [0,1]:

        if side==0:
            xyz = action[0:3]
            quat = action[3:7]
            grip = action[7]
            cmap = ColorMapLeft
        else:
            xyz = action[8:11]
            quat = action[11:15]
            grip = action[15]
            cmap = ColorMapRight

        T = W2C @ transform(xyz,quat) @ Grip2EEF
        pts = (T @ EndPts)[:3]

        uv = K @ pts
        uv = (uv[:2]/uv[2:3]).T.astype(int)

        c = tuple(int(v*255) for v in cmap(grip/120)[:3])

        x,y = uv[0]
        draw.ellipse([x-40,y-40,x+40,y+40],fill=c)

        for k in range(1,4):
            draw.line(
                [(x,y),(uv[k][0],uv[k][1])],
                fill=(255,255,255),
                width=5
            )

    return img

def save_temp_video(frames,name):

    path = os.path.join(tempfile.gettempdir(),name)
    save_video(frames,path,fps=8,quality=5)
    return path

# ============================================================
# MAIN STEP
# ============================================================

def step(*vals):

    global current_context

    action = build_action(vals)

    controls = [draw_map(action) for _ in range(9)]

    generated = pipe(
        prompt="机械臂按照要求移动夹爪执行任务",
        vace_video=controls,
        vace_reference_image=current_context,
        height=H,
        width=W,
        num_frames=9,
        num_inference_steps=NUM_STEPS,
        cfg_scale=CFG,
        seed=SEED,
        tiled=True,
    )

    new_frames = generated[1:]

    current_context = new_frames[-1]

    gen_path = save_temp_video(new_frames,"gen.mp4")
    ctrl_path = save_temp_video(controls[1:],"ctrl.mp4")

    return gen_path, ctrl_path, current_context

# ============================================================
# UI INIT VALUES
# ============================================================

lxyz = init_pos[0]
rxyz = init_pos[1]

le = pose_to_euler(init_quat[0])
re = pose_to_euler(init_quat[1])

lg = float(init_grip[0])
rg = float(init_grip[1])

def slider(v):
    return gr.Slider(v-0.3,v+0.3,value=v,step=0.01)

# ============================================================
# UI
# ============================================================

with gr.Blocks() as demo:

    gr.Markdown("# AgiBot Interactive Wan Demo")

    with gr.Row():

        with gr.Column():

            gr.Markdown("## Left Arm")

            lx=slider(lxyz[0]); ly=slider(lxyz[1]); lz=slider(lxyz[2])
            lr=gr.Slider(-180,180,value=le[0])
            lp=gr.Slider(-180,180,value=le[1])
            lyaw=gr.Slider(-180,180,value=le[2])
            lgr=gr.Slider(0,120,value=lg)

            gr.Markdown("## Right Arm")

            rx=slider(rxyz[0]); ry=slider(rxyz[1]); rz=slider(rxyz[2])
            rr=gr.Slider(-180,180,value=re[0])
            rp=gr.Slider(-180,180,value=re[1])
            ryaw=gr.Slider(-180,180,value=re[2])
            rgr=gr.Slider(0,120,value=rg)

            btn = gr.Button("Generate Next 8 Frames")

        with gr.Column():

            out_video = gr.Video(label="Generated Video")
            ctrl_video = gr.Video(label="Control Video")
            last_img = gr.Image(value=init_img,label="Current Last Frame")

    btn.click(
        fn=step,
        inputs=[
            lx,ly,lz,lr,lp,lyaw,lgr,
            rx,ry,rz,rr,rp,ryaw,rgr
        ],
        outputs=[out_video,ctrl_video,last_img]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)