import subprocess

def trim_mp4(input_path, output_path, n_frames):
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-vf', f'select=lte(n\\,{n_frames-1})',
        '-vsync', 'vfr', '-an',
        '-c:v', 'libx264', output_path
    ]
    subprocess.run(cmd, check=True)

# 使用
trim_mp4('/mnt/workspace/zsq/DiffSynth-Studio/examples/wanvideo/model_inference/data/examples/agi/action_map.mp4', '/mnt/workspace/zsq/DiffSynth-Studio/examples/wanvideo/model_inference/data/examples/agi/action_map_trim.mp4', 9)