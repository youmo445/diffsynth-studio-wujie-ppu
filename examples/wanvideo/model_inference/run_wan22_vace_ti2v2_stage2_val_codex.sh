#!/usr/bin/env bash
set -euo pipefail

WORKDIR=/mnt/data/zsq/DiffSynth-Studio
OUT_BASE="${WORKDIR}/outputs/val_wan22_vace_ti2v2_stage2_epoch29_codex"
OUT_DIR="${OUT_BASE}_perspective_raymap"
PID_FILE=/tmp/wan22_vace_ti2v2_stage2_val.pid

mkdir -p "${OUT_DIR}"
cd "${WORKDIR}"

PYTHONUNBUFFERED=1 PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}" \
python examples/wanvideo/model_inference/Wan2.2-TI2V-5B_VACE_TI2V2_val_three_episodes_stage2_codex.py \
  --parallel_episodes \
  --episode_limit 3 \
  --gpus 0,1,2 \
  --num_inference_steps 5 \
  > "${OUT_DIR}/launcher.log" 2>&1 &

echo "$!" > "${PID_FILE}"
echo "PID=$(cat "${PID_FILE}")"
echo "LOG=${OUT_DIR}/launcher.log"
