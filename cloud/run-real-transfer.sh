#!/bin/bash
set -euo pipefail
cd /opt/aerorozum/source
export AEROROZUM_DATA_ROOT=/opt/aerorozum/artifacts
export AEROROZUM_LOCAL_MODEL_PATH=/opt/aerorozum/models/gemma-4-12B-it
export AEROROZUM_DEVICE_MAP=balanced
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME=/opt/aerorozum/hf
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
manifest=/opt/aerorozum/artifacts/datasets/real-transfer-machine-labels-v1/frozen-seed17-v1.json
destination=gs://aerorozum-thermal-rl-avid-relic-509408-e4/runs/real-transfer-seed17-v1
sync_results() {
    gcloud storage rsync -r /opt/aerorozum/artifacts/training/real-transfer-machine-labels-v1 "$destination/training" --quiet || true
    gcloud storage rsync -r /opt/aerorozum/artifacts/jobs "$destination/jobs" --quiet || true
}
(
    while true; do
        sleep 60
        sync_results
    done
) &
uploader_pid=$!
finish() {
    code=$?
    trap - EXIT
    kill "$uploader_pid" 2>/dev/null || true
    sync_results
    exit "$code"
}
trap finish EXIT
# Approximately one pass through 270 frozen frames at accumulation four.
/opt/aerorozum/venv/bin/python -u -m aerorozum.training --manifest "$manifest" --stage real_sft --seed 17 --steps 68
# This loads the matching SFT adapter and verifies the frozen dataset identity.
/opt/aerorozum/venv/bin/python -u -m aerorozum.training --manifest "$manifest" --stage real_grpo --seed 17 --steps 32
