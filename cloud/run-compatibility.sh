#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/aerorozum-training.log) 2>&1
until test -f /opt/aerorozum/ready.json; do sleep 5; done
cd /opt/aerorozum
gcloud storage cp gs://aerorozum-thermal-rl-avid-relic-509408-e4/bootstrap/worker-source.tar.gz gs://aerorozum-thermal-rl-avid-relic-509408-e4/bootstrap/worker-real-training.tar.gz . --quiet
tar -xzf worker-source.tar.gz
tar -xzf worker-real-training.tar.gz
/opt/aerorozum/venv/bin/python -m pip install -e /opt/aerorozum/source
export AEROROZUM_DATA_ROOT=/opt/aerorozum/artifacts
export AEROROZUM_LOCAL_MODEL_PATH=/opt/aerorozum/models/gemma-4-12B-it
export AEROROZUM_DEVICE_MAP=balanced
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME=/opt/aerorozum/hf
set +e
/opt/aerorozum/venv/bin/python -u -m aerorozum.training --manifest /opt/aerorozum/artifacts/datasets/real/manifest.json --stage compatibility --seed 17 --steps 10
code=$?
set -e
printf '%s\n' "$code" > /opt/aerorozum/compatibility-exit-code.txt
gcloud storage rsync -r /opt/aerorozum/artifacts/training gs://aerorozum-thermal-rl-avid-relic-509408-e4/runs/compatibility/training --quiet || true
gcloud storage rsync -r /opt/aerorozum/artifacts/jobs gs://aerorozum-thermal-rl-avid-relic-509408-e4/runs/compatibility/jobs --quiet || true
gcloud storage cp /var/log/aerorozum-training.log /opt/aerorozum/compatibility-exit-code.txt /opt/aerorozum/bootstrap-packages.txt gs://aerorozum-thermal-rl-avid-relic-509408-e4/runs/compatibility/ --quiet || true
exit "$code"
