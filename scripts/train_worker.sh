#!/usr/bin/env bash
set -euo pipefail
: "${AEROROZUM_DATA_ROOT:?Set the artifact directory containing the dataset and checkpoints}"
: "${HF_HOME:=${AEROROZUM_DATA_ROOT}/model-cache}"
export HF_HOME
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
exec "${AEROROZUM_PYTHON:-python3}" -u -m aerorozum.training "$@"
