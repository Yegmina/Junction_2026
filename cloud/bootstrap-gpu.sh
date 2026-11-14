#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/aerorozum-bootstrap.log) 2>&1
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3-venv python3-pip git
install -d /opt/aerorozum
nvidia-smi
python3 --version
python3 -m venv --system-site-packages /opt/aerorozum/venv
/opt/aerorozum/venv/bin/python -m pip install --upgrade pip
/opt/aerorozum/venv/bin/python -m pip install 'git+https://github.com/huggingface/transformers.git' trl peft accelerate datasets pillow scipy huggingface_hub torch "jinja2>=3.1.6" librosa
/opt/aerorozum/venv/bin/python -m pip install --force-reinstall --no-deps torchaudio==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu129
/opt/aerorozum/venv/bin/python -m pip freeze > /opt/aerorozum/bootstrap-packages.txt
/opt/aerorozum/venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
import json,torch
path=snapshot_download('google/gemma-4-12B-it',revision='43c9d014b231474fa7212dbb48a762566c2b255d',local_dir='/opt/aerorozum/models/gemma-4-12B-it',allow_patterns=['*.json','*.safetensors','*.model','*.jinja','tokenizer*'])
Path('/opt/aerorozum/ready.json').write_text(json.dumps({'weights':path,'torch':torch.__version__,'cuda':torch.cuda.is_available(),'gpus':torch.cuda.device_count()}))
PY
