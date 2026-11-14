#!/bin/bash
set -euo pipefail
mkdir -p /scene/media/captures /scene/logs /scene/cache /scene/config
Xvfb :99 -screen 0 1280x960x24 -nolisten tcp > /scene/logs/xvfb.log 2>&1 &
sleep 1
gz sim -s -r -v 2 /scene/worlds/aerorozum.sdf > /scene/logs/gazebo.log 2>&1 &
python3 /runtime/runtime.py > /scene/logs/adapter.log 2>&1 &
exec python3 /runtime/server.py
