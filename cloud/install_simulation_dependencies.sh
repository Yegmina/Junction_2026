#!/bin/bash
# Run on the Ubuntu 24.04 VM; do not restart unrelated long-running research jobs.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=l
curl -fsSL https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
printf '%s\n' 'deb [arch=amd64 signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable noble main' > /etc/apt/sources.list.d/gazebo-stable.list
apt-get update
apt-get install -y --no-install-recommends gz-harmonic python3-gz-transport13 python3-gz-msgs10 \
  python3-pil python3-numpy python3-venv libegl-mesa0 libgl1-mesa-dri mesa-utils \
  xvfb xauth nginx
