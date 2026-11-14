FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-numpy python3-scipy python3-pil curl ca-certificates git bash jq && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 10001 agent && mkdir -p /workspace/evidence && chown -R agent:agent /workspace
COPY drone.py /usr/local/bin/drone
COPY camera_geometry.py /opt/drone-sdk/camera_geometry.py
ENV PYTHONPATH=/opt/drone-sdk
RUN chmod 755 /usr/local/bin/drone
WORKDIR /workspace
