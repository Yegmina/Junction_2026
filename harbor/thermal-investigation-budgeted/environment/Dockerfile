FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-numpy python3-scipy python3-pil curl ca-certificates git bash jq nodejs npm && rm -rf /var/lib/apt/lists/*
RUN npm install --global opencode-ai@1.19.5 && opencode --version
RUN useradd -m -u 10001 agent && mkdir -p /workspace/evidence && chown -R agent:agent /workspace
COPY drone.py /usr/local/bin/drone
COPY camera_geometry.py /opt/drone-sdk/camera_geometry.py
COPY flight_plan.py /opt/drone-sdk/flight_plan.py
ENV PYTHONPATH=/opt/drone-sdk
RUN chmod 755 /usr/local/bin/drone /opt/drone-sdk && chmod 644 /opt/drone-sdk/*.py
WORKDIR /workspace
# Harbor enables Git-based OpenCode snapshots. A real public workspace avoids
# snapshot subprocesses trying to read ignore rules from a missing repository.
USER agent
RUN git init . && git -c user.name='AeroRozum Benchmark' -c user.email=benchmark@localhost commit --allow-empty -m 'Initialize public agent workspace'
USER root
