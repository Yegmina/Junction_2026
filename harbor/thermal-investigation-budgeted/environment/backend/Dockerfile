FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates gnupg && curl -fsSL https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable noble main' > /etc/apt/sources.list.d/gazebo-stable.list && apt-get update && apt-get install -y --no-install-recommends gz-harmonic python3-gz-transport13 python3-gz-msgs10 python3-pil python3-numpy python3-scipy xvfb xauth libegl-mesa0 libgl1-mesa-dri && rm -rf /var/lib/apt/lists/*
COPY assets/ /scene/
COPY runtime/ /runtime/
COPY server.py /runtime/server.py
COPY start.sh /start.sh
RUN chmod 755 /start.sh
ENV AEROROZUM_SIM_ROOT=/scene GZ_PARTITION=aerorozum-mission-local GZ_IP=127.0.0.1 GZ_DISCOVERY_MSG_PORT=18935 GZ_DISCOVERY_SRV_PORT=18936 GZ_SIM_RESOURCE_PATH=/scene/models LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe LP_NUM_THREADS=4 DISPLAY=:99
EXPOSE 8080
ENTRYPOINT ["/start.sh"]
