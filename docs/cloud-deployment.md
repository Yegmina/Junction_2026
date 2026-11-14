# Cloud deployment and the 48-hour change

Active deployment: `aerorozum-mission-48h-20261114`, `us-central1-b`, four
A100-SXM4-40GB GPUs, 48 vCPUs. Standard provisioning and the fixed stop deadline
are set to **2026-11-16 16:46:54 UTC** (18:46 Helsinki time). The original
Netherlands VM is stopped and retained for rollback. A full disk snapshot is
also retained. Capacity for the replacement VM is reserved through its runtime
window; the reservation automatically expires five seconds after that deadline.

The deployment runs the dashboard, mission API, native Gazebo dynamics, paired
RGB/thermal adapter, telemetry mirror and a private Gemma inference endpoint on
the GPU VM. Gemini remains a Google-hosted API; only its caller moves to the VM.
The default live mission controller remains Gemini. Gemma is available as an
explicit alternative, without changing the existing model precision or weights.

## Services and storage

`cloud/install_mission_stack.py` installs enabled system services with automatic
restart. The deployment is isolated under `/opt/aerorozum/deployment`; existing
research code, training adapters, source recordings and physics outputs stay in
their original directories. The service account has its own writable runtime
home. Application processes do not need a laptop session to remain running.

| Service | Purpose | Loopback port |
| --- | --- | --- |
| `aerorozum-cloud-gazebo` | Native x500 dynamics and RGB rendering | Gazebo transport |
| `aerorozum-cloud-adapter` | Pose control and paired physics thermal/RGB capture | 18930 |
| `aerorozum-cloud-api` | Mission management, saved reports and built dashboard | 18901 |
| `aerorozum-cloud-telemetry` | Copy measured worker state and previews locally | observer on 18920 |
| `aerorozum-cloud-gemma` | Gemma 4-12B real-SFT adapter, bf16, GPU 0 | 18940 |
| nginx | Same-origin dashboard proxy | 18900 |

A100 does not have a graphics engine. Gazebo uses Mesa llvmpipe in a private
Xvfb display, while Gemma uses CUDA. Native camera images and integrated motion
are retained. The thermal image is the existing experimental DuMux-derived
composed field; moving it to the cloud does not qualify it as field validated.

The simulator accepts `AEROROZUM_SIM_ROOT`; the API accepts
`AEROROZUM_DATA_ROOT`. The deployment rewrites copied scene manifests/SDF paths
and verifies the immutable thermal field checksums during camera initialization.
No laptop filesystem paths or mounted-SSD symlinks are required on the VM.

## Installation

Use the existing Ubuntu 24.04 GPU environment for model inference. Install native
simulation packages using `sudo bash cloud/install_simulation_dependencies.sh`.
This explicitly sets `NEEDRESTART_MODE=l`: dependency installation must not
silently restart unrelated long-running solver jobs.

Build `frontend/dist` with `npm run build` in `frontend`, transfer the source
release and prepared `models`, `worlds`, `sensor-scene` assets. Place a
`simulation/deployment-source.json` containing the original `simulation_root`
alongside the copied assets. Transfer only the isolated hackathon key into
`/etc/aerorozum/hackathon-key`, root-owned mode 600. The API receives it through
systemd `LoadCredential`; the Gemma process receives no API key.

Run:

```sh
sudo python3 cloud/install_mission_stack.py --release /opt/aerorozum/deployment/releases/48h-v1
```

The default `--provider gemini_api` preserves the live agent. To explicitly use
the VM Gemma service for subsequent missions, install with
`--provider local_gemma`. Invalid model JSON remains a failed agent action and
the complete raw generation is saved; no hidden substitute controller is used.
GPU 0 needs a coordinator external lease for the lifetime of the inference
service, so future training dispatch does not compete for its memory.

The base installer keeps access on loopback. An authenticated Google Cloud SSH/IAP tunnel remains available:

```sh
gcloud compute ssh INSTANCE --project PROJECT --zone ZONE -- -N -L 127.0.0.1:18910:127.0.0.1:18900
```

Open `http://127.0.0.1:18910` through that tunnel.

## Public IP demo

The demo is also available at **https://136.115.179.240** directly from the VM.
`cloud/publish_ip_demo.py --ip 136.115.179.240` configures nginx with a trusted
Let's Encrypt IP certificate, HTTP-to-HTTPS redirect, a separate demo login,
request rate limiting and twice-daily certificate renewal. The web login is
generated on the VM and stored only in `/etc/aerorozum/demo-access.json` (0600).
It is unrelated to the Gemini key, which remains in the backend credential store.
No password or private certificate key is checked into this repository.

The firewall rule `aerorozum-public-demo-web` allows only TCP 80/443 to instances
with the `aerorozum-public-demo` tag. API, simulator and Gemma inference ports
remain on loopback. Publishing the proxy does not restart research workers.

## VM policy and recovery

The requested policy is `STANDARD`, `preemptible=false`,
`automaticRestart=true`, `maxRunDuration.seconds=172800`,
`instanceTerminationAction=STOP`, VM deletion protection enabled, and boot-disk
automatic deletion disabled. Google requires stopping the VM to change the
runtime limit. Standard provisioning removes Spot preemption; GPU maintenance,
host failures and cloud capacity shortages can still interrupt service.
An enabled persistent guest timer also enforces a fixed 48-hour stop deadline,
because host recovery can reset the platform's relative runtime counter.
Reinstalling application services preserves that deadline.

The original four-A100 VM's region initially allowed only one standard A100 and
12 A2 vCPUs. The requests for four GPUs and 48 vCPUs were approved. Both A100
zones in that region then rejected standard capacity allocation. A stopped-disk
snapshot was requested before attempting relocation; the original disk is kept.
The private `cloud/48h-extension-verification.json` records the actual final VM
state and expiration time, rather than treating saved settings as proof of uptime.

Ubuntu's first dependency installation automatically restarted two native
physics runs. Their incomplete directories and logs are retained. The current
DuMux binary cannot resume its full interior state from its surface/VTK output.
`cloud/restart_physics_case.py` therefore archives an incomplete trajectory and
starts the identical full-day case in a fresh directory. It refuses to move a
directory locked by an active runner and never joins partial runs or marks them
accepted. This is a full rerun, not checkpoint resumption.

Cost reservations are conservative allowances, not actual billing. The local
Google Cloud ledger reserves €900 of the existing €1,000 cap for this work.
No new committed-use contract is purchased. Any temporary capacity reservation
must automatically expire within the requested deployment window.
