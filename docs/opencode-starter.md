# One-command OpenCode starter

Run this in an interactive terminal on an x86_64 Linux machine:

```bash
bash <(curl -fsSL https://136.115.179.240/start.sh)
```

The wizard asks where to save the setup, checks disk space and Docker resources,
offers to install missing Docker/Compose components, and requests administrator
access only when needed. It asks privately for a Gemini API key, verifies model
access without issuing an inference request, and lets you choose a model.
The key is kept in memory and the child process environment; the wizard does
not save it to disk or put it in command arguments.

The launcher verifies its Python helper by SHA-256, downloads the published V6
judging package, verifies its archive hash and all packaged file hashes, and
creates an isolated Harbor 0.26.0 installation. Python 3.12 is obtained through
UV when needed. OpenCode 1.19.5 installs inside the task container through
Harbor. There is no need to install OpenCode separately.

After confirmation it starts the real OpenCode mission. The first Docker build
can take several minutes. Scores and logs are saved under the chosen directory
in `jobs/opencode-<timestamp>/`; the terminal prints the verifier score at the
end. The flight budget and scoring remain the unchanged judging defaults.
Cached downloads and installations are reused on subsequent launches.

Allow 20 GiB of free space, preferably 8 CPU cores and 16 GB RAM. The prepared
Gazebo task requires an amd64 Linux Docker engine; an ARM Docker engine is not
supported by these scene images. The shell bootstrap requires Bash and curl.
It can install missing Python on apt-based Linux; other systems need Python
3.9+ available to start the wizard. A GPU is unnecessary.

For a setup-only check without an API key or model run:

```bash
bash <(curl -fsSL https://136.115.179.240/start.sh) --prepare-only
```

The durable bootstrap is also hosted at
<https://storage.googleapis.com/aerorozum-junction2026-submission-avid-relic-509408-e4/start-opencode-v4.sh>.
Source: `cloud/opencode_wizard.py`; `cloud/build_opencode_launcher.py` binds the
bootstrap to the exact helper hash. Published submission ZIPs remain unchanged.

Version 3 fixes terminal prompting when launched through process substitution.
The Python helper opens separate read and write handles for `/dev/tty`; a
buffered `r+` handle requires seeking and caused a false noninteractive-terminal
error. The exact public command was checked under real Bash and Zsh terminals,
then interrupted at the first prompt before setup or inference. The V5 task ZIP
and earlier versioned launcher artifacts remain unchanged. Verification:
`docs/evidence/opencode-wizard-terminal-fix-v3.json`.

## Repaired flight execution (V6)

The updated package explicitly makes the public geometry and flight-plan SDK
readable to the agent. Its waypoint collector saves actual camera evidence and
reserves time to return. The recommended starter uses a flight-operator system
prompt and a 1,024-token thinking budget for the default Gemini 4 Argon profile, instead
of letting setup reasoning exhaust the response allowance before takeoff.
The 75-second simulated-flight limit and verifier are unchanged.

OpenCode 1.19.5 is preinstalled in the agent image. Package versions are extracted
into separate cache directories; existing jobs and V5 packages are retained.
Repeat installations with the existing Harbor environment need 8 GiB of free
workspace, while fresh setups still require 20 GiB. The separate 15 GiB Docker
storage check remains in effect.
