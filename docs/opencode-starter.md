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

The launcher verifies its Python helper by SHA-256, downloads the published V5
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
<https://storage.googleapis.com/aerorozum-junction2026-submission-avid-relic-509408-e4/start-opencode-v2.sh>.
Source: `cloud/opencode_wizard.py`; `cloud/build_opencode_launcher.py` binds the
bootstrap to the exact helper hash. Published submission ZIPs remain unchanged.
