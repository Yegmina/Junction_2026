# Live research monitor

Public dashboard: https://136.115.179.240/research-live/

This standalone, read-only dashboard shows the active cloud experiments without changing the mission demo. It makes no model requests and has no job-control endpoints.

- Actual `nvidia-smi` utilization, VRAM, temperature and power for each GPU; host CPU, RAM and artifact-volume disk space.
- Three Gemma training continuations, SFT and GRPO updates, gradient health, numeric worker events, recent loss and observed-rate estimates.
- All 88 Aki-validated Telegram development images across three prompts, four thinking levels and five repeats. The 5,280-result denominator comes from the frozen protocol. Prior paid responses are identified separately from new requests.
- Per-setting provisional F2, with matched/missed counts and precision/recall available on hover. All images are positive; these results cannot estimate confirmed-negative specificity.
- The separate 4,799-frame airborne HIGH-thinking review, including valid/failed attempts and cloud upload progress.
- Recent original thermal frames, model annotations, and Aki-validated reference boxes overlaid on the same pixels.

The background collector samples every eight seconds and caches compact per-result metrics. It checks systemd worker state to avoid describing stopped workers as running. Unavailable sections preserve their previous snapshot and are explicitly flagged. Estimates need at least 15 seconds of observed progress; training estimates cover SFT only, with GRPO and evaluation listed separately.

## Run

```sh
/opt/aerorozum/venv/bin/python server.py --root /opt/aerorozum --port 18960
```

Python's standard library serves HTTP and collects metrics. Pillow renders annotation overlays. `nvidia-smi` and systemd are optional telemetry dependencies. The service binds only to loopback, with nginx publishing `/research-live/`.

Deployment uses `aerorozum-research-live.service` with a 512 MiB memory ceiling, one CPU core limit, a read-only filesystem and no API credential loaded. The public JSON intentionally excludes private paths, provider raw responses, exception text and credentials. Images are accessible only through opaque IDs from an explicit evidence-root registry; arbitrary filesystem paths are never accepted.

## Verify

```sh
python -m unittest discover -s cloud/research_live -p 'test_*.py' -v
```

Tests cover frozen denominators, result-backed completion counts, reference-image discovery and rejection of image paths outside evidence roots. Deployment checks also verify the public API, JPEG delivery, unknown-route rejection, desktop rendering, and continued availability of the mission demo.
