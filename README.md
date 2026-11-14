# AeroRozum thermal research lab

Coupled soil heat/moisture simulation, a six-sortie survey environment, high-volume
hackathon Gemini supervision, and Gemma 4 12B LoRA/SFT/GRPO experiments.

## Run locally

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/aerorozum preflight --check-models
.venv/bin/aerorozum audit --export
.venv/bin/aerorozum freeze
.venv/bin/aerorozum serve
```

In a second terminal, run `npm install` and `npm run dev -- --host 127.0.0.1`
inside `frontend/`. API: http://127.0.0.1:18901; dashboard:
http://127.0.0.1:18900. Docker/MCP adapters expose the same campaign actions.

The active workspace uses a shared virtual environment and artifacts on the SSD:

```bash
TASK_PYTHON='/media/yehormsi/External SSD Volume/AeroRozum-thermal-rl/venv/bin/python'
"$TASK_PYTHON" -m pytest -q
"$TASK_PYTHON" -m aerorozum.cli preflight
```

Set `AEROROZUM_DATA_ROOT` to choose artifact storage. Default is the mounted SSD
when available, otherwise ignored `artifacts/`. The source recordings are never
modified. Cloud model caches and large generated data do not belong in Git.

## Authentication and billing

The Gemini adapter explicitly selects the Developer API with
`enterprise=False, vertexai=False`. It reads `GEMINI_API_KEY` from the process,
a private file named by `AEROROZUM_GEMINI_KEY_FILE`, or a literal assignment in a
local ignored `.env` / parent `.env`. Dotenv is parsed, never sourced or executed.
There is no fallback to startup ADC/Vertex credentials. Do not commit keys.

Google Cloud compute uses `gcloud auth login` separately. The run's resources
carry `project=aerorozum-thermal-rl` labels. The four-A100 Spot worker's original
two-hour timer was extended to 20 hours at the user's request. Its verified
scheduled deletion is 2026-11-15 06:30:50 UTC, subject to earlier Spot preemption.
The initial EUR100 reservation plus EUR500 extension allowance are conservative
budget reservations, not quoted or actual charges. The
SQLite ledger counts outstanding reservations and settled costs against the
EUR1000 limit. Reservations are not claims about actual billing.

## Truth and research gates

The ordinary browser campaign uses an explicitly unvalidated recorded-ground composite.
Physical NPZ observations are separately identified. No generated field becomes
research-accepted solely because it rendered successfully or conserved energy.

Pinned DuMux/DUNE upstream tests and a separate experimental Python reference
have separate provenance. The release verifier records every required check,
with missing checks marked pending. Read `physics/README.md` and the canonical
`physics/verification.json` under the artifact root for actual validation state.

Research campaigns additionally require a frozen observability certificate
from the same field, truth, seed, configuration and feasible public action
trace. Hidden evaluator data is never an MCP tool. Tests cover truth access,
action overrides, repeated coverage, battery reserves and duplicate detections.

Original real annotations are user-supplied labels, not independently verified
object identities. They include manual and reviewed AI annotations. Real data
splits hold out entire zones; demos and unaccepted physics cannot silently enter
research training. Teacher text cannot overwrite deterministic synthetic labels.

## Reproduce the fixed protocol

`configs/research.json` defines all counts and training seeds. `freeze` emits
immutable split/layout manifests and refuses configuration drift. It does not
claim that planned cases have been solved. `teacher MANIFEST` runs resumable
image analyses/reviews. `train --stage compatibility` performs ten optimizer
updates on 16 training-zone images; research stages require accepted synthetic
data. Metrics exist only after real execution.

`package-submissions` generates a reviewable evidence ZIP and per-partner
checklists. It neither submits projects nor claims that unconfigured providers
are integrated. Additional partner credentials and official rules are required
for those entries. The Gemini key cannot pay for other providers' services.

## Tests and external dependencies

Run `python -m pytest -q`; build the frontend with `npm run build`. Heavy training
dependencies are installed only on GPU workers. Gemma 4 Unified currently needs
a sufficiently recent Transformers build; worker manifests retain exact package
versions and the pinned model revision. An incompatible trainer is a failed
compatibility run, not a reason to report fictional training progress.

DuMux-derived code and upstream dependencies retain their original licences.
See `docs/THIRD_PARTY.md`; a blanket application licence does not relicense them.
# Current paired-camera demo

The native Gazebo mission dashboard now pairs RGB with grayscale thermal
rendered from experimental DuMux surface responses. It supports warming and
cooling snapshots, synchronized evidence, agent-selected revisits and reports.
See [camera sources and limitations](docs/paired-physics-camera.md) and the
[presentation comparison](docs/presentation/README.md). This is a research
prototype; complete numerical and field validation remains pending.

