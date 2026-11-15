# Development and research setup

Run the commands below from the repository root.

A Harbor environment for a drone agent that chooses where to fly, which thermal
sightings to inspect, and which evidence belongs in its final report. The new
close-inspection task combines a 75-second flight budget, native Gazebo dynamics,
paired thermal/RGB cameras, and an authoritative scalar verifier.

## Junction / AfterQuery submission

[Demo, downloads and measured results](https://136.115.179.240/presentation/submission/index.html)
· [Task description](../harbor/thermal-investigation-budgeted/README.md)
· [Integration evidence](submission-integration-progress.md)

Start the verified package with one command; the wizard checks prerequisites
and asks privately for a Gemini API key:

```bash
bash <(curl -fsSL https://136.115.179.240/start.sh)
```

[Live overnight research](https://136.115.179.240/presentation/research-overnight/index.html)
· [Seven-slide PDF brief](https://136.115.179.240/presentation/research-overnight/brief.pdf)
· [Methods, source hashes and current limitations](overnight-research-20261115.md)

The prepared ZIP includes scene assets and runtime sources; a source checkout
requires staging those large files. From an extracted close-inspection bundle:

```bash
harbor run -p ./thermal-investigation-budgeted -a oracle
harbor run -p ./thermal-investigation-budgeted -a opencode -m <provider/model>
```

Tested: Harbor 0.26.0, OpenCode 1.19.5, Docker and Compose. The reference uses
public images and telemetry, with no target locations or counts. Fresh-seed
reference runs scored 0.733/1 on both instances. All completed no-op, early-stop,
reward-file, forged-evidence and private-state controls scored zero. Model
results and process failures are retained explicitly. The earlier baseline
task has different rules and clean frontier scores up to 1.0; no frontier
superiority is claimed.

Supporting research includes coupled soil heat/moisture simulation, recorded
thermal labels and Gemma 4 12B visual LoRA/SFT/GRPO experiments. Development
perception F2 and native mission reward are separate metrics. Local physical
admission does not establish heterogeneous-field or real mine accuracy.

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

Run tests using the virtual environment created above:

```bash
.venv/bin/python -m pytest -q
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

Google Cloud compute uses `gcloud auth login` separately. The current four-A100
deployment is `aerorozum-mission-48h-20261114`: STANDARD provisioning,
`preemptible: false`, automatic restart enabled, and a 172800-second maximum
runtime with STOP on termination. It started on 14 November at 16:46:58 UTC;
the runtime limit falls on 16 November at about 16:46:58 UTC. Host maintenance
can still terminate the GPU instance. Earlier Spot-worker timers are historical.
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
See [camera sources and limitations](paired-physics-camera.md) and the
[presentation comparison](presentation/README.md). This is a research
prototype; complete numerical and field validation remains pending.

## Source-data requirements for research tests

The portable Harbor task and starter checks do not need the original flight archive.
The broader research suite includes tests that depend on a recorded thermal frame
bank and local input archives. Configure `AEROROZUM_RECORDED_BACKGROUND_DIR` with
original `frame_<digits>.jpg` inputs when exercising those paths. The prepared
Harbor ZIP includes its own scene inputs and is the recommended judging entry point.

During main-branch integration, the scoped Harbor/starter tests and frontend build
passed. The broader cloud sweep was incomplete: 293 tests passed and 38 failed
before it was stopped. Most failures reported missing recorded background frames;
four streaming assertions also failed and remain unverified. See
[the merge validation record](evidence/main-merge-validation.json).
