# Implementation contracts

Python package `aerorozum`, Python 3.12. Project root `/home/yehormsi/codex/aerorozum-thermal-rl`.

Ownership: physics agent owns `physics/`, `aerorozum/physics.py`, `tests/test_physics.py`; environment agent owns `aerorozum/environment.py`, `aerorozum/renderer.py`, `aerorozum/evaluation.py`, `aerorozum/mcp_server.py`, `tests/test_environment.py`; UI agent owns `frontend/`, `aerorozum/api.py`, `tests/test_api.py`; root owns all other files. Do not overwrite other agents' files.

Data root from `AEROROZUM_DATA_ROOT`, default repository `artifacts/` (ignored). Secrets never in repo/artifacts. JSON native values, SI physical units, normalized image boxes named x_min/y_min/x_max/y_max in 0..1000.

Campaign interface: `Campaign(config: dict | None = None, seed: int = 0, output_dir: Path | None = None)`; methods `status() -> dict`, `act(action: str, arguments: dict | None = None) -> dict`, `reset(seed: int | None = None) -> dict`. Action names launch, move_to, capture, inspect_observation, inspect_crop, wait_until, land, record_candidate, submit_report. Invalid actions return explicit errors and never advance truth incorrectly. Status/act results JSON serializable. Each campaign has opaque id, authoritative event list, observations, no truth in agent-facing responses. `evaluator_snapshot()` privileged method for dashboard evaluator and scoring, never exposed through agent MCP. API may introspect optional methods conservatively. Output images use relative artifact paths, never arbitrary user filesystem paths.

Environment config includes width_m=64,height_m=64,day_seconds=86400,max_sorties=6,sortie_seconds=600,max_actions=128,turnaround_seconds=1800,camera width=640 height=512 nominal hfov_deg=100 vfov_deg=82, optional thermal-field path. Physics surrogate/demo must always be honestly labelled; never call unchecked analytic images direct physical truth. Renderer can start in explicitly labelled demonstration mode, final research generation requires accepted physical fields.

UI API adapter wraps environment, dataset audit results, provider/model/job status and artifact manifests. FastAPI endpoints `/api/status`, `/api/campaigns` POST, `/api/campaigns/{id}` GET, `/api/campaigns/{id}/actions` POST {action,arguments}, `/api/campaigns/{id}/evaluator` GET with evaluator-mode guard, `/api/events` SSE, `/artifacts/...` contained paths. UI dev port 18900, API port 18901. All actual statuses honest: planned/not_configured/blocked/running/completed/failed/demo. No fake learning curves, resource use, confirmations or model detections.

Physics field exchange: compressed NPZ arrays `times_s` [T], `x_m` [W], `y_m` [H], `temperature_k` [T,H,W], `moisture` [T,H,W], metadata JSON sidecar with fidelity, solver version, config hash, units, validation state, input parameters. Direct truth/object metadata private separate file. Emit CLI and documented call from physics module; no claim DuMux verification without actual logs/results.
