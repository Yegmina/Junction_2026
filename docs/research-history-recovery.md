# Deployed research history recovery — 14 November 2026

The public API reads `/opt/aerorozum/deployment/artifacts`. The deployment had
preserved missions, telemetry and the recorded-flight gallery but omitted the
experiment ledger, research snapshot and earlier campaign states. Consequently
the API returned its empty defaults even though experiment data still existed.

Recovered the SSD artifact metadata into
`/opt/aerorozum/deployment/recovery/ssd-history-20261114`, then ran
`cloud/restore_research_history.py` with that directory, the original worker
root `/opt/aerorozum/artifacts`, and the deployed artifact root. The original
SSD archive and VM job files were retained. Existing deployed campaign and
validation files are never overwritten by the recovery script.

The deployed ledger now contains 44 jobs: 33 original VM jobs and 11 SSD
records, including the completed 264-prediction held-out comparison. Both
saved development campaigns and their images were recovered. SFT/GRPO
completion metadata was restored, including the missing worker-side
`training/status.json` that feeds the telemetry mirror. Historical SSD records
whose last status was running are explicitly labelled `historical_snapshot`,
with their original status/time retained; current labelling remains in Pipeline.

This was a data restoration. No API, simulator or training service restart
was required. A public-browser check confirmed 44 table rows, the saved
base/SFT/GRPO comparison, and no JavaScript errors. Recovered campaign image
URLs returned HTTP 200. Presentation evidence is saved as
`artifacts/presentation/restored-experiments.png`. The deployed recovery receipt
is `recovery/research-history.json`, including the source snapshot SHA-256.

Future deployments must retain this persistent artifact root, including
`status.json`, `jobs/`, `campaigns/`, `training/` and `validation/`, alongside
missions and pipeline state. A reachable API alone does not verify migration:
compare job/campaign counts and load a historical image and validation result.
