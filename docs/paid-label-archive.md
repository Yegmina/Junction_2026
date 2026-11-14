# Paid thermal-label archive

Robotics ER 2 labels and provider logs are archived in the source video bucket:

`gs://aerorozum-jetson-testdata-smart-bird/robotics-er2-labels/5fps-v1/`

- `results/`: complete cached inference records, image SHA-256 and source lineage.
- `labels/`: normalized accepted/uncertain boxes and frame-specific mappings.
- `logs/`: original provider response text, prompt, exact model, generation settings,
  token usage and timing, where available.
- `zip-crops/`: all 455 completed annotation-excluded crop records and available logs.
- `audit/<run>/logs/`: content-addressed versions of local provider raw attempts,
  successful responses, errors, paid records, labels and progress documents.
- `audit/<run>/progress/`: immutable indexes mapping original local relative paths
  and JSON hashes to archive objects.

The per-frame worker writes paid output locally before schema validation, then
uploads results, labels and logs. Local/cloud cache lookup precedes inference.
Failed uploads retain the local output and can retry without another model call.
Malformed or refused output never becomes an empty/background label.

An independent `aerorozum-robotics-cloud-audit.service` scans both active labeling
runs every 30 seconds after each completed pass. It also preserves raw retry and
error attempts that have no successful normalized label. It does not instantiate
a model client, issue inference, or modify files used by the active workers.
Failed archival attempts remain eligible for the next pass.

Local receipts are under each run's `cloud-audit/state.json`. Per-frame upload
status is in `labels/results/*.json`, with aggregate counters in
`labels/progress.json`. A remote readback receipt is saved at
`labelling/cloud-readback-verification.json` under the artifact root.

The 455 older crop responses predate verbatim-text logging. Their original stored
parsed output, prompt, usage and timing are archived with that limitation recorded;
missing raw text is neither reconstructed nor purchased again.

Labels retain `provenance: user_accepted_model`, `human_reviewed: false` and the
user's 0.60 acceptance threshold. A model-screened empty frame is not described as
an independently verified negative.

To perform a single archive pass without inference:

```sh
python scripts/archive_robotics_logs.py --once
```

Check the archive worker:

```sh
systemctl --user status aerorozum-robotics-cloud-audit.service
journalctl --user -u aerorozum-robotics-cloud-audit.service -n 20
```
