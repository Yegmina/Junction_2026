# Aiven: a versioned investigation evidence ledger

AeroRozum uses Aiven PostgreSQL to store verified Harbor mission outcomes and separately measured held-out image benchmarks. SQL queries expose flight-budget failures, model errors and mean mission rewards without dropping unsuccessful runs. JSONB retains evidence hashes, frozen scenario identities and verifier details. A content-addressed primary key makes imports idempotent and retains changed record versions.

The public page shows a timestamped query snapshot from the database, not a simulated connection or fabricated metrics. Historical imports are identified as such. The integration does not change the frozen Harbor task or its reward.

Install `psycopg[binary]>=3.2,<4` in the backend Python environment, set `AIVEN_DATABASE_URL` privately, download the project's CA certificate, then run:

```sh
python cloud/aiven_evidence.py --input docs/evidence/submission-measurements-latest.json --ca /private/aiven-ca.pem --output docs/presentation/aiven/report.json
```

Alternatively `--connection-file /private/connection.json` reads an object with a `uri` key. Connections enforce `sslmode=verify-full`. Credentials are never written into the public report. Only the `aerorozum` schema is created; existing application tables are untouched.

## Submission wording

We use Aiven PostgreSQL as our investigation evidence ledger. It stores versioned, hash-linked mission outcomes and evaluation results, including failures. SQL queries reveal when agents exceed their flight budget and compare mission rewards independently from image-detection scores. Our demo renders actual database query results through a backend snapshot, with verified TLS and no database credentials in the browser.
