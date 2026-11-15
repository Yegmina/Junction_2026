"""Versioned mission evidence in Aiven PostgreSQL; no model calls or raw credentials."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


DDL = """
CREATE SCHEMA IF NOT EXISTS aerorozum;
CREATE TABLE IF NOT EXISTS aerorozum.evidence (
  kind text NOT NULL, record_id text NOT NULL, sha256 text NOT NULL,
  model text NOT NULL, payload jsonb NOT NULL,
  inserted_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (kind, record_id, sha256)
);
CREATE INDEX IF NOT EXISTS evidence_model_idx ON aerorozum.evidence(model);
"""


def records(document):
    """Explicit field allowlists exclude credentials and arbitrary source metadata."""
    specs = {
        'trials': ('mission', ['job', 'trial', 'task_checksum', 'model', 'rewards',
                              'exception_type', 'finished_at', 'details', 'instance', 'result_sha256']),
        'held_out_perception': ('held_out_image', ['model', 'status', 'examples', 'tp', 'fp',
                              'fn', 'micro_f2', 'valid_json_fraction', 'source_sha256',
                              'frozen_test_contract', 'protocol_hash', 'decoding_protocol']),
    }
    output = []
    for source, (kind, fields) in specs.items():
        for row in document.get(source, []):
            payload = {k: row[k] for k in fields if k in row}
            model = payload.get('model') or 'unattributed/control'
            identity = (str(row['job']) + '/' + str(row['trial'])) if kind == 'mission' else model
            encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
            output.append((kind, identity, hashlib.sha256(encoded.encode()).hexdigest(), model, encoded))
    return output


def summarize(connection):
    # Latest version of each identity, while retaining every previous version.
    latest = """WITH latest AS (
      SELECT DISTINCT ON (kind, record_id) * FROM aerorozum.evidence
      ORDER BY kind, record_id, inserted_at DESC, sha256 DESC
    ) """
    with connection.cursor() as c:
        c.execute(latest + """SELECT model, payload->>'task_checksum',
          payload->'instance'->>'budget_s',
          payload->'instance'->'thermal_condition'->>'phase', count(*) AS trials,
          count(*) FILTER (WHERE payload->>'exception_type' IS NOT NULL) AS errors,
          count(*) FILTER (WHERE (payload->'details'->>'flight_elapsed_s')::float8 >
                                 (payload->'details'->>'budget_s')::float8) AS over_budget,
          avg((payload->'rewards'->>'reward')::float8) AS mean_reward
          FROM latest WHERE kind='mission' GROUP BY model, payload->>'task_checksum',
          payload->'instance'->>'budget_s',
          payload->'instance'->'thermal_condition'->>'phase' ORDER BY model""")
        missions = [dict(zip(['model', 'task_checksum', 'budget_s', 'phase', 'trials',
                             'errors', 'over_budget', 'mean_reward'], r)) for r in c.fetchall()]
        c.execute(latest + "SELECT model, payload FROM latest WHERE kind='held_out_image' ORDER BY model")
        images = [r[1] for r in c.fetchall()]
        c.execute('SELECT kind, count(*) FROM aerorozum.evidence GROUP BY kind ORDER BY kind')
        counts = dict(c.fetchall())
        c.execute("SELECT ssl FROM pg_stat_ssl WHERE pid=pg_backend_pid()")
        tls = c.fetchone()[0]
    return dict(generated_at=datetime.now(timezone.utc).isoformat(), backend='Aiven PostgreSQL',
                tls=tls, stored_versions=counts, missions=missions, held_out_images=images,
                scope='Mission rewards and held-out image label agreement are separate measurements. Historical import; not field validation.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--connection-file', type=Path)
    p.add_argument('--ca', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import psycopg
    uri = os.environ.get('AIVEN_DATABASE_URL')
    if not uri and a.connection_file:
        uri = json.loads(a.connection_file.read_text())['uri']
    if not uri:
        p.error('Set AIVEN_DATABASE_URL or supply a private --connection-file')
    rows = records(json.loads(a.input.read_text()))
    # Override URI SSL options: hostname and Aiven project CA are both checked.
    with psycopg.connect(uri, sslmode='verify-full', sslrootcert=str(a.ca.resolve()),
                         connect_timeout=15, application_name='aerorozum-evidence') as conn:
        with conn.cursor() as c:
            c.execute(DDL)
            for row in rows:
                c.execute('INSERT INTO aerorozum.evidence (kind,record_id,sha256,model,payload) '
                          'VALUES (%s,%s,%s,%s,%s::jsonb) ON CONFLICT DO NOTHING', row)
        report = summarize(conn)
    report['imported_records'] = len(rows)
    report['input_sha256'] = hashlib.sha256(a.input.read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = a.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(a.output)
    print(json.dumps({'backend': report['backend'], 'tls': report['tls'],
                      'stored_versions': report['stored_versions'], 'output': str(a.output)}))


if __name__ == '__main__':
    main()
