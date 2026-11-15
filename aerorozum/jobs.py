"""Durable, inspectable job state and transactional committed-cost accounting."""
import json
import sqlite3
import uuid
from contextlib import contextmanager

from .common import data_root, save, utcnow


@contextmanager
def ledger(root=None):
    root = root or data_root()
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / 'ledger.sqlite3', timeout=30)
    try:
        db.execute('CREATE TABLE IF NOT EXISTS costs (id TEXT PRIMARY KEY, provider TEXT, reserved REAL, spent REAL, status TEXT, detail TEXT)')
        db.commit()
        yield db
    finally:
        db.close()


def reserve(provider, euros, detail, cap=1000, root=None):
    if euros < 0:
        raise ValueError('Negative reservation')
    with ledger(root) as db:
        db.execute('BEGIN IMMEDIATE')
        used = db.execute("SELECT COALESCE(SUM(spent + CASE WHEN status='active' THEN reserved ELSE 0 END),0) FROM costs WHERE provider=?", (provider,)).fetchone()[0]
        if used + euros > cap:
            raise ValueError(f'{provider} reservation exceeds {cap} EUR cap')
        key = uuid.uuid4().hex
        db.execute('INSERT INTO costs VALUES (?,?,?,?,?,?)', (key, provider, euros, 0, 'active', json.dumps(detail)))
        db.commit()
        return key


def settle(key, actual_euros, root=None):
    if actual_euros < 0:
        raise ValueError('Negative expenditure')
    with ledger(root) as db:
        if db.execute("UPDATE costs SET spent=?,reserved=0,status='closed' WHERE id=? AND status='active'", (actual_euros, key)).rowcount != 1:
            raise ValueError('Unknown or already closed reservation')
        db.commit()


def cost_status(root=None):
    with ledger(root) as db:
        return [dict(zip(('provider', 'reserved_eur', 'spent_eur'), row)) for row in db.execute("SELECT provider,SUM(CASE WHEN status='active' THEN reserved ELSE 0 END),SUM(spent) FROM costs GROUP BY provider")]


def job(kind, state='running', **fields):
    record = dict(id=uuid.uuid4().hex, kind=kind, status=state, created_at=utcnow(), updated_at=utcnow(), **fields)
    save(data_root() / 'jobs' / (record['id'] + '.json'), record)
    return record


def update(record, state=None, **fields):
    record.update(fields, updated_at=utcnow())
    if state:
        record['status'] = state
    save(data_root() / 'jobs' / (record['id'] + '.json'), record)
    return record
