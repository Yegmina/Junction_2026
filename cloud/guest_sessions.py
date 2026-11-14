"""Loopback-only automatic visitor identities for nginx auth_request.

Visitors enter anonymously. The generated credential is carried in a secure,
HttpOnly cookie; it is never displayed. This is not a private access gate or
workspace isolation: the demo simulator and research gallery remain shared.
"""
import hashlib
import hmac
import os
from pathlib import Path
import secrets
import sqlite3
import time
import uuid
from urllib.parse import urlsplit

from fastapi import FastAPI, Request, Response

COOKIE = 'aerorozum_guest'
LIFETIME_S = 86400


def credential_hash(username, password):
    return hashlib.sha256((username+':'+password).encode()).hexdigest()


def create_app(root=None):
    root=Path(root or os.environ.get('AEROROZUM_GUEST_ROOT','/opt/aerorozum/deployment/guest-sessions'))
    root.mkdir(parents=True,exist_ok=True,mode=0o700)
    database=root/'guests.sqlite3'
    with sqlite3.connect(database) as db:
        db.execute('CREATE TABLE IF NOT EXISTS guests (username TEXT PRIMARY KEY, password_hash TEXT NOT NULL, created_at INTEGER NOT NULL, expires_at INTEGER NOT NULL)')
        db.execute('CREATE INDEX IF NOT EXISTS guests_expiry ON guests(expires_at)')
    database.chmod(0o600)
    app=FastAPI(docs_url=None,redoc_url=None,openapi_url=None)

    @app.get('/session')
    def session(request: Request):
        now=int(time.time())
        opening=urlsplit(request.headers.get('x-original-uri','')).path=='/'
        value=request.cookies.get(COOKIE,'')
        username,sep,password=value.partition('.')
        valid=False
        with sqlite3.connect(database,timeout=10) as db:
            if not opening and sep and len(value)<256:
                row=db.execute('SELECT password_hash,expires_at FROM guests WHERE username=?',(username,)).fetchone()
                valid=bool(row and row[1]>now and hmac.compare_digest(row[0],credential_hash(username,password)))
            if not valid:
                username='guest_'+uuid.uuid4().hex
                password=secrets.token_urlsafe(24)
                db.execute('DELETE FROM guests WHERE expires_at<=?',(now,))
                db.execute('INSERT INTO guests VALUES (?,?,?,?)',
                           (username,credential_hash(username,password),now,now+LIFETIME_S))
        response=Response(status_code=200,headers={'X-AeroRozum-Guest':username,'Cache-Control':'no-store'})
        if not valid:
            response.set_cookie(COOKIE,username+'.'+password,max_age=LIFETIME_S,
                                httponly=True,secure=True,samesite='lax',path='/')
        return response

    return app


if __name__=='__main__':
    import uvicorn
    uvicorn.run(create_app(),host='127.0.0.1',port=18950,access_log=False)
