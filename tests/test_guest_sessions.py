import hashlib
import sqlite3
from fastapi.testclient import TestClient
from cloud.guest_sessions import create_app, COOKIE


def test_each_opening_creates_credentials_but_subrequests_keep_the_visitor(tmp_path):
    with TestClient(create_app(tmp_path),base_url='https://demo.test') as client:
        first=client.get('/session',headers={'X-Original-URI':'/?mission=example'})
        user=first.headers['x-aerorozum-guest']
        credential=client.cookies[COOKIE]
        again=client.get('/session',headers={'X-Original-URI':'/api/missions'})
        assert again.headers['x-aerorozum-guest']==user
        assert 'set-cookie' not in again.headers
        reopened=client.get('/session',headers={'X-Original-URI':'/'})
        assert reopened.headers['x-aerorozum-guest']!=user
        assert client.cookies[COOKIE]!=credential
        assert 'www-authenticate' not in first.headers
        assert 'Secure' in first.headers['set-cookie'] and 'HttpOnly' in first.headers['set-cookie']
        assert not first.content


def test_password_is_hashed_and_a_changed_cookie_cannot_reuse_the_identity(tmp_path):
    with TestClient(create_app(tmp_path),base_url='https://demo.test') as client:
        first=client.get('/session',headers={'X-Original-URI':'/'})
        username,password=client.cookies[COOKIE].split('.',1)
        with sqlite3.connect(tmp_path/'guests.sqlite3') as db:
            stored=db.execute('SELECT password_hash FROM guests WHERE username=?',(username,)).fetchone()[0]
        assert stored==hashlib.sha256((username+':'+password).encode()).hexdigest()
        assert password not in stored
        client.cookies.clear()
        changed=client.get('/session',headers={'X-Original-URI':'/api/missions','Cookie':COOKIE+'='+username+'.incorrect'})
        assert changed.headers['x-aerorozum-guest']!=first.headers['x-aerorozum-guest']
        assert changed.status_code==200


def test_independent_browsers_get_different_users(tmp_path):
    app=create_app(tmp_path)
    with TestClient(app,base_url='https://demo.test') as a,TestClient(app,base_url='https://demo.test') as b:
        assert a.get('/session').headers['x-aerorozum-guest']!=b.get('/session').headers['x-aerorozum-guest']
