"""Immutable paid-inference cache and label archive in user-owned GCS buckets."""
from __future__ import annotations
import json
import re
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from .preflight import sdk_path


class GCSRecordStore:
    def __init__(self,bucket,prefix='robotics-er2-labels/5fps-v1'):
        if not re.fullmatch(r'[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]',bucket):raise ValueError('Invalid GCS bucket')
        if '..' in prefix.split('/') or prefix.startswith('/'):raise ValueError('Invalid artifact prefix')
        self.bucket=bucket;self.prefix=prefix.rstrip('/');self._token=None;self._expiry=0.;self._lock=threading.Lock()

    def _headers(self):
        with self._lock:
            if not self._token or time.monotonic()>self._expiry:
                self._token=subprocess.check_output([sdk_path(),'auth','print-access-token','--quiet'],text=True,timeout=30).strip()
                self._expiry=time.monotonic()+2400
            token=self._token
        return {'Authorization':'Bearer '+token,'Content-Type':'application/json'}

    def _invalidate(self, rejected_headers):
        """A gcloud token can already be aged when first obtained.

        Refresh after an actual 401, rather than assuming our local TTL is its
        remaining lifetime. A concurrent refresh must not be invalidated by an
        older request that used the rejected token.
        """
        with self._lock:
            if rejected_headers.get('Authorization') == 'Bearer '+str(self._token):
                self._token = None
                self._expiry = 0.

    def _object(self,key,kind='results'):
        if not re.fullmatch(r'[a-zA-Z0-9_-]{8,128}',key):raise ValueError('Unsafe cache key')
        if kind not in ('results','labels','logs','progress'):raise ValueError('Unknown artifact type')
        return f'{self.prefix}/{kind}/{key}.json'

    def get(self,key,kind='results'):
        name=self._object(key,kind)
        url=f'https://storage.googleapis.com/storage/v1/b/{self.bucket}/o/{urllib.parse.quote(name,safe="")}?alt=media'
        for attempt in range(2):
            headers=self._headers()
            try:
                with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=45) as r:return json.load(r)
            except urllib.error.HTTPError as e:
                if e.code==404:return None
                if e.code==401 and attempt==0:
                    self._invalidate(headers)
                    continue
                raise RuntimeError(f'GCS cache read failed: HTTP {e.code}; paid inference was not issued') from None

    def put(self,key,record,kind='results'):
        name=self._object(key,kind)
        payload=json.dumps(record,allow_nan=False,separators=(',',':'),default=str).encode()
        url=f'https://storage.googleapis.com/upload/storage/v1/b/{self.bucket}/o?uploadType=media&ifGenerationMatch=0&name={urllib.parse.quote(name,safe="")}'
        refreshed=False
        for attempt in range(4):
            headers=self._headers()
            try:
                request=urllib.request.Request(url,data=payload,headers=headers,method='POST')
                with urllib.request.urlopen(request,timeout=60) as r:json.load(r)
                return f'gs://{self.bucket}/{name}'
            except urllib.error.HTTPError as e:
                if e.code==401 and not refreshed:
                    self._invalidate(headers)
                    refreshed=True
                    continue
                if e.code==412:
                    existing=self.get(key,kind)
                    if existing is None:raise RuntimeError('Cache creation race without readable object') from None
                    return f'gs://{self.bucket}/{name}'
                if e.code not in (429,500,502,503,504):raise RuntimeError(f'GCS archive failed: HTTP {e.code}; local paid result retained') from None
            except (OSError,TimeoutError):
                if attempt==3:raise
            time.sleep(2**attempt)
        raise RuntimeError('GCS upload retries exhausted; local paid result retained')

    def write_frame(self,key,record,raw_log=None):
        uris={'result':self.put(key,record)}
        if raw_log is not None:uris['raw_log']=self.put(key,raw_log,'logs')
        uris['labels']=self.put(key,{k:v for k,v in record.items() if k not in ('model_output','raw_response')},'labels')
        return uris
