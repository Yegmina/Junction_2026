import json,subprocess,urllib.error
from pathlib import Path
from aerorozum.common import save
from aerorozum.additional_sources import prepare_additional
from aerorozum.cloud_cache import GCSRecordStore


def test_new_source_uses_verified_video_clock_without_fabricated_receipts(tmp_path,monkeypatch):
    video=tmp_path/'source.avi';subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc=size=64x64:rate=10:duration=1','-c:v','mjpeg','-threads','1','-y',str(video)],check=True)
    original={'videos':[{'object':'jetson/old/recordings/old/thermal-0000.avi','generation':'1','session_id':'old'}]};save(tmp_path/'source-videos/labelling-sources.json',original)
    save(tmp_path/'labelling/robotics-er2-5fps/session-splits.json',{'seed':17,'sessions':{'old':'test'}})
    source={'bucket':'source-bucket','object':'jetson/new/recordings/new/thermal-0000.avi','generation':'2','session_id':'new','segment':0,'path':str(video),'sha256':'a'*64,'native_frames':10,'pts_strictly_monotonic':True,'duration_s':.9,'recovered_variant':False}
    out=tmp_path/'labelling/additional-bucket-sources';save(out/'native-probe.json',{'sources':[source]})
    monkeypatch.setattr(GCSRecordStore,'_headers',lambda self:{})
    def unavailable(*args,**kwargs):raise urllib.error.HTTPError('https://example.invalid',404,'missing',None,None)
    monkeypatch.setattr('urllib.request.urlopen',unavailable)
    doc=prepare_additional(tmp_path)
    assert len(doc['images'])==5 and doc['session_splits']['old']=='test'
    assert all(r['timestamp_s'] is None and not r['wallclock_alignment_available'] and not r['gps_alignment_available'] for r in doc['images'])
    assert all(r['time_basis']=='container_pts_seconds' for r in doc['images'])
    assert json.loads((tmp_path/'source-videos/labelling-sources-original-150.json').read_text())==original
    again=prepare_additional(tmp_path);assert [r['sha256'] for r in again['images']]==[r['sha256'] for r in doc['images']]


def test_nonmonotonic_source_is_explicitly_quarantined(tmp_path):
    save(tmp_path/'source-videos/labelling-sources.json',{'videos':[]})
    source={'bucket':'b','object':'jetson/new/recordings/new/thermal-0000.avi','generation':'2','session_id':'new','segment':0,'pts_strictly_monotonic':False,'recovered_variant':False}
    save(tmp_path/'labelling/additional-bucket-sources/native-probe.json',{'sources':[source]})
    doc=prepare_additional(tmp_path)
    assert not doc['images'] and doc['sources'][0]['status']=='quarantined_no_monotonic_native_PTS'
