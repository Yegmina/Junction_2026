"""Recover only provably aligned portions outside receipt-clock ambiguity.

Native counts and original contiguous frame counters must match one-to-one.
No timestamp is edited. Rollback overlap plus the full two-second freshness
margin in either thermal receipts or telemetry is excluded before sampling.
"""
from pathlib import Path
import hashlib
import json
import os
from datetime import datetime
import re
import subprocess
import tempfile
from PIL import Image


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def prepare(folder,output,sessions):
    from aerorozum.common import save,digest
    from aerorozum.labelling import _probe,sample_indices
    from aerorozum.receipt_clock import safe_native_indices
    from aerorozum.flight_filter import filter_rows,read_telemetry
    original=json.loads((folder/'report.json').read_text())
    objects={}
    for file in folder.glob('objects/*/source.json'):
        meta=json.loads(file.read_text());path=file.parent/Path(meta['object']).name
        if sha(path)!=meta['sha256']:raise ValueError('Pinned source object changed')
        objects[Path(meta['object']).name]=(path,meta)
    video,video_meta=objects[Path(original['object']).name]
    receipts,receipt_meta=objects['thermal-timestamps.jsonl']
    if 'telemetry.jsonl' not in objects:
        from cloud.prepare_followup_sources import PinnedObjects
        client=PinnedObjects(folder,original['bucket'],datetime.fromisoformat('2026-11-15T03:30:00+00:00'))
        telemetry,telemetry_meta=client.object(original['object'].rsplit('/',1)[0]+'/telemetry.jsonl')
    else:telemetry,telemetry_meta=objects['telemetry.jsonl']
    records=[json.loads(line) for line in receipts.read_bytes().rstrip(b'\x00\r\n').splitlines() if line.strip()]
    local=[row for row in records if int(row.get('segment',0))==original['segment']]
    telemetry_times=[]
    with telemetry.open('rb') as stream:
        for line in stream:
            line=line.rstrip(b'\x00\r\n')
            if line.strip():telemetry_times.append(json.loads(line)['received_at_ns'])
    width,height,pts=_probe(video)
    if not pts or any(b<=a for a,b in zip(pts,pts[1:])):raise ValueError('Native PTS is not strictly increasing')
    safe,proof=safe_native_indices(local,len(pts),[[r['received_at_ns'] for r in records],telemetry_times])
    if not safe:raise ValueError('No unambiguous original timestamps survive the complete freshness margin')
    times=[local[i]['received_at_ns']/1e9 for i in safe]
    indices=[safe[i] for i in sample_indices(times,5,origin=times[0])]
    proof.update(native_pts_strict=True,thermal_receipts=receipt_meta,telemetry=telemetry_meta,
        video=video_meta,sampling_fps=5,sampled_frames=len(indices),
        original_whole_source_quarantine_preserved=True,protocol='unambiguous-clock-portions-v1')
    target=output/hashlib.sha256(original['object'].encode()).hexdigest()[:20];target.mkdir(parents=True,exist_ok=True)
    save(target/'clock-proof.json',proof)
    source={k:original[k] for k in ['bucket','object','generation','session_id','segment']}
    source.update(video_sha256=video_meta['sha256'],clock_recovery=dict(protocol=proof['protocol'],
        proof_sha256=sha(target/'clock-proof.json'),timestamps_modified=False,guard_s=2))
    selected=set(indices);rows=[];index=0
    with tempfile.TemporaryFile() as errors:
        process=subprocess.Popen(['ffmpeg','-v','error','-threads','1','-i',str(video),'-map','0:v:0','-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE,stderr=errors)
        size=width*height*3
        try:
            while True:
                raw=process.stdout.read(size)
                if not raw:break
                if len(raw)!=size:raise ValueError('Truncated native frame')
                if index in selected:
                    stamp=local[index]['received_at_ns'];identity=digest([source,index,stamp]);image=target/(identity+'.jpg')
                    Image.frombytes('RGB',(width,height),raw).save(image,quality=95)
                    rows.append(dict(id=identity,image=str(image),sha256=sha(image),width=width,height=height,
                        source=source,split=sessions[original['session_id']],frame_index=index,
                        original_frame_counter=local[index]['frame'],source_pts_s=pts[index],
                        receipt_timestamp_ns=stamp,timestamp_s=stamp/1e9,
                        source_offset_s=(stamp-local[indices[0]]['received_at_ns'])/1e9,
                        time_basis='jetson_original_receipt_outside_clock_ambiguity',
                        wallclock_alignment_available=True,sampling_fps=5,sample_grid_interval_s=.2))
                index+=1
            if process.wait(timeout=30) or index!=len(pts):raise ValueError('Native decoding count differs')
        except BaseException:
            process.terminate();process.wait();raise
        finally:process.stdout.close()
    parsed=read_telemetry(telemetry,telemetry_meta)
    audit,eligible=filter_rows(rows,{original['session_id']:parsed},session_splits=sessions)
    report=dict(source=source,status='qualified_only_outside_clock_ambiguity',proof=proof,counts=audit['counts'])
    save(target/'report.json',report)
    return report,audit['frames'],eligible['images']


if __name__=='__main__':
    from aerorozum.common import save,utcnow
    from aerorozum.flight_filter import POLICY
    root=Path('/opt/aerorozum');out=Path(os.environ.get('AEROROZUM_CLOCK_OUTPUT_ROOT',str(root/'airborne-clock-portions-v1/artifacts')));out.mkdir(parents=True,exist_ok=True)
    registry=json.loads((root/'airborne-labeler/artifacts/labelling/robotics-er2-airborne/session-splits.json').read_text());sessions=registry['sessions']
    reports=[];frames=[];images=[]
    for file in (root/'airborne-requalification-v1/artifacts/sources').glob('*/report.json'):
        original=json.loads(file.read_text())
        if 'Receipt timestamps are not strictly increasing' not in original.get('error',''):continue
        try:
            report,decisions,eligible=prepare(file.parent,out,sessions);reports.append(report);frames.extend(decisions);images.extend(eligible)
        except Exception as error:
            reports.append(dict(source=original['object'],status='remains_quarantined',error=str(error)[:500]))
        save(out/'progress.json',dict(at=utcnow(),sources=len(reports),eligible=len(images),paid_inference_issued=False))
        print(json.dumps(dict(source=original['object'],status=reports[-1]['status'],eligible_so_far=len(images))),flush=True)
    audit=dict(schema_version=1,at=utcnow(),policy=POLICY,sources=reports,frames=frames,paid_inference_issued=False)
    save(out/'airborne-eligibility.json',audit);save(out/'airborne-eligible.json',dict(images=images,session_splits=sessions))
    save(out/'result.json',dict(at=utcnow(),sources=len(reports),eligible_frames=len(images),source_reports=reports,
        paid_inference_issued=False,timestamps_modified=False,original_quarantines_preserved=True))
