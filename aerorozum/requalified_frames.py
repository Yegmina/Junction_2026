"""Admit newly qualified frames without overriding any known flight decision."""
from collections import Counter
import copy


def key(row):
    return row['source']['object'],str(row['source']['generation']),row['frame_index']


def stage(original_audits,new_audit,eligible,already_high,session_splits=None):
    old={key(row):row for audit in original_audits for row in audit['frames']}
    known={(row['sha256'],str(row['source']['generation'])) for row in already_high}
    conflicts=[];blocked=set()
    for row in new_audit['frames']:
        prior=old.get(key(row))
        if not prior or prior['eligibility']['status']=='unknown':continue
        before,after=prior['eligibility']['status'],row['eligibility']['status']
        if after!='unknown' and before!=after:
            blocked.add(key(row)[:2]);conflicts.append(dict(source=key(row),prior_status=before,new_status=after))
    selected=[];excluded=Counter()
    for original in eligible:
        row=copy.deepcopy(original);identity=key(row);prior=old.get(identity)
        if identity[:2] in blocked:excluded['source_has_conflicting_known_decision']+=1;continue
        if prior and prior['eligibility']['status']!='unknown':excluded['previously_known_frame']+=1;continue
        if (row['sha256'],str(row['source']['generation'])) in known:
            excluded['existing_high_thinking_pixels']+=1;continue
        if row['split'] not in ('train','validation'):excluded['held_out_split']+=1;continue
        session=row['source'].get('session_id')
        if session_splits is not None and session_splits.get(session)!=row['split']:
            raise ValueError('Frozen session registry disagrees with the requalified frame')
        decision=row.get('eligibility',{})
        if row.get('airborne_eligible') is not True or decision.get('status')!='eligible' or not decision.get('telemetry',{}).get('sha256'):
            raise ValueError('A requalified frame lacks fresh airborne telemetry')
        if prior:
            old_video=prior['source'].get('video_sha256');new_video=row['source'].get('video_sha256')
            if old_video and new_video and old_video!=new_video:
                raise ValueError('Pinned native video bytes differ between flight audits')
            previous_split=prior.get('split')
            if previous_split is None:
                previous_split=(session_splits or {}).get(session)
            if previous_split!=row['split']:raise ValueError('Frozen session split changed or lacks a registry binding')
        row['flight_requalification']=dict(prior_status=prior['eligibility']['status'] if prior else 'not_previously_sampled',
            prior_reason=prior['eligibility'].get('reason') if prior else None,
            known_decisions_overridden=False,reason='Current pinned original receipts and fresh primary IN_AIR state',
            sampling_grid='5 FPS on the original receipt clock; additional native indices can differ from earlier container-PTS sampling')
        selected.append(row);known.add((row['sha256'],str(row['source']['generation'])))
    return selected,dict(new_unique_eligible=len(selected),excluded=dict(excluded),conflicts=conflicts,
        original_audits_modified=False,known_decisions_overridden=False,paid_inference_issued=False)
