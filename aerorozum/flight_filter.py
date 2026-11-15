"""Fail-closed thermal/flight-state joining using original Jetson receipt clocks."""
from __future__ import annotations
import bisect,hashlib,json,math
from collections import Counter,defaultdict
from pathlib import Path
from .common import save,utcnow

POLICY={'version':'airborne-only-v2-primary-state','state_max_age_s':2.,'altitude_max_age_s':2.,'primary':'fresh unambiguous autopilot EXTENDED_SYS_STATE IN_AIR; armed alone never establishes flight','relative_altitude_policy':'HOME-relative diagnostic, never an AGL threshold or eligibility requirement','unknown_policy':'skip paid inference; pending review','transition_policy':'takeoff/landing/undefined states require review','references':['https://mavlink.io/en/messages/common.html#MAV_LANDED_STATE','https://mavlink.io/en/messages/common.html#ALTITUDE']}


def read_telemetry(path,source=None):
    raw=Path(path).read_bytes();states=defaultdict(list);altitudes=defaultdict(list);counts=Counter();malformed=0
    for line in raw.rstrip(b'\x00\r\n').splitlines():
        if not line.strip():continue
        try:row=json.loads(line);stamp=int(row['received_at_ns']);message=row.get('message',{});system=int(row.get('system',0));component=int(row.get('component',1))
        except (ValueError,KeyError,TypeError):malformed+=1;continue
        if stamp<=0 or component!=1:continue
        packet=message.get('mavpackettype');counts[str(packet)]+=1
        if packet=='EXTENDED_SYS_STATE':
            state=message.get('landed_state')
            if isinstance(state,int) and not isinstance(state,bool):states[system].append((stamp,state))
        value=None;basis=None
        if packet=='GLOBAL_POSITION_INT' and isinstance(message.get('relative_alt'),(float,int)):value=message['relative_alt']/1000;basis='GLOBAL_POSITION_INT.relative_alt_mm'
        elif packet=='ALTITUDE' and isinstance(message.get('altitude_relative'),(float,int)):value=message['altitude_relative'];basis='ALTITUDE.altitude_relative_m'
        if value is not None and math.isfinite(value):altitudes[system].append((stamp,float(value),basis))
    for mapping in (states,altitudes):
        for events in mapping.values():events.sort(key=lambda event:event[0])
    return {'states':dict(states),'altitudes':dict(altitudes),'state_times':{s:[x[0] for x in v] for s,v in states.items()},'altitude_times':{s:[x[0] for x in v] for s,v in altitudes.items()},'source':{**(source or {}),'sha256':hashlib.sha256(raw).hexdigest()},'message_counts':dict(counts),'malformed_lines':malformed,'timestamp_policy':'original received_at_ns sorted for lookup; values never modified'}


def frame_receipt_ns(row):
    if row.get('wallclock_alignment_available') is False or row.get('source',{}).get('wallclock_alignment_available') is False:return None,'wallclock_alignment_unavailable'
    exact=row.get('receipt_timestamp_ns')
    if isinstance(exact,int) and not isinstance(exact,bool) and exact>0 and row.get('wallclock_alignment_available') is not False:return exact,'original_integer_receipt'
    basis=row.get('time_basis','')
    if basis.startswith('jetson') and row.get('wallclock_alignment_available') is not False:
        value=row.get('timestamp_s')
        if isinstance(value,(int,float)) and math.isfinite(value) and value>0:return round(value*1e9),'validated_receipt_seconds_rounded_to_ns'
    return None,'wallclock_alignment_unavailable'


def classify_frame(row,telemetry,policy=None):
    rules={**POLICY,**(policy or {})};stamp,basis=frame_receipt_ns(row)
    result={'status':'unknown','reason':'thermal_wallclock_alignment_unavailable','airborne_eligible':False,'thermal_receipt_ns':stamp,'thermal_time_basis':basis,'policy_version':rules['version']}
    if stamp is None:return result
    if not telemetry:return {**result,'reason':'telemetry_unavailable'}
    result['telemetry']=telemetry['source'];fresh=[]
    for system,events in telemetry['states'].items():
        index=bisect.bisect_right(telemetry['state_times'][system],stamp)-1
        if index<0:continue
        event=events[index];age=(stamp-event[0])/1e9
        if 0<=age<=rules['state_max_age_s']:fresh.append((system,event[1],age))
    if not fresh:return {**result,'reason':'no_fresh_primary_landed_state'}
    if len({state for _,state,_ in fresh})!=1:return {**result,'reason':'conflicting_vehicle_flight_states'}
    system,state,age=min(fresh,key=lambda item:item[2]);result.update(vehicle_system=system,landed_state=state,state_age_s=age)
    if state==1:return {**result,'status':'ground','reason':'autopilot_on_ground'}
    if state!=2:return {**result,'reason':'takeoff_landing_or_undefined_state'}
    altitudes=telemetry['altitudes'].get(system,[]);index=bisect.bisect_right(telemetry['altitude_times'].get(system,[]),stamp)-1
    result['relative_altitude_reference']='HOME, not height above ground'
    if index<0:result['altitude_diagnostic_status']='unavailable'
    else:
        event=altitudes[index];alt_age=(stamp-event[0])/1e9;result.update(relative_altitude_m=event[1],altitude_age_s=alt_age,altitude_basis=event[2],altitude_diagnostic_status='fresh' if 0<=alt_age<=rules['altitude_max_age_s'] else 'stale')
    return {**result,'status':'eligible','reason':'fresh_unambiguous_autopilot_in_air','airborne_eligible':True}


def filter_rows(rows,telemetry_by_session,*,session_splits=None,output_dir=None,source_reports=None,prior_paid_identities=None):
    decisions=[];eligible=[];counts=Counter();reasons=Counter();per_source=defaultdict(Counter)
    for original in rows:
        row=dict(original);source=row.get('source',{});sid=source.get('session_id');result=classify_frame(row,telemetry_by_session.get(sid));counts[result['status']]+=1;reasons[result['reason']]+=1;per_source[source.get('object','unknown')][result['status']]+=1
        decisions.append({'id':row.get('id'),'sha256':row.get('sha256'),'source':source,'frame_index':row.get('frame_index'),'eligibility':result})
        if result['status']=='eligible':
            row.update(airborne_eligible=True,eligibility=result)
            if prior_paid_identities is not None:
                previous=prior_paid_identities.get((row.get('sha256'),source.get('generation')))
                row['prior_paid_inference']=previous or {'status':'not_found_in_canonical_paid_cache'}
            eligible.append(row)
    audit={'schema_version':1,'status':'airborne_only_audit','created_at':utcnow(),'policy':POLICY,'counts':{'total_frames':len(decisions),'airborne_eligible':counts['eligible'],'ground_excluded':counts['ground'],'unknown_skipped':counts['unknown']},'reasons':dict(reasons),'per_source':{k:dict(v) for k,v in per_source.items()},'sources':source_reports or [],'frames':decisions,'previous_paid_labels_preserved':True,'paid_inference_issued':False}
    if prior_paid_identities is not None:
        pending=[r for r in eligible if (r.get('sha256'),r.get('source',{}).get('generation')) not in prior_paid_identities]
        audit['counts'].update(airborne_with_prior_paid_cache=len(eligible)-len(pending),airborne_pending_frames=len(pending),estimated_new_unique_inference_requests=len({(r['sha256'],r['source']['generation']) for r in pending}),prior_paid_invalid_requires_explicit_review=sum(r.get('prior_paid_inference',{}).get('status')=='invalid_paid_output_retained' for r in eligible))
    manifest={'schema_version':1,'status':'airborne_only','created_at':audit['created_at'],'policy':POLICY,'counts':audit['counts'],'session_splits':session_splits or {},'images':eligible,'previous_paid_labels_preserved':True}
    if output_dir:
        folder=Path(output_dir);save(folder/'airborne-eligibility.json',audit);manifest['eligibility_manifest_sha256']=hashlib.sha256((folder/'airborne-eligibility.json').read_bytes()).hexdigest();manifest['eligibility_manifest']='pipeline/inputs/airborne-eligibility.json';save(folder/'airborne-eligible.json',manifest)
    return audit,manifest
