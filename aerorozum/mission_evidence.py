"""Sensor-bound mission evidence. Detector input never contains evaluator truth.

No client is constructed here. A caller-injected provider can issue a paid
request only after a coordinated, rate-bounded lease authorizes admission.
"""
from __future__ import annotations
import asyncio,copy,hashlib,inspect,json,math,os,re
from datetime import datetime,timezone
from pathlib import Path
from PIL import Image,ImageDraw
from .common import data_root,digest,save,utcnow,redacted_error
from .evaluation import valid_box,score_submission
from .labelling import MODEL,classify_response
from .label_prompts import PROMPTS,PROMPT_VERSION,prompt_metadata

ROLE='mission_thermal_evidence'
SOURCE_KINDS={'gazebo_camera','dumux_field_crop','domain_randomized_thermal_world','recorded_reference'}
_LOCKS={}


def _sha(raw):return hashlib.sha256(raw).hexdigest()

def _finite(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)

def _identifier(value,name):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',value):raise ValueError(f'Invalid {name}')
    return value


def public_context(context):
    """Deep allowlist; no mission task, target list, segmentation, or truth enters."""
    if not isinstance(context,dict):raise ValueError('Observation context must be an object')
    result={key:_identifier(context.get(key),key) for key in ('mission_id','observation_id')}
    kind=context.get('source_kind')
    if kind not in SOURCE_KINDS:raise ValueError('Explicit supported sensor source_kind required')
    result['source_kind']=kind
    modality=context.get('sensor_modality','simulated_thermal_intensity')
    if modality not in ('rgb','thermal_intensity','simulated_thermal_intensity'):raise ValueError('Unsupported sensor modality')
    result['sensor_modality']=modality
    result['pose_frame']=str(context.get('pose_frame','unspecified'))[:200]
    if context.get('pose') is not None:
        if kind=='recorded_reference':raise ValueError('Recorded reference frames cannot acquire a simulated moved pose')
        pose=context['pose']
        if not isinstance(pose,dict):raise ValueError('Pose must be an object')
        allowed=('x_m','y_m','z_m','altitude_m','roll_rad','pitch_rad','yaw_rad','yaw_deg')
        if any(k in pose and not _finite(pose[k]) for k in allowed):raise ValueError('Capture pose coordinates must be finite')
        result['pose']={k:float(pose[k]) for k in allowed if k in pose and _finite(pose[k])}
        if not all(k in result['pose'] for k in ('x_m','y_m')):raise ValueError('Camera pose needs x_m and y_m')
        if not any(k in result['pose'] for k in ('z_m','altitude_m')):raise ValueError('Camera pose needs z_m or altitude_m')
    elif kind!='recorded_reference':raise ValueError('Simulator/field observations require their actual capture pose')
    for key in ('sim_time_s','captured_at_s'):
        if key in context:
            if not _finite(context[key]) or context[key]<0:raise ValueError('Invalid capture time')
            result[key]=float(context[key])
    capture=context.get('capture',{})
    if not isinstance(capture,dict):raise ValueError('Capture provenance must be an object')
    result['capture']={k:str(capture[k])[:500] for k in ('event_id','camera_topic','sensor_frame','bridge','renderer_version','field_sha256','field_metadata_sha256','source_image_sha256','source_archive_sha256','binding') if k in capture and isinstance(capture[k],str)}
    for stamp_key in ('sim_time_s','pose_sim_time_s','pose_time_offset_s'):
        if stamp_key in capture:
            if not _finite(capture[stamp_key]) or (stamp_key!='pose_time_offset_s' and capture[stamp_key]<0):raise ValueError('Invalid sensor simulation time')
            result['capture'][stamp_key]=float(capture[stamp_key])
    if kind=='gazebo_camera' and not capture.get('camera_topic'):raise ValueError('Gazebo camera evidence requires its real sensor topic')
    if kind=='dumux_field_crop' and not re.fullmatch('[a-f0-9]{64}',str(capture.get('field_sha256',''))):raise ValueError('Physical field crops require a bound field SHA256')
    mount=context.get('camera_mount',{})
    if not isinstance(mount,dict):raise ValueError('Camera mounting metadata must be an object')
    result['camera_mount']={k:float(mount[k]) for k in ('x_m','y_m','z_m','roll_rad','pitch_rad','yaw_rad') if k in mount and _finite(mount[k])}
    camera=context.get('camera',{})
    if not isinstance(camera,dict):raise ValueError('Camera metadata must be an object')
    result['camera']={k:float(camera[k]) for k in ('hfov_deg','vfov_deg','fx_px','fy_px','cx_px','cy_px','near_m','far_m') if k in camera and _finite(camera[k])}
    mapping=context.get('display_mapping',{})
    if not isinstance(mapping,dict):raise ValueError('Display mapping must be an object')
    result['display_mapping']={k:float(mapping[k]) for k in ('low_k','high_k','low_gray','high_gray','clipped_fraction','kelvin_per_unit','display_min_k','display_max_k') if k in mapping and _finite(mapping[k])}
    result['display_mapping'].update({k:str(mapping[k])[:200] for k in ('method','input_unit','output_unit','lut_sha256','native_encoding','display') if k in mapping and isinstance(mapping[k],str)})
    appearance=context.get('appearance',{})
    if not isinstance(appearance,dict):raise ValueError('Appearance provenance must be an object')
    result['appearance']={k:copy.deepcopy(appearance[k]) for k in ('kind','source_sha256','source_kind','screening_status') if k in appearance and isinstance(appearance[k],str)}
    if appearance.get('kind')=='recorded_texture':
        result['appearance']['recorded_pixels_used_as_texture_only']=True
        result['appearance']['background_identity_status']='not_verified_target_free'
    # Numerical admission is not physical field validation. These are declared
    # upstream receipts, never inferred from image appearance or provider output.
    field=context.get('field_validation',{})
    result['field_validation']={k:str(field[k])[:200] for k in ('numerical_status','field_status','receipt_sha256') if isinstance(field,dict) and k in field and isinstance(field[k],str)}
    result['validity']={'pixel_source':kind,'research_validated':False,'interpretation':'Reviewed recorded-image evidence; pose-independent replay' if kind=='recorded_reference' else 'Direct field display; numerical/field validation follows bound upstream receipts' if kind=='dumux_field_crop' else 'Simulator camera demonstration; thermal world is not field-validated','radiometric_temperature_verified':False}
    return result


def _immutable_bytes(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    try:
        with path.open('xb') as stream:stream.write(payload)
    except FileExistsError:
        if path.read_bytes()!=payload:raise ValueError('Immutable evidence artifact identity collision')


def _image_info(path):
    raw=Path(path).read_bytes()
    with Image.open(Path(path)) as im:
        if im.mode not in ('L','RGB','RGBA','P'):raise ValueError('Detector needs an explicitly mapped 8-bit display image; high-bit thermal fields must not be silently truncated')
        if min(im.size)<1 or max(im.size)>8192:raise ValueError('Unsupported observation dimensions')
        im.load();info={'width':im.width,'height':im.height,'mode':im.mode,'format':im.format}
    return raw,info


def _path_ref(path,root):
    path=Path(path).resolve();base=data_root().resolve()
    return path.relative_to(base).as_posix() if path.is_relative_to(base) else path.relative_to(root.resolve()).as_posix()


def _overlay(raw_path,detections,path,provisional=True,local_reference=False):
    with Image.open(raw_path) as source:im=source.convert('RGB')
    draw=ImageDraw.Draw(im);w,h=im.size
    for detection in detections:
        box=detection['bbox'];accepted=not local_reference and detection['confidence']>=.6;color='#36e280' if accepted else '#ffbc57'
        bounds=(box['x_min']*w/1000,box['y_min']*h/1000,box['x_max']*w/1000,box['y_max']*h/1000)
        draw.rectangle(bounds,outline=color,width=2);text=('candidate ' if accepted else 'uncertain ')+f"{detection['confidence']:.2f}"
        x,y=bounds[:2];draw.text((max(0,x),max(0,y-12)),text,fill=color)
    footer='LOCAL CONTRAST | uncalibrated' if local_reference else 'PROVISIONAL | simulator' if provisional else 'PROVISIONAL | recorded reference'
    draw.rectangle((0,max(0,h-17),min(w,245),h),fill='#101824');draw.text((4,max(0,h-14)),footer,fill='white')
    path.parent.mkdir(parents=True,exist_ok=True);im.save(path,format='PNG')


async def _maybe(function,*args,**kwargs):
    if inspect.iscoroutinefunction(function):return await function(*args,**kwargs)
    result=await asyncio.to_thread(function,*args,**kwargs)
    return await result if inspect.isawaitable(result) else result


def _lease_receipt(decision,provider):
    if not isinstance(decision,dict) or decision.get('allowed') is not True or decision.get('global_quota_coordinated') is not True:return None
    if not isinstance(decision.get('lease_id'),str) or not decision['lease_id']:return None
    rpm=decision.get('allowance_rpm')
    if not _finite(rpm) or not 0<rpm<=120:return None
    try:expires=datetime.fromisoformat(decision['expires_at'].replace('Z','+00:00'))
    except (KeyError,TypeError,ValueError):return None
    if expires.tzinfo is None or expires<=datetime.now(timezone.utc):return None
    # Every retry must pass a rate limiter confined to this allocated slice.
    actual=getattr(getattr(provider,'limiter',None),'rpm',None)
    if not _finite(actual) or actual>rpm:return None
    return {'lease_id':decision['lease_id'][:200],'allowance_rpm':float(rpm),'expires_at':expires.isoformat(),'global_quota_coordinated':True,'provider_limiter_rpm':float(actual)}


def local_thermal_proposals(image_path,*,max_proposals=12):
    """Pixel-only reference proposals; scores are uncalibrated contrast, not mines."""
    import numpy as np
    from scipy.ndimage import gaussian_filter,label,find_objects
    if type(max_proposals) is not int or not 1<=max_proposals<=64:raise ValueError('Bounded proposal count required')
    raw,info=_image_info(image_path)
    with Image.open(image_path) as source:pixels=np.asarray(source.convert('L'),dtype=np.float32)
    height,width=pixels.shape;sigma=max(3.,min(height,width)*.025)
    residual=pixels-gaussian_filter(pixels,sigma=sigma,mode='reflect')
    noise=float(np.median(np.abs(residual-np.median(residual)))*1.4826)
    threshold=max(6.,3.5*noise);minimum=max(4,int(width*height*.000015));maximum=max(minimum,int(width*height*.015));proposals=[]
    for polarity,name in ((1,'warmer'),(-1,'colder')):
        components,n=label(polarity*residual>threshold)
        for index,slices in enumerate(find_objects(components),1):
            if slices is None:continue
            ys,xs=slices;mask=components[slices]==index;area=int(mask.sum());bh=ys.stop-ys.start;bw=xs.stop-xs.start
            if not minimum<=area<=maximum or max(bw/width,bh/height)>.18:continue
            if min(xs.start,ys.start,width-xs.stop,height-ys.stop)<2:continue
            if max(bw,bh)/max(1,min(bw,bh))>5 or area/(bw*bh)<.08:continue
            contrast=float(np.median(np.abs(residual[slices][mask])));strength=float(1-np.exp(-contrast/threshold))
            bbox={'x_min':max(0,int((xs.start-1)*1000/width)),'y_min':max(0,int((ys.start-1)*1000/height)),'x_max':min(1000,int(np.ceil((xs.stop+1)*1000/width))),'y_max':min(1000,int(np.ceil((ys.stop+1)*1000/height)))}
            proposals.append({'bbox':bbox,'confidence':round(strength,4),'confidence_kind':'uncalibrated_contrast_score','label':'local thermal contrast proposal','polarity':name,'evidence':f'{name} compact component; median display contrast {contrast:.1f}/255, area {area}px; no object identity inferred','area_px':area,'contrast_gray':round(contrast,3)})
    proposals.sort(key=lambda p:(-p['confidence'],-p['area_px'],p['bbox']['x_min'],p['bbox']['y_min']))
    return {'status':'analyzed_reference','boxes':proposals[:max_proposals],'uncertain_boxes':[],
        'model':'local_thermal_contrast','detector_version':'adaptive-highpass-components-v1',
        'provenance':'pixel_only_heuristic_not_validated_mine_detection','image_sha256':_sha(raw),
        'image':info,'confidence_kind':'uncalibrated_contrast_score','zero_candidates':None,
        'physical_mine_confirmed':False,'human_reviewed':False,'paid_request_issued':False,
        'parameters':{'background_sigma_px':sigma,'robust_noise_gray':noise,'threshold_gray':threshold,'minimum_component_px':minimum,'maximum_component_px':maximum,'max_proposals':max_proposals},
        'limitation':'Compact contrast proposals can be rocks, vegetation, shadows or sensor artifacts; no proposals does not establish absence.'}


async def analyze_observation(image_path,context,*,provider=None,request_authorizer=None,output_dir=None,prompt_version=PROMPT_VERSION):
    """Exact ER2 output or an explicit cache/lease wait; never creates a client."""
    if prompt_version not in PROMPTS:raise ValueError('Registered detector prompt version required')
    public=public_context(context);raw,info=_image_info(image_path);sha=_sha(raw)
    if context.get('image_sha256') not in (None,sha):raise ValueError('Capture receipt image SHA mismatch')
    root=Path(output_dir or data_root()/'missions/evidence').resolve();root.mkdir(parents=True,exist_ok=True)
    prompt=PROMPTS[prompt_version];metadata=prompt_metadata(prompt)
    if public['sensor_modality']=='rgb':
        prompt=prompt.replace('ORIGINAL thermal drone frame','ORIGINAL RGB simulator camera frame').replace('ORIGINAL thermal frame','ORIGINAL RGB simulator camera frame').replace('both hotter and colder anomalies','both light and dark local contrasts')
        metadata={'label_prompt_version':'mission-rgb-derived:'+prompt_version,'label_prompt_sha256':_sha(prompt.encode()),'base_label_prompt_version':prompt_version}
    key=digest({'image_sha256':sha,'model':MODEL,'prompt_sha256':metadata['label_prompt_sha256']})
    evidence_id=digest({'context':public,'image_sha256':sha,'inference_key':key})[:32]
    suffix={'PNG':'.png','JPEG':'.jpg','WEBP':'.webp'}.get(info['format'])
    if suffix is None:raise ValueError('PNG/JPEG/WEBP observation required')
    image=root/'raw'/(sha+suffix);_immutable_bytes(image,raw)
    base={'evidence_id':evidence_id,'observation_id':public['observation_id'],'mission_id':public['mission_id'],'image_sha256':sha,'raw_image':_path_ref(image,root),'image':info,'context':public,'requested_model':MODEL,**metadata,'boxes':[],'uncertain_boxes':[],'zero_candidates':None,'model':None,'human_reviewed':False,'physical_mine_confirmed':False,'candidate_identity':'provisional','detection_status':'unavailable'}
    cache=root/'inferences'/(key+'.json');paid=root/'private/paid'/(key+'.json');attempt=root/'private/attempts'/(key+'.json')
    def deferred(status,reason,called=False):
        result={**base,'status':status,'reason':reason,'analyzed':False,'paid_request_issued':called};save(root/'observations'/(evidence_id+'.json'),result);return result
    async with _LOCKS.setdefault((str(root),key),asyncio.Lock()):
        import fcntl
        lockfile=root/'locks'/(key+'.lock');lockfile.parent.mkdir(parents=True,exist_ok=True);fd=os.open(lockfile,os.O_CREAT|os.O_RDWR,0o600)
        try:
            try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return deferred('analysis_pending','These exact pixels are being analyzed by another process')
            stored=None;raw_log=None;origin=None;lease=None;called=False
            if cache.exists():stored=json.loads(cache.read_text());origin='mission_cache'
            elif paid.exists():stored=json.loads(paid.read_text());origin='retained_paid_response'
            elif provider is not None and hasattr(provider,'response_record'):
                raw_log=await _maybe(provider.response_record,ROLE,prompt,[image],model=MODEL)
                if raw_log is not None:
                    if raw_log.get('model')!=MODEL or raw_log.get('image_hashes')!=[sha]:raise ValueError('Provider cache identity mismatch')
                    stored={'model':MODEL,'image_sha256':sha,'prompt_sha256':metadata['label_prompt_sha256'],'output':raw_log.get('output'),'raw_response':raw_log};origin='provider_cache';save(paid,stored)
            if stored is None:
                if provider is None:return deferred('cache_only','No cached output; no provider supplied')
                if request_authorizer is None:return deferred('awaiting_request_lease','No coordinated API allowance; no request dispatched')
                decision=await _maybe(request_authorizer,{'scope':'mission_detection','mission_id':public['mission_id'],'observation_id':public['observation_id'],'image_sha256':sha,'model':MODEL,'inference_key':key,**metadata});lease=_lease_receipt(decision,provider)
                if lease is None:return deferred('awaiting_request_lease','Lease missing/expired/uncoordinated, or provider exceeds its reserved rate')
                if attempt.exists():return deferred('prior_attempt_unresolved','Prior reservation retained; automatic repayment blocked')
                _immutable_bytes(attempt,(json.dumps({'at':utcnow(),'inference_key':key,'image_sha256':sha,'lease':lease,'status':'reserved'})+'\n').encode())
                try:
                    called=True;output=await provider.generate(ROLE,prompt,[image],model=MODEL)
                    raw_log=await _maybe(provider.response_record,ROLE,prompt,[image],model=MODEL) if hasattr(provider,'response_record') else None
                    stored={'model':MODEL,'image_sha256':sha,'prompt_sha256':metadata['label_prompt_sha256'],'output':output,'raw_response':raw_log,'lease':lease,'generation_config':{k:v for k,v in getattr(provider,'cfg',{}).items() if k in ('api_version','thinking_budget','reserve_output_tokens','max_retries')}};origin='live_provider'
                    # The exact model output is durable before validation or overlays.
                    save(paid,stored)
                except Exception as error:
                    save(root/'private/errors'/(key+'.json'),{'at':utcnow(),'error':redacted_error(error),'inference_key':key});return deferred('provider_error','Request failed; provider logs/reservation retained; no automatic retry',True)
            if stored.get('model')!=MODEL or stored.get('image_sha256')!=sha or stored.get('prompt_sha256')!=metadata['label_prompt_sha256']:raise ValueError('Detector cache identity mismatch')
            raw_log=stored.get('raw_response')
            try:
                if raw_log is not None and raw_log.get('model')!=MODEL:raise ValueError('Wrong returned model')
                classification=classify_response(stored.get('output'),.6)
            except (ValueError,TypeError):return deferred('invalid_model_response','Original paid output retained; invalid/refused output is never negative',called)
            if not cache.exists():save(cache,stored)
            overlay=root/'overlays'/(evidence_id+'.png');_overlay(image,stored['output']['detections'],overlay,public['source_kind']!='recorded_reference')
            result={**base,'status':'analyzed','analyzed':True,'detection_status':'available','model':MODEL,'boxes':copy.deepcopy(classification['accepted']),'uncertain_boxes':copy.deepcopy(classification['uncertain']),'zero_candidates':classification['zero_candidates'],'overlay_image':_path_ref(overlay,root),'overlay_geometry':{'width':info['width'],'height':info['height'],'normalized_box_scale':1000,'resized':False},'provenance':'gemini_robotics_er2_visible_image_output','cache_source':origin,'paid_request_issued':called,'provider_log_available':raw_log is not None,'request_lease':lease,'raw_response_retained':paid.exists(),'generation_config':stored.get('generation_config'),'analyzed_at':utcnow()};save(root/'observations'/(evidence_id+'.json'),result);return result
        finally:fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)


def record_observation_evidence(image_path,context,detection,*,output_dir=None):
    """Bind caller-supplied local/frontier proposals to the exact captured pixels.

    This helper issues no model calls. The caller retains any actual provider
    receipt; model/provenance are recorded as caller declarations.
    """
    public=public_context(context);raw,info=_image_info(image_path);sha=_sha(raw)
    if context.get('image_sha256') not in (None,sha) or detection.get('image_sha256') not in (None,sha):raise ValueError('Proposal image SHA differs from captured image')
    if not isinstance(detection.get('model'),str) or not isinstance(detection.get('provenance'),str):raise ValueError('Explicit detector model and provenance required')
    boxes=copy.deepcopy(detection.get('boxes',[]));uncertain=copy.deepcopy(detection.get('uncertain_boxes',[]))
    for candidate in boxes+uncertain:
        if not isinstance(candidate,dict) or not valid_box(candidate.get('bbox')) or not _finite(candidate.get('confidence')) or not 0<=candidate['confidence']<=1:raise ValueError('Invalid proposal geometry or score')
    if len(boxes)+len(uncertain)>128:raise ValueError('Bounded evidence proposals required')
    root=Path(output_dir or data_root()/'missions/evidence').resolve();root.mkdir(parents=True,exist_ok=True)
    evidence_id=digest({'context':public,'image_sha256':sha,'model':detection['model'],'provenance':detection['provenance'],'boxes':boxes,'uncertain_boxes':uncertain})[:32]
    suffix={'PNG':'.png','JPEG':'.jpg','WEBP':'.webp'}.get(info['format'])
    if suffix is None:raise ValueError('PNG/JPEG/WEBP observation required')
    image=root/'raw'/(sha+suffix);_immutable_bytes(image,raw);overlay=root/'overlays'/(evidence_id+'.png')
    _overlay(image,boxes+uncertain,overlay,public['source_kind']!='recorded_reference',local_reference=detection['model']=='local_thermal_contrast')
    result={'evidence_id':evidence_id,'mission_id':public['mission_id'],'observation_id':public['observation_id'],'context':public,
        'image_sha256':sha,'image':info,'raw_image':_path_ref(image,root),'overlay_image':_path_ref(overlay,root),
        'status':'analyzed_reference' if detection['model']=='local_thermal_contrast' else 'analyzed','analyzed':True,
        'boxes':boxes,'uncertain_boxes':uncertain,'model':detection['model'],'provenance':detection['provenance'],
        'confidence_kind':detection.get('confidence_kind','uncalibrated_contrast_score' if detection['model']=='local_thermal_contrast' else 'uncalibrated_model_confidence'),
        'zero_candidates':None,'physical_mine_confirmed':False,'human_reviewed':False,
        'paid_request_issued':detection.get('paid_request_issued') if isinstance(detection.get('paid_request_issued'),bool) else None,'recording_helper_issued_no_requests':True,
        'provider_identity_verified_by_helper':False,'created_at':utcnow(),
        'overlay_geometry':{'width':info['width'],'height':info['height'],'normalized_box_scale':1000,'resized':False}}
    for name in ('label_prompt_version','label_prompt_sha256','detector_version'):
        if isinstance(detection.get(name),str):result[name]=detection[name]
    save(root/'observations'/(evidence_id+'.json'),result);return result


def build_mission_report(mission_id,evidence,*,candidate_tracks=None,output_dir=None):
    """Evidence-linked candidate report, with no inference or invented location."""
    _identifier(mission_id,'mission_id');records=list(evidence);lookup={}
    for record in records:
        if record.get('mission_id')!=mission_id:raise ValueError('Cross-mission evidence is forbidden')
        identity=_identifier(record.get('evidence_id'),'evidence_id')
        if identity in lookup and lookup[identity]!=record:raise ValueError('Conflicting evidence identity')
        lookup[identity]=record
    tracks=[];track_ids=set()
    for candidate in candidate_tracks or []:
        identity=_identifier(candidate.get('candidate_id'),'candidate_id');references=[]
        if identity in track_ids:raise ValueError('Duplicate candidate track identity')
        track_ids.add(identity)
        for item in candidate.get('evidence',[]):
            ref=lookup.get(item.get('evidence_id'))
            if ref is None or ref.get('status') not in ('analyzed','analyzed_reference'):raise ValueError('Candidate cites missing or unanalyzed evidence')
            if not valid_box(item.get('bbox')):raise ValueError('Candidate evidence needs a valid measured image box')
            references.append({'evidence_id':item['evidence_id'],'bbox':copy.deepcopy(item['bbox'])})
        if not references:raise ValueError('Candidate requires captured evidence')
        tracks.append({'candidate_id':identity,'evidence':references,'identity_status':'provisional','notes':str(candidate.get('notes',''))[:1000]})
    report={'schema_version':1,'mission_id':mission_id,'created_at':utcnow(),'status':'candidate_evidence_report',
        'candidate_tracks':tracks,'provisional_track_count':len(tracks),'unique_physical_mines_confirmed':0,
        'observations':[{k:r[k] for k in ('evidence_id','observation_id','image_sha256','raw_image','overlay_image','status','model','provenance','context','boxes','uncertain_boxes','confidence_kind','label_prompt_version','label_prompt_sha256') if k in r} for r in lookup.values()],
        'counts':{'captured_observations':len(lookup),'analyzed_observations':sum(r.get('status') in ('analyzed','analyzed_reference') for r in lookup.values()),'candidate_box_measurements':sum(len(r.get('boxes',[])) for r in lookup.values()),'uncertain_box_measurements':sum(len(r.get('uncertain_boxes',[])) for r in lookup.values())},
        'limitations':['Image proposals are provisional; repeated boxes do not establish distinct physical objects.','Simulator thermal appearance is not field-validated mine detection.','Camera pose is capture provenance; geographic mine coordinates are not inferred.','Deferred/invalid analyses and an empty heuristic proposal set do not establish absence.']}
    if output_dir:save(Path(output_dir)/(mission_id+'-report.json'),report)
    return report


def score_mission_report(report,evidence,private_reference,*,qualified_coverage,completed=True):
    """Evaluator-only scoring. Never call this from a detector/agent tool.

    Image hashes bind projections/annotations to actual captures. Target
    eligibility must be fixed independently of the agent route before the run.
    """
    reference=private_reference;kind=reference.get('reference_kind');scope=reference.get('reference_scope','complete')
    if kind not in ('simulator_private','reviewed_recorded'):raise ValueError('Independent simulator truth or reviewed-image reference required; teacher labels are not independent accuracy')
    if reference.get('frozen_before_run') is not True:raise ValueError('Reference eligibility must be frozen before the agent route')
    if scope not in ('complete','positive_only','inserted_objects_only'):raise ValueError('Explicit reference completeness required')
    if not _finite(qualified_coverage) or not 0<=qualified_coverage<=1:raise ValueError('Coverage must be a measured fraction in [0,1]')
    if kind=='reviewed_recorded' and reference.get('human_reviewed') is not True:raise ValueError('Reviewed-real benchmark requires independent human-reviewed annotations')
    lookup={r['evidence_id']:r for r in evidence};truth={};converted=[];observations=reference.get('observations',{})
    for candidate in report.get('candidate_tracks',[]):
        refs=[]
        for item in candidate.get('evidence',[]):
            record=lookup.get(item.get('evidence_id'))
            if record is None or record.get('mission_id')!=report.get('mission_id'):raise ValueError('Candidate evidence does not belong to this mission')
            if record.get('status') not in ('analyzed','analyzed_reference'):raise ValueError('Unanalyzed evidence cannot score a candidate')
            obs=record['observation_id'];reference_obs=observations.get(obs)
            if reference_obs is None or reference_obs.get('image_sha256')!=record.get('image_sha256'):raise ValueError('Reference image hash differs from actual observation')
            if kind=='reviewed_recorded' and record['context']['source_kind']!='recorded_reference':raise ValueError('Reviewed recorded frames cannot score simulator-camera imagery')
            if not valid_box(item.get('bbox')):raise ValueError('Invalid candidate image box')
            refs.append({'observation_id':obs,'box':item['bbox']})
        converted.append({'candidate_id':candidate['candidate_id'],'evidence':refs})
    for obs,record in observations.items():
        targets=[]
        for target in record.get('targets',[]):
            if not valid_box(target.get('bbox')):raise ValueError('Reference projection needs a valid image box')
            targets.append({'target_id':str(target['target_id']),'box':target['bbox'],'observable':target.get('observable') is True})
        truth[obs]={'truth':targets}
    eligible=reference.get('eligible_target_ids')
    if not isinstance(eligible,list) or len(set(map(str,eligible)))!=len(eligible):raise ValueError('Frozen unique eligible target IDs required')
    metrics=score_submission(converted,truth,eligible,qualified_coverage,completed=completed,observability_verified=reference.get('observability_verified') is True)
    captured={record['observation_id'] for record in evidence if record.get('mission_id')==report.get('mission_id') and record.get('status') in ('analyzed','analyzed_reference') and observations.get(record['observation_id'],{}).get('image_sha256')==record.get('image_sha256')}
    negative_observations=[obs for obs,record in observations.items() if obs in captured and not record.get('targets')]
    alerted={r['observation_id'] for c in converted for r in c['evidence']}
    metrics.update(reference_kind=kind,reference_scope=scope,reference_id=reference.get('reference_id'),negative_scene_count=len(negative_observations),negative_scene_false_alarm_rate=sum(obs in alerted for obs in negative_observations)/len(negative_observations) if negative_observations and scope=='complete' else None,independent_real_mine_accuracy_claim=False,
        interpretation='Agreement with reviewed recorded-image annotations; no independent physical mine confirmation' if kind=='reviewed_recorded' else 'Simulation object-retrieval score; no real mine-detection accuracy claim')
    if scope=='inserted_objects_only':
        metrics.update(unmatched_predictions=metrics['false_positives'],false_positives=None,precision=None,f2=None,reward=None,negative_scene_false_alarm_rate=None,interpretation='Recall of explicitly inserted objects only; unknown background identities prevent closed-world false-positive scoring')
    return metrics


def context_from_capture(image_path,receipt,*,mission_id,observation_id,
                         source_kind='gazebo_camera',sensor_modality='simulated_thermal_intensity',appearance=None):
    """Strict bridge receipt adapter; never substitutes pre-command/current pose.

    The bridge must issue the image and metadata atomically and bind their SHA.
    Unknown receipt schemas are rejected instead of guessing units or frames.
    """
    if not isinstance(receipt,dict) or receipt.get('atomic_capture') is not True:raise ValueError('Atomic image/pose capture receipt required')
    actual=_sha(Path(image_path).read_bytes());expected=receipt.get('image_sha256',receipt.get('sha256'))
    if expected!=actual:raise ValueError('Atomic capture receipt image SHA mismatch')
    pose=receipt.get('pose');stamp=receipt.get('sim_time_s');topic=receipt.get('camera_topic')
    if not isinstance(pose,dict) or not _finite(stamp) or stamp<0 or not isinstance(topic,str) or not topic:raise ValueError('Capture receipt needs explicit pose, sim_time_s and camera_topic')
    context={'mission_id':mission_id,'observation_id':observation_id,'source_kind':source_kind,
        'sensor_modality':sensor_modality,'image_sha256':actual,'sim_time_s':float(stamp),
        'pose':copy.deepcopy(pose),'pose_frame':str(receipt.get('pose_frame','unspecified'))[:200],
        'capture':{'camera_topic':topic,'sim_time_s':float(stamp),'event_id':str(receipt.get('event_id') or digest([actual,stamp,topic])[:24]),'source_image_sha256':actual,'binding':'atomic_image_pose_sha256_verified'}}
    for key in ('bridge','sensor_frame','field_sha256','field_metadata_sha256'):
        if isinstance(receipt.get(key),str):context['capture'][key]=receipt[key]
    if isinstance(receipt.get('camera_mount'),dict):context['camera_mount']=copy.deepcopy(receipt['camera_mount'])
    if isinstance(receipt.get('camera'),dict):context['camera']=copy.deepcopy(receipt['camera'])
    elif isinstance(receipt.get('intrinsics'),dict):
        intrinsic=receipt['intrinsics'];context['camera']={out:intrinsic[key] for key,out in (('fx','fx_px'),('fy','fy_px'),('cx','cx_px'),('cy','cy_px')) if key in intrinsic}
        if _finite(intrinsic.get('hfov_rad')):context['camera']['hfov_deg']=math.degrees(intrinsic['hfov_rad'])
    for stamp_key in ('pose_sim_time_s','pose_time_offset_s'):
        if stamp_key in receipt:context['capture'][stamp_key]=receipt[stamp_key]
    if isinstance(receipt.get('display_mapping'),dict):context['display_mapping']=copy.deepcopy(receipt['display_mapping'])
    if appearance is not None:context['appearance']=copy.deepcopy(appearance)
    # Validate the public part now, while retaining the expected hash for the
    # recorder's independent second check if the source file changes.
    public_context(context);return context
