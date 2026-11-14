"""Deterministic six-sortie thermal-survey campaign with private evaluation.

Public methods expose sensor data and the agent's records only. Ground truth,
signature boxes, and scores are available exclusively through the privileged
evaluator_snapshot method, which the MCP adapter never registers.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import tempfile
import threading

import numpy as np
from PIL import Image

from .renderer import ThermalRenderer, DEMO_FIDELITY
from .evaluation import valid_box, score_submission


DEFAULTS = {
    "width_m":64., "height_m":64., "day_seconds":86400., "max_sorties":6,
    "sortie_seconds":600., "max_actions":128, "turnaround_seconds":1800.,
    "speed_m_s":3., "vertical_speed_m_s":1., "landing_seconds":20.,
    "coverage_phase_edges_s":[0.,36000.,54000.,86400.],
    "camera":{"width":640,"height":512,"hfov_deg":100.,"vfov_deg":82.,"calibrated":False},
}


def _finite(value, label):
    try:
        number = float(value)
    except (ValueError,TypeError):
        raise ValueError(f"{label} must be a finite number") from None
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _protocol_config(config,seed):
    keys = ("width_m","height_m","day_seconds","max_sorties","sortie_seconds","max_actions",
            "turnaround_seconds","speed_m_s","vertical_speed_m_s","landing_seconds","camera",
            "coverage_phase_edges_s")
    return {**{key:config[key] for key in keys},"start_time_s":config.get("start_time_s",21600 if config["day_seconds"]>21600 else 0),
            "sensor_seed":int(seed),"reference_policy":"six_sortie_sweep_v2","signature_protocol":"contrast_0.12k_min3pixels_v1"}


class Campaign:
    def __init__(self, config: dict | None = None, seed: int = 0, output_dir: Path | None = None):
        self._lock = threading.RLock()
        source = copy.deepcopy(config or {})
        self.config = copy.deepcopy(DEFAULTS)
        nested = source.pop("campaign", {})
        self.config.update(source)
        self.config.update(nested)
        self.config["camera"] = {**DEFAULTS["camera"], **source.get("camera",{}), **nested.get("camera",{})}
        for key in ("width_m","height_m","day_seconds","sortie_seconds","speed_m_s","vertical_speed_m_s"):
            self.config[key] = _finite(self.config[key], key)
            if self.config[key] <= 0:
                raise ValueError(f"{key} must be positive")
        for key in ("max_sorties","max_actions"):
            self.config[key] = int(self.config[key])
            if self.config[key] < 1:
                raise ValueError(f"{key} must be positive")
        for key in ("turnaround_seconds","landing_seconds"):
            self.config[key] = _finite(self.config[key],key)
            if self.config[key] < 0:
                raise ValueError(f"{key} cannot be negative")
        for key in ("width","height"):
            self.config["camera"][key] = int(self.config["camera"][key])
            if not 16 <= self.config["camera"][key] <= 2048:
                raise ValueError("Camera dimensions must be between 16 and 2048")
        for key in ("hfov_deg","vfov_deg"):
            value = _finite(self.config["camera"][key], key)
            if not 1 <= value < 175:
                raise ValueError("Camera FOV must be in [1,175) degrees")
        edges = self.config["coverage_phase_edges_s"]
        if len(edges)!=4 or any(not math.isfinite(float(x)) for x in edges) or any(b<=a for a,b in zip(edges,edges[1:])):
            raise ValueError("Three coverage phases need four increasing finite time boundaries")
        self.config["coverage_phase_edges_s"] = [float(x) for x in edges]
        self.output_dir = Path(output_dir or os.environ.get("AEROROZUM_DATA_ROOT", Path(__file__).resolve().parents[1]/"artifacts")).resolve()
        self.seed = int(seed)
        self.reset()

    def reset(self, seed: int | None = None):
        with self._lock:
            return self._reset(seed)

    def _reset(self, seed=None):
        if seed is not None:
            self.seed = int(seed)
        self.id = secrets.token_hex(12)
        self._dir = self.output_dir/self.id
        self._dir.mkdir(parents=True, exist_ok=True)
        self.time_s = _finite(self.config.get("start_time_s",21600 if self.config["day_seconds"]>21600 else 0),"start_time_s")
        if not 0 <= self.time_s < self.config["day_seconds"]:
            raise ValueError("Start time must be within campaign day")
        self.home = {"x_m":self.config["width_m"]/2,"y_m":self.config["height_m"]/2}
        self.pose = {**self.home,"altitude_m":0.,"yaw_deg":0.}
        self.airborne = False
        self.sorties_used = 0
        self.actions_used = 0
        self.sortie_elapsed_s = 0.
        self.total_airborne_s = 0.
        self.next_launch_s = self.time_s
        self.completed = False
        self.completion_reason = None
        self._events = []
        self._observations = {}
        self._candidates = {}
        self._targets = self._load_targets()
        self.renderer = ThermalRenderer(self.config, self.seed, self._targets)
        self._coverage = np.zeros((3,32,32),dtype=bool)
        self._coverage_denominator = np.ones_like(self._coverage,dtype=bool)
        self._observable_ids = {str(t["id"]) for t in self._targets if t.get("observable",True)}
        self._observability_verified = False
        self._observability_manifest = None
        self._load_observability()
        if self.config.get("mode") == "research" and not (
                self.renderer.metadata.get("research_accepted") is True and self._observability_verified):
            raise ValueError("Research mode requires accepted physical fields and a frozen verified observability manifest")
        self._log("reset",{},True)
        return self.status()

    def _load_targets(self):
        path = self.config.get("thermal_field_path") or self.config.get("physics",{}).get("thermal_field_path")
        if path:
            truth_file = Path(path).expanduser().resolve().with_suffix(".truth.json")
            if truth_file.exists():
                content = json.loads(truth_file.read_text())
                targets = content.get("targets",[]) if isinstance(content,dict) else content
            else:
                targets = []
        elif "_targets" in self.config:
            targets = copy.deepcopy(self.config["_targets"])
        elif self.config.get("demo_background_mode", "recorded") != "procedural_legacy" and "demo_target_count" not in self.config:
            # Empty synthetic truth means none injected, not verified mine-free.
            targets = []
        else:
            rng = np.random.default_rng(self.seed)
            targets = [{"id":f"private-{i+1}","x_m":float(rng.uniform(4,self.config["width_m"]-4)),
                        "y_m":float(rng.uniform(4,self.config["height_m"]-4)),
                        "radius_m":float(rng.uniform(.12,.25)),"depth_m":float(rng.uniform(.01,.08)),
                        "material":"demonstration_surrogate"}
                       for i in range(int(self.config.get("demo_target_count",6)))]
        for i,target in enumerate(targets):
            target.setdefault("id",f"private-{i+1}")
            for key,bound in (("x_m",self.config["width_m"]),("y_m",self.config["height_m"])):
                target[key] = _finite(target[key],key)
                if not 0 <= target[key] <= bound:
                    raise ValueError("Private target is outside the configured terrain")
        return targets

    def _load_observability(self):
        path = self.config.get("observability_manifest_path")
        if not path:
            return
        manifest = json.loads(Path(path).read_text())
        if not isinstance(manifest,dict):
            raise ValueError("Invalid observability manifest")
        expected = manifest.get("manifest_sha256")
        content = {k:v for k,v in manifest.items() if k!="manifest_sha256"}
        if expected != _canonical_hash(content):
            raise ValueError("Observability manifest integrity hash does not match")
        field = self.config.get("thermal_field_path") or self.config.get("physics",{}).get("thermal_field_path")
        if not field or self.renderer.metadata.get("research_accepted") is not True:
            raise ValueError("Certified observability requires accepted physical fields")
        field = Path(field).expanduser().resolve()
        for key,p in (("field_sha256",field),("truth_sha256",field.with_suffix(".truth.json")),
                      ("metadata_sha256",field.with_suffix(".json"))):
            if manifest.get(key)!=_file_hash(p):
                raise ValueError("Observability manifest does not match physical field and truth files")
        if manifest.get("protocol_config_sha256")!=_canonical_hash(_protocol_config(self.config,self.seed)):
            raise ValueError("Observability manifest does not match campaign configuration and sensor seed")
        if manifest.get("verified") is not True or manifest.get("protocol")!="feasible_reference_sensor_evidence_v1":
            raise ValueError("Observability manifest has not passed reference verification")
        mask = np.asarray(manifest.get("qualified_cells"),dtype=bool)
        if mask.shape!=(3,32,32) or not mask.any():
            raise ValueError("Observability manifest must contain nonempty three-phase qualified cells")
        ids = {str(t["id"]) for t in self._targets}
        certified = set(manifest.get("observable_target_ids",[]))
        if not certified.issubset(ids):
            raise ValueError("Observability manifest references unknown targets")
        if any(not event.get("ok") for event in manifest.get("reference_trace",[])):
            raise ValueError("Observability reference contains failed actions")
        self._coverage_denominator = mask
        self._observable_ids = certified
        self._observability_verified = True
        self._observability_manifest = manifest

    def _log(self, action, arguments, ok, error=None):
        event = {"index":len(self._events),"time_s":float(self.time_s),"action":action,
                 "arguments":copy.deepcopy(arguments),"ok":bool(ok)}
        if error:
            event["error"] = error
        self._events.append(event)

    def _public_observation(self, record):
        public = {k:copy.deepcopy(v) for k,v in record.items() if k != "truth"}
        public["age_s"] = float(self.time_s-record["time_s"])
        public["fresh"] = public["age_s"] <= 30
        return public

    def status(self):
        with self._lock:
            return self._status()

    def _status(self):
        hour = int(self.time_s//3600); minute = int(self.time_s%3600//60)
        phase = 2*math.pi*(self.time_s/86400-.25)
        metadata = self.renderer.metadata
        research = bool(metadata.get("research_accepted",False)) and self._observability_verified
        return {"id":self.id,"campaign_id":self.id,"mode":"demo" if self.renderer.fidelity==DEMO_FIDELITY else "physical_field",
                "fidelity":self.renderer.fidelity,"validation_state":metadata.get("validation_state","unverified"),
                "research_accepted":research,"time_s":float(self.time_s),"time_hhmm":f"{hour:02d}:{minute:02d}",
                "day_seconds":self.config["day_seconds"],"airborne":self.airborne,"pose":copy.deepcopy(self.pose),
                "sorties_used":self.sorties_used,"max_sorties":self.config["max_sorties"],
                "remaining_sortie_s":float(max(0,self.config["sortie_seconds"]-self.sortie_elapsed_s)) if self.airborne else self.config["sortie_seconds"],
                "next_launch_s":float(self.next_launch_s),"total_airborne_s":float(self.total_airborne_s),
                "actions_used":self.actions_used,"max_actions":self.config["max_actions"],
                "width_m":self.config["width_m"],"height_m":self.config["height_m"],
                "coverage_by_phase":{name:float(self._coverage[i].mean()) for i,name in enumerate(("morning","heating","cooling"))},
                "weather":{"air_temperature_c":round(12+5*math.sin(phase),2),
                           "solar_fraction":round(max(0,math.sin(phase)),3),
                           "moisture_index":round(.2+.03*math.cos(phase),3),
                           "source":"illustrative_weather_not_measured"},
                "observations":[self._public_observation(o) for o in self._observations.values()],
                "candidates":copy.deepcopy(list(self._candidates.values())),"completed":self.completed,
                "completion_reason":self.completion_reason,
                "message":"Analytic demonstration; research acceptance pending" if not research else "Accepted physical fields and observability protocol"}

    def _advance(self, seconds, *, reserve_return=True, pose_after=None):
        seconds = _finite(seconds,"duration")
        if seconds < 0:
            raise ValueError("Duration cannot be negative")
        if self.time_s+seconds > self.config["day_seconds"]:
            raise ValueError("Action would exceed campaign day")
        if self.airborne:
            destination = pose_after or self.pose
            reserve = (self._travel_seconds(destination,{**self.home,"altitude_m":0.}) + self.config["landing_seconds"]) if reserve_return else 0.
            if self.time_s+seconds+reserve > self.config["day_seconds"]:
                raise ValueError("Insufficient day time including return and landing reserve")
            if self.sortie_elapsed_s+seconds+reserve > self.config["sortie_seconds"]+1e-8:
                raise ValueError("Insufficient sortie battery including return and landing reserve")
            self.sortie_elapsed_s += seconds
            self.total_airborne_s += seconds
        self.time_s += seconds

    def _travel_seconds(self, start, end):
        horizontal = math.hypot(end["x_m"]-start["x_m"],end["y_m"]-start["y_m"])/self.config["speed_m_s"]
        vertical = abs(end["altitude_m"]-start["altitude_m"])/self.config["vertical_speed_m_s"]
        return max(horizontal,vertical)

    def _mark_coverage(self):
        ys = (np.arange(32)+.5)/32*self.config["height_m"]
        xs = (np.arange(32)+.5)/32*self.config["width_m"]
        xx,yy = np.meshgrid(xs,ys)
        angle = math.radians(self.pose["yaw_deg"]); c,s = math.cos(angle),math.sin(angle)
        dx,dy = xx-self.pose["x_m"],yy-self.pose["y_m"]
        fw,fh = self.renderer.footprint(self.pose)
        if fw/self.config["camera"]["width"] <= .05:
            edges = self.config["coverage_phase_edges_s"]
            phase = int(np.searchsorted(edges,self.time_s,side="right")-1)
            if 0 <= phase < 3:
                self._coverage[phase] |= (np.abs(dx*c-dy*s)<=fw/2)&(np.abs(dx*s+dy*c)<=fh/2)

    def _capture(self):
        if not self.airborne:
            raise ValueError("Launch before capturing a thermal observation")
        # Preflight field coverage before advancing state or writing output.
        self.renderer.temperature(self.pose["x_m"],self.pose["y_m"],self.time_s+2)
        self._advance(2.)
        number = len(self._observations)
        oid = hashlib.sha256(f"{self.id}:{number}".encode()).hexdigest()[:20]
        relative = Path(self.id)/"observations"/f"{oid}.png"
        info = self.renderer.render(self.pose,self.time_s,self.output_dir/relative,number)
        record = {"id":oid,"time_s":float(self.time_s),"image":relative.as_posix(),
                  "pose":copy.deepcopy(self.pose),**info,
                  "truth":self.renderer.evidence_targets(self.pose,self.time_s)}
        self._observations[oid] = record
        self._mark_coverage()
        return self._public_observation(record)

    def act(self, action: str, arguments: dict | None = None):
        with self._lock:
            return self._act(action,arguments)

    def _act(self, action, arguments=None):
        if not isinstance(action,str):
            return {"ok":False,"error":"Action must be a string","status":self.status()}
        arguments = copy.deepcopy(arguments if arguments is not None else {})
        if not isinstance(arguments,dict):
            return {"ok":False,"error":"Action arguments must be an object","status":self.status()}
        try:
            json.dumps(arguments,allow_nan=False)
        except (TypeError,ValueError):
            return {"ok":False,"error":"Action arguments must contain finite JSON values","status":self.status()}
        allowed = {"launch":{"altitude_m"},"move_to":set(self.pose),"capture":set(),
                   "inspect_observation":{"observation_id"},"inspect_crop":{"observation_id","box"},
                   "wait_until":{"time_s"},"land":set(),
                   "record_candidate":{"candidate_id","evidence","confidence","notes"},"submit_report":set()}
        if action in allowed and set(arguments)-allowed[action]:
            return {"ok":False,"error":"Unexpected action arguments; campaign configuration is immutable","status":self.status()}
        if self.completed:
            return {"ok":False,"error":"Campaign is complete; reset to start another","status":self.status()}
        if self.actions_used >= self.config["max_actions"]:
            return {"ok":False,"error":"Action budget exhausted","status":self.status()}
        self.actions_used += 1
        extra = {}
        try:
            if action == "launch":
                if self.airborne:
                    raise ValueError("Already airborne")
                if self.sorties_used >= self.config["max_sorties"]:
                    raise ValueError("All six-sortie campaign allocations have been used")
                if self.time_s < self.next_launch_s:
                    raise ValueError("Turnaround is not complete; wait before next launch")
                altitude = _finite(arguments.get("altitude_m",6),"altitude_m")
                if not 2 <= altitude <= 30:
                    raise ValueError("Altitude must be between 2 and 30 metres")
                duration = 20+altitude/self.config["vertical_speed_m_s"]
                if duration+altitude/self.config["vertical_speed_m_s"]+self.config["landing_seconds"] > self.config["sortie_seconds"]:
                    raise ValueError("Sortie battery cannot support launch and landing")
                if self.time_s+duration+altitude/self.config["vertical_speed_m_s"]+self.config["landing_seconds"] > self.config["day_seconds"]:
                    raise ValueError("Insufficient day time for launch and landing")
                self._advance(duration)
                self.airborne = True; self.sorties_used += 1
                self.sortie_elapsed_s = duration; self.total_airborne_s += duration
                self.pose = {**self.home,"altitude_m":altitude,"yaw_deg":0.}
            elif action == "move_to":
                if not self.airborne:
                    raise ValueError("Launch before moving")
                pose = {key:_finite(arguments.get(key,self.pose[key]),key) for key in self.pose}
                if not 0 <= pose["x_m"] <= self.config["width_m"] or not 0 <= pose["y_m"] <= self.config["height_m"]:
                    raise ValueError("Waypoint is outside terrain bounds")
                if not 2 <= pose["altitude_m"] <= 30:
                    raise ValueError("Altitude must be between 2 and 30 metres")
                pose["yaw_deg"] %= 360
                self._advance(self._travel_seconds(self.pose,pose)+2.,pose_after=pose)
                self.pose = pose
            elif action == "capture":
                extra["observation"] = self._capture()
            elif action in ("inspect_observation","inspect_crop"):
                record = self._observations.get(arguments.get("observation_id"))
                if not record:
                    raise ValueError("Unknown observation ID")
                if action == "inspect_crop":
                    box = arguments.get("box")
                    if not valid_box(box):
                        raise ValueError("Crop needs a normalized named box in 0..1000")
                    name = hashlib.sha256(json.dumps(box,sort_keys=True).encode()).hexdigest()[:12]
                    relative = Path(self.id)/"crops"/f"{record['id']}-{name}.png"
                    with Image.open(self.output_dir/record["image"]) as im:
                        w,h = im.size
                        crop = im.crop((math.floor(box["x_min"]*w/1000),math.floor(box["y_min"]*h/1000),
                                        math.ceil(box["x_max"]*w/1000),math.ceil(box["y_max"]*h/1000)))
                        self._advance(5.)
                        (self.output_dir/relative).parent.mkdir(parents=True,exist_ok=True)
                        crop.save(self.output_dir/relative)
                    extra["crop"] = {"image":relative.as_posix(),"observation_id":record["id"],"box":box,
                                      "note":"Crop of recorded pixels; no added physical resolution"}
                else:
                    self._advance(5.)
                    extra["observation"] = self._public_observation(record)
            elif action == "wait_until":
                when = _finite(arguments.get("time_s"),"time_s")
                if when <= self.time_s:
                    raise ValueError("Wait time must be in the future")
                self._advance(when-self.time_s)
            elif action == "land":
                if not self.airborne:
                    raise ValueError("Already on the ground")
                home = {**self.home,"altitude_m":0.,"yaw_deg":self.pose["yaw_deg"]}
                self._advance(self._travel_seconds(self.pose,home)+self.config["landing_seconds"],reserve_return=False)
                self.pose = home; self.airborne = False
                self.next_launch_s = self.time_s+self.config["turnaround_seconds"]
            elif action == "record_candidate":
                evidence = arguments.get("evidence",[])
                if not isinstance(evidence,list) or not evidence or len(evidence)>32:
                    raise ValueError("Candidate needs between 1 and 32 observation/box evidence records")
                for item in evidence:
                    if not isinstance(item,dict) or item.get("observation_id") not in self._observations or not valid_box(item.get("box")):
                        raise ValueError("Candidate evidence must reference recorded observations and valid normalized boxes")
                cid = arguments.get("candidate_id") or f"CAND-{len(self._candidates)+1:03d}"
                if not isinstance(cid,str) or len(cid)>80:
                    raise ValueError("Invalid candidate ID")
                confidence = _finite(arguments.get("confidence",.5),"confidence")
                if not 0 <= confidence <= 1:
                    raise ValueError("Confidence must be in [0,1]")
                self._advance(5.)
                self._candidates[cid] = {"id":cid,"confidence":confidence,"evidence":evidence,
                                         "notes":str(arguments.get("notes",""))[:2000]}
                extra["candidate_id"] = cid
            elif action == "submit_report":
                self.completed = True; self.completion_reason = "submitted"
                extra["submission"] = {"accepted":True,"candidate_count":len(self._candidates),
                                        "note":"Scoring is available only to the isolated evaluator"}
            else:
                raise ValueError("Unknown campaign action")
            self._log(action,arguments,True)
            if self.actions_used >= self.config["max_actions"] and not self.completed:
                self.completed = True; self.completion_reason = "action_budget_exhausted"
            if self.time_s >= self.config["day_seconds"] and not self.completed:
                self.completed = True; self.completion_reason = "day_ended"
            return {"ok":True,"status":self.status(),**extra}
        except (ValueError,TypeError,KeyError,OverflowError) as exc:
            error = str(exc)
            self._log(action,arguments,False,error)
            if self.actions_used >= self.config["max_actions"]:
                self.completed = True; self.completion_reason = "action_budget_exhausted"
            return {"ok":False,"error":error,"status":self.status()}

    def evaluator_snapshot(self):
        with self._lock:
            return self._evaluator_snapshot()

    def _evaluator_snapshot(self):
        denominator = int(self._coverage_denominator.sum())
        coverage = float((self._coverage & self._coverage_denominator).sum()/denominator) if denominator else 0.
        score = score_submission(list(self._candidates.values()),self._observations,self._observable_ids,
                                 coverage,completed=self.completed,observability_verified=self._observability_verified)
        return {"status":self.status(),"targets":copy.deepcopy(self._targets),"coverage":coverage,
                "score":score,"events":copy.deepcopy(self._events),
                "observations":copy.deepcopy(list(self._observations.values())),
                "observability":{"verified":self._observability_verified,
                                 "protocol":"feasible_reference_sensor_evidence_v1" if self._observability_verified else "unverified_demo_presence_denominator",
                                 "manifest_sha256":self._observability_manifest.get("manifest_sha256") if self._observability_manifest else None,
                                 "qualified_cells_by_phase":{name:int(self._coverage_denominator[i].sum()) for i,name in enumerate(("morning","heating","cooling"))},
                                 "evaluable_target_ids":sorted(self._observable_ids),
                                 "note":"Presence alone is not proof of thermal observability; unverified scores cannot support research acceptance"}}


def baseline_policy(campaign: Campaign, *, include_detection=False):
    """Reproducible observation-only sweep; never reads evaluator truth.

    Optional detector must be supplied as a callable accepting a public
    observation and returning normalized boxes. There is no oracle detector.
    """
    events = []
    for sortie in range(campaign.config["max_sorties"]):
        status = campaign.status()
        if status["completed"]:
            break
        schedule = (21600,28800,39600,46800,57600,64800)
        when = max(status["time_s"],status["next_launch_s"],schedule[sortie % len(schedule)])
        if when >= status["day_seconds"]:
            break
        if when > status["time_s"]:
            events.append(campaign.act("wait_until",{"time_s":when}))
        launch = campaign.act("launch",{"altitude_m":8})
        events.append(launch)
        if not launch["ok"]:
            break
        width,height = campaign.config["width_m"],campaign.config["height_m"]
        for x,y in ((.2,.2),(.5,.2),(.8,.2),(.8,.5),(.5,.5),(.2,.5),(.2,.8),(.5,.8),(.8,.8)):
            move = campaign.act("move_to",{"x_m":width*x,"y_m":height*y})
            events.append(move)
            if not move["ok"] or campaign.completed:
                break
            capture = campaign.act("capture")
            events.append(capture)
            if capture["ok"] and callable(include_detection):
                observation = capture["observation"]
                for box in include_detection(observation):
                    events.append(campaign.act("record_candidate",{"evidence":[{"observation_id":observation["id"],"box":box}]}))
            if campaign.completed:
                break
        if not campaign.completed:
            events.append(campaign.act("land"))
    if not campaign.completed:
        events.append(campaign.act("submit_report"))
    return {"status":campaign.status(),"actions":len(events),"ok":all(e["ok"] for e in events)}


def certify_observability(config: dict, output_path: Path, *, seed: int = 0, output_dir: Path | None = None):
    """Run a real feasible public reference and freeze its private sensor labels.

    This certifies only a declared sensor-evidence protocol, not that thermal
    imagery uniquely identifies physical mines. Accepted solver metadata is a
    prerequisite. The demonstrated baseline never receives evaluator labels.
    """
    config = copy.deepcopy(config)
    config.pop("observability_manifest_path",None)
    if isinstance(config.get("campaign"),dict):
        config["campaign"].pop("observability_manifest_path",None)
    temporary = tempfile.TemporaryDirectory(prefix="aerorozum-certification-") if output_dir is None else None
    try:
        campaign = Campaign(config,seed=seed,output_dir=output_dir or Path(temporary.name))
        field = campaign.config.get("thermal_field_path") or campaign.config.get("physics",{}).get("thermal_field_path")
        if not field or campaign.renderer.metadata.get("research_accepted") is not True:
            raise ValueError("Reference certification requires explicitly accepted physical field metadata")
        outcome = baseline_policy(campaign)
        if not outcome["ok"] or not campaign.completed or campaign.airborne or campaign.sorties_used!=campaign.config["max_sorties"]:
            raise ValueError("Reference campaign did not complete every feasible sortie and land")
        if not all(campaign._coverage[i].any() for i in range(3)):
            raise ValueError("Reference campaign did not qualify coverage in all three thermal phases")
        sightings = {}
        for observation in campaign._observations.values():
            for target in observation["truth"]:
                if target["observable"]:
                    sightings.setdefault(target["target_id"],[]).append({"observation_id":observation["id"],
                        "time_s":observation["time_s"],"pose":observation["pose"],"box":target["box"],
                        "contrast_k":target["contrast_k"],"pixels":target["pixels"]})
        field = Path(field).expanduser().resolve()
        manifest = {"protocol":"feasible_reference_sensor_evidence_v1","verified":True,
                    "interpretation":"Feasible reference thermal evidence; not unique material identity or real mine confirmation",
                    "field_sha256":_file_hash(field),"truth_sha256":_file_hash(field.with_suffix(".truth.json")),
                    "metadata_sha256":_file_hash(field.with_suffix(".json")),
                    "protocol_config":_protocol_config(campaign.config,seed),
                    "protocol_config_sha256":_canonical_hash(_protocol_config(campaign.config,seed)),
                    "observable_target_ids":sorted(sightings),"reference_evidence":sightings,
                    "qualified_cells":campaign._coverage.tolist(),"reference_trace":copy.deepcopy(campaign._events),
                    "reference_airborne_seconds":campaign.total_airborne_s,
                    "reference_sorties":campaign.sorties_used,"reference_actions":campaign.actions_used,
                    "research_claim":"Conditional on accepted field/camera inputs and the stated observability thresholds"}
        manifest["manifest_sha256"] = _canonical_hash(manifest)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True,exist_ok=True)
        output_path.write_text(json.dumps(manifest,indent=2,allow_nan=False))
        output_path.chmod(0o600)
        return manifest
    finally:
        if temporary is not None:
            temporary.cleanup()
