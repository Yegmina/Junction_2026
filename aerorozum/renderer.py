"""Thermal camera rendering; analytic demonstration imagery is explicitly labelled.

The camera sees a ground temperature field, not hidden object colours. Optional
NPZ inputs follow docs/CONTRACTS.md. World coordinates are east/north, metres.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


DEMO_FIDELITY = "demonstration_not_physics_validated"


def _bilinear(field, xs, ys, x_axis, y_axis):
    xi = np.interp(xs, x_axis, np.arange(len(x_axis)))
    yi = np.interp(ys, y_axis, np.arange(len(y_axis)))
    x0 = np.floor(xi).astype(int); y0 = np.floor(yi).astype(int)
    x1 = np.minimum(x0 + 1, len(x_axis) - 1)
    y1 = np.minimum(y0 + 1, len(y_axis) - 1)
    wx = xi - x0; wy = yi - y0
    return ((1-wx)*(1-wy)*field[y0, x0] + wx*(1-wy)*field[y0, x1]
            + (1-wx)*wy*field[y1, x0] + wx*wy*field[y1, x1])


class ThermalRenderer:
    def __init__(self, config: dict, seed: int, targets: list[dict]):
        self.config = config
        self.seed = int(seed)
        self.targets = targets
        self.camera = config["camera"]
        self.field = None
        self.metadata = {"fidelity": DEMO_FIDELITY, "validation_state": "not_validated"}
        field_path = config.get("thermal_field_path") or config.get("physics", {}).get("thermal_field_path")
        if field_path:
            path = Path(field_path).expanduser().resolve()
            with np.load(path, allow_pickle=False) as data:
                self.field = {k: np.asarray(data[k], dtype=np.float64) for k in
                              ("times_s", "x_m", "y_m", "temperature_k", "moisture")}
            f = self.field
            shape = (len(f["times_s"]), len(f["y_m"]), len(f["x_m"]))
            if f["temperature_k"].shape != shape or f["moisture"].shape != shape:
                raise ValueError("Thermal NPZ arrays do not have [time,y,x] shape")
            for name in ("times_s", "x_m", "y_m"):
                axis = f[name]
                if axis.ndim != 1 or len(axis) < 2 or not np.all(np.diff(axis) > 0):
                    raise ValueError(f"Invalid monotone NPZ axis {name}")
            if not all(np.isfinite(v).all() for v in f.values()):
                raise ValueError("Thermal NPZ contains non-finite values")
            # Finite-volume exports use cell centres; their boundary half-cells
            # belong to the physical domain and are reconstructed constantly.
            x_margin = .5*min(np.diff(f["x_m"])[0],np.diff(f["x_m"])[-1])
            y_margin = .5*min(np.diff(f["y_m"])[0],np.diff(f["y_m"])[-1])
            if (f["x_m"][0]-x_margin > 1e-8 or f["x_m"][-1]+x_margin < config["width_m"]-1e-8
                    or f["y_m"][0]-y_margin > 1e-8 or f["y_m"][-1]+y_margin < config["height_m"]-1e-8):
                raise ValueError("Physical field does not cover the configured terrain")
            sidecar = path.with_suffix(".json")
            if sidecar.exists():
                self.metadata = json.loads(sidecar.read_text())
                self.metadata.setdefault("fidelity", "physical_field_unverified")
                self.metadata.setdefault("validation_state", "not_validated")
            else:
                self.metadata = {"fidelity": "physical_field_unverified", "validation_state": "missing_sidecar"}
        self.fidelity = str(self.metadata["fidelity"])
        self.recorded_background = None
        if self.field is None and config.get("demo_background_mode", "recorded") != "procedural_legacy":
            from .recorded_backgrounds import RecordedBackground
            self.recorded_background = RecordedBackground(config,self.seed)
            self.metadata.update(self.recorded_background.metadata)
            self.metadata['fidelity']=DEMO_FIDELITY
            self.metadata['appearance_source']='recorded_background_composite_unvalidated'
            self.metadata['appearance_is_physical_validation']=False
        elif self.field is None:
            from .appearance import terrain
            self._ground,self._ground_x,self._ground_y,self._appearance_source=terrain(config,self.seed)
            self.metadata['appearance_source']=self._appearance_source
            self.metadata['appearance_is_physical_validation']=False
            from .common import data_root
            profile=Path(config.get('appearance_profile_path',data_root()/'appearance/profile.json'))
            self._display_profile=json.loads(profile.read_text()) if profile.is_file() else {}

    def temperature(self, xs, ys, time_s):
        xs = np.asarray(xs, dtype=float); ys = np.asarray(ys, dtype=float)
        if self.field is not None:
            f = self.field; times = f["times_s"]
            if time_s < times[0] or time_s > times[-1]:
                raise ValueError("Requested time is outside physical-field time coverage")
            j = min(max(1, int(np.searchsorted(times, time_s))), len(times)-1)
            weight = (time_s-times[j-1])/(times[j]-times[j-1])
            a = _bilinear(f["temperature_k"][j-1], xs, ys, f["x_m"], f["y_m"])
            b = _bilinear(f["temperature_k"][j], xs, ys, f["x_m"], f["y_m"])
            return a*(1-weight) + b*weight
        if self.recorded_background is not None:
            # Raw display intensities for compatibility preflight, never kelvin.
            return self.recorded_background.sample(xs,ys)
        # EXPLICIT LEGACY DEMO ONLY: smooth varying ground, harmless clutter, and a time-varying
        # perturbation. This is not a heat-equation solver or research dataset.
        phase = 2*math.pi*(time_s/86400 - 0.39)
        ground = 286 + 7*math.sin(phase) + .32*_bilinear(self._ground,xs,ys,self._ground_x,self._ground_y)
        contrast = 1.5*math.sin(phase-.8)
        for target in self.targets:
            sigma = float(target.get("radius_m", .18)) + 1.5*float(target.get("depth_m", .03))
            ground = ground + contrast*np.exp(-((xs-target["x_m"])**2+(ys-target["y_m"])**2)/(2*sigma*sigma))
        return ground

    def footprint(self, pose):
        height = float(pose["altitude_m"])
        return (2*height*math.tan(math.radians(self.camera["hfov_deg"])/2),
                2*height*math.tan(math.radians(self.camera["vfov_deg"])/2))

    def ground_grid(self, pose, width=None, height=None):
        width = width or self.camera["width"]; height = height or self.camera["height"]
        fw, fh = self.footprint(pose)
        u = ((np.arange(width)+.5)/width-.5)*fw
        v = (.5-(np.arange(height)+.5)/height)*fh
        uu, vv = np.meshgrid(u, v)
        angle = math.radians(pose.get("yaw_deg", 0)); c, s = math.cos(angle), math.sin(angle)
        return pose["x_m"]+uu*c+vv*s, pose["y_m"]-uu*s+vv*c

    def projected_box(self, target, pose):
        fw, fh = self.footprint(pose)
        if fw <= 0 or fh <= 0:
            return None
        dx = target["x_m"]-pose["x_m"]; dy = target["y_m"]-pose["y_m"]
        angle = math.radians(pose.get("yaw_deg", 0)); c,s = math.cos(angle),math.sin(angle)
        u = dx*c-dy*s; v = dx*s+dy*c
        # The box encloses a thermal footprint, which can extend beyond the body.
        radius = float(target.get("signature_radius_m", target.get("radius_m", .18)+1.5*target.get("depth_m", .03)))
        box = {"x_min":1000*(.5+(u-radius)/fw), "x_max":1000*(.5+(u+radius)/fw),
               "y_min":1000*(.5-(v+radius)/fh), "y_max":1000*(.5-(v-radius)/fh)}
        if box["x_max"] <= 0 or box["x_min"] >= 1000 or box["y_max"] <= 0 or box["y_min"] >= 1000:
            return None
        return {k: float(max(0, min(1000, value))) for k,value in box.items()}

    def evidence_targets(self, pose, time_s):
        if self.recorded_background is not None:
            # Unlabelled footage cannot provide synthetic positive/negative gold.
            return []
        result = []
        for target in self.targets:
            box = self.projected_box(target, pose)
            if not box:
                continue
            pixel_size = min((box["x_max"]-box["x_min"])*self.camera["width"]/1000,
                             (box["y_max"]-box["y_min"])*self.camera["height"]/1000)
            radius = target.get("signature_radius_m", target.get("radius_m", .18)+1.5*target.get("depth_m", .03))
            angles = np.arange(8)*math.pi/4
            center = float(self.temperature(target["x_m"], target["y_m"], time_s))
            ring = self.temperature(target["x_m"]+3*radius*np.cos(angles),
                                    target["y_m"]+3*radius*np.sin(angles), time_s)
            contrast = abs(center-float(np.median(ring)))
            visibility = target.get("observable", True) and pixel_size >= 3 and contrast >= .12
            result.append({"target_id":str(target["id"]), "box":box, "observable":bool(visibility),
                           "contrast_k":float(contrast), "pixels":float(pixel_size)})
        return result

    def render(self, pose, time_s, output: Path, observation_index=0):
        xs, ys = self.ground_grid(pose)
        if self.recorded_background is not None:
            pixels=self.recorded_background.sample(xs,ys)
            inside=((xs>=0)&(xs<=self.config['width_m'])&(ys>=0)&(ys<=self.config['height_m']))
            pixels=np.clip(np.rint(pixels),0,255).astype(np.uint8)
            pixels[~inside]=4
            image=Image.fromarray(pixels).convert('RGB')
            output.parent.mkdir(parents=True,exist_ok=True)
            image.save(output,format='PNG')
            provenance={**self.recorded_background.metadata,'fidelity':self.fidelity,'pose':pose,'time_s':float(time_s),'camera':self.camera}
            output.with_suffix('.source.json').write_text(json.dumps(provenance,indent=2))
            return {'width':int(image.width),'height':int(image.height),'fidelity':self.fidelity,
                    'sensor_model':'recorded_raw_thermal_pixels_quilted_and_pose_resampled',
                    'temperature_measurement':'display_only_not_radiometric',
                    'background_source':'recorded_background_composite_unvalidated',
                    'ground_truth':'unknown_recorded_background','synthetic_targets_injected':False,
                    'source_frame_count':self.recorded_background.metadata['unique_source_frames_used'],
                    'ground_sample_distance_m':float(self.footprint(pose)[0]/image.width)}
        field = self.temperature(xs, ys, time_s)
        inside = ((xs>=0)&(xs<=self.config["width_m"])&(ys>=0)&(ys<=self.config["height_m"]))
        low, high = np.percentile(field[inside] if inside.any() else field, (2,98))
        span = max(float(high-low), .5)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed & 0xffffffff, observation_index]))
        sensor = field + rng.normal(0, .04, field.shape)
        display_span,display_center=230.,127.
        if self.field is None and self._display_profile:
            display_span=float(self._display_profile['display_contrast_p02_p98'])*255
            display_center=float(self._display_profile.get('display_median',.5))*255
        pixels = np.clip(((sensor-low)/span-.5)*display_span+display_center, 0,255).astype(np.uint8)
        pixels[~inside] = 4
        image = Image.fromarray(pixels).filter(ImageFilter.GaussianBlur(.65)).convert("RGB")
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output, format="PNG")
        return {"width":int(image.width), "height":int(image.height), "fidelity":self.fidelity,
                "sensor_model":"nominal_nadir_planar_camera_with_agc_noise_blur",
                "temperature_measurement":"display_only_not_radiometric",
                "ground_sample_distance_m":float(self.footprint(pose)[0]/image.width)}
