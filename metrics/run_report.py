
"""
metrics/run_report.py
---------------------
Mission-run evidence recorder for the FSOC PAT prototype.

Captures a bounded, judge-readable time series for a user-defined START -> STOP
window and exports:
  * JSON evidence bundle
  * summary CSV
  * frame-by-frame CSV
  * printable HTML report with inline SVG charts
  * optional PDF report when ReportLab is installed
"""
from __future__ import annotations

import csv
import html
import json
import math
import os
import statistics
import time
import webbrowser
from pathlib import Path
from typing import Any


LOCK_STATES = {"LOCKED", "DEGRADED_LOCK"}


def _f(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _i(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def _pct(values):
    return round((sum(values) / len(values)) * 100.0, 2) if values else 0.0


def _mean(values):
    vals = [float(v) for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _rms(values):
    vals = [float(v) for v in values if v is not None]
    return math.sqrt(sum(v * v for v in vals) / len(vals)) if vals else None


def _max(values):
    vals = [float(v) for v in values if v is not None]
    return max(vals) if vals else None


def _p95(values):
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None
    return vals[min(len(vals) - 1, max(0, int(math.ceil(0.95 * len(vals))) - 1))]


class MissionRunRecorder:
    """Capture one explicit START/STOP evidence window and retain recent runs."""

    def __init__(self, log_dir="logs", history_limit=24):
        self.log_dir = os.path.abspath(str(log_dir))
        os.makedirs(self.log_dir, exist_ok=True)
        self.history_limit = int(max(4, history_limit))
        self.active = False
        self.samples = []
        self.meta = {}
        self.started_wall = None
        self.started_sim = None
        self.stopped_wall = None
        self.stopped_sim = None
        self.latest_report = None
        self.history = []
        self._run_seq = 0
        self._load_history()

    # ------------------------------------------------------------------ state
    @property
    def elapsed_s(self):
        if self.started_wall is None:
            return 0.0
        end = self.stopped_wall if self.stopped_wall is not None else time.time()
        return max(0.0, end - self.started_wall)

    @property
    def latest_html_path(self):
        return (self.latest_report or {}).get("html_path")

    @property
    def latest_pdf_path(self):
        return (self.latest_report or {}).get("pdf_path")

    def _load_history(self):
        items = []
        try:
            for name in os.listdir(self.log_dir):
                if not name.endswith("_report.json"):
                    continue
                path = os.path.join(self.log_dir, name)
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        doc = json.load(fh)
                    if isinstance(doc, dict) and doc.get("summary"):
                        doc["_json_path"] = path
                        items.append(doc)
                except Exception:
                    continue
        except Exception:
            pass
        items.sort(key=lambda x: str(x.get("run_id", "")))
        self.history = items[-self.history_limit:]
        if self.history:
            self.latest_report = self.history[-1]

    # ------------------------------------------------------------------- run
    def start(self, meta=None, sim_time=None):
        if self.active:
            return False
        self._run_seq += 1
        run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime()) + f"_{self._run_seq:02d}"
        self.active = True
        self.samples = []
        self.meta = dict(meta or {})
        self.meta["run_id"] = run_id
        self.meta["started_local"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.started_wall = time.time()
        self.started_sim = _f(sim_time)
        self.stopped_wall = None
        self.stopped_sim = None
        self.latest_report = None
        return True

    def record(self, sim, perf=None, stress_mgr=None):
        if not self.active:
            return
        r = getattr(sim, "last_result", None) or {}
        trk = getattr(sim, "tracker", None)
        unc = getattr(trk, "unc", None)
        trust = getattr(trk, "trust", None)
        conf = getattr(trk, "conf", None)
        scene = getattr(sim, "scene", None)
        beacon = getattr(scene, "beacon", None)
        orbit = getattr(scene, "orbit", None) or getattr(beacon, "orbit", None)
        dist = getattr(sim, "disturbance", None)

        active_stress = []
        stress_levels = {}
        if stress_mgr is not None:
            for key, sc in getattr(stress_mgr, "scenarios", {}).items():
                level = _f(sc.get("level", 0.0))
                if sc.get("active"):
                    active_stress.append(key)
                    stress_levels[key] = round(level, 3)

        sim_t = _f(r.get("t", getattr(sim, "t", 0.0)))
        if self.started_sim is None:
            self.started_sim = sim_t
        if self.meta.get("start_sim_t") is None:
            self.meta["start_sim_t"] = sim_t

        disturbance_values = {
            "turbulence": _f(getattr(dist, "turbulence", 0.0)),
            "vibration": _f(getattr(dist, "vibration", 0.0)),
            "sensor_noise": _f(getattr(dist, "sensor_noise", 0.0)),
            "jerk_prob": _f(getattr(dist, "jerk_prob", 0.0)),
            "beacon_fade": _f(getattr(dist, "beacon_fade", 0.0)),
        }
        disturbance_load = sum(disturbance_values.values()) / max(1, len(disturbance_values))

        tracker_conf = _f(getattr(conf, "overall", r.get("confidence", 0.0)))
        snr = _f(getattr(getattr(trk, "associated", None), "snr", r.get("snr", 0.0)))
        mod_score = _f(getattr(getattr(trk, "associated", None), "mod_score", 0.0))
        prior_resid = _f(getattr(trk, "_prior_resid_deg", 0.0))
        sigma = _f(getattr(unc, "display_sigma_px", 0.0))
        vision_trust = _f(getattr(trust, "vision_trust", 0.0))
        model_trust = _f(getattr(trust, "model_trust", 0.0))

        wall_fps = _f(getattr(perf, "last_tick", 0.0), 0.0) if perf else 0.0
        if wall_fps <= 0:
            wall_fps = _f(getattr(sim, "dt", 0.0))
            wall_fps = (1.0 / wall_fps) if wall_fps > 0 else 0.0

        shape = str(getattr(beacon, "shape", self.meta.get("target_shape", "SQUARE"))).upper()
        motion = str(getattr(orbit, "_motion_type", self.meta.get("trajectory", "straight_line")))

        pointing_deg = r.get("pointing_err_deg")
        pointing_px = r.get("boresight_error_px")
        measurement_valid = bool(r.get("measurement_valid", False))
        visible = bool(r.get("beacon_visible", True))
        aligned = pointing_px is not None and _f(pointing_px) <= 15.0

        sample = {
            "t_s": round(sim_t - _f(self.started_sim), 4),
            "state": str(r.get("state", "SEARCHING")),
            "phase": str(r.get("tracking_phase", r.get("state", "SEARCHING"))),
            "confidence": round(tracker_conf, 5),
            "confidence_pct": round(tracker_conf * 100.0, 2),
            "measurement_valid": measurement_valid,
            "visible": visible,
            "boresight_error_px": None if pointing_px is None else round(_f(pointing_px), 4),
            "pointing_error_deg": None if pointing_deg is None else round(_f(pointing_deg), 6),
            "estimate_error_deg": None if r.get("est_err_deg") is None else round(_f(r.get("est_err_deg")), 6),
            "centroid_error_px": None if r.get("centroid_error_px") is None else round(_f(r.get("centroid_error_px")), 4),
            "snr_db": round(snr, 3),
            "modulation_score": round(mod_score, 4),
            "prediction_residual_deg": round(prior_resid, 6),
            "uncertainty_sigma_px": round(sigma, 4),
            "vision_trust_pct": round(vision_trust * 100.0, 2),
            "model_trust_pct": round(model_trust * 100.0, 2),
            "fps": round(wall_fps, 3),
            "gimbal_pan_deg": round(_f(r.get("gimbal_pan", getattr(getattr(sim, "gimbal", None), "pan", 0.0))), 5),
            "gimbal_tilt_deg": round(_f(r.get("gimbal_tilt", getattr(getattr(sim, "gimbal", None), "tilt", 0.0))), 5),
            "gimbal_v_pan_deg_s": round(_f(r.get("gimbal_v_pan", getattr(getattr(sim, "gimbal", None), "v_pan", 0.0))), 5),
            "gimbal_v_tilt_deg_s": round(_f(r.get("gimbal_v_tilt", getattr(getattr(sim, "gimbal", None), "v_tilt", 0.0))), 5),
            "gimbal_sat_pan_pct": round(_f(r.get("gimbal_sat_pan", getattr(getattr(sim, "gimbal", None), "pan_sat", 0.0))) * 100.0, 2),
            "gimbal_sat_tilt_pct": round(_f(r.get("gimbal_sat_tilt", getattr(getattr(sim, "gimbal", None), "tilt_sat", 0.0))) * 100.0, 2),
            "target_az_deg": None if r.get("truth_az") is None else round(_f(r.get("truth_az")), 5),
            "target_el_deg": None if r.get("truth_el") is None else round(_f(r.get("truth_el")), 5),
            "target_shape": shape,
            "trajectory": motion,
            "platform_mode": str(self.meta.get("platform_mode", "")),
            "preset": str(self.meta.get("preset", "")),
            "atmosphere": str(self.meta.get("atmosphere", "")),
            "target_count": _i(self.meta.get("target_count", getattr(scene, "num_targets", 1)), 1),
            "primary_target_id": str(r.get("primary_target_id", "")),
            "disturbance": disturbance_values,
            "disturbance_load_pct": round(disturbance_load, 2),
            "active_stress": list(active_stress),
            "stress_count": len(active_stress),
            "stress_mode": "NONE" if not active_stress else ("MULTIPLE" if len(active_stress) > 1 else "SINGLE"),
            "stress_levels": stress_levels,
            "alignment_pass": bool(aligned) if measurement_valid else False,
        }
        self.samples.append(sample)

    # ---------------------------------------------------------------- summary
    def _summary(self, samples):
        if not samples:
            return {
                "duration_s": 0.0,
                "frames": 0,
                "fps_mean": 0.0,
                "acquisition_time_s": None,
                "pat_error_mean_px": None,
                "pat_error_rms_px": None,
                "pat_error_p95_px": None,
                "pat_error_max_px": None,
                "pointing_error_mean_deg": None,
                "pointing_error_rms_deg": None,
                "confidence_mean_pct": 0.0,
                "alignment_accuracy_pct": 0.0,
                "lock_retention_pct": 0.0,
                "visible_lock_retention_pct": 0.0,
                "reacquisition_count": 0,
                "mean_reacquisition_s": None,
                "max_uncertainty_px": None,
                "mean_uncertainty_px": None,
                "gimbal_pan_range_deg": 0.0,
                "gimbal_tilt_range_deg": 0.0,
                "gimbal_saturation_pct": 0.0,
                "disturbance_mean_pct": 0.0,
                "disturbance_peak_pct": 0.0,
                "stress_single_frames": 0,
                "stress_multiple_frames": 0,
                "stress_combinations": [],
                "stress_response": {},
                "states": {},
            }

        ts = [_f(s.get("t_s")) for s in samples]
        fps = [_f(s.get("fps")) for s in samples if _f(s.get("fps")) > 0]
        pat_px = [s.get("boresight_error_px") for s in samples]
        pat_px = [v for v in pat_px if v is not None]
        pat_deg = [s.get("pointing_error_deg") for s in samples]
        pat_deg = [v for v in pat_deg if v is not None]
        conf = [_f(s.get("confidence")) for s in samples]
        locks = [s for s in samples if s.get("state") in LOCK_STATES]
        visible = [s for s in samples if s.get("visible")]
        visible_locks = [s for s in visible if s.get("state") in LOCK_STATES]
        aligned = [s for s in samples if s.get("measurement_valid")]
        aligned_pass = [s for s in aligned if s.get("alignment_pass")]
        unc = [_f(s.get("uncertainty_sigma_px")) for s in samples]
        pan = [_f(s.get("gimbal_pan_deg")) for s in samples]
        tilt = [_f(s.get("gimbal_tilt_deg")) for s in samples]
        sat = [max(_f(s.get("gimbal_sat_pan_pct")), _f(s.get("gimbal_sat_tilt_pct"))) for s in samples]
        dload = [_f(s.get("disturbance_load_pct")) for s in samples]

        acq = None
        prev_locked = False
        reacq_start = None
        reacq_times = []
        for s in samples:
            locked = s.get("state") in LOCK_STATES
            t = _f(s.get("t_s"))
            if locked and not prev_locked and acq is None:
                acq = t
            if not locked and prev_locked:
                reacq_start = t
            if locked and not prev_locked and reacq_start is not None:
                reacq_times.append(max(0.0, t - reacq_start))
                reacq_start = None
            prev_locked = locked

        state_counts = {}
        for s in samples:
            state = s.get("state", "SEARCHING")
            state_counts[state] = state_counts.get(state, 0) + 1

        combinations = {}
        response = {}
        single_frames = multi_frames = 0
        for s in samples:
            active = list(s.get("active_stress", []))
            if not active:
                continue
            key = "+".join(sorted(active))
            combinations[key] = combinations.get(key, 0) + 1
            if len(active) == 1:
                single_frames += 1
            else:
                multi_frames += 1
            for stress in active:
                bucket = response.setdefault(stress, {"frames": 0, "lock_frames": 0, "confidence": [], "pat_error_px": [], "fps": [], "uncertainty_px": []})
                bucket["frames"] += 1
                if s.get("state") in LOCK_STATES:
                    bucket["lock_frames"] += 1
                bucket["confidence"].append(_f(s.get("confidence_pct")))
                if s.get("boresight_error_px") is not None:
                    bucket["pat_error_px"].append(_f(s.get("boresight_error_px")))
                bucket["fps"].append(_f(s.get("fps")))
                bucket["uncertainty_px"].append(_f(s.get("uncertainty_sigma_px")))

        stress_response = {}
        for key, b in response.items():
            stress_response[key] = {
                "frames": b["frames"],
                "lock_retention_pct": round(b["lock_frames"] / max(1, b["frames"]) * 100.0, 2),
                "mean_confidence_pct": round(_mean(b["confidence"]) or 0.0, 2),
                "mean_pat_error_px": None if not b["pat_error_px"] else round(_mean(b["pat_error_px"]), 3),
                "mean_fps": round(_mean(b["fps"]) or 0.0, 2),
                "mean_uncertainty_px": None if not b["uncertainty_px"] else round(_mean(b["uncertainty_px"]), 3),
            }

        shape = str(samples[-1].get("target_shape", self.meta.get("target_shape", "SQUARE")))
        trajectory = str(samples[-1].get("trajectory", self.meta.get("trajectory", "straight_line")))

        return {
            "duration_s": round(max(ts[-1] if ts else 0.0, 0.0), 3),
            "frames": len(samples),
            "fps_mean": round(_mean(fps) or 0.0, 2),
            "fps_min": round(min(fps), 2) if fps else 0.0,
            "fps_max": round(max(fps), 2) if fps else 0.0,
            "acquisition_time_s": None if acq is None else round(acq, 3),
            "pat_error_mean_px": None if not pat_px else round(_mean(pat_px), 3),
            "pat_error_rms_px": None if not pat_px else round(_rms(pat_px), 3),
            "pat_error_p95_px": None if not pat_px else round(_p95(pat_px), 3),
            "pat_error_max_px": None if not pat_px else round(_max(pat_px), 3),
            "pointing_error_mean_deg": None if not pat_deg else round(_mean(pat_deg), 6),
            "pointing_error_rms_deg": None if not pat_deg else round(_rms(pat_deg), 6),
            "confidence_mean_pct": round(_mean(conf) * 100.0 if conf else 0.0, 2),
            "alignment_accuracy_pct": round(len(aligned_pass) / max(1, len(aligned)) * 100.0, 2),
            "lock_retention_pct": round(len(locks) / max(1, len(samples)) * 100.0, 2),
            "visible_lock_retention_pct": round(len(visible_locks) / max(1, len(visible)) * 100.0, 2),
            "reacquisition_count": len(reacq_times),
            "mean_reacquisition_s": None if not reacq_times else round(_mean(reacq_times), 3),
            "max_reacquisition_s": None if not reacq_times else round(_max(reacq_times), 3),
            "mean_uncertainty_px": round(_mean(unc) or 0.0, 3),
            "max_uncertainty_px": round(_max(unc) or 0.0, 3),
            "gimbal_pan_range_deg": round((max(pan) - min(pan)) if pan else 0.0, 3),
            "gimbal_tilt_range_deg": round((max(tilt) - min(tilt)) if tilt else 0.0, 3),
            "gimbal_saturation_pct": round(_mean(sat) or 0.0, 2),
            "disturbance_mean_pct": round(_mean(dload) or 0.0, 2),
            "disturbance_peak_pct": round(_max(dload) or 0.0, 2),
            "stress_single_frames": single_frames,
            "stress_multiple_frames": multi_frames,
            "stress_combinations": [{"name": k or "NONE", "frames": v} for k, v in sorted(combinations.items(), key=lambda kv: (-kv[1], kv[0]))],
            "stress_response": stress_response,
            "states": {k: round(v / len(samples) * 100.0, 2) for k, v in sorted(state_counts.items(), key=lambda kv: (-kv[1], kv[0]))},
            "target_shape": shape,
            "trajectory": trajectory,
        }

    def live_summary(self):
        return self._summary(self.samples)

    def stop(self, sim=None, reason="MANUAL STOP"):
        if not self.active:
            return self.latest_report
        if sim is not None:
            self.stopped_sim = _f(getattr(sim, "t", None))
        self.stopped_wall = time.time()
        self.active = False

        summary = self._summary(self.samples)
        summary["wall_duration_s"] = round(self.elapsed_s, 3)
        summary["stop_reason"] = str(reason)

        run_id = str(self.meta.get("run_id", time.strftime("%Y%m%d_%H%M%S")))
        stem = os.path.join(self.log_dir, f"run_{run_id}")
        json_path = stem + "_report.json"
        summary_csv = stem + "_summary.csv"
        series_csv = stem + "_timeseries.csv"
        html_path = stem + "_report.html"
        pdf_path = stem + "_report.pdf"

        doc = {
            "run_id": run_id,
            "meta": dict(self.meta),
            "summary": summary,
            "samples": list(self.samples),
            "exported_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "summary_csv": summary_csv,
            "timeseries_csv": series_csv,
            "html_path": html_path,
            "pdf_path": None,
        }
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)

        self._write_summary_csv(summary_csv, doc)
        self._write_series_csv(series_csv, self.samples)
        self._write_html(html_path, doc)
        if self._write_pdf(pdf_path, doc):
            doc["pdf_path"] = pdf_path
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(doc, fh, indent=2)

        doc["_json_path"] = json_path
        self.latest_report = doc
        self.history.append(doc)
        self.history = self.history[-self.history_limit:]
        return doc

    # --------------------------------------------------------------- exporting
    def _write_summary_csv(self, path, doc):
        summary = doc["summary"]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["metric", "value"])
            for key, value in summary.items():
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False)
                w.writerow([key, value])
            for key, value in doc.get("meta", {}).items():
                w.writerow([f"meta.{key}", value])

    def _write_series_csv(self, path, samples):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            if not samples:
                w = csv.writer(fh)
                w.writerow(["t_s"])
                return
            fields = [
                "t_s", "state", "phase", "confidence_pct", "measurement_valid", "visible",
                "boresight_error_px", "pointing_error_deg", "estimate_error_deg",
                "centroid_error_px", "snr_db", "modulation_score",
                "uncertainty_sigma_px", "vision_trust_pct", "model_trust_pct",
                "fps", "gimbal_pan_deg", "gimbal_tilt_deg", "gimbal_v_pan_deg_s",
                "gimbal_v_tilt_deg_s", "gimbal_sat_pan_pct", "gimbal_sat_tilt_pct",
                "target_az_deg", "target_el_deg", "target_shape", "trajectory",
                "platform_mode", "preset", "atmosphere", "target_count",
                "primary_target_id", "disturbance_load_pct", "stress_mode",
                "stress_count", "active_stress", "alignment_pass",
            ]
            fields += [f"disturbance.{k}" for k in ("turbulence", "vibration", "sensor_noise", "jerk_prob", "beacon_fade")]
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for s in samples:
                row = {k: s.get(k) for k in fields if not k.startswith("disturbance.")}
                for k in ("turbulence", "vibration", "sensor_noise", "jerk_prob", "beacon_fade"):
                    row[f"disturbance.{k}"] = s.get("disturbance", {}).get(k, 0.0)
                row["active_stress"] = "+".join(s.get("active_stress", []))
                w.writerow(row)

    # ------------------------------------------------------------- html charts
    @staticmethod
    def _svg_line(samples, key, title, unit="", width=880, height=220, color="#31d7ff", secondary=None):
        vals = [_f(s.get(key), float("nan")) for s in samples]
        if secondary:
            vals2 = [_f(s.get(secondary), float("nan")) for s in samples]
        else:
            vals2 = []
        allv = [v for v in vals + vals2 if math.isfinite(v)]
        if not allv:
            return f"<div class='chart empty'>{html.escape(title)}<div>No samples captured.</div></div>"
        lo, hi = min(allv), max(allv)
        if abs(hi - lo) < 1e-9:
            pad = max(1.0, abs(hi) * 0.1)
            lo, hi = lo - pad, hi + pad
        left, right, top, bottom = 48, width - 12, 26, height - 32

        def path_for(series):
            pts = []
            n = max(1, len(series) - 1)
            for i, v in enumerate(series):
                if not math.isfinite(v):
                    continue
                x = left + (right-left) * i / n
                y = bottom - (bottom-top) * (v-lo) / (hi-lo)
                pts.append(f"{x:.1f},{y:.1f}")
            return " ".join(pts)

        p1 = path_for(vals)
        p2 = path_for(vals2)
        label_lo = f"{lo:.2f}"
        label_hi = f"{hi:.2f}"
        extra = ""
        if secondary:
            extra = f"<polyline points='{p2}' fill='none' stroke='#ffb84d' stroke-width='2'/>" if p2 else ""
        return (
            f"<div class='chart'><div class='chart-title'>{html.escape(title)}</div>"
            f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>"
            f"<line x1='{left}' y1='{top}' x2='{left}' y2='{bottom}' stroke='#223652'/>"
            f"<line x1='{left}' y1='{bottom}' x2='{right}' y2='{bottom}' stroke='#223652'/>"
            f"<line x1='{left}' y1='{(top+bottom)/2:.1f}' x2='{right}' y2='{(top+bottom)/2:.1f}' stroke='#162844'/>"
            f"<polyline points='{p1}' fill='none' stroke='{color}' stroke-width='2.5'/>"
            f"<text x='6' y='{top+4}' fill='#8ca3bd' font-size='11'>{html.escape(label_hi)} {html.escape(unit)}</text>"
            f"<text x='6' y='{bottom}' fill='#8ca3bd' font-size='11'>{html.escape(label_lo)} {html.escape(unit)}</text>"
            f"<text x='{left}' y='{height-8}' fill='#657f9e' font-size='10'>START</text>"
            f"<text x='{right-30}' y='{height-8}' fill='#657f9e' font-size='10'>STOP</text>"
            f"{extra}</svg></div>"
        )

    def _write_html(self, path, doc):
        meta = doc.get("meta", {})
        s = doc["summary"]
        cards = [
            ("FPS", f"{s.get('fps_mean', 0):.1f}"),
            ("PAT mean error", "n/a" if s.get("pat_error_mean_px") is None else f"{s['pat_error_mean_px']:.2f} px"),
            ("Acquisition", "never" if s.get("acquisition_time_s") is None else f"{s['acquisition_time_s']:.2f} s"),
            ("Tracking confidence", f"{s.get('confidence_mean_pct', 0):.1f}%"),
            ("Alignment accuracy", f"{s.get('alignment_accuracy_pct', 0):.1f}%"),
            ("Lock retention", f"{s.get('lock_retention_pct', 0):.1f}%"),
            ("Reacquisitions", str(s.get("reacquisition_count", 0))),
            ("Peak uncertainty", "n/a" if s.get("max_uncertainty_px") is None else f"{s['max_uncertainty_px']:.2f} px"),
        ]
        card_html = "".join(f"<div class='metric'><div>{html.escape(k)}</div><strong>{html.escape(v)}</strong></div>" for k, v in cards)

        stress_rows = ""
        for key, row in (s.get("stress_response") or {}).items():
            stress_rows += (
                f"<tr><td>{html.escape(key)}</td><td>{row['frames']}</td>"
                f"<td>{row['lock_retention_pct']:.1f}%</td><td>{row['mean_confidence_pct']:.1f}%</td>"
                f"<td>{'n/a' if row['mean_pat_error_px'] is None else f\"{row['mean_pat_error_px']:.2f} px\"}</td>"
                f"<td>{row['mean_uncertainty_px']:.2f} px</td></tr>"
            )
        if not stress_rows:
            stress_rows = "<tr><td colspan='6'>No stress injection occurred during this run.</td></tr>"

        comparison_rows = ""
        for old in self.history[-10:]:
            sm = old.get("summary", {})
            mm = old.get("meta", {})
            comparison_rows += (
                f"<tr><td>{html.escape(str(old.get('run_id','')))}</td>"
                f"<td>{html.escape(str(mm.get('platform_mode','')))}</td>"
                f"<td>{html.escape(str(mm.get('preset','')))}</td>"
                f"<td>{html.escape(str(sm.get('trajectory', mm.get('trajectory',''))))}</td>"
                f"<td>{sm.get('fps_mean',0):.1f}</td>"
                f"<td>{'n/a' if sm.get('pat_error_mean_px') is None else f\"{sm['pat_error_mean_px']:.2f}\"}</td>"
                f"<td>{sm.get('alignment_accuracy_pct',0):.1f}%</td>"
                f"<td>{sm.get('lock_retention_pct',0):.1f}%</td></tr>"
            )

        charts = [
            self._svg_line(doc.get("samples", []), "boresight_error_px", "PAT PERFORMANCE ERROR", "px"),
            self._svg_line(doc.get("samples", []), "confidence_pct", "TRACKING CONFIDENCE", "%"),
            self._svg_line(doc.get("samples", []), "gimbal_pan_deg", "GIMBAL PAN / TILT", "deg", secondary="gimbal_tilt_deg", color="#31d7ff"),
            self._svg_line(doc.get("samples", []), "disturbance_load_pct", "DISTURBANCE LOAD", "%", color="#ff6b8a"),
            self._svg_line(doc.get("samples", []), "uncertainty_sigma_px", "PREDICTION UNCERTAINTY", "px", color="#b8a0ff"),
            self._svg_line(doc.get("samples", []), "snr_db", "SNR", "dB", color="#66e0a3"),
        ]
        chart_html = "".join(charts)

        body = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>FSOC PAT Run {html.escape(str(doc.get('run_id','')))}</title>
<style>
@page{{size:A4;margin:12mm}}*{{box-sizing:border-box}}body{{font-family:Inter,Segoe UI,Arial,sans-serif;background:#07101f;color:#eaf2ff;margin:0;padding:28px}}
h1,h2{{margin:0 0 10px}}h1{{font-size:24px}}h2{{font-size:15px;color:#31d7ff;margin-top:26px}}
.meta{{color:#91a7c1;font-size:12px;margin-bottom:18px}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}}
.metric{{background:#0d1a2d;border:1px solid #1d3453;border-radius:7px;padding:12px}}.metric div{{font-size:10px;color:#8aa0bb;text-transform:uppercase}}.metric strong{{font-size:20px;display:block;margin-top:5px}}
.charts{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}}.chart{{background:#0d1a2d;border:1px solid #1d3453;border-radius:7px;padding:8px}}.chart-title{{font-size:11px;font-weight:700;color:#b9c9dc;margin:2px 0 5px;letter-spacing:.08em}}svg{{width:100%;height:auto}}
table{{width:100%;border-collapse:collapse;background:#0d1a2d;border:1px solid #1d3453;font-size:11px}}th,td{{padding:7px;border-bottom:1px solid #1a2d47;text-align:left}}th{{color:#84a0be;text-transform:uppercase;font-size:9px}}td{{color:#e4eefb}}
.note{{padding:10px 12px;background:#0b1728;border-left:3px solid #31d7ff;color:#a9bad0;font-size:11px;margin-top:12px}}
@media print{{body{{background:#fff;color:#111;padding:0}}.metric,.chart,table{{background:#fff;border-color:#bbb}}.metric div,th,.meta{{color:#444}}td,.metric strong{{color:#111}}.charts{{break-inside:avoid}}h2{{color:#075d78}}.note{{color:#333;background:#f5f5f5}}}}
</style></head><body>
<h1>FSOC PAT MISSION RUN REPORT</h1>
<div class='meta'>Run ID: {html.escape(str(doc.get('run_id','')))} · {html.escape(str(meta.get('started_local','')))} → STOP · reason: {html.escape(str(s.get('stop_reason','MANUAL STOP')))}</div>
<div class='meta'>Platform: <b>{html.escape(str(meta.get('platform_mode','')))}</b> · Scenario: <b>{html.escape(str(meta.get('preset','')))}</b> · Atmosphere: <b>{html.escape(str(meta.get('atmosphere','')))}</b> · Target: <b>{html.escape(str(s.get('target_shape',meta.get('target_shape',''))))}</b> · Trajectory: <b>{html.escape(str(s.get('trajectory',meta.get('trajectory',''))))}</b> · Targets: <b>{html.escape(str(meta.get('target_count','')))}</b></div>
<div class='grid'>{card_html}</div>
<h2>TIME-SERIES ANALYSIS</h2><div class='charts'>{chart_html}</div>
<h2>STRESS INJECTION RESPONSE — SINGLE vs MULTIPLE</h2>
<div class='meta'>Single-stress frames: {s.get('stress_single_frames',0)} · Multi-stress frames: {s.get('stress_multiple_frames',0)} · Disturbance mean/peak: {s.get('disturbance_mean_pct',0):.1f}% / {s.get('disturbance_peak_pct',0):.1f}%</div>
<table><thead><tr><th>Stress</th><th>Frames</th><th>Lock retention</th><th>Confidence</th><th>PAT error</th><th>Uncertainty</th></tr></thead><tbody>{stress_rows}</tbody></table>
<h2>MODE / RUN COMPARISON</h2>
<table><thead><tr><th>Run</th><th>Platform</th><th>Scenario</th><th>Trajectory</th><th>FPS</th><th>PAT mean px</th><th>Accuracy</th><th>Lock retention</th></tr></thead><tbody>{comparison_rows or "<tr><td colspan='8'>No previous completed runs in this session.</td></tr>"}</tbody></table>
<h2>RECORDED STATE MIX</h2>
<table><thead><tr><th>Tracking state</th><th>Share</th></tr></thead><tbody>
{''.join(f"<tr><td>{html.escape(str(k))}</td><td>{v:.1f}%</td></tr>" for k,v in s.get('states',{}).items())}
</tbody></table>
<div class='note'>This report is generated from the explicit START → STOP evidence window. Ground-truth metrics remain separated from the tracker path; values shown here are the measurements recorded by the running prototype.</div>
</body></html>"""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(body)

    def _write_pdf(self, path, doc):
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import mm
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        except Exception:
            return False

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
        styles.add(ParagraphStyle(name="Tiny", parent=styles["BodyText"], fontSize=7, leading=9))
        s = doc["summary"]
        meta = doc.get("meta", {})
        story = [
            Paragraph("FSOC PAT MISSION RUN REPORT", styles["Title"]),
            Paragraph(
                f"Run {html.escape(str(doc.get('run_id','')))} · "
                f"{html.escape(str(meta.get('platform_mode','')))} · "
                f"{html.escape(str(meta.get('preset','')))} · "
                f"Target {html.escape(str(s.get('target_shape','')))} · "
                f"Trajectory {html.escape(str(s.get('trajectory','')))}",
                styles["Small"]),
            Spacer(1, 8),
        ]

        rows = [["Metric", "Value"], ["Duration", f"{s.get('duration_s',0):.2f} s"], ["Frames", str(s.get("frames",0))],
                ["Average FPS", f"{s.get('fps_mean',0):.2f}"], ["PAT mean error", "n/a" if s.get("pat_error_mean_px") is None else f"{s['pat_error_mean_px']:.2f} px"],
                ["PAT RMS error", "n/a" if s.get("pat_error_rms_px") is None else f"{s['pat_error_rms_px']:.2f} px"],
                ["Acquisition", "never" if s.get("acquisition_time_s") is None else f"{s['acquisition_time_s']:.2f} s"],
                ["Confidence", f"{s.get('confidence_mean_pct',0):.1f}%"], ["Alignment accuracy", f"{s.get('alignment_accuracy_pct',0):.1f}%"],
                ["Lock retention", f"{s.get('lock_retention_pct',0):.1f}%"], ["Reacquisitions", str(s.get("reacquisition_count",0))],
                ["Mean uncertainty", f"{s.get('mean_uncertainty_px',0):.2f} px"], ["Peak uncertainty", f"{s.get('max_uncertainty_px',0):.2f} px"]]
        tbl = Table(rows, colWidths=[62*mm, 95*mm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#16324f")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#9aa7b4")),
            ("FONTSIZE",(0,0),(-1,-1),8),
            ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f4f7fa")]),
        ]))
        story += [tbl, Spacer(1,10), Paragraph("Stress injection response", styles["Heading2"])]
        stress_rows = [["Stress","Frames","Lock %","Conf %","PAT px","Uncertainty px"]]
        for key,row in (s.get("stress_response") or {}).items():
            stress_rows.append([key,row["frames"],f"{row['lock_retention_pct']:.1f}",f"{row['mean_confidence_pct']:.1f}",
                                "n/a" if row["mean_pat_error_px"] is None else f"{row['mean_pat_error_px']:.2f}",
                                f"{row['mean_uncertainty_px']:.2f}"])
        if len(stress_rows)==1:
            stress_rows.append(["No stress injection","-","-","-","-","-"])
        t2=Table(stress_rows,colWidths=[40*mm,18*mm,20*mm,20*mm,22*mm,25*mm])
        t2.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#16324f")),
                                ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                                ("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#9aa7b4")),
                                ("FONTSIZE",(0,0),(-1,-1),7),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f4f7fa")])]))
        story += [t2, Spacer(1,10), Paragraph("Run / mode comparison", styles["Heading2"])]
        cmp=[["Run","Platform","Scenario","Trajectory","FPS","PAT px","Accuracy"]]
        for old in self.history[-10:]:
            sm=old.get("summary",{}); mm=old.get("meta",{})
            cmp.append([str(old.get("run_id",""))[-14:], str(mm.get("platform_mode",""))[-14:], str(mm.get("preset","")),
                        str(sm.get("trajectory","")), f"{sm.get('fps_mean',0):.1f}",
                        "n/a" if sm.get("pat_error_mean_px") is None else f"{sm['pat_error_mean_px']:.2f}",
                        f"{sm.get('alignment_accuracy_pct',0):.1f}%"])
        t3=Table(cmp,colWidths=[28*mm,28*mm,25*mm,32*mm,18*mm,20*mm,22*mm])
        t3.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#16324f")),
                                ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                                ("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#9aa7b4")),
                                ("FONTSIZE",(0,0),(-1,-1),7),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f4f7fa")])]))
        story.append(t3)
        SimpleDocTemplate(path, pagesize=A4, rightMargin=14*mm, leftMargin=14*mm, topMargin=14*mm, bottomMargin=14*mm).build(story)
        return True

    # -------------------------------------------------------------- utilities
    def open_latest(self, prefer_pdf=True):
        doc = self.latest_report
        if not doc:
            return None
        path = doc.get("pdf_path") if prefer_pdf and doc.get("pdf_path") else doc.get("html_path")
        if not path or not os.path.isfile(path):
            path = doc.get("html_path")
        if not path:
            return None
        try:
            if os.name == "nt" and hasattr(os, "startfile"):
                os.startfile(os.path.abspath(path))
            else:
                webbrowser.open(Path(path).resolve().as_uri())
        except Exception:
            try:
                webbrowser.open(Path(path).resolve().as_uri())
            except Exception:
                pass
        return path

    def history_rows(self):
        rows = []
        for doc in self.history:
            sm = doc.get("summary", {})
            mm = doc.get("meta", {})
            rows.append({
                "run_id": doc.get("run_id", ""),
                "platform": mm.get("platform_mode", ""),
                "preset": mm.get("preset", ""),
                "trajectory": sm.get("trajectory", mm.get("trajectory", "")),
                "shape": sm.get("target_shape", mm.get("target_shape", "")),
                "fps": sm.get("fps_mean", 0.0),
                "pat_error_px": sm.get("pat_error_mean_px"),
                "accuracy": sm.get("alignment_accuracy_pct", 0.0),
                "retention": sm.get("lock_retention_pct", 0.0),
                "confidence": sm.get("confidence_mean_pct", 0.0),
                "reacq": sm.get("mean_reacquisition_s"),
                "report": doc.get("pdf_path") or doc.get("html_path"),
            })
        return rows[-self.history_limit:]
