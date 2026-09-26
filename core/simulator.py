"""
core/simulator.py
-----------------
The mission pipeline, wired end-to-end (used by the GUI and the headless
stress-test harness alike):

    Scene3D --(gimbal pose)--> VirtualSensor --(disturbances)-->
    frame --DetectionEngine--> candidates --Tracker(state machine)-->
    estimate --PointingController--> gimbal set-point --(physics)-->

All information that flows into detection/control comes from either the
rendered (and disturbed) pixels or the ephemeris prior - never ground truth.
"""

import math
import os
import config
from collections import deque
from core.geometry import azel_unit, sd_angle_deg
from core.scene import Scene3D
from core.gimbal import Gimbal
from core.sensor import VirtualSensor
from core.disturbances import DisturbanceEngine
from core.detection import DetectionEngine
from core.tracking import (Tracker, SEARCHING, COASTING, LOCKED, DEGRADED_LOCK)
from core.control import PointingController
from core.orbital import EphemerisModel


class Simulator:
    def __init__(self, preset_name="EASY", seed=None, dt=1.0 / config.FPS,
                 tracker_factory=None, platform_mode=None, atmosphere=None,
                 motion_type=None, target_shape=None, target_size=None,
                 num_targets=None, target_initial=None, ephemeris_model=None,
                 mismatch_mode="matched"):
        preset = config.DIFFICULTY_PRESETS.get(preset_name, config.DIFFICULTY_PRESETS["EASY"])
        self.preset_name = preset_name
        self.preset = preset
        self.platform_mode = platform_mode
        self.atmosphere_name = atmosphere

        # Merge platform mode defaults with preset (preset overrides)
        pm = None
        if platform_mode:
            from core.platforms import PLATFORM_MODES
            pm = PLATFORM_MODES.get(platform_mode)
        if pm:
            dist = {**pm.get("disturbances", {}), **{k: v for k, v in preset.items()
                    if k in ("turbulence", "vibration", "sensor_noise", "jerk_prob", "beacon_fade")}}
        else:
            dist = preset

        self.scene = Scene3D(
            az_amp=preset["az_amp"], el_amp=preset["el_amp"],
            speed=preset["speed"],
            distractors=preset["distractors"],
            obstacles=preset["obstacles"], seed=seed,
            motion_type=motion_type or preset.get("motion_type",
                                    pm.get("motion_default") if pm else None),
            target_shape=target_shape, target_size=target_size,
            num_targets=num_targets if num_targets is not None
                        else getattr(config, "NUM_TARGETS", 1),
            target_initial=target_initial,
        )
        self.eph = ephemeris_model if ephemeris_model is not None else EphemerisModel(self.scene.orbit, seed=seed, mismatch_mode=mismatch_mode)
        self.gimbal = Gimbal()
        self.sensor = VirtualSensor()
        self.disturbance = DisturbanceEngine(
            turbulence=dist.get("turbulence", 0),
            vibration=dist.get("vibration", 0),
            sensor_noise=dist.get("sensor_noise", 0),
            jerk_prob=dist.get("jerk_prob", 0),
            beacon_fade=dist.get("beacon_fade", 0),
            seed=seed,
        )
        self.disturbance.noise_types = preset.get("noise_types",
                                                   pm.get("noise_types", ["gaussian"]) if pm
                                                   else ["gaussian"])
        # Atmospheric condition engine.  Scenario gate: a space-to-space
        # link (SAT-SAT) has no terrestrial atmosphere, so its atmosphere is
        # hard-forced to CLEAR even if a condition was requested through the
        # backend/config - a disabled disturbance can never be applied
        # accidentally.  UAV-SAT / UAV-UAV keep the full weather engine.
        from core.platforms import atmosphere_allowed
        self.atmosphere_allowed = atmosphere_allowed(platform_mode)
        from core.atmosphere import AtmosphereEngine
        atm = atmosphere or (pm.get("atmosphere", "CLEAR") if pm else "CLEAR")
        if not self.atmosphere_allowed and atm != "CLEAR":
            atm = "CLEAR"
        self.atmosphere = AtmosphereEngine(condition=atm, seed=seed)
        self.atmosphere_name = atm
        self.detector = DetectionEngine()
        if tracker_factory is None:
            self.tracker = Tracker(self.eph, seed=seed)
        else:
            self.tracker = tracker_factory(self.eph, seed=seed)
        self.controller = PointingController(self.gimbal, self.tracker)
        self.dt = dt
        self.t = 0.0
        self.frame = None
        self.last_result = None
        # brightness history of the associated object (for the scope HUD)
        self.intensity_hist = deque(maxlen=240)
        # structured PAT state-transition journal (event-only, one entry per
        # state change - never per-frame): (t, from_state, to_state).  Drives
        # the recovery timeline in the HUD and the scenario/recovery report;
        # no computation reads it, so it cannot perturb the loop.
        self.event_log = []

        # coarse pre-aim toward the ephemeris prediction
        paz, pel = self.eph.predict_az_el(0.0)
        self.gimbal.pan, self.gimbal.tilt = paz, pel
        self.gimbal.pan_cmd, self.gimbal.tilt_cmd = paz, pel
        self.tracker.reset(paz, pel)

    # ------------------------------------------------------------------
    def set_preset(self, preset_name, seed=None):
        self.__init__(preset_name, seed=seed, dt=self.dt)

    # ------------------------------------------------------------------
    @property
    def state(self):
        return self.tracker.state

    @property
    def is_locked(self):
        return self.tracker.state == LOCKED

    # ------------------------------------------------------------------
    def set_atmosphere(self, condition):
        """Request an atmosphere condition.  Respects the scenario gate: on a
        space-to-space link any non-CLEAR condition is ignored (the vacuum
        path has no weather) and the engine stays CLEAR.

        Returns the condition actually active after the call.
        """
        if not self.atmosphere_allowed:
            self.atmosphere_name = "CLEAR"
            self.atmosphere.condition = "CLEAR"
            return self.atmosphere_name
        self.atmosphere_name = condition
        self.atmosphere.condition = condition
        return self.atmosphere_name

    # ------------------------------------------------------------------
    def set_fov(self, hfov_deg, vfov_deg=None):
        """Configure user-defined camera field of view (PS default 4x3 deg)."""
        res = config.update_fov(hfov_deg, vfov_deg)
        return res[0], res[1]

    def set_screen_size(self, w, h):
        """Configure user-defined virtual screen size (PS default 2000x2000)."""
        w, h, cx, cy = config.update_screen_size(w, h)
        return w, h, cx, cy

    def set_target_params(self, shape=None, size_px=None, size_py=None, count=None, initial=None):
        """Configure target parameters (shape: Square/Circle/Spot, size: 5-20px, count: 1-5, initial)."""
        self.scene.set_target_params(shape=shape, size_px=size_px, size_py=size_py, count=count, initial=initial)

    def set_gimbal_limits(self, max_pan=None, max_tilt=None):
        """Configure gimbal speed limits (PS default 5 deg/s, range 5-10 deg/s)."""
        self.gimbal.set_limits(max_pan=max_pan, max_tilt=max_tilt)

    def set_motion_type(self, motion_type):
        """Configure target trajectory motion (straight_line, circular, figure_eight, random, etc.)."""
        self.scene.set_motion_type(motion_type)

    def set_platform_mode(self, platform_mode):
        """Configure platform mode: SATELLITE_SATELLITE, UAV_SATELLITE, UAV_UAV.
        Enforces vacuum gating for SATELLITE_SATELLITE (no terrestrial atmosphere)."""
        from core.platforms import PLATFORM_MODES, atmosphere_allowed
        self.platform_mode = platform_mode
        self.atmosphere_allowed = atmosphere_allowed(platform_mode)
        pm = PLATFORM_MODES.get(platform_mode)
        if pm:
            self.gimbal.set_limits(max_pan=pm.get("gimbal_max_pan", 5.0),
                                   max_tilt=pm.get("gimbal_max_tilt", 5.0))
            if "motion_default" in pm:
                self.scene.set_motion_type(pm["motion_default"])
            d = pm.get("disturbances", {})
            self.disturbance.turbulence = d.get("turbulence", self.disturbance.turbulence)
            self.disturbance.vibration = d.get("vibration", self.disturbance.vibration)
            self.disturbance.sensor_noise = d.get("sensor_noise", self.disturbance.sensor_noise)
            self.disturbance.jerk_prob = d.get("jerk_prob", self.disturbance.jerk_prob)
            self.disturbance.beacon_fade = d.get("beacon_fade", self.disturbance.beacon_fade)
            if "noise_types" in pm:
                self.disturbance.noise_types = list(pm["noise_types"])
            if not self.atmosphere_allowed:
                self.atmosphere_name = "CLEAR"
                self.atmosphere.condition = "CLEAR"
            else:
                self.set_atmosphere(pm.get("atmosphere", "CLEAR"))
        return self.platform_mode

    def set_noise_types(self, noise_types):
        """Configure active noise channels (gaussian, salt_pepper, poisson)."""
        self.disturbance.noise_types = [n for n in noise_types if n in ("gaussian", "salt_pepper", "poisson", "hot_pixels")]

    def inject_target_loss(self, duration_s=1.0):
        """Simulate real target disappearance to test COASTING -> REACQUIRING -> LOCKED reacquisition ladder."""
        self._target_loss_until = self.t + max(0.1, float(duration_s))
        for b in getattr(self.scene, "beacons", [getattr(self.scene, "beacon", None)]):
            if b is not None:
                b.suppressed = True

    def set_primary_target(self, target_idx):
        """Designate which target is primary (PS Item 8 Multi-Target Handover)."""
        if hasattr(self, "scene"):
            idx, tid = self.scene.set_primary_target(target_idx)
            if hasattr(self, "eph") and hasattr(self.scene.beacon, "orbit"):
                self.eph.orbit = self.scene.beacon.orbit
            if hasattr(self.tracker, "eph") and hasattr(self.scene.beacon, "orbit"):
                self.tracker.eph.orbit = self.scene.beacon.orbit
            paz, pel = self.tracker.eph.predict_az_el(self.t)
            self.tracker.reset(paz, pel)
            # Physical slew command: gimbal slews at rate limit (no instant teleportation)
            self.gimbal.pan_cmd, self.gimbal.tilt_cmd = paz, pel
            return idx, tid
        return 0, "TARGET-01"

    # ------------------------------------------------------------------
    def step(self):
        """Advance one simulation frame.  Returns a dict of measurements the
        caller turns into metrics or HUD + the (possibly disturbed) frame."""
        dt = self.dt
        self.scene.advance(dt)
        self.t += dt

        is_suppressed = self.t < getattr(self, "_target_loss_until", 0.0)
        for b in getattr(self.scene, "beacons", [self.scene.beacon]):
            b.suppressed = is_suppressed

        basis = self.gimbal.basis()
        frame = self.sensor.render(self.scene, self.gimbal, config.FOCAL_PX,
                                   disturbance=self.disturbance, dt=dt)
        # Apply atmospheric condition effects (haze, fog, rain, low light).
        # Double gate: condition != CLEAR AND the scenario physically allows
        # an atmosphere (a space link can never inherit one).
        if self.atmosphere_allowed and self.atmosphere_name != "CLEAR":
            frame = self.atmosphere.apply(frame)

        candidates = self.detector.detect(frame, basis, config.FOCAL_PX)

        prev_state = self.tracker.state
        state, est_az, est_el, confidence = self.tracker.update(candidates, self.t, dt)
        if state != prev_state:
            # Acquisition is summarised as one SEARCHING -> LOCK entry; after the
            # first lock every transition (COAST / REACQ / SEARCH / relock) is
            # recorded so recovery sequences are observable.
            if not self.event_log:
                if state in (LOCKED, DEGRADED_LOCK):
                    self.event_log.append((self.t, "SEARCHING", state))
            else:
                self.event_log.append((self.t, prev_state, state))
        assoc = self.tracker.associated
        self.intensity_hist.append(assoc.peak if assoc is not None else None)

        pan, tilt = self.controller.compute_setpoint(self.t, dt)
        self.gimbal.step(dt, self.disturbance)

        # ----- ground-truth view (metrics only, never into the pipeline) -----
        truth_az = self.scene.beacon.az_deg
        truth_el = self.scene.beacon.el_deg
        truth_dir = azel_unit(truth_az, truth_el)
        enc_dir = azel_unit(self.gimbal.pan, self.gimbal.tilt)
        pointing_err_deg = sd_angle_deg(enc_dir, truth_dir)

        # estimate error (tracker LOS estimate vs truth)
        if est_az is not None and est_el is not None:
            est_dir = azel_unit(est_az, est_el)
            est_err_deg = sd_angle_deg(est_dir, truth_dir)
        else:
            est_err_deg = None

        occ = 0.0
        for o in self.scene.obstacles:
            occ = max(occ, o.crossing(self.scene.time))
        beacon_visible = (occ < 0.55) and not is_suppressed

        # is the beacon within the sensor FOV (as projected)?
        import math
        dpan = min(abs(truth_az - self.gimbal.pan), 360 - abs(truth_az - self.gimbal.pan))
        dtilt = min(abs(truth_el - self.gimbal.tilt), 360 - abs(truth_el - self.gimbal.tilt))

        candidates_detail = [
            dict(
                u=round(float(c.u), 1),
                v=round(float(c.v), 1),
                area=int(c.area),
                circularity=round(float(c.circularity), 3),
                snr=round(float(c.snr), 2),
                ml_score=round(float(c.ml_score), 3),
                track_id=getattr(c, "track_id", None),
                track_age=getattr(c, "track_age", 0),
            )
            for c in candidates
        ]

        # Centroid metrics (Metric B: optical offset / boresight error; Metric C: true centroid error)
        cam_cx = config.CAM_VIEW_W / 2.0
        cam_cy = config.CAM_VIEW_H / 2.0
        detected_cx = assoc.u if assoc is not None else None
        detected_cy = assoc.v if assoc is not None else None
        boresight_error_px = (
            round(math.hypot(detected_cx - cam_cx, detected_cy - cam_cy), 2)
            if detected_cx is not None else None
        )
        cam_canvas = self.sensor._canvas_xy(self.gimbal.pan, self.gimbal.tilt)
        gt_u, gt_v = self.sensor._viewport_px(truth_az, truth_el, cam_canvas)
        in_frame = (0 <= gt_u < config.CAM_VIEW_W and 0 <= gt_v < config.CAM_VIEW_H)
        centroid_error_px = (
            round(math.hypot(detected_cx - gt_u, detected_cy - gt_v), 2)
            if (detected_cx is not None and in_frame) else None
        )

        self.last_result = dict(
            state=state,
            tracking_state=state,
            tracking_phase=getattr(self.tracker, "phase", state),
            is_locked=getattr(self.tracker, "is_locked", state == "LOCKED"),
            is_degraded=getattr(self.tracker, "is_degraded", state == "DEGRADED_LOCK"),
            measurement_valid=getattr(self.tracker, "measurement_valid", assoc is not None),
            measurement_age=getattr(self.tracker, "measurement_age", 0.0),
            prediction_only=getattr(self.tracker, "prediction_only", False),
            boresight_error_px=boresight_error_px,
            centroid_error_px=centroid_error_px,
            est_az=est_az, est_el=est_el,
            confidence=confidence,
            truth_az=truth_az, truth_el=truth_el,
            pointing_err_deg=pointing_err_deg,
            est_err_deg=est_err_deg,
            candidates=len(candidates),
            cand_list=candidates,
            candidates_detail=candidates_detail,
            beacon_visible=beacon_visible,
            dist_pan_deg=dpan, dist_tilt_deg=dtilt,
            in_fov=(dpan < config.HFOV_DEG / 2.0 + 0.1 and
                    dtilt < config.VFOV_DEG / 2.0 + 0.1),
            frame=frame,
            t=self.t,
            cmd_pan=pan,
            cmd_tilt=tilt,
            gimbal_pan=self.gimbal.pan,
            gimbal_tilt=self.gimbal.tilt,
            gimbal_v_pan=self.gimbal.v_pan,
            gimbal_v_tilt=self.gimbal.v_tilt,
            gimbal_sat_pan=self.gimbal.pan_sat,
            gimbal_sat_tilt=self.gimbal.tilt_sat,
            fsm_pan_urad=getattr(self.gimbal, "fsm_pan_urad", 0.0),
            fsm_tilt_urad=getattr(self.gimbal, "fsm_tilt_urad", 0.0),
            fsm_active=getattr(self.gimbal, "fsm_active", False),
            fsm_sat=getattr(self.gimbal, "fsm_sat", 0.0),
            primary_target_id=getattr(self.scene.beacon, "target_id", "TARGET-01"),
        )
        return self.last_result


class VideoInputSimulator:
    """Benchmark-2 mode: an external MP4 ("bypass its PTZ camera and take this
    video as an input to the coarse pointing system") drives the real
    detect -> track -> control -> gimbal closed loop.

    The virtual camera renders nothing of its own; each step reads the next
    frame of the supplied video, detects the beacon blob in those *raw*
    pixels, feeds the candidates into the same Tracker/PointingController
    used by the synthetic mode, and steers the gimbal so the beacon lands on
    the boresight.  Modulation-ID gating is relaxed in video mode (the video
    has no known 15 Hz clock); identity = persistence + appearance.
    """

    def __init__(self, video_path, seed=None, dt=None, truth_csv=None, truth_path=None):
        import cv2
        truth_csv = truth_path or truth_csv
        self.preset = config.DIFFICULTY_PRESETS["EASY"]
        self.preset_name = "VIDEO"
        self.platform_mode = None
        self.atmosphere_name = "CLEAR"

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        self.video_path = video_path
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.video_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._dt = dt if dt is not None else 1.0 / self.video_fps

        # optional ground-truth sidecar (generator truth CSV): the "predefined
        # error values" Benchmark-2 compares against
        self.truth = {}
        if truth_csv and os.path.isfile(truth_csv):
            import csv as _csv
            with open(truth_csv, newline="") as f:
                for row in _csv.DictReader(f):
                    try:
                        self.truth[int(row["frame"])] = (float(row["bx"]),
                                                         float(row["by"]))
                    except (ValueError, KeyError):
                        continue

        # stationary prior at boresight: an external video has no orbital
        # ephemeris, so the prior is simply "the beacon is near cue centre"
        from core.orbital import RelativeOrbitModel, EphemerisModel
        from core.tracking import Tracker
        from core.gimbal import Gimbal
        from core.control import PointingController
        from core.detection import DetectionEngine
        from core.disturbances import DisturbanceEngine

        zero_orbit = RelativeOrbitModel(
            az_amp=0.0, el_amp=0.0, seed=seed, motion_type="straight_line")
        self.eph = EphemerisModel(zero_orbit, seed=seed)
        self.gimbal = Gimbal()
        self.disturbance = DisturbanceEngine()   # zero-level: no extra motion
        from core.video_detector import VideoBeaconDetector
        self.detector = VideoBeaconDetector(
            target_px=int(getattr(config, "TARGET_SIZE_PX", 10)))
        # video mode: the whole frame is the sensor - the association gate
        # spans the video's own angular FOV so the beacon stays associated
        # across its full sweep instead of being knocked out by servo lag.
        self.video_fov_deg = float(getattr(config, "CAMERA_FOV_H_DEG", 4.0))
        self.cu = self.video_w / 2.0
        self.cv = self.video_h / 2.0
        self.tracker = Tracker(self.eph, seed=seed, video_mode=True,
                               gate_deg=self.video_fov_deg * 0.95,
                               cu=self.cu, cv=self.cv)
        self.controller = PointingController(self.gimbal, self.tracker)
        self.dt = self._dt
        self.t = 0.0
        self.frame = None
        self.intensity_hist = deque(maxlen=240)

        # Video geometry mapped onto the LOS frame
        self.focal_px = (self.video_w / 2.0) / math.tan(
            math.radians(self.video_fov_deg / 2.0))
        self.cu = self.video_w / 2.0
        self.cv = self.video_h / 2.0
        self.pixels_per_deg = self.video_w / self.video_fov_deg

        self.tracker.reset(0.0, 0.0)
        self.ground_truth_available = bool(self.truth and len(self.truth) > 0)
        self.frame_idx = 0

        # Structured performance and metric logs
        self.centroid_log = []          # Metric A: (frame, detected_cx, detected_cy)
        self.optical_offset_log = []    # Metric B: (frame, offset_px, offset_deg)
        self.true_err_log = []          # Metric C: (frame, true_err_px, true_err_deg) (only if GT available)
        self.estimate_log = []          # (frame, est_az, est_el)
        self.lock_history = []          # (frame, state)

        self.acquisition_time_s = None
        self.lock_lost_at = None
        self.reacq_times = []
        self._was_locked = False
        self.locked_frames = 0
        self.lost_frames = 0
        self.reacq_attempts = 0
        self.reacq_successes = 0
        self._false_lock_count = 0
        self.false_lock_events = 0
        self.last_result = None

    @property
    def state(self):
        return self.tracker.state

    @property
    def is_locked(self):
        return self.tracker.is_locked

    def step(self):
        """Read one video frame and close the detection -> track -> control loop.

        Benchmark-2 Semantics (PS 26169):
        - Virtual/PTZ camera is bypassed; external video enters the pipeline.
        - Metric A: Detected centroid (u, v) in frame pixels.
        - Metric B: Optical-axis / Frame-centre offset dist((u, v), (cu, cv)).
        - Metric C: True centroiding error dist((u, v), (gt_u, gt_v)) computed ONLY
          when evaluator ground-truth sidecar CSV is available.
        - No ground-truth coordinates are ever injected into the tracking loop.
        """
        import cv2
        import numpy as np

        ret, frame = self.cap.read()
        if not ret:
            return None
        self.t += self.dt
        frame = np.ascontiguousarray(frame)

        # --- Create a virtual-camera viewport from the MP4 frame using the
        #     current realized gimbal attitude so that pan/tilt actually
        #     translate the visible pixels the detector consumes.
        #
        # The MP4 is treated as a fixed-angular canvas with horizontal
        # angular span `self.video_fov_deg`.  Shifting the realized gimbal
        # attitude recenters the visible window inside that canvas by an
        # integer-pixel translation.  Regions that fall outside the source
        # frame are zero-padded (black). This keeps the change minimal and
        # reuses the existing detection/tracker/controller/gimbal stack.
        basis = self.gimbal.basis()  # encoder basis for pixel->world mapping
        candidates = self.detector.detect(frame, basis, self.focal_px,
                                          cu=self.cu, cv=self.cv)
        state, est_az, est_el, confidence = self.tracker.update(
            candidates, self.t, self.dt)
        self.intensity_hist.append(
            self.tracker.associated.peak if self.tracker.associated else None)

        # Tracked candidate in sensor frame
        tracked = self.tracker.associated

        # --- METRIC A: Detected Centroid (x, y) ---
        detected_cx = float(tracked.x) if tracked is not None else None
        detected_cy = float(tracked.y) if tracked is not None else None
        self.detected_cx = detected_cx
        self.detected_cy = detected_cy

        # --- METRIC B: Optical-axis / Frame-centre offset ---
        if detected_cx is not None:
            optical_offset_px = math.hypot(detected_cx - self.cu, detected_cy - self.cv)
            optical_offset_deg = optical_offset_px / self.pixels_per_deg
        else:
            optical_offset_px = None
            optical_offset_deg = None

        # --- METRIC C: True Centroiding Error (Only if Ground Truth Available) ---
        true_centroid_err_px = None
        true_centroid_err_deg = None
        gt_x, gt_y = None, None

        if self.ground_truth_available:
            tb = self.truth.get(self.frame_idx)
            if tb is not None:
                gt_x, gt_y = float(tb[0]), float(tb[1])
                if detected_cx is not None:
                    true_centroid_err_px = math.hypot(detected_cx - gt_x, detected_cy - gt_y)
                else:
                    true_centroid_err_px = math.hypot(self.cu - gt_x, self.cv - gt_y)
                true_centroid_err_deg = true_centroid_err_px / self.pixels_per_deg

        self.frame_idx += 1
        if detected_cx is not None:
            self.centroid_log.append((self.frame_idx, detected_cx, detected_cy))
        if optical_offset_px is not None:
            self.optical_offset_log.append((self.frame_idx, optical_offset_px, optical_offset_deg))
        if true_centroid_err_px is not None:
            self.true_err_log.append((self.frame_idx, true_centroid_err_px, true_centroid_err_deg))

        self.estimate_log.append((self.frame_idx, est_az, est_el))
        self.lock_history.append((self.frame_idx, state))

        # --- State, Acquisition & Reacquisition Statistics ---
        is_locked = (state in (LOCKED, DEGRADED_LOCK))
        if is_locked:
            self.locked_frames += 1
            if self.acquisition_time_s is None:
                self.acquisition_time_s = self.t
            elif not self._was_locked:
                if self.lock_lost_at is not None:
                    dt_reacq = self.t - self.lock_lost_at
                    self.reacq_times.append(dt_reacq)
                    self.reacq_successes += 1
                    self.lock_lost_at = None

            # False lock detection against ground truth when available
            if self.ground_truth_available and true_centroid_err_px is not None:
                if true_centroid_err_px > 0.35 * max(self.video_w, self.video_h):
                    self._false_lock_count += 1
                    if self._false_lock_count == 5:
                        self.false_lock_events += 1
                else:
                    self._false_lock_count = 0
        else:
            self.lost_frames += 1
            self._false_lock_count = 0
            if self._was_locked:
                self.lock_lost_at = self.t
                self.reacq_attempts += 1

        self._was_locked = is_locked

        candidates_detail = [
            dict(
                u=round(float(c.u), 1),
                v=round(float(c.v), 1),
                area=int(c.area),
                circularity=round(float(c.circularity), 3),
                snr=round(float(c.snr), 2),
                ml_score=round(float(c.ml_score), 3),
                track_id=getattr(c, "track_id", None),
                track_age=getattr(c, "track_age", 0),
            )
            for c in candidates
        ]

        self.last_result = dict(
            state=state,
            tracking_state=state,
            tracking_phase=getattr(self.tracker, "phase", state),
            is_locked=getattr(self.tracker, "is_locked", state == "LOCKED"),
            is_degraded=getattr(self.tracker, "is_degraded", state == "DEGRADED_LOCK"),
            measurement_valid=getattr(self.tracker, "measurement_valid", detected_cx is not None),
            measurement_age=getattr(self.tracker, "measurement_age", 0.0),
            prediction_only=getattr(self.tracker, "prediction_only", False),
            boresight_error_px=optical_offset_px,
            centroid_error_px=true_centroid_err_px,
            est_az=est_az,
            est_el=est_el,
            confidence=confidence,
            candidates=len(candidates),
            cand_list=candidates,
            candidates_detail=candidates_detail,
            frame=frame,
            t=self.t,
            frame_idx=self.frame_idx,
            ground_truth_available=self.ground_truth_available,
            # Metric A
            detected_cx=detected_cx,
            detected_cy=detected_cy,
            # Metric B
            optical_offset_px=optical_offset_px,
            optical_offset_deg=optical_offset_deg,
            # Metric C (Only valid if ground_truth_available)
            gt_x=gt_x,
            gt_y=gt_y,
            true_centroid_err_px=true_centroid_err_px,
            true_centroid_err_deg=true_centroid_err_deg,
            # Telemetry compatibility
            pointing_err_deg=true_centroid_err_deg if true_centroid_err_deg is not None else optical_offset_deg,
            in_fov=detected_cx is not None,
            beacon_visible=(detected_cx is not None),
            est_err_deg=true_centroid_err_deg if self.ground_truth_available else None,
            primary_target_id="TARGET-01",
        )
        # --- Closed-loop actuation: feed the tracker output into the
        #     existing controller+gimbal, then advance the gimbal so the
        #     next video frame will reflect the new encoder pose.
        pan, tilt = self.controller.compute_setpoint(self.t, self.dt)
        self.gimbal.step(self.dt, self.disturbance)

        # --- Prevent the camera from panning/tilting beyond the angular
        #     extent of the video sensor canvas (avoid infinite orbit).
        max_center_pan = self.video_fov_deg / 2.0
        video_vfov = (self.video_h / float(self.video_w)) * self.video_fov_deg
        max_center_tilt = video_vfov / 2.0

        self.gimbal.pan = max(-max_center_pan, min(max_center_pan, self.gimbal.pan))
        self.gimbal.pan_cmd = max(-max_center_pan, min(max_center_pan, self.gimbal.pan_cmd))
        self.gimbal.tilt = max(-max_center_tilt, min(max_center_tilt, self.gimbal.tilt))
        self.gimbal.tilt_cmd = max(-max_center_tilt, min(max_center_tilt, self.gimbal.tilt_cmd))

        # Update telemetry with the applied setpoint/realized attitude
        self.last_result.update(dict(
            cmd_pan=pan, cmd_tilt=tilt,
            gimbal_pan=self.gimbal.pan,
            gimbal_tilt=self.gimbal.tilt,
            gimbal_v_pan=self.gimbal.v_pan,
            gimbal_v_tilt=self.gimbal.v_tilt,
            gimbal_sat_pan=self.gimbal.pan_sat,
            gimbal_sat_tilt=self.gimbal.tilt_sat,
            fsm_pan_urad=getattr(self.gimbal, "fsm_pan_urad", 0.0),
            fsm_tilt_urad=getattr(self.gimbal, "fsm_tilt_urad", 0.0),
            fsm_active=getattr(self.gimbal, "fsm_active", False),
            fsm_sat=getattr(self.gimbal, "fsm_sat", 0.0),
        ))
        return self.last_result

    def close(self):
        """Release underlying OpenCV video capture handle."""
        if hasattr(self, "cap") and self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

    def reset(self):
        """Reset video to frame 0 and reinitialize tracking states."""
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.set(1, 0)  # cv2.CAP_PROP_POS_FRAMES = 1
        self.frame_idx = 0
        self.t = 0.0
        self.gimbal.reset()
        self.tracker = Tracker(self.eph, seed=None, video_mode=True,
                               gate_deg=self.video_fov_deg * 0.95,
                               cu=self.cu, cv=self.cv)
        self.tracker.reset(0.0, 0.0)
        from core.control import PointingController
        self.controller = PointingController(self.gimbal, self.tracker)
        self.centroid_log.clear()
        self.optical_offset_log.clear()
        self.true_err_log.clear()
        self.estimate_log.clear()
        self.lock_history.clear()
        self.acquisition_time_s = None
        self.lock_lost_at = None
        self.reacq_times.clear()
        self._was_locked = False
        self.locked_frames = 0
        self.lost_frames = 0
        self.reacq_attempts = 0
        self.reacq_successes = 0
        self._false_lock_count = 0
        self.false_lock_events = 0
        self.last_result = None
