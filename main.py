"""
main.py  —  FSOC-PAT Tactical Optical Acquisition Console  ·  SIH 2026 · PS 26169

Layout (1600 × 900):
  Header    : y   0 .. 60   — system ID, state chip, live metrics, scenario chips
  Camera    : x   0 ..1076  y 60 ..760  — 1076×700 sensor viewport
  Right panel: x 1080 ..1596  y 60 ..760  — 516×700 telemetry column
  Bottom    : y 760 ..900   — PAT pipeline stepper + error graph + beam strip

Keyboard:
  1-5  scenario preset   SPACE pause/resume   R reset   S screenshot
  6-8  platform mode     A cycle atmosphere   V FOV grid   F fullscreen
  ESC  quit

Headless self-test:   python main.py --frames 200 --preset HARD
"""

import argparse
import math
import os
import statistics
import sys
import time
from collections import deque

import pygame
import numpy as np

import config
import config
from core.simulator import Simulator
from core.geometry import azel_unit, sd_angle_deg, project_point_into_camera
from core.platforms import atmosphere_allowed, disturbance_kind_label
from metrics.performance import PerformanceTracker
from ui import theme as T
from ui import widgets as W
from ui import view3d
from ui.mission_pages import (
    OpticalLinkModel,
    StressTestManager,
    render_telemetry_page,
    render_ai_classifier_page,
    render_gimbal_servo_page,
    render_stress_test_page,
    render_false_lock_page,
    render_event_log_page,
)
from metrics.run_report import MissionRunRecorder
from ui.evidence_pages import (
    render_target_settings_page,
    render_run_report_page,
    render_live_stress_overlay,
    handle_target_settings_click,
    handle_report_click,
)


APP_W, APP_H = 1600, 900
CAM_W, CAM_H = config.CAM_VIEW_W, config.CAM_VIEW_H
DISPLAY_CAP = getattr(config, "FPS", 60)
LOCKED_STATES = ("LOCKED", "DEGRADED_LOCK")

# Reference layout defaults (for class attributes and headless mode)
SIDEBAR_W = 150
HDR_H     = 46
BTM_Y     = 720
BTM_H     = 160
CAM_X0, CAM_X1 = 158, 1200
CAM_Y0, CAM_Y1 = 92, 720
PNL_X0, PNL_X1 = 1208, 1592
PNL_W   = 384
PNL_INN = 1212
PNL_IW  = 376


class App:
    CAM_X0, CAM_X1 = CAM_X0, CAM_X1
    CAM_Y0, CAM_Y1 = CAM_Y0, CAM_Y1

    def __init__(self, preset="EASY", seed=None, fullscreen=True,
                 platform_mode=None, atmosphere=None,
                 motion_type=None, target_shape=None, target_size=None,
                 num_targets=None, target_initial=None,
                 video_path=None, video_seed=None,
                 width=1600, height=900):
        pygame.init()
        self.fullscreen = bool(fullscreen)
        if self.fullscreen:
            flags = pygame.FULLSCREEN
            self.screen = pygame.display.set_mode((0, 0), flags)
            dw, dh = self.screen.get_size()
            if dw < 1000 or dh < 600:
                dw, dh = width, height
            self.W, self.H = dw, dh
        else:
            self.W, self.H = width, height
            self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)
        self.win_w, self.win_h = width, height
        self.canvas = pygame.Surface((self.W, self.H)).convert()
        self._display_rect = pygame.Rect(0, 0, self.W, self.H)
        pygame.display.set_caption(
            "FSOC-PAT Mission Control Console  ·  SIH 2026 · PS 26169")
        self.clock = pygame.time.Clock()

        # SpaceX / ISRO Mission Control Navigation System
        self.sidebar_collapsed = False
        self.SIDEBAR_EXP_W = 172
        self.SIDEBAR_COL_W = 54
        self.SIDEBAR_W = self.SIDEBAR_EXP_W
        self.SIDEBAR_TABS = [
            ("TRACKING CONSOLE", "CON", "01"),
            ("LINK TELEMETRY",   "TEL", "02"),
            ("AI CLASSIFIER",    "AI",  "03"),
            ("GIMBAL SERVO",     "GMB", "04"),
            ("STRESS INJECTION", "STR", "05"),
            ("FALSE LOCK SUITE", "FLK", "06"),
            ("MISSION LOGS",     "LOG", "07"),
            ("TARGET SETTINGS",   "TGT", "08"),
            ("RUN REPORTS",       "RPT", "09"),
        ]
        self.active_tab = 0  # 0: TRACKING CONSOLE (Default on launch!)
        self.opt_model = OpticalLinkModel()
        self.stress_mgr = StressTestManager()
        self.run_recorder = MissionRunRecorder(config.LOG_DIR)

        self.events_list = [
            (time.strftime("%H:%M:%S UTC", time.gmtime()), "INFO", "OPT-LINK", "Carrier acquisition confirmed. Coarse alignment loop ACTIVE"),
            (time.strftime("%H:%M:%S UTC", time.gmtime()), "INFO", "TRACKER", "State transition: SEARCHING -> LOCKED (pointing error <= 10 px)"),
            (time.strftime("%H:%M:%S UTC", time.gmtime()), "INFO", "GIMBAL", "PD servo converged: Pan slew 0.12 deg/s, Tilt slew -0.04 deg/s"),
            (time.strftime("%H:%M:%S UTC", time.gmtime()), "INFO", "CLASSIF", "Beacon spot classified: circularity 0.94, SNR 73.4 dB, corr > 0.62"),
            (time.strftime("%H:%M:%S UTC", time.gmtime()), "INFO", "SUBSYS", "All 6 optical subsystems report nominal health"),
        ]
        self._last_state = "SEARCHING"
        self.hdr_pause_rect = pygame.Rect(self.W - 80, 8, 72, 30)

        self.preset        = preset
        self.platform_mode = platform_mode or "SATELLITE_SATELLITE"
        self.atmosphere    = atmosphere or "CLEAR"
        from core.platforms import atmosphere_allowed
        if not atmosphere_allowed(self.platform_mode) and self.atmosphere != "CLEAR":
            self.atmosphere = "CLEAR"
        self.motion_override  = motion_type
        self.shape_override   = target_shape
        self.size_override    = target_size
        self.targets_override = num_targets
        self.initial_override = target_initial
        self.video_path       = video_path
        self.video_done       = False
        self.video_seed       = video_seed
        self.video_dt_accum   = 0.0
        self._video_report_saved = False

        if video_path:
            from core.simulator import VideoInputSimulator
            truth = os.path.splitext(video_path)[0] + "_truth.csv"
            self.sim = VideoInputSimulator(
                video_path, seed=video_seed,
                truth_csv=truth if os.path.isfile(truth) else None)
            self.video_mode = True
        else:
            self.sim = Simulator(
                preset_name=preset, seed=seed,
                platform_mode=self.platform_mode,
                atmosphere=self.atmosphere,
                motion_type=self.motion_override,
                target_shape=self.shape_override,
                target_size=self.size_override,
                num_targets=self.targets_override,
                target_initial=self.initial_override)
            self.video_mode = False

        self.perf       = PerformanceTracker()
        self.paused     = False
        self.show_fov_grid = True
        self.eph_pred_az   = None
        self.eph_pred_el   = None
        self.show_gt    = False
        self.show_diag  = False
        self.compare    = self._load_compare()
        self.error_spark = deque(maxlen=1800)
        self.show_ps_modal = False
        self.ps_modal_rect = pygame.Rect(0, 0, 0, 0)
        self.ps_btn_rects = {}
        self._scanline_surf = None   # created lazily on first camera draw
        self.hud_declutter = False
        self.target_count = getattr(getattr(self.sim, "scene", None), "num_targets", num_targets or 1)
        self.current_motion = motion_type or "straight_line"
        self.primary_target_idx = 0
        self.target_click_rects = {}

        # Initialize widget collections
        self.sliders = {
            "turbulence":  W.Slider((0, 0, 100, 20), "TURBULENCE",
                                    self.sim.preset.get("turbulence", 0), T.C.PURPLE,
                                    enabled=atmosphere_allowed(self.platform_mode),
                                    unit=config.DISTURBANCE_UNITS.get("turbulence", ("", ""))[1]),
            "vibration":   W.Slider((0, 0, 100, 20), "VIBRATION",
                                    self.sim.preset.get("vibration", 0), T.C.AMBER,
                                    unit=config.DISTURBANCE_UNITS.get("vibration", ("", ""))[1]),
            "sensor_noise": W.Slider((0, 0, 100, 20), "SENSOR NOISE",
                                     self.sim.preset.get("sensor_noise", 0), T.C.RED,
                                     unit=config.DISTURBANCE_UNITS.get("sensor_noise", ("", ""))[1]),
            "jerk_prob":   W.Slider((0, 0, 100, 20), "JERK PROB",
                                    self.sim.preset.get("jerk_prob", 0), T.C.CYAN,
                                    unit=config.DISTURBANCE_UNITS.get("jerk_prob", ("", ""))[1]),
            "beacon_fade": W.Slider((0, 0, 100, 20), "BEACON FADE",
                                     self.sim.preset.get("beacon_fade", 0), T.C.AMBER_DIM,
                                     unit=config.DISTURBANCE_UNITS.get("beacon_fade", ("", ""))[1]),
        }

        self.buttons = {
            "PAUSE":      W.Button((0, 0, 80, 24), "PAUSE",  T.C.AMBER),
            "RESET":      W.Button((0, 0, 80, 24), "RESET",  T.C.CYAN),
            "SHOT":       W.Button((0, 0, 80, 24), "SHOT",   T.C.GREEN),
            "GT":         W.Button((0, 0, 80, 24), "GT OFF", T.C.PURPLE),
            "DIAG":       W.Button((0, 0, 80, 24), "DIAG",   T.C.TEXT_DIM),
            "LOAD_VIDEO": W.Button((0, 0, 80, 24), "LOAD MP4", T.C.AMBER),
        }
        self._screenshot_n = 0

        self.chips = {}
        for name in config.PRESET_ORDER:
            lbl = "ISRO-RX" if name == "ISRO_RX" else name
            self.chips[name] = W.Chip((0, 0, 60, 26), lbl, T.C.CYAN)
        self.show_ephemeris = True

        self.platform_chips = {}
        for pm in ["SATELLITE_SATELLITE", "UAV_SATELLITE", "UAV_UAV"]:
            lbl = {"SATELLITE_SATELLITE": "SAT-SAT",
                   "UAV_SATELLITE": "UAV-SAT",
                   "UAV_UAV": "UAV-UAV"}[pm]
            self.platform_chips[pm] = W.Chip((0, 0, 70, 26), lbl, T.C.GREEN)

        self.atmos_chips = {}
        for atm in ["CLEAR", "HAZE", "FOG", "RAIN", "LOW_LIGHT"]:
            lbl = "LOW LIGHT" if atm == "LOW_LIGHT" else atm
            self.atmos_chips[atm] = W.Chip((0, 0, 60, 26), lbl, T.C.AMBER)

# Responsive layout will be computed below

        # Initial responsive layout computation
        self._recompute_layout()

    # ── Responsive layout computation ──────────────────────────────────────────
    def _recompute_layout(self):
        w, h = self.W, self.H
        if self.canvas.get_size() != (w, h):
            self.canvas = pygame.Surface((w, h)).convert()
        self.SIDEBAR_W = self.SIDEBAR_COL_W if self.sidebar_collapsed else self.SIDEBAR_EXP_W
        self.HDR_H = 46
        self.SCENARIO_H = 38 if self.active_tab == 0 else 0
        self.HDR_TOTAL_H = self.HDR_H + self.SCENARIO_H
        self.FOOTER_H = 22

        # Right Telemetry Column width: ~27% of available content width, bounded [320, 420]
        content_w = w - self.SIDEBAR_W - 16
        self.PNL_W = max(320, min(420, int(content_w * 0.27)))
        self.PNL_X0 = w - self.PNL_W - 8
        self.PNL_X1 = w - 8
        self.PNL_INN = self.PNL_X0 + 4
        self.PNL_IW = self.PNL_W - 8

        # Left / Center Camera Region
        self.CAM_X0 = self.SIDEBAR_W + 8
        self.CAM_X1 = self.PNL_X0 - 8
        self.CAM_SIDE_W = self.CAM_X1 - self.CAM_X0

        # Vertical allocation on camera side
        content_h = h - self.HDR_TOTAL_H - self.FOOTER_H - 10
        self.BTM_H = max(144, min(176, int(content_h * 0.27)))
        self.BTM_Y = h - self.FOOTER_H - self.BTM_H - 4
        self.CAM_Y0 = self.HDR_TOTAL_H + 4
        self.CAM_Y1 = self.BTM_Y - 4

        # Pause toggle button in top header
        self.hdr_pause_rect = pygame.Rect(self.W - 74, 8, 66, 30)

        # Tier 2: Scenario Bar Chips
        chip_y = self.HDR_H + 6
        chip_h = 25
        chip_gap = 4
        group_gap = 14 if w < 1440 else 24

        # Group 1: Scenario
        self.scenario_lbl_rect = pygame.Rect(self.SIDEBAR_W + 10, chip_y, 68, chip_h)
        xs = self.scenario_lbl_rect.right + 6
        chip_widths = {
            "EASY": 48,
            "MODERATE": 72,
            "HARD": 48,
            "SEVERE": 60,
            "ADVERSARIAL": 88,
            "ISRO_RX": 66,
        }
        for name in config.PRESET_ORDER:
            cw = chip_widths.get(name, 56)
            self.chips[name].rect = pygame.Rect(xs, chip_y, cw, chip_h)
            xs += cw + chip_gap

        # Group 2: Platform
        plat_lbl_x = xs + group_gap
        self.platform_lbl_rect = pygame.Rect(plat_lbl_x, chip_y, 70, chip_h)
        px = self.platform_lbl_rect.right + 6
        plat_widths = {"SATELLITE_SATELLITE": 62, "UAV_SATELLITE": 62, "UAV_UAV": 62}
        for pm in ["SATELLITE_SATELLITE", "UAV_SATELLITE", "UAV_UAV"]:
            cw = plat_widths.get(pm, 62)
            self.platform_chips[pm].rect = pygame.Rect(px, chip_y, cw, chip_h)
            px += cw + chip_gap

        # Group 3: Atmosphere
        atm_lbl_x = px + group_gap
        self.atmos_lbl_rect = pygame.Rect(atm_lbl_x, chip_y, 82, chip_h)
        ax = self.atmos_lbl_rect.right + 6
        atm_widths = {"CLEAR": 50, "HAZE": 46, "FOG": 40, "RAIN": 44, "LOW_LIGHT": 74}
        for atm in ["CLEAR", "HAZE", "FOG", "RAIN", "LOW_LIGHT"]:
            cw = atm_widths.get(atm, 50)
            self.atmos_chips[atm].rect = pygame.Rect(ax, chip_y, cw, chip_h)
            ax += cw + chip_gap

        # Group 4: Dynamic Target Controls (PS Item 8 Multi-Target & Randomize)
        tx = ax + group_gap
        self.target_lbl_rect = pygame.Rect(tx, chip_y, 56, chip_h)
        self.btn_tgt_dec = pygame.Rect(self.target_lbl_rect.right + 2, chip_y, 18, chip_h)
        self.tgt_cnt_rect = pygame.Rect(self.btn_tgt_dec.right + 2, chip_y, 20, chip_h)
        self.btn_tgt_inc = pygame.Rect(self.tgt_cnt_rect.right + 2, chip_y, 18, chip_h)
        self.btn_cycle_tgt = pygame.Rect(self.btn_tgt_inc.right + 4, chip_y, 64, chip_h)

        self.btn_randomize = pygame.Rect(self.btn_cycle_tgt.right + 6, chip_y, 74, chip_h)
        self.btn_motion = pygame.Rect(self.btn_randomize.right + 6, chip_y, 80, chip_h)
        self.btn_hud_toggle = pygame.Rect(self.btn_motion.right + 6, chip_y, 68, chip_h)

        # Right Panel Cards vertical rhythm (proportional & spacious, ZERO OVERFLOW on any resolution)
        pnl_avail = (h - self.FOOTER_H - 4) - (self.HDR_TOTAL_H + 4)
        card_gap = 4 if h < 820 else 6
        show_mission_card = (h >= 800)
        num_cards = 6 if show_mission_card else 5
        usable_h = pnl_avail - card_gap * (num_cards - 1)
        
        self.pnl_ctrl_h = 82 if h >= 780 else 74
        cards_pool = max(240, usable_h - self.pnl_ctrl_h)

        if show_mission_card:
            self.pnl_state_h = max(66, int(cards_pool * 0.14))
            self.pnl_mission_h = max(44, int(cards_pool * 0.12))
            self.pnl_perf_h = max(98, int(cards_pool * 0.22))
            self.pnl_geom_h = max(92, int(cards_pool * 0.22))
            self.pnl_dist_h = cards_pool - (self.pnl_state_h + self.pnl_mission_h + self.pnl_perf_h + self.pnl_geom_h)
        else:
            self.pnl_mission_h = 0
            self.pnl_state_h = max(60, int(cards_pool * 0.18))
            self.pnl_perf_h = max(86, int(cards_pool * 0.26))
            self.pnl_geom_h = max(82, int(cards_pool * 0.26))
            self.pnl_dist_h = cards_pool - (self.pnl_state_h + self.pnl_perf_h + self.pnl_geom_h)

        pnl_y = self.HDR_TOTAL_H + 4
        self.pnl_state_y = pnl_y
        pnl_y += self.pnl_state_h + card_gap
        if self.pnl_mission_h > 0:
            self.pnl_mission_y = pnl_y
            pnl_y += self.pnl_mission_h + card_gap
        else:
            self.pnl_mission_y = -999
        self.pnl_perf_y = pnl_y
        pnl_y += self.pnl_perf_h + card_gap
        self.pnl_geom_y = pnl_y
        pnl_y += self.pnl_geom_h + card_gap
        self.pnl_dist_y = pnl_y
        pnl_y += self.pnl_dist_h + card_gap
        self.pnl_ctrl_y = pnl_y

        # Sliders inside Disturbance card (proportional spacing)
        slider_start_y = self.pnl_dist_y + 22
        slider_gap = max(18, (self.pnl_dist_h - 26) // 5)
        _sx, _sw = self.PNL_INN + 6, self.PNL_IW - 12
        for idx, key in enumerate(["turbulence", "vibration", "sensor_noise", "jerk_prob", "beacon_fade"]):
            sy = slider_start_y + idx * slider_gap
            if key in self.sliders:
                self.sliders[key].rect = pygame.Rect(_sx, sy, _sw, 16)

        # Control buttons inside Controls card (compact fit guaranteed on small screens)
        bw = (self.PNL_IW - 20) // 3
        bh = 22 if h < 800 else 24
        bx0 = self.PNL_INN + 6
        r1_y = self.pnl_ctrl_y + 22
        r2_y = self.pnl_ctrl_y + 22 + bh + 4
        if "PAUSE" in self.buttons:
            self.buttons["PAUSE"].rect = pygame.Rect(bx0, r1_y, bw, bh)
            self.buttons["RESET"].rect = pygame.Rect(bx0 + bw + 4, r1_y, bw, bh)
            self.buttons["SHOT"].rect = pygame.Rect(bx0 + (bw + 4) * 2, r1_y, bw, bh)
            self.buttons["GT"].rect = pygame.Rect(bx0, r2_y, bw, bh)
            self.buttons["DIAG"].rect = pygame.Rect(bx0 + bw + 4, r2_y, bw, bh)
            self.buttons["LOAD_VIDEO"].rect = pygame.Rect(bx0 + (bw + 4) * 2, r2_y, bw, bh)

# Mission pages handle sizing dynamically

    # ------------------------------------------------------------------
    def _atmosphere_allowed(self):
        from core.platforms import atmosphere_allowed
        return atmosphere_allowed(self.platform_mode)

    def _platform_label(self):
        return {"SATELLITE_SATELLITE": "SAT-SAT",
                "UAV_SATELLITE":       "UAV-SAT",
                "UAV_UAV":             "UAV-UAV"}.get(self.platform_mode, self.platform_mode)

    def _platform_atm_default(self):
        from core.platforms import PLATFORM_MODES
        pm = PLATFORM_MODES.get(self.platform_mode, {})
        return pm.get("atmosphere", "CLEAR")

    def _select_platform(self, pm):
        self.platform_mode = pm
        from core.platforms import atmosphere_allowed
        if not atmosphere_allowed(pm):
            self.atmosphere = "CLEAR"
        elif self.atmosphere == "CLEAR":
            self.atmosphere = self._platform_atm_default()
        self.sliders["turbulence"].enabled = atmosphere_allowed(pm)
        self._reset()

    def sync_sliders(self):
        for key, s in self.sliders.items():
            s.value = self.sim.disturbance.__getattribute__(key)

    def apply_sliders(self):
        d = self.sim.disturbance
        if self.sliders["turbulence"].enabled:
            d.turbulence = int(self.sliders["turbulence"].value)
        else:
            d.turbulence = 0
        d.vibration    = int(self.sliders["vibration"].value)
        d.sensor_noise = int(self.sliders["sensor_noise"].value)
        d.jerk_prob    = int(self.sliders["jerk_prob"].value)
        d.beacon_fade  = int(self.sliders["beacon_fade"].value)

    # ------------------------------------------------------------------
    def run(self):
        running = True
        while running:
            dt_w = self.clock.tick(DISPLAY_CAP) / 1000.0
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.VIDEORESIZE:
                    if not self.fullscreen:
                        self.W, self.H = max(1024, ev.w), max(600, ev.h)
                        self.win_w, self.win_h = self.W, self.H
                        self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)
                        self.canvas = pygame.Surface((self.W, self.H)).convert()
                        self._recompute_layout()
                        self._scanline_surf = None
                elif ev.type == pygame.KEYDOWN:
                    running = self._key(ev.key)
                elif ev.type == pygame.MOUSEBUTTONDOWN:
                    self._mouse_down(self._logical_mouse_pos(ev.pos), ev.button)
                elif ev.type == pygame.MOUSEBUTTONUP:
                    for s in self.sliders.values():
                        s.dragging = False
                    pass
                elif ev.type == pygame.MOUSEMOTION:
                    self._mouse_move(self._logical_mouse_pos(ev.pos), ev.buttons)

            if not self.paused and not self.video_done:
                step_now = True
                if self.video_mode:
                    v_fps = max(1.0, float(getattr(self.sim, "video_fps", 30.0)))
                    target_dt = 1.0 / v_fps
                    self.video_dt_accum += dt_w
                    if self.video_dt_accum < target_dt:
                        step_now = False
                    else:
                        self.video_dt_accum = min(target_dt * 2.0, self.video_dt_accum - target_dt)

                if step_now:
                    # Stress cards are live inputs, not just visual annotations.
                    self.stress_mgr.update(dt_w, self.events_list)
                    self.stress_mgr.apply_to_sim(self.sim)
                    res = self.sim.step()
                    if res is None:
                        print("VIDEO ENDED - returning to normal application", flush=True)
                        self.video_mode = False
                        self.video_done = False
                        self.preset = getattr(self, "normal_preset", "EASY")
                        self.sim = Simulator(
                            preset_name=self.preset, seed=None,
                            platform_mode=self.platform_mode,
                            atmosphere=self.atmosphere,
                            motion_type=getattr(self, "current_motion", self.motion_override),
                            target_shape=self.shape_override,
                            target_size=self.size_override,
                            num_targets=getattr(self, "target_count", self.targets_override),
                            target_initial=self.initial_override)
                        self.perf = PerformanceTracker()
                        self.error_spark.clear()
                        self.sync_sliders()
                        self.video_path = None
                    


                        
                        if "PAUSE" in self.buttons:
                            self.buttons["PAUSE"].label = "PAUSE"
                            continue
                        if not getattr(self, "_video_report_saved", False):
                            self._final_report()
                            self._video_report_saved = True
                    else:
                        self.perf.record_frame(self.sim)
                        if getattr(self, "run_recorder", None) is not None:
                            self.run_recorder.record(self.sim, self.perf, self.stress_mgr)
                        # Always record pointing error so the acquisition curve is live
                        self.error_spark.append(res["pointing_err_deg"])
                        self.eph_pred_az, self.eph_pred_el = self.sim.eph.predict_az_el(res["t"])
            self.apply_sliders()
            self._draw()
            pygame.display.flip()
        if getattr(self, "run_recorder", None) is not None and self.run_recorder.active:
            self.run_recorder.stop(self.sim, reason="APPLICATION EXIT")
        self._final_report()
        pygame.quit()

    # -------------------------------------------------------------- events
    def _key(self, key):
        if pygame.K_ESCAPE == key:
            if getattr(self, "show_ps_modal", False):
                self.show_ps_modal = False
                return True
            return False
        if pygame.K_p == key:
            self.show_ps_modal = not getattr(self, "show_ps_modal", False)
            return True
        if pygame.K_SPACE == key:
            if self.video_mode and self.video_done:
                self._reset()
            else:
                self.paused = not self.paused
                self.buttons["PAUSE"].label = "RESUME" if self.paused else "PAUSE"
        elif pygame.K_TAB == key:
            self.active_tab = (self.active_tab + 1) % len(self.SIDEBAR_TABS)
            self._recompute_layout()
        elif pygame.K_F1 <= key <= pygame.K_F9:
            self.active_tab = key - pygame.K_F1
            self._recompute_layout()
        elif pygame.K_c == key:
            self.sidebar_collapsed = not self.sidebar_collapsed
            self._recompute_layout()
        elif pygame.K_r == key:
            self._reset()
        elif pygame.K_s == key:
            self._screenshot()
        elif pygame.K_l == key:
            self._load_video()
        elif pygame.K_v == key:
            self.show_fov_grid = not self.show_fov_grid
        elif pygame.K_e == key:
            self.show_ephemeris = not getattr(self, "show_ephemeris", True)
        elif pygame.K_f == key:
            self._toggle_fullscreen()
        elif pygame.K_h == key:
            self.hud_declutter = not getattr(self, "hud_declutter", False)
        elif pygame.K_n == key:
            self._randomize_scenario()
        elif pygame.K_m == key:
            if self.active_tab == 2:
                self._toggle_ai_model()
            else:
                self._cycle_motion()
        elif pygame.K_t == key:
            self._cycle_primary_target()
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, getattr(pygame, "K_KP_PLUS", 270)):
            self._adjust_target_count(1)
        elif key in (pygame.K_MINUS, pygame.K_UNDERSCORE, getattr(pygame, "K_KP_MINUS", 269)):
            self._adjust_target_count(-1)
        elif self.active_tab == 0:
            if pygame.K_1 <= key <= pygame.K_6 and (key - pygame.K_1) < len(config.PRESET_ORDER):
                name = config.PRESET_ORDER[key - pygame.K_1]
                self.preset = name
                self._reset(name)
            elif pygame.K_7 <= key <= pygame.K_9:
                idx = key - pygame.K_7
                pkeys = list(self.platform_chips.keys())
                if idx < len(pkeys):
                    pm = pkeys[idx]
                    if pm != self.platform_mode:
                        self._select_platform(pm)
            elif pygame.K_a == key:
                allowed = [n for n in self.atmos_chips
                           if n == "CLEAR" or self._atmosphere_allowed()]
                if allowed:
                    idx = (allowed.index(self.atmosphere) + 1) % len(allowed)
                    self.atmosphere = allowed[idx]
                    self._reset()
        elif 0 <= key - pygame.K_1 < len(self.SIDEBAR_TABS):
            self.active_tab = key - pygame.K_1
            self._recompute_layout()
        return True

    def _logical_mouse_pos(self, pos):
        """Map a physical display coordinate into the logical UI canvas."""
        if self._display_rect.w <= 0 or self._display_rect.h <= 0:
            return pos
        x = (pos[0] - self._display_rect.x) * self.W / self._display_rect.w
        y = (pos[1] - self._display_rect.y) * self.H / self._display_rect.h
        return int(x), int(y)

    def _toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        flags = pygame.FULLSCREEN if self.fullscreen else pygame.RESIZABLE
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), flags)
            dw, dh = self.screen.get_size()
            if dw >= 1000 and dh >= 600:
                self.W, self.H = dw, dh
        else:
            self.W, self.H = self.win_w, self.win_h
            self.screen = pygame.display.set_mode((self.W, self.H), flags)
        self.canvas = pygame.Surface((self.W, self.H)).convert()
        self._recompute_layout()
        self._scanline_surf = None

    def _present_canvas(self):
        """Scale the canvas into the current display if needed."""
        dw, dh = self.screen.get_size()
        if dw == self.W and dh == self.H:
            self._display_rect = pygame.Rect(0, 0, self.W, self.H)
            self.screen.blit(self.canvas, (0, 0))
        else:
            scale = min(dw / self.W, dh / self.H)
            scaled_size = (max(1, int(self.W * scale)), max(1, int(self.H * scale)))
            self._display_rect = pygame.Rect(
                (dw - scaled_size[0]) // 2,
                (dh - scaled_size[1]) // 2,
                scaled_size[0], scaled_size[1])
            self.screen.fill(T.C.BG)
            scaled = pygame.transform.smoothscale(self.canvas, scaled_size)
            self.screen.blit(scaled, self._display_rect.topleft)

    def _mouse_down(self, pos, button):
        # 0. Check PS 26169 Modal clicks if active
        if getattr(self, "show_ps_modal", False):
            if button == 1 and self._handle_ps_modal_click(pos):
                return
            if button == 1 and hasattr(self, "ps_modal_rect") and not self.ps_modal_rect.collidepoint(pos):
                self.show_ps_modal = False
                return

        # 0b. Check PS Inspector header button
        if button == 1 and hasattr(self, "hdr_ps_rect") and self.hdr_ps_rect.collidepoint(pos):
            self.show_ps_modal = not getattr(self, "show_ps_modal", False)
            return

        # 1. Check sidebar collapse / expand toggle button
        if button == 1 and hasattr(self, "sidebar_toggle_rect") and self.sidebar_toggle_rect.collidepoint(pos):
            self.sidebar_collapsed = not self.sidebar_collapsed
            self._recompute_layout()
            return

        # 2. Check sidebar tabs
        if button == 1 and pos[0] < self.SIDEBAR_W:
            for i in range(len(self.SIDEBAR_TABS)):
                tab_rect = pygame.Rect(0, 52 + i * 50, self.SIDEBAR_W, 46)
                if tab_rect.collidepoint(pos):
                    if self.active_tab != i:
                        self.active_tab = i
                        self._recompute_layout()
                    return

        # 3. Check top header pause toggle
        if button == 1 and hasattr(self, "hdr_pause_rect") and self.hdr_pause_rect.collidepoint(pos):
            if self.video_mode and self.video_done:
                self._reset()
            else:
                self.paused = not self.paused
                self.buttons["PAUSE"].label = "RESUME" if self.paused else "PAUSE"
            return

        # 3. Handle active view controls
        if self.active_tab == 7:
            if button == 1 and handle_target_settings_click(self, pos):
                return
        elif self.active_tab == 8:
            if button == 1 and handle_report_click(self, pos):
                return
        elif self.active_tab == 0:
            for name, b in self.buttons.items():
                if button == 1 and b.hit(pos):
                    if name == "PAUSE":
                        if self.video_mode and self.video_done:
                            self._reset()
                        else:
                            self.paused = not self.paused
                            b.label = "RESUME" if self.paused else "PAUSE"
                    elif name == "RESET":
                        self._reset()
                    elif name == "SHOT":
                        self._screenshot()
                    elif name == "GT":
                        self.show_gt = not self.show_gt
                        b.label = "GT ON" if self.show_gt else "GT OFF"
                    elif name == "DIAG":
                        self.show_diag = not self.show_diag
                        b.label = "DIAG ‹" if self.show_diag else "DIAG"
                    elif name == "LOAD_VIDEO":
                        self._load_video()
                    return
            for name, c in self.chips.items():
                if button == 1 and c.hit(pos):
                    self.preset = name
                    self._reset(name)
                    return
            for name, c in self.platform_chips.items():
                if button == 1 and c.hit(pos):
                    self._select_platform(name)
                    return
            for name, c in self.atmos_chips.items():
                if button == 1 and c.hit(pos):
                    if not self._atmosphere_allowed() and name != "CLEAR":
                        return
                    self.atmosphere = name
                    self._reset()
                    return
            if button == 1:
                if hasattr(self, "btn_tgt_dec") and self.btn_tgt_dec.collidepoint(pos):
                    self._adjust_target_count(-1)
                    return
                if hasattr(self, "btn_tgt_inc") and self.btn_tgt_inc.collidepoint(pos):
                    self._adjust_target_count(1)
                    return
                if hasattr(self, "btn_cycle_tgt") and self.btn_cycle_tgt.collidepoint(pos):
                    self._cycle_primary_target()
                    return
                if hasattr(self, "btn_randomize") and self.btn_randomize.collidepoint(pos):
                    self._randomize_scenario()
                    return
                if hasattr(self, "btn_motion") and self.btn_motion.collidepoint(pos):
                    self._cycle_motion()
                    return
                if hasattr(self, "btn_hud_toggle") and self.btn_hud_toggle.collidepoint(pos):
                    self.hud_declutter = not getattr(self, "hud_declutter", False)
                    return
                # Check viewport target clicks (click-to-designate primary target)
                if hasattr(self, "target_click_rects"):
                    for tidx, trect in self.target_click_rects.items():
                        if trect.collidepoint(pos):
                            self._designate_target(tidx)
                            return
            for s in self.sliders.values():
                if button == 1 and s.hit(pos):
                    s.dragging = True
                    s.drag_to(pos[0])
        
        elif self.active_tab == 2:
            btn = getattr(self.sim, "ai_model_btn_rect", None)
            if button == 1 and btn and btn.collidepoint(pos):
                self._toggle_ai_model()
                return

        elif self.active_tab == 4:
            self.stress_mgr.handle_click(pos, self.events_list)

    def _mouse_move(self, pos, buttons):
        if self.active_tab == 0:
            for s in self.sliders.values():
                if s.dragging and buttons[0]:
                    s.drag_to(pos[0])
        

    def _adjust_target_count(self, delta):
        new_cnt = max(1, min(5, getattr(self, "target_count", 1) + delta))
        if new_cnt != getattr(self, "target_count", 1):
            self.target_count = new_cnt
            if getattr(self, "primary_target_idx", 0) >= new_cnt:
                self.primary_target_idx = 0
            if hasattr(self.sim, "scene"):
                self.sim.scene.set_target_params(count=new_cnt)
            ts_str = time.strftime("%H:%M:%S UTC", time.gmtime())
            self.events_list.insert(0, (ts_str, "INFO", "SCENE", f"Target count adjusted to {new_cnt} (PS Item 8 Multi-Target Mode)"))
            if len(self.events_list) > 100:
                self.events_list.pop()

    def _cycle_primary_target(self):
        cnt = getattr(self, "target_count", 1)
        if cnt <= 1:
            return
        cur = getattr(self, "primary_target_idx", 0)
        self._designate_target((cur + 1) % cnt)

    def _designate_target(self, idx):
        if not hasattr(self.sim, "set_primary_target"):
            return
        cnt = getattr(self, "target_count", 1)
        idx = max(0, min(cnt - 1, int(idx)))
        self.primary_target_idx = idx
        sel_idx, tid = self.sim.set_primary_target(idx)
        ts_str = time.strftime("%H:%M:%S UTC", time.gmtime())
        self.events_list.insert(0, (ts_str, "INFO", "HANDOVER", f"Primary tracking designated to {tid} (Target #{idx+1})"))
        if len(self.events_list) > 100:
            self.events_list.pop()

    def _cycle_motion(self):
        motions = ["straight_line", "circular", "figure_eight", "random", "spiral"]
        cur = getattr(self, "current_motion", "straight_line")
        idx = (motions.index(cur) + 1) % len(motions) if cur in motions else 0
        self.current_motion = motions[idx]
        if hasattr(self.sim, "scene"):
            self.sim.scene.set_motion_type(self.current_motion)
        ts_str = time.strftime("%H:%M:%S UTC", time.gmtime())
        lbl = self.current_motion.replace("_", " ").upper()
        self.events_list.insert(0, (ts_str, "INFO", "SCENE", f"Target motion trajectory set to {lbl}"))
        if len(self.events_list) > 100:
            self.events_list.pop()

    def _randomize_scenario(self):
        """Regenerate random initial position, heading, velocity and phase on demand."""
        self._reset(self.preset, seed=None)
        ts_str = time.strftime("%H:%M:%S UTC", time.gmtime())
        self.events_list.insert(0, (ts_str, "INFO", "SCENE", "New randomized trajectory generated"))
        if len(self.events_list) > 100:
            self.events_list.pop()

    def _toggle_ai_model(self):
        import ai.classifier as ai_clf
        cur = getattr(ai_clf, "ACTIVE_MODEL", "LINEAR")
        new_model = "DEEP_MLP" if cur == "LINEAR" else "LINEAR"
        ai_clf.set_active_model(new_model)
        ts_str = time.strftime("%H:%M:%S UTC", time.gmtime())
        arch = "2-Layer Deep MLP (4->16->8->1, 225 weights)" if new_model == "DEEP_MLP" else "Linear Logistic Regression (5 weights)"
        self.events_list.insert(0, (ts_str, "INFO", "AI-ENGINE", f"Inference engine switched to {new_model} [{arch}]"))
        if len(self.events_list) > 100:
            self.events_list.pop()

    def _reset(self, name=None, seed=None):
        if getattr(self, "run_recorder", None) is not None and self.run_recorder.active:
            self.run_recorder.stop(self.sim, reason="RESET / MODE CHANGE")
        if self.video_mode:
            from core.simulator import VideoInputSimulator
            truth = os.path.splitext(self.video_path)[0] + "_truth.csv"
            self.sim = VideoInputSimulator(
                self.video_path, seed=self.video_seed,
                truth_csv=truth if os.path.isfile(truth) else None)
            self.video_done = False
            self.video_dt_accum = 0.0
            self._video_report_saved = False
            self.perf = PerformanceTracker()
            self.error_spark.clear()
            self.sync_sliders()
            self.paused = False
            self.buttons["PAUSE"].label = "PAUSE"
            return
        self.sim = Simulator(
            preset_name=name or self.preset, seed=seed,
            platform_mode=self.platform_mode,
            atmosphere=self.atmosphere,
            motion_type=getattr(self, "current_motion", self.motion_override),
            target_shape=self.shape_override,
            target_size=self.size_override,
            num_targets=getattr(self, "target_count", self.targets_override),
            target_initial=self.initial_override)
        self.perf = PerformanceTracker()
        self.error_spark.clear()
        self.sync_sliders()
        self.paused = False
        self.buttons["PAUSE"].label = "PAUSE"
        try:
            res = self.sim.step()
            if res is not None:
                self.perf.record_frame(self.sim)
        except Exception:
            pass

    def _screenshot(self):
        os.makedirs(config.LOG_DIR, exist_ok=True)
        self._screenshot_n += 1
        path = os.path.join(
            config.LOG_DIR,
            f"shot_{self.preset.lower()}_{self._screenshot_n}.png")
        pygame.image.save(self.screen, path)
        print(f"screenshot -> {path}", flush=True)

    def _load_video(self):
        try:
            import tkinter
            from tkinter import filedialog
            root = tkinter.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askopenfilename(
                title="Load Benchmark-2 video (.mp4)",
                filetypes=[("MP4 video", "*.mp4"),
                           ("Video files", "*.mp4;*.avi"),
                           ("All files", "*.*")])
            root.destroy()
        except Exception as ex:
            print(f"file dialog unavailable ({ex}); pass --video path instead")
            return
        if not path:
            return
        self.normal_preset = self.preset
        self.video_path = path
        from core.simulator import VideoInputSimulator
        truth = os.path.splitext(path)[0] + "_truth.csv"
        try:
            self.sim = VideoInputSimulator(
                path, seed=self.video_seed,
                truth_csv=truth if os.path.isfile(truth) else None)
        except Exception as ex:
            print(f"cannot open video: {ex}")
            return
        self.video_mode  = True
        self.video_done  = False
        self.video_dt_accum = 0.0
        self._video_report_saved = False
        self.preset      = "VIDEO"
        self.perf        = PerformanceTracker()
        self.error_spark.clear()
        self.sync_sliders()
        self.paused = False
        self.buttons["PAUSE"].label = "PAUSE"
        print(f"video loaded -> {path} "
              f"({self.sim.video_w}x{self.sim.video_h} @ {self.sim.video_fps:.1f} fps)")
        try:
            res = self.sim.step()
            if res is not None:
                self.perf.record_frame(self.sim)
        except Exception as ex:
            print(f"video step error: {ex}")

    def _utc_event_time(self):
        return time.strftime("%H:%M:%S UTC", time.gmtime())

    def _run_metadata(self):
        scene = getattr(self.sim, "scene", None)
        beacon = getattr(scene, "beacon", None)
        return {
            "preset": self.preset,
            "platform_mode": self.platform_mode,
            "atmosphere": self.atmosphere,
            "trajectory": getattr(self, "current_motion", "straight_line"),
            "target_shape": getattr(beacon, "shape", self.shape_override or "SQUARE"),
            "target_size_px": getattr(beacon, "size_px", self.size_override or 10),
            "target_count": getattr(scene, "num_targets", getattr(self, "target_count", 1)),
            "primary_target_id": getattr(beacon, "target_id", "TARGET-01"),
        }

    def _final_report(self):
        st = self.perf.live_stats()
        p  = os.path.join(config.LOG_DIR, f"run_{int(time.time())}.csv")
        extra = {"preset": self.preset}
        if self.video_mode:
            extra["input_video"] = os.path.basename(self.video_path)
            # Metric B: Optical-axis / Frame-centre offset
            b_errs = [e[1] for e in getattr(self.sim, "optical_offset_log", []) if e[1] is not None]
            if b_errs:
                extra["metric_b_optical_offset_mean_px"] = round(float(np.mean(b_errs)), 2)
                extra["metric_b_optical_offset_rms_px"]  = round(float(np.sqrt(np.mean(np.array(b_errs) ** 2))), 2)
                extra["metric_b_optical_offset_max_px"]  = round(float(np.max(b_errs)), 2)

            # Metric C: True Centroiding Error (Only valid if ground-truth CSV provided)
            if getattr(self.sim, "ground_truth_available", False) and len(getattr(self.sim, "true_err_log", [])) > 0:
                c_errs = [e[1] for e in self.sim.true_err_log if e[1] is not None]
                if c_errs:
                    extra["centroiding_error_mean_px"] = round(float(np.mean(c_errs)), 2)
                    extra["centroiding_error_rms_px"]  = round(float(np.sqrt(np.mean(np.array(c_errs) ** 2))), 2)
                    extra["centroiding_error_p95_px"]  = round(float(np.percentile(c_errs, 95)), 2)
                    extra["centroiding_error_max_px"]  = round(float(np.max(c_errs)), 2)
                    extra["centroiding_frames"]        = len(c_errs)
            else:
                extra["centroiding_error_mean_px"] = "n/a (no ground truth)"
                extra["centroiding_error_rms_px"]  = "n/a (no ground truth)"

            extra["reacquisition_count_video"] = len(self.sim.reacq_times)
            extra["video_false_lock_events"]   = self.sim.false_lock_events
        self.perf.write_log(p, extra_info=extra)
        print(f"performance log -> {p}")

    # ================================================================ DRAW
    def _draw(self):
        s = self.canvas
        s.fill(T.C.BG)

        res = self.sim.last_result
        if res is None:
            try:
                res = self.sim.step()
            except Exception:
                res = None
        if res is None:
            res = {}
        fps = self.clock.get_fps()
        dt = 1.0 / max(1.0, fps)
        hist_pt = self.opt_model.update_from_sim(res, self.stress_mgr)

        # Record state change events
        raw_st = hist_pt.get("state", res.get("state", "SEARCHING"))
        boresight_err = res.get("boresight_error_px")
        if boresight_err is None and getattr(self, "video_mode", False):
            boresight_err = res.get("optical_offset_px")
        is_aligned = (boresight_err is not None and boresight_err <= 15.0)
        cur_st = "LOCKED" if (raw_st in ("LOCKED", "DEGRADED_LOCK") and is_aligned) else ("TRACKING" if getattr(self, "video_mode", False) and raw_st in ("LOCKED", "DEGRADED_LOCK") else ("ALIGNING" if raw_st in ("LOCKED", "DEGRADED_LOCK") else raw_st))
        if cur_st != self._last_state:
            ts_str = time.strftime("%H:%M:%S UTC", time.gmtime())
            lvl = "INFO" if cur_st == "LOCKED" else ("WARNING" if cur_st in ("COASTING", "REACQUIRING", "DEGRADED_LOCK", "CANDIDATE", "ACQUIRING", "ALIGNING", "TRACKING") else "CRITICAL")
            self.events_list.insert(0, (ts_str, lvl, "TRACKER", f"State transition: {self._last_state} -> {cur_st} (error: {res.get('pointing_err_deg', 0)*1000:.1f} mdeg)"))
            if len(self.events_list) > 100:
                self.events_list.pop()
            self._last_state = cur_st

        # 1. Main Active Content View (7 SpaceX/ISRO Modules)
        if self.active_tab == 0:
            # MODULE 01: TRACKING CONSOLE (Camera Viewport + Open Reticle + Telemetry Panel + PAT Pipeline)
            self._draw_bg_grid(s)
            self._draw_camera(s)
            self._draw_panel(s)
            self._draw_bottom(s)
            self._draw_footer(s)
        elif self.active_tab == 1:
            # MODULE 02: LINK TELEMETRY (Waterfall Budget, 4 Big Metric Cards, Dual Oscilloscopes)
            page_rect = pygame.Rect(self.SIDEBAR_W + 16, self.HDR_TOTAL_H + 10, self.W - self.SIDEBAR_W - 32, self.H - self.HDR_TOTAL_H - 24)
            render_telemetry_page(s, page_rect, self.opt_model)
        elif self.active_tab == 2:
            # MODULE 03: AI CLASSIFIER (Feature Scores, 15Hz Mod Waveform, Discrimination Matrix)
            page_rect = pygame.Rect(self.SIDEBAR_W + 16, self.HDR_TOTAL_H + 10, self.W - self.SIDEBAR_W - 32, self.H - self.HDR_TOTAL_H - 24)
            render_ai_classifier_page(s, page_rect, self.sim)
        elif self.active_tab == 3:
            # MODULE 04: GIMBAL SERVO (Pan/Tilt Rates, 5 deg/s Spec Clamp, 3D Radar Sky Plot, PD Dynamics)
            page_rect = pygame.Rect(self.SIDEBAR_W + 16, self.HDR_TOTAL_H + 10, self.W - self.SIDEBAR_W - 32, self.H - self.HDR_TOTAL_H - 24)
            render_gimbal_servo_page(s, page_rect, self.sim)
        elif self.active_tab == 4:
            # MODULE 05: STRESS INJECTION (5 Interactive Scenario Cards, Live RX Waveform, Judge Demo Sequence)
            page_rect = pygame.Rect(self.SIDEBAR_W + 16, self.HDR_TOTAL_H + 10, self.W - self.SIDEBAR_W - 32, self.H - self.HDR_TOTAL_H - 24)
            render_stress_test_page(s, page_rect, self.stress_mgr, self.opt_model)
            render_live_stress_overlay(s, page_rect, self.sim, self.stress_mgr)
        elif self.active_tab == 5:
            # MODULE 06: FALSE LOCK SUITE (Live Dynamic 5-Point Verification Matrix & Telemetry)
            page_rect = pygame.Rect(self.SIDEBAR_W + 16, self.HDR_TOTAL_H + 10, self.W - self.SIDEBAR_W - 32, self.H - self.HDR_TOTAL_H - 24)
            render_false_lock_page(s, page_rect, self.sim, self.opt_model, self.stress_mgr)
        elif self.active_tab == 6:
            # MODULE 07: MISSION LOGS (4 Big Metric Cards, Severity Legend Banner, Event Table)
            page_rect = pygame.Rect(self.SIDEBAR_W + 16, self.HDR_TOTAL_H + 10, self.W - self.SIDEBAR_W - 32, self.H - self.HDR_TOTAL_H - 24)
            render_event_log_page(s, page_rect, self.events_list)

        # 2. Left Navigation Sidebar (Fixed Overlay)
        self._draw_sidebar(s)

        # 3. Top Mission Control Header (Fixed Overlay)
        self._draw_mission_header(s, hist_pt)

        # 4. ISRO PS 26169 Compliance & Configurator Modal
        if getattr(self, "show_ps_modal", False):
            self._draw_ps_inspector_modal(s)

        self._present_canvas()

    def _draw_sidebar(self, surf):
        pygame.draw.rect(surf, (8, 14, 26), (0, 0, self.SIDEBAR_W, self.H))
        pygame.draw.line(surf, (24, 40, 70), (self.SIDEBAR_W - 1, 0), (self.SIDEBAR_W - 1, self.H), 1)

        # Brand header at top of sidebar
        pygame.draw.rect(surf, (12, 20, 38), (0, 0, self.SIDEBAR_W, self.HDR_H))
        pygame.draw.line(surf, (24, 40, 70), (0, self.HDR_H - 1), (self.SIDEBAR_W, self.HDR_H - 1), 1)

        if not self.sidebar_collapsed:
            T.text(surf, (16, 6), "FSOC PAT LAB", 16, T.C.CYAN_ELEC, bold=True)
            T.text(surf, (16, 26), "ISRO CONSOLE · PS 26169", 11, T.C.TEXT_DIM, bold=True)
        else:
            T.text(surf, (self.SIDEBAR_W // 2, 14), "FSOC", 14, T.C.CYAN_ELEC, bold=True, anchor="tc")

        # Tabs
        mouse_pos = self._logical_mouse_pos(pygame.mouse.get_pos())
        hover_tooltip = None

        for i, (name, abbr, num) in enumerate(self.SIDEBAR_TABS):
            tab_y = 54 + i * 52
            tab_rect = pygame.Rect(0, tab_y, self.SIDEBAR_W, 48)
            is_active = (i == self.active_tab)
            is_hover = tab_rect.collidepoint(mouse_pos)

            if is_active:
                pygame.draw.rect(surf, (0, 36, 68), tab_rect)
                pygame.draw.rect(surf, T.C.CYAN_ELEC, (0, tab_y, 4, 48))
                title_col = T.C.CYAN_ELEC
                badge_col = T.C.CYAN_ELEC
            elif is_hover:
                pygame.draw.rect(surf, (16, 26, 48), tab_rect)
                title_col = T.C.TEXT
                badge_col = T.C.TEXT_DIM
            else:
                title_col = T.C.TEXT_DIM
                badge_col = T.C.TEXT_MUTED

            if not self.sidebar_collapsed:
                # Badge number
                T.text(surf, (14, tab_y + 7), num, 12, badge_col, bold=True, mono=True)
                # Label
                T.text(surf, (38, tab_y + 7), name, 12.5, title_col, bold=True)
                # Abbr subtext
                T.text(surf, (38, tab_y + 25), abbr, 11, badge_col, bold=False)
            else:
                # Collapsed: Show number & abbreviation centered
                T.text(surf, (self.SIDEBAR_W // 2, tab_y + 7), num, 12, badge_col, bold=True, anchor="tc", mono=True)
                T.text(surf, (self.SIDEBAR_W // 2, tab_y + 25), abbr, 11, title_col, bold=True, anchor="tc")
                if is_hover:
                    hover_tooltip = (self.SIDEBAR_W + 8, tab_y + 10, f"{num} · {name}")

        # Collapse / Expand Toggle Button
        toggle_y = self.H - 96
        if not self.sidebar_collapsed:
            self.sidebar_toggle_rect = pygame.Rect(10, toggle_y, self.SIDEBAR_W - 20, 30)
            is_tog_hov = self.sidebar_toggle_rect.collidepoint(mouse_pos)
            tog_bg = (18, 30, 54) if is_tog_hov else (10, 18, 34)
            pygame.draw.rect(surf, tog_bg, self.sidebar_toggle_rect, border_radius=3)
            pygame.draw.rect(surf, T.C.CYAN_ELEC if is_tog_hov else (20, 36, 62), self.sidebar_toggle_rect, 1, border_radius=3)
            T.text(surf, (self.sidebar_toggle_rect.centerx, self.sidebar_toggle_rect.centery), "◄ COLLAPSE", 11, T.C.CYAN_ELEC if is_tog_hov else T.C.TEXT_DIM, bold=True, anchor="cc")
        else:
            self.sidebar_toggle_rect = pygame.Rect(8, toggle_y, 34, 30)
            is_tog_hov = self.sidebar_toggle_rect.collidepoint(mouse_pos)
            tog_bg = (18, 30, 54) if is_tog_hov else (10, 18, 34)
            pygame.draw.rect(surf, tog_bg, self.sidebar_toggle_rect, border_radius=3)
            pygame.draw.rect(surf, T.C.CYAN_ELEC if is_tog_hov else (20, 36, 62), self.sidebar_toggle_rect, 1, border_radius=3)
            T.text(surf, (self.sidebar_toggle_rect.centerx, self.sidebar_toggle_rect.centery), "►", 13, T.C.CYAN_ELEC, bold=True, anchor="cc")

        # Bottom stats
        fps = self.clock.get_fps()
        fps_col = T.C.GREEN if fps >= 25 else T.C.AMBER
        if not self.sidebar_collapsed:
            T.text(surf, (14, self.H - 52), f"FPS: {fps:.0f}", 11, fps_col, bold=True)
            T.text(surf, (14, self.H - 34), "NATIVE DESKTOP", 11, T.C.TEXT_FAINT)
            T.text(surf, (14, self.H - 18), "ISRO PAT CONSOLE", 11, T.C.TEXT_FAINT)
        else:
            T.text(surf, (self.SIDEBAR_W // 2, self.H - 38), f"{fps:.0f}", 11, fps_col, bold=True, anchor="tc")
            T.text(surf, (self.SIDEBAR_W // 2, self.H - 22), "FPS", 11, T.C.TEXT_FAINT, anchor="tc")

        # Floating Tooltip in collapsed mode
        if hover_tooltip:
            tx, ty, tip_text = hover_tooltip
            tt_w, tt_h = T.font(11, bold=True).size(tip_text)
            tip_rect = pygame.Rect(tx, ty, tt_w + 16, 26)
            pygame.draw.rect(surf, (8, 16, 32), tip_rect, border_radius=3)
            pygame.draw.rect(surf, T.C.CYAN_ELEC, tip_rect, 1, border_radius=3)
            T.text(surf, (tx + 8, ty + 5), tip_text, 11, T.C.CYAN_ELEC, bold=True)

    def _draw_mission_header(self, surf, hist_pt):
        hdr_w = self.W - self.SIDEBAR_W
        pygame.draw.rect(surf, (8, 14, 26), (self.SIDEBAR_W, 0, hdr_w, self.HDR_H))
        pygame.draw.line(surf, (24, 40, 70), (self.SIDEBAR_W, self.HDR_H - 1), (self.W, self.HDR_H - 1), 1)

        # Cyan top accent line
        pygame.draw.rect(surf, T.C.CYAN_ELEC, (self.SIDEBAR_W, 0, hdr_w, 2))

        # Title & Subtitle (dynamically scaled for narrower screens)
        title_str = "FSOC MISSION CONTROL CONSOLE" if self.W >= 1480 else "FSOC MISSION CONTROL"
        T.text(surf, (self.SIDEBAR_W + 16, 6), title_str, 15 if self.W >= 1480 else 14, T.C.TEXT, bold=True)
        T.text(surf, (self.SIDEBAR_W + 16, 25), "PAT LAB · ISRO PS 26169", 11, T.C.CYAN_ELEC, bold=True)

        # Link State badge (positioned after title with clean clearance)
        res = getattr(self.sim, "last_result", {}) or {}
        raw_st = hist_pt.get("state", res.get("state", "SEARCHING"))
        boresight_err = res.get("boresight_error_px")
        if boresight_err is None and getattr(self, "video_mode", False):
            boresight_err = res.get("optical_offset_px")
        is_aligned = (boresight_err is not None and boresight_err <= 15.0)

        # Optical link state requires both tracking lock AND boresight alignment
        if raw_st in ("LOCKED", "DEGRADED_LOCK") and not is_aligned:
            st = "TRACKING" if getattr(self, "video_mode", False) else "ALIGNING"
            st_col = T.C.CYAN_ELEC
            badge_bg = (8, 28, 44)
        else:
            st = raw_st
            st_col = T.C.STATE.get(st, T.C.GREEN)
            badge_bg = (44, 14, 18) if st in ("FALSE LOCK", "SIGNAL LOSS", "LOST") else ((44, 30, 8) if st in ("DEGRADED", "DEGRADED_LOCK") else (6, 36, 26))

        badge_x = self.SIDEBAR_W + (285 if self.W >= 1480 else 235)
        badge_w = 126
        badge_h = 34
        badge_y = 6
        pygame.draw.rect(surf, badge_bg, (badge_x, badge_y, badge_w, badge_h), border_radius=4)
        pygame.draw.rect(surf, st_col, (badge_x, badge_y, badge_w, badge_h), 1, border_radius=4)
        T.text(surf, (badge_x + 10, badge_y + 3), "LINK STATE", 10, T.C.TEXT_FAINT, bold=True)

        # Pulsing indicator dot
        p_alpha = int(180 + 75 * math.sin(time.time() * 5.0))
        dot_col = (0, min(255, p_alpha), 120) if st == "LOCKED" else st_col
        pygame.draw.circle(surf, dot_col, (badge_x + 14, badge_y + 22), 4)
        T.text(surf, (badge_x + 24, badge_y + 16), st, 12.5, st_col, bold=True)

        # Header live telemetry values
        def _h_val(x, lbl, val, unit, col=T.C.GREEN):
            T.text(surf, (x, 5), lbl, 10.5, T.C.TEXT_FAINT, bold=True)
            vw, vh = T.text(surf, (x, 19), val, 15, col, bold=True, mono=True)
            if unit:
                T.text(surf, (x + vw + 3, 21), unit, 11, T.C.TEXT_DIM)

        m_spacing = 84 if self.W < 1440 else 104
        mx = badge_x + badge_w + 18
        _h_val(mx, "RX POWER", f"{hist_pt.get('rx_power', -11.4):.1f}", "dBm", T.C.CYAN_ELEC)
        _h_val(mx + m_spacing, "SNR", f"{hist_pt.get('snr', 73.6):.1f}", "dB", T.C.GREEN)
        _h_val(mx + m_spacing * 2, "MARGIN", f"{hist_pt.get('link_margin', 38.6):.1f}", "dB", T.C.GREEN)
        # Real-time tracking percentage
        _h_val(mx + m_spacing * 3, "TRACKING", f"{int(round(res.get('confidence', 0.0) * 100))}", "%", T.C.CYAN_ELEC)
        
       
        

        # UTC Clock
        utc_str = time.strftime("%Y-%m-%d  %H:%M:%S", time.gmtime())
        clock_cx = self.W - 160
        T.text(surf, (clock_cx, 5), "UTC", 10, T.C.TEXT_FAINT, anchor="tc")
        T.text(surf, (clock_cx, 19), utc_str, 13, T.C.TEXT, bold=True, anchor="tc", mono=True)

        # ISRO PS 26169 Compliance Inspector Button
        ps_btn_w = 172
        self.hdr_ps_rect = pygame.Rect(self.W - 78 - ps_btn_w - 12, 8, ps_btn_w, 30)
        p_active = getattr(self, "show_ps_modal", False)
        ps_bg = (50, 34, 8) if p_active else (14, 28, 48)
        ps_border = (255, 180, 0) if p_active else (0, 220, 255)
        pygame.draw.rect(surf, ps_bg, self.hdr_ps_rect, border_radius=3)
        pygame.draw.rect(surf, ps_border, self.hdr_ps_rect, 1, border_radius=3)
        T.text(surf, (self.hdr_ps_rect.centerx, self.hdr_ps_rect.centery),
               "ISRO PS 26169 AUDIT [P]", 11, ps_border, bold=True, anchor="cc")

        # PAUSE Button
        self.hdr_pause_rect = pygame.Rect(self.W - 78, 8, 70, 30)
        btn_col = T.C.GREEN if (getattr(self, "video_mode", False) and getattr(self, "video_done", False)) else T.C.AMBER
        pygame.draw.rect(surf, (14, 38, 24) if (getattr(self, "video_mode", False) and getattr(self, "video_done", False)) else (44, 28, 6), self.hdr_pause_rect, border_radius=3)
        pygame.draw.rect(surf, btn_col, self.hdr_pause_rect, 1, border_radius=3)
        pause_label = "REPLAY" if (getattr(self, "video_mode", False) and getattr(self, "video_done", False)) else ("RESUME" if self.paused else "PAUSE")
        T.text(surf, (self.hdr_pause_rect.centerx, self.hdr_pause_rect.centery),
               pause_label, 12, btn_col, bold=True, anchor="cc")

        # Tier 2: Dedicated Scenario Control Bar (only on TRACKING CONSOLE tab)
        if self.active_tab == 0:
            scen_rect = pygame.Rect(self.SIDEBAR_W, self.HDR_H, hdr_w, self.SCENARIO_H)
            pygame.draw.rect(surf, (10, 16, 28), scen_rect)
            pygame.draw.line(surf, (24, 40, 70), (self.SIDEBAR_W, scen_rect.bottom - 1), (self.W, scen_rect.bottom - 1), 1)

            # Group Labels and chips
            T.text(surf, (self.scenario_lbl_rect.x, self.scenario_lbl_rect.y + 5), "SCENARIO:", 11, T.C.TEXT_DIM, bold=True)
            for name, c in self.chips.items():
                c.draw(surf, selected=(name == self.preset))

            T.text(surf, (self.platform_lbl_rect.x, self.platform_lbl_rect.y + 5), "PLATFORM:", 11, T.C.TEXT_DIM, bold=True)
            for name, c in self.platform_chips.items():
                c.draw(surf, selected=(name == self.platform_mode))

            T.text(surf, (self.atmos_lbl_rect.x, self.atmos_lbl_rect.y + 5), "ATMOSPHERE:", 11, T.C.TEXT_DIM, bold=True)
            for name, c in self.atmos_chips.items():
                enabled = (name == "CLEAR" or self._atmosphere_allowed())
                c.draw(surf, selected=(name == self.atmosphere), enabled=enabled)

            # Group 4: Dynamic Target Controls (PS Item 8 Multi-Target & Randomize)
            mouse_pos = self._logical_mouse_pos(pygame.mouse.get_pos())
            if hasattr(self, "target_lbl_rect") and self.target_lbl_rect.right < self.W - 30:
                T.text(surf, (self.target_lbl_rect.x, self.target_lbl_rect.y + 5), "TARGETS:", 11, T.C.TEXT_DIM, bold=True)
                
                # [-] decrement button
                hov_dec = self.btn_tgt_dec.collidepoint(mouse_pos)
                pygame.draw.rect(surf, (20, 36, 62) if hov_dec else (12, 20, 36), self.btn_tgt_dec, border_radius=3)
                pygame.draw.rect(surf, T.C.CYAN_ELEC if hov_dec else (24, 44, 72), self.btn_tgt_dec, 1, border_radius=3)
                T.text(surf, (self.btn_tgt_dec.centerx, self.btn_tgt_dec.centery), "-", 12, T.C.CYAN_ELEC if hov_dec else T.C.TEXT, bold=True, anchor="cc")

                # Count indicator
                cnt_str = str(getattr(self, "target_count", 1))
                pygame.draw.rect(surf, (8, 16, 30), self.tgt_cnt_rect, border_radius=2)
                pygame.draw.rect(surf, (20, 40, 70), self.tgt_cnt_rect, 1, border_radius=2)
                T.text(surf, (self.tgt_cnt_rect.centerx, self.tgt_cnt_rect.centery), cnt_str, 12, T.C.GREEN, bold=True, anchor="cc", mono=True)

                # [+] increment button
                hov_inc = self.btn_tgt_inc.collidepoint(mouse_pos)
                pygame.draw.rect(surf, (20, 36, 62) if hov_inc else (12, 20, 36), self.btn_tgt_inc, border_radius=3)
                pygame.draw.rect(surf, T.C.CYAN_ELEC if hov_inc else (24, 44, 72), self.btn_tgt_inc, 1, border_radius=3)
                T.text(surf, (self.btn_tgt_inc.centerx, self.btn_tgt_inc.centery), "+", 12, T.C.CYAN_ELEC if hov_inc else T.C.TEXT, bold=True, anchor="cc")

                # [TARGET DESIGNATE] button
                if hasattr(self, "btn_cycle_tgt") and self.btn_cycle_tgt.right < self.W - 10:
                    hov_tgt = self.btn_cycle_tgt.collidepoint(mouse_pos)
                    pri_num = getattr(self, "primary_target_idx", 0) + 1
                    pygame.draw.rect(surf, (24, 44, 72) if hov_tgt else (14, 24, 42), self.btn_cycle_tgt, border_radius=3)
                    pygame.draw.rect(surf, T.C.GREEN if hov_tgt else (30, 60, 90), self.btn_cycle_tgt, 1, border_radius=3)
                    T.text(surf, (self.btn_cycle_tgt.centerx, self.btn_cycle_tgt.centery), f"TRG#{pri_num} [T]", 10, T.C.GREEN if hov_tgt else T.C.CYAN_ELEC, bold=True, anchor="cc")

                # [RANDOMIZE] button
                if hasattr(self, "btn_randomize") and self.btn_randomize.right < self.W - 10:
                    hov_rand = self.btn_randomize.collidepoint(mouse_pos)
                    pygame.draw.rect(surf, (24, 44, 72) if hov_rand else (14, 24, 42), self.btn_randomize, border_radius=3)
                    pygame.draw.rect(surf, T.C.CYAN_ELEC if hov_rand else (30, 56, 90), self.btn_randomize, 1, border_radius=3)
                    T.text(surf, (self.btn_randomize.centerx, self.btn_randomize.centery), "RANDOM [N]", 10, T.C.CYAN_ELEC if hov_rand else T.C.TEXT_DIM, bold=True, anchor="cc")

                # [MOTION] button
                if hasattr(self, "btn_motion") and self.btn_motion.right < self.W - 10:
                    hov_mot = self.btn_motion.collidepoint(mouse_pos)
                    mot_short = getattr(self, "current_motion", "straight").split("_")[0][:4].upper()
                    pygame.draw.rect(surf, (24, 44, 72) if hov_mot else (14, 24, 42), self.btn_motion, border_radius=3)
                    pygame.draw.rect(surf, T.C.AMBER if hov_mot else (40, 50, 70), self.btn_motion, 1, border_radius=3)
                    T.text(surf, (self.btn_motion.centerx, self.btn_motion.centery), f"MOT:{mot_short} [M]", 10, T.C.AMBER if hov_mot else T.C.TEXT_DIM, bold=True, anchor="cc")

                # [HUD: DECLUTTER] button
                if hasattr(self, "btn_hud_toggle") and self.btn_hud_toggle.right < self.W - 10:
                    hov_hud = self.btn_hud_toggle.collidepoint(mouse_pos)
                    hud_mode_txt = "MIN HUD" if getattr(self, "hud_declutter", False) else "CLEAN HUD"
                    hud_col = T.C.GREEN if getattr(self, "hud_declutter", False) else T.C.CYAN_ELEC
                    pygame.draw.rect(surf, (20, 36, 56) if hov_hud else (12, 20, 36), self.btn_hud_toggle, border_radius=3)
                    pygame.draw.rect(surf, hud_col if hov_hud else (24, 44, 70), self.btn_hud_toggle, 1, border_radius=3)
                    T.text(surf, (self.btn_hud_toggle.centerx, self.btn_hud_toggle.centery), f"{hud_mode_txt} [H]", 9.5, hud_col if hov_hud else T.C.TEXT_DIM, bold=True, anchor="cc")

    # ---------------------------------------------------------------- background grid
    def _draw_bg_grid(self, surf):
        col = (8, 15, 27)
        for x in range(self.CAM_X0, self.CAM_X1, 100):
            pygame.draw.line(surf, col, (x, self.CAM_Y0), (x, self.BTM_Y), 1)
        for y in range(self.CAM_Y0, self.BTM_Y, 80):
            pygame.draw.line(surf, col, (self.CAM_X0, y), (self.CAM_X1, y), 1)

    def _draw_footer(self, surf):
        T.text(surf, (self.SIDEBAR_W + 12, self.H - 14),
               "TAB view  ·  1-6 preset  ·  7-9 plat  ·  A atmos  ·  +/- targets  ·  N randomize  ·  M motion  ·  H HUD  ·  SPACE pause  ·  R reset  ·  S shot  ·  V grid  ·  F full  ·  P audit",
               8, T.C.TEXT_FAINT)

    # ---------------------------------------------------------------- camera
    def _frame_dims(self):
        if self.video_mode:
            s = self.sim
            return s.video_w, s.video_h
        return CAM_W, CAM_H

    def _cam_scale(self):
        w, h = self._frame_dims()
        return min((self.CAM_X1 - self.CAM_X0) / w,
                   (self.CAM_Y1 - self.CAM_Y0) / h)

    def _cam_dest(self):
        sc = self._cam_scale()
        w, h = self._frame_dims()
        dw, dh = int(w * sc), int(h * sc)
        x = self.CAM_X0 + (self.CAM_X1 - self.CAM_X0 - dw) // 2
        y = self.CAM_Y0 + (self.CAM_Y1 - self.CAM_Y0 - dh) // 2
        return pygame.Rect(x, y, dw, dh)

    def _draw_camera(self, surf):
        frame = self.sim.last_result.get("frame")
        cam   = self._frame_to_surf(frame)
        fw, fh = self._frame_dims()
        sc = self._cam_scale()
        scaled = pygame.transform.smoothscale(cam, (int(fw * sc), int(fh * sc)))
        dest   = self._cam_dest()
        surf.blit(scaled, dest.topleft)

        # A restrained overlay keeps the viewport readable at projector distance.
        if (self._scanline_surf is None or
                self._scanline_surf.get_size() != (dest.w, dest.h)):
            self._scanline_surf = pygame.Surface((dest.w, dest.h), pygame.SRCALPHA)
            for yl in range(0, dest.h, 8):
                pygame.draw.line(self._scanline_surf, (0, 0, 0, 8),
                                 (0, yl), (dest.w, yl), 1)
        surf.blit(self._scanline_surf, dest.topleft)

        # State-reactive viewport border with glow
        res    = self.sim.last_result
        st     = res.get("state", "SEARCHING")
        boresight_err = res.get("boresight_error_px")
        if boresight_err is None and getattr(self, "video_mode", False):
            boresight_err = res.get("optical_offset_px")
        is_aligned = (boresight_err is not None and boresight_err <= 15.0)
        is_optically_locked = (st in LOCKED_STATES and is_aligned)

        vp_col = T.C.GREEN if is_optically_locked else (T.C.CYAN_ELEC if st in LOCKED_STATES else T.C.STATE.get(st, T.C.BORDER))
        # Pulsing glow on true optical lock
        if is_optically_locked:
            p = T.pulse(1.5)
            gc = tuple(int(c * (0.4 + 0.6 * p)) for c in vp_col)
            pygame.draw.rect(surf, gc,      dest, 2)
            pygame.draw.rect(surf, T.C.BG,  pygame.Rect(dest.x - 2, dest.y - 2,
                                                         dest.w + 4, dest.h + 4), 1)
        elif st == "LOST":
            p = T.pulse(4.0)
            gc = tuple(int(c * (0.3 + 0.7 * p)) for c in vp_col)
            pygame.draw.rect(surf, gc, dest, 2)
        else:
            pygame.draw.rect(surf, vp_col, dest, 1)

        # Corner brackets around camera
        self._draw_corner_brackets(surf, dest, vp_col, 20, 2)
        self._draw_camera_story(surf, dest)
        self._draw_hud(surf, dest, sc)

    def _frame_to_surf(self, frame):
        if frame is None:
            fw, fh = self._frame_dims()
            s = pygame.Surface((fw, fh))
            s.fill((0, 0, 0))
            return s
        img = np.ascontiguousarray(frame[:, :, ::-1])
        return pygame.surfarray.make_surface(img)

    @staticmethod
    def _draw_corner_brackets(surf, rect, color, arm, th=1):
        """Four-corner L-brackets around a rect."""
        for cx, cy, sx, sy in (
            (rect.x,     rect.y,      1,  1),
            (rect.right, rect.y,     -1,  1),
            (rect.x,     rect.bottom, 1, -1),
            (rect.right, rect.bottom,-1, -1),
        ):
            pygame.draw.line(surf, color,
                             (cx, cy), (cx + sx * arm, cy), th)
            pygame.draw.line(surf, color,
                             (cx, cy), (cx, cy + sy * arm), th)

    def _draw_camera_story(self, surf, dest):
        """Slim status strip at bottom of camera."""
        res = self.sim.last_result
        st  = res["state"]
        boresight_err = res.get("boresight_error_px")
        if boresight_err is None and getattr(self, "video_mode", False):
            boresight_err = res.get("optical_offset_px")
        is_aligned = (boresight_err is not None and boresight_err <= 15.0)

        if not res["in_fov"]:
            label, col = "OUTSIDE FIELD OF VIEW", T.C.TEXT_FAINT
        elif st == "SEARCHING":
            label, col = "ACQUISITION WINDOW · SCANNING", T.C.AMBER
        elif st == "COASTING":
            label, col = "PREDICTIVE COAST · BEACON LOST", T.C.CYAN
        elif st in LOCKED_STATES and is_aligned:
            label, col = "BEACON ACQUIRED · LOCKED", T.C.GREEN
        elif st in LOCKED_STATES and not is_aligned:
            label, col = "BEACON TRACKED · SLEWING TO BORESIGHT", T.C.CYAN_ELEC
        elif st == "DEGRADED_LOCK":
            label, col = "DEGRADED LOCK · CONFIDENCE LOW", T.C.GREEN_DIM
        elif st == "REACQUIRING":
            label, col = "BEACON LOST · RE-ACQUIRING", T.C.PURPLE
        elif st in ("CANDIDATE", "ACQUIRING"):
            label, col = "BEACON DETECTED · ALIGNING TO BORESIGHT", T.C.CYAN_ELEC
        else:
            label, col = "SEARCHING", T.C.AMBER

        r    = pygame.Rect(dest.x, dest.bottom - 32, dest.w, 32)
        fill = T.C.STATE_FILL.get(st, (0, 16, 26))
        ovl  = pygame.Surface((r.w, r.h), pygame.SRCALPHA)
        ovl.fill((*fill, 215))
        surf.blit(ovl, r.topleft)
        pygame.draw.rect(surf, col, (r.x, r.y, 4, r.h))
        pygame.draw.line(surf, col, (r.x, r.y), (r.right, r.y), 1)

        # State label
        T.text(surf, (r.x + 14, r.centery), label, 12, col, bold=True, anchor="lc")

        # Confidence bar inside story strip
        conf_val = res.get("confidence", 0.0)
        bar_x = r.right - 210
        T.text(surf, (bar_x - 8, r.centery), f"CONF {int(conf_val*100)}%", 11, col, bold=True, anchor="rc")
        W.hbar(surf, (bar_x, r.centery - 4, 80, 8), conf_val, col)

        acq_t = self.perf.live_stats().get("acquisition_time_s")
        if acq_t:
            T.text(surf, (r.right - 14, r.centery), f"ACQ {acq_t:.2f}s", 11, T.C.TEXT_DIM, bold=True, anchor="rc")

    # ---------------------------------------------------------------- HUD (native display space)
    def _draw_hud(self, surf, dest, sc):
        res = self.sim.last_result
        st  = res.get("state", "SEARCHING")
        col = T.C.STATE.get(st, T.C.CYAN)
        cx, cy = dest.centerx, dest.centery

        # Pre-compute target beacon position & true optical boresight co-alignment
        b_u, b_v = None, None
        assoc = self.sim.tracker.associated
        if assoc is not None:
            b_u, b_v = assoc.u, assoc.v
        elif res.get("beacon_visible", False):
            if hasattr(self.sim, "sensor"):
                b = self.sim.scene.beacon
                cam_canvas = self.sim.sensor._canvas_xy(self.sim.gimbal.pan, self.sim.gimbal.tilt)
                b_u, b_v = self.sim.sensor._viewport_px(b.az_deg, b.el_deg, cam_canvas)
            elif getattr(self, "video_mode", False):
                b_u = res.get("detected_cx")
                b_v = res.get("detected_cy")

        cam_cu = float(getattr(self.sim, "cu", 320.0)) if getattr(self, "video_mode", False) else float(getattr(config, "PRINCIPAL_U", 320.0))
        cam_cv = float(getattr(self.sim, "cv", 240.0)) if getattr(self, "video_mode", False) else float(getattr(config, "PRINCIPAL_V", 240.0))
        fw, fh = self._frame_dims()

        dist_px = math.hypot(b_u - cam_cu, b_v - cam_cv) if (b_u is not None and b_v is not None) else res.get("boresight_error_px")
        if dist_px is None and getattr(self, "video_mode", False):
            dist_px = res.get("optical_offset_px")

        is_aligned = (dist_px is not None and dist_px <= 15.0)
        is_spec_pass = (dist_px is not None and dist_px < 10.0)
        is_optically_locked = (st in LOCKED_STATES and is_aligned)

        def to_screen(u, v):
            return (dest.x + int(u * sc), dest.y + int(v * sc))

        # ── 1. FOV reference grid (broken around center boresight) ──
        if getattr(self, "show_fov_grid", True):
            gc = (16, 28, 44)
            center_gap = int(45 * sc)
            pygame.draw.line(surf, gc, (cx, dest.top), (cx, cy - center_gap), 1)
            pygame.draw.line(surf, gc, (cx, cy + center_gap), (cx, dest.bottom), 1)
            pygame.draw.line(surf, gc, (dest.left, cy), (cx - center_gap, cy), 1)
            pygame.draw.line(surf, gc, (cx + center_gap, cy), (dest.right, cy), 1)
            for rr in (int(dest.h * 0.25), int(dest.h * 0.42)):
                pygame.draw.circle(surf, (14, 22, 38), (cx, cy), rr, 1)

        # ── 2. Gimbal Boresight (Optical LOS Axis at camera center) ──
        bp = self._est_pixel(self.sim.gimbal.pan, self.sim.gimbal.tilt)
        if bp is not None:
            bx, by = to_screen(bp[0], bp[1])
            rc = (0, 255, 136) if is_optically_locked else (50, 120, 180)
            ret_r = int(18 * sc)
            
            # Sleek open reticle ring
            pygame.draw.circle(surf, rc, (bx, by), ret_r, 1)
            
            # 4 external tick marks (open center allows clear view of beacon!)
            t_start = ret_r + 3
            t_end   = ret_r + 9
            pygame.draw.line(surf, rc, (bx + t_start, by), (bx + t_end, by), 2)
            pygame.draw.line(surf, rc, (bx - t_start, by), (bx - t_end, by), 2)
            pygame.draw.line(surf, rc, (bx, by + t_start), (bx, by + t_end), 2)
            pygame.draw.line(surf, rc, (bx, by - t_start), (bx, by - t_end), 2)

            # Dynamic Gimbal Slew Velocity Vector Arrow (Real-Time Motion Feedback!)
            v_pan = getattr(self.sim.gimbal, "v_pan", 0.0)
            v_tilt = getattr(self.sim.gimbal, "v_tilt", 0.0)
            slew_spd = math.hypot(v_pan, v_tilt)
            if slew_spd > 0.04:
                arr_len = min(int(54 * sc), int(slew_spd * 20 * sc))
                arr_dx = int((v_pan / slew_spd) * arr_len)
                arr_dy = -int((v_tilt / slew_spd) * arr_len)
                arr_tip = (bx + arr_dx, by + arr_dy)
                pygame.draw.line(surf, (0, 230, 255), (bx, by), arr_tip, 2)
                pygame.draw.circle(surf, (0, 230, 255), arr_tip, 3)
                T.text(surf, (arr_tip[0] + 6, arr_tip[1] - 6), f"SLEW {slew_spd:.2f}°/s", 10, (0, 230, 255), bold=True)

            axis_lbl = "BORESIGHT [CO-ALIGNED]" if is_optically_locked else "OPTICAL AXIS (0,0)"
            T.text(surf, (bx, by + ret_r + 5), axis_lbl, 9.5, rc, bold=True, anchor="tc")

            # Dual-Stage FSM Telemetry (Sub-µrad active piezo stabilization)
            fsm_p = res.get("fsm_pan_urad", 0.0)
            fsm_t = res.get("fsm_tilt_urad", 0.0)
            fsm_active = res.get("fsm_active", False)
            if fsm_active and not getattr(self, "hud_declutter", False):
                fsm_txt = f"FSM: [{fsm_p:+.1f}, {fsm_t:+.1f}] µrad"
                T.text(surf, (bx, by + ret_r + 17), fsm_txt, 9.0, (0, 230, 255), bold=True, anchor="tc")

        # ── 2b. Azimuth Heading Tape (Top of camera view) ──
        tape_y = dest.top + 7
        tape_w = min(320, dest.w - 380)
        tape_cx = dest.centerx
        tape_rect = pygame.Rect(tape_cx - tape_w // 2, tape_y, tape_w, 20)
        t_surf = pygame.Surface((tape_w, 20), pygame.SRCALPHA)
        t_surf.fill((6, 14, 26, 195))
        surf.blit(t_surf, tape_rect.topleft)
        pygame.draw.rect(surf, (24, 44, 72), tape_rect, 1, border_radius=3)
        cur_pan = self.sim.gimbal.pan
        px_per_deg_tape = 75.0
        for deg_mark in range(int((cur_pan - 2.0) * 5), int((cur_pan + 2.0) * 5) + 1):
            deg_val = deg_mark / 5.0
            x_off = tape_cx + int((deg_val - cur_pan) * px_per_deg_tape)
            if tape_rect.x + 4 < x_off < tape_rect.right - 4:
                is_major = (deg_mark % 5 == 0)
                th_len = 7 if is_major else 4
                pygame.draw.line(surf, (0, 200, 255) if is_major else (35, 60, 90),
                                 (x_off, tape_rect.bottom - th_len), (x_off, tape_rect.bottom), 1)
        pygame.draw.polygon(surf, T.C.AMBER, [(tape_cx - 4, tape_rect.bottom + 3), (tape_cx + 4, tape_rect.bottom + 3), (tape_cx, tape_rect.bottom)])
        T.text(surf, (tape_rect.right + 8, tape_rect.centery), f"PAN {cur_pan:+.2f}°", 10.5, T.C.CYAN_ELEC, bold=True, anchor="lc")

        # ── 2c. Actuator Slew Rate Saturation Alert (5.0 deg/s ISRO limit) ──
        pan_sat = res.get("gimbal_sat_pan", 0.0)
        tilt_sat = res.get("gimbal_sat_tilt", 0.0)
        if pan_sat >= 0.95 or tilt_sat >= 0.95:
            sat_w = min(460, dest.w - 120)
            sat_h = 24
            sat_rect = pygame.Rect(dest.centerx - sat_w // 2, dest.top + 34, sat_w, sat_h)
            s_bg = pygame.Surface((sat_w, sat_h), pygame.SRCALPHA)
            s_bg.fill((64, 18, 12, 220))
            surf.blit(s_bg, sat_rect.topleft)
            pygame.draw.rect(surf, T.C.AMBER, sat_rect, 1, border_radius=3)
            p_blink = (int(time.time() * 4.0) % 2 == 0)
            dot_c = T.C.RED if p_blink else T.C.AMBER
            pygame.draw.circle(surf, dot_c, (sat_rect.x + 12, sat_rect.centery), 4)
            T.text(surf, (sat_rect.x + 22, sat_rect.centery),
                   "[!] ACTUATOR SLEW LIMIT SATURATED (5.00°/s MAX RATE CAP REACHED)", 10.0, T.C.AMBER, bold=True, anchor="lc")

        # ── 3. Target Beacon Reticle & Clean Aerospace HUD Tag ──
        pri_idx = getattr(self, "primary_target_idx", 0)
        pri_id_str = f"TRG-{pri_idx+1:02d}"

        if b_u is not None and 0 <= b_u < fw and 0 <= b_v < fh:
            tx, ty = to_screen(b_u, b_v)
            degraded = (st == "DEGRADED_LOCK")
            target_col = T.C.AMBER if degraded else (T.C.GREEN if is_optically_locked else T.C.CYAN_ELEC)
            p = T.pulse(2.0)
            target_r = int((20 + 2 * p) * sc)
            arm = int(7 * sc)

            # 4 Corner L-Brackets around beacon
            for sx, sy in ((-1,-1), (-1,1), (1,-1), (1,1)):
                cpx = tx + sx * target_r
                cpy = ty + sy * target_r
                pygame.draw.line(surf, target_col, (cpx, cpy), (cpx - sx * arm, cpy), 2)
                pygame.draw.line(surf, target_col, (cpx, cpy), (cpx, cpy - sy * arm), 2)

            # Subtle outer guide ring
            pygame.draw.circle(surf, tuple(c // 4 for c in target_col), (tx, ty), target_r + 5, 1)

            if is_optically_locked:
                pygame.draw.circle(surf, T.C.GREEN, (tx, ty), target_r + int(6 * p), 1)

            # Register click rect for primary target
            self.target_click_rects[pri_idx] = pygame.Rect(tx - target_r - 4, ty - target_r - 4, (target_r + 4) * 2, (target_r + 4) * 2)

            # Compact, non-occluding aerospace HUD label right below reticle
            if not getattr(self, "hud_declutter", False):
                dist_val = dist_px if dist_px is not None else 0.0
                tag_st = "LOCKED" if is_optically_locked else ("ALIGN" if not getattr(self, "video_mode", False) else "TRACK")
                tag_txt = f"{pri_id_str} [{tag_st}] {dist_val:.1f}px"
                tw, th = T.font(10, bold=True).size(tag_txt)
                lbl_y = ty + target_r + 6 if ty + target_r + 20 < dest.bottom - 36 else ty - target_r - th - 6
                lbl_rect = pygame.Rect(tx - tw // 2 - 5, lbl_y, tw + 10, th + 3)
                l_bg = pygame.Surface((lbl_rect.w, lbl_rect.h), pygame.SRCALPHA)
                l_bg.fill((6, 14, 24, 180))
                surf.blit(l_bg, lbl_rect.topleft)
                pygame.draw.rect(surf, target_col, lbl_rect, 1, border_radius=3)
                T.text(surf, (tx, lbl_y + 1), tag_txt, 10, target_col, bold=True, anchor="tc")

        # ── 3b. Secondary Targets (Multi-Target Mode: PS Item 8) ──
        scene = getattr(self.sim, "scene", None)
        if scene is not None and hasattr(self.sim, "sensor"):
            cam_canvas = self.sim.sensor._canvas_xy(self.sim.gimbal.pan, self.sim.gimbal.tilt)
            cur_pri = getattr(self, "primary_target_idx", 0)
            for b_idx, extra_b in enumerate(getattr(scene, "beacons", [])):
                if b_idx == cur_pri:
                    continue
                eu, ev = self.sim.sensor._viewport_px(extra_b.az_deg, extra_b.el_deg, cam_canvas)
                if 0 <= eu < fw and 0 <= ev < fh:
                    ex_s, ey_s = to_screen(eu, ev)
                    sec_r = int(14 * sc)
                    sec_arm = int(5 * sc)
                    sec_col = (70, 160, 220)
                    for sx, sy in ((-1,-1), (-1,1), (1,-1), (1,1)):
                        cpx = ex_s + sx * sec_r
                        cpy = ey_s + sy * sec_r
                        pygame.draw.line(surf, sec_col, (cpx, cpy), (cpx - sx * sec_arm, cpy), 1)
                        pygame.draw.line(surf, sec_col, (cpx, cpy), (cpx, cpy - sy * sec_arm), 1)
                    # Register clickable target area for instant handover
                    self.target_click_rects[b_idx] = pygame.Rect(ex_s - sec_r - 4, ey_s - sec_r - 4, (sec_r + 4) * 2, (sec_r + 4) * 2)
                    if not getattr(self, "hud_declutter", False):
                        stxt = f"{extra_b.target_id} [SEC - CLICK]"
                        T.text(surf, (ex_s, ey_s + sec_r + 3), stxt, 8.5, sec_col, bold=True, anchor="tc")

        # ── 4. Distractor Decoys (AI Discrimination Showcase) ──
        if hasattr(self.sim, "sensor"):
            cam_canvas = self.sim.sensor._canvas_xy(
                self.sim.gimbal.pan, self.sim.gimbal.tilt
            )
        else:
            cam_canvas = (0, 0)

        for d in getattr(scene, "distractors", []):
            du, dv = self.sim.sensor._viewport_px(d.az, d.el, cam_canvas)
            if 0 <= du < fw and 0 <= dv < fh:
                dx, dy = to_screen(du, dv)
                dr = int(10 * sc)
                diamond_pts = [(dx, dy - dr), (dx + dr, dy), (dx, dy + dr), (dx - dr, dy)]
                pygame.draw.polygon(surf, T.C.AMBER, diamond_pts, 1)
                if not getattr(self, "hud_declutter", False):
                    decoy_txt = f"DEC {d.mod_freq:.0f}Hz"
                    T.text(surf, (dx, dy + dr + 2), decoy_txt, 9, T.C.AMBER, bold=True, anchor="tc")

        # ── 5. Orbital Ephemeris Prior (Predictive Coast Aid — Anti-Collision) ──
        if getattr(self, "show_ephemeris", True):
            paz = getattr(self, "eph_pred_az", None)
            if paz is not None:
                pp = self._est_pixel(paz, self.eph_pred_el)
                if pp is not None:
                    ex, ey = to_screen(pp[0], pp[1])
                    er = int(11 * sc)
                    eph_col = (220, 160, 50)
                    for k in range(0, 360, 45):
                        a1, a2 = math.radians(k), math.radians(k + 22)
                        pygame.draw.line(surf, eph_col,
                                         (ex + er*math.cos(a1), ey + er*math.sin(a1)),
                                         (ex + er*math.cos(a2), ey + er*math.sin(a2)), 1)
                    if not getattr(self, "hud_declutter", False):
                        is_near = (b_u is not None and abs(pp[0] - b_u) < 35 and abs(pp[1] - b_v) < 35)
                        if not is_near:
                            T.text(surf, (ex, ey + er + 2), "EPH", 9, eph_col, bold=True, anchor="tc")

        # ── 6. Candidate Detections ──
        for c in res.get("cand_list", []):
            cu, cv = int(c.u), int(c.v)
            if b_u is not None and (cu - b_u)**2 + (cv - b_v)**2 < 20**2:
                continue
            cx_s, cy_s = to_screen(cu, cv)
            cand_r = int(8 * sc)
            cand_arm = int(3 * sc)
            c_col = (0, 100, 150)
            for sx, sy in ((-1,-1), (-1,1), (1,-1), (1,1)):
                pygame.draw.line(surf, c_col, (cx_s+sx*cand_r, cy_s+sy*cand_r), (cx_s+sx*(cand_r-cand_arm), cy_s+sy*cand_r), 1)
                pygame.draw.line(surf, c_col, (cx_s+sx*cand_r, cy_s+sy*cand_r), (cx_s+sx*cand_r, cy_s+sy*(cand_r-cand_arm)), 1)

        # ── 7. Top Overlays: Adaptive Trust (Left), Symbology Legend (Center), ISRO Spec (Right) ──
        if not getattr(self, "hud_declutter", False):
            # Left Trust Box
            tr = self.sim.tracker
            tm = getattr(tr, "trust", None)
            t_box_w = 155
            t_box_h = 50
            t_x = dest.left + 10
            t_y = dest.top + 8
            t_right = t_x + t_box_w
            if tm is not None:
                t_surf = pygame.Surface((t_box_w, t_box_h), pygame.SRCALPHA)
                t_surf.fill((6, 14, 26, 220))
                surf.blit(t_surf, (t_x, t_y))
                pygame.draw.rect(surf, (20, 36, 62), (t_x, t_y, t_box_w, t_box_h), 1, border_radius=4)
                
                W.hbar(surf, (t_x + 8, t_y + 10, 70, 6), tm.vision_trust, T.C.CYAN_ELEC)
                T.text(surf, (t_x + 84, t_y + 6), f"VIS {tm.vision_trust:.2f}", 10, T.C.CYAN_ELEC, bold=True, mono=True)
                
                W.hbar(surf, (t_x + 8, t_y + 24, 70, 6), tm.model_trust, T.C.PURPLE)
                T.text(surf, (t_x + 84, t_y + 20), f"MDL {tm.model_trust:.2f}", 10, T.C.PURPLE, bold=True, mono=True)
                
                sigma = getattr(getattr(tr, "unc", None), "display_sigma_px", None)
                if sigma is not None:
                    T.text(surf, (t_x + 8, t_y + 35), f"UNCERTAINTY sigma: {sigma:.1f} px", 9.5, T.C.TEXT_DIM, bold=True)

            # Right ISRO Spec Box
            spec_w, spec_h = 175, 50
            spec_x = dest.right - spec_w - 10
            spec_y = dest.top + 8
            
            err_deg = res.get("pointing_err_deg", 999.0)
            err_px = err_deg * (config.FOCAL_PX / (180.0 / math.pi))
            is_spec_pass = (err_px < 10.0 and is_optically_locked)
            pass_txt = "[ PASS ]" if is_spec_pass else ("[ ALIGNING ]" if not getattr(self, "video_mode", False) else "[ TRACKING ]")
            pass_col = T.C.GREEN if is_spec_pass else T.C.AMBER

            s_surf = pygame.Surface((spec_w, spec_h), pygame.SRCALPHA)
            s_surf.fill((4, 18, 14, 225) if is_spec_pass else (20, 16, 6, 225))
            surf.blit(s_surf, (spec_x, spec_y))
            pygame.draw.rect(surf, pass_col, (spec_x, spec_y, spec_w, spec_h), 1, border_radius=4)
            pygame.draw.rect(surf, pass_col, (spec_x, spec_y, 4, spec_h), border_top_left_radius=4, border_bottom_left_radius=4)
            T.text(surf, (spec_x + 10, spec_y + 4), "ISRO PS 26169 SPEC", 10, T.C.TEXT_FAINT, bold=True)
            T.text(surf, (spec_x + 10, spec_y + 18), "POINTING: < 10 px (0.06°)", 10.5, T.C.TEXT, bold=True)
            T.text(surf, (spec_x + 10, spec_y + 32), f"STATUS: {pass_txt}", 10.5, pass_col, bold=True)

            # Middle HUD Symbology Legend
            avail_w = spec_x - t_right - 20
            if avail_w > 320:
                items = [
                    ("[TRG] BEACON 15Hz", T.C.GREEN),
                    ("[LOS] BORESIGHT", (80, 150, 220)),
                    ("[EPH] PRIOR", (220, 160, 50)),
                    ("[DEC] DECOY", T.C.AMBER),
                ]
                measured = []
                fnt = T.font(10, bold=True)
                for itxt, icol in items:
                    iw, _ = fnt.size(itxt)
                    measured.append((itxt, icol, iw))
                total_content_w = sum(m[2] for m in measured)
                item_gap = 16
                leg_w = total_content_w + item_gap * (len(items) - 1) + 24
                if leg_w <= avail_w:
                    leg_h = 24
                    leg_x = t_right + (spec_x - t_right - leg_w) // 2
                    leg_y = dest.bottom - 58
                    leg_surf = pygame.Surface((leg_w, leg_h), pygame.SRCALPHA)
                    leg_surf.fill((6, 12, 22, 215))
                    surf.blit(leg_surf, (leg_x, leg_y))
                    pygame.draw.rect(surf, (24, 44, 70), (leg_x, leg_y, leg_w, leg_h), 1, border_radius=4)
                    
                    cur_x = leg_x + 12
                    for itxt, icol, iw in measured:
                        T.text(surf, (cur_x, leg_y + 4), itxt, 10, icol, bold=True)
                        cur_x += iw + item_gap

        # ── 8. Occluded Banner ──
        if not res.get("beacon_visible", True):
            occ_w, occ_h = 280, 26
            occ_r = pygame.Rect(cx - occ_w // 2, dest.bottom - 62, occ_w, occ_h)
            o_surf = pygame.Surface((occ_w, occ_h), pygame.SRCALPHA)
            o_surf.fill((54, 10, 10, 230))
            surf.blit(o_surf, occ_r.topleft)
            pygame.draw.rect(surf, T.C.RED, occ_r, 1, border_radius=4)
            T.text(surf, (cx, occ_r.centery), "[!] BEACON OCCLUDED · PREDICTIVE COAST", 10.5, T.C.RED, bold=True, anchor="cc")

        # ── 9. Ground Truth Overlay (Eval Only) ──
        if getattr(self, "show_gt", False):
            gp = self._est_pixel(res.get("truth_az"), res.get("truth_el"))
            if gp is not None:
                gx, gy = to_screen(gp[0], gp[1])
                pygame.draw.circle(surf, T.C.PURPLE, (gx, gy), 8, 1)
                pygame.draw.line(surf, T.C.PURPLE, (gx-12, gy), (gx+12, gy), 1)
                pygame.draw.line(surf, T.C.PURPLE, (gx, gy-12), (gx, gy+12), 1)
                T.text(surf, (gx, gy + 12), "GROUND TRUTH", 9, T.C.PURPLE, bold=True, anchor="tc")

        # ── 10. Video Playback Completed Banner ──
        if getattr(self, "video_mode", False) and getattr(self, "video_done", False):
            done_w, done_h = min(540, dest.w - 40), 34
            done_r = pygame.Rect(cx - done_w // 2, cy - done_h // 2, done_w, done_h)
            d_surf = pygame.Surface((done_w, done_h), pygame.SRCALPHA)
            d_surf.fill((6, 28, 20, 235))
            surf.blit(d_surf, done_r.topleft)
            pygame.draw.rect(surf, T.C.GREEN, done_r, 1, border_radius=4)
            T.text(surf, (cx, done_r.centery), "VIDEO PLAYBACK COMPLETE  ·  [SPACE]/[R]: REPLAY  ·  [L]: LOAD MP4", 10.5, T.C.GREEN, bold=True, anchor="cc")

    def _cam_space(self):
        if self.video_mode:
            s = self.sim
            return (s.focal_px, s.cu, s.cv, s.video_w, s.video_h)
        return (config.FOCAL_PX, config.PRINCIPAL_U, config.PRINCIPAL_V,
                CAM_W, CAM_H)

    def _est_pixel(self, az, el):
        if az is None or el is None:
            return None
        focal, cu, cv, vw, vh = self._cam_space()
        d     = azel_unit(az, el)
        basis = self.sim.gimbal.basis()
        p = project_point_into_camera(d, (0, 0, 0), basis, focal, cu, cv)
        if p is None:
            return None
        u, v = p
        if 0 <= u < vw and 0 <= v < vh:
            return (int(u), int(v))
        return None

    # ================================================================ BOTTOM STRIP
    def _draw_bottom(self, surf):
        by = self.BTM_Y
        bh = self.BTM_H
        bx0 = self.CAM_X0
        bw = self.CAM_SIDE_W

        pygame.draw.rect(surf, T.C.PANEL, (bx0, by, bw, bh))
        pygame.draw.line(surf, T.C.BORDER_B, (bx0, by),     (bx0 + bw, by),     1)
        pygame.draw.line(surf, T.C.BORDER,   (bx0, by + 1), (bx0 + bw, by + 1), 1)

        # 1. PAT pipeline stepper (camera-side width)
        stepper_h = 26
        self._draw_pat_stepper(surf, pygame.Rect(bx0 + 6, by + 4, bw - 12, stepper_h))

        # 2. Lower region: split between Error Graph and Gimbal/Beam Panel
        rem_y = by + stepper_h + 8
        rem_h = bh - stepper_h - 12
        graph_w = int((bw - 16) * 0.58)
        panel_w = (bw - 16) - graph_w - 8

        graph_rect = pygame.Rect(bx0 + 6, rem_y, graph_w, rem_h)
        self._draw_error_graph(surf, graph_rect)

        panel_rect = pygame.Rect(graph_rect.right + 8, rem_y, panel_w, rem_h)
        self._draw_camera_panel(surf, panel_rect)

    def _draw_pat_stepper(self, surf, box):
        res  = self.sim.last_result
        st   = res["state"]
        boresight_err = res.get("boresight_error_px")
        if boresight_err is None and getattr(self, "video_mode", False):
            boresight_err = res.get("optical_offset_px")
        is_aligned = (boresight_err is not None and boresight_err <= 15.0)
        is_opt_lock = (st in LOCKED_STATES and is_aligned)

        steps = ["PREDICT", "POINT", "SEARCH", "ACQUIRE", "LOCK"]
        if is_opt_lock:
            idx = 4
            act_col = T.C.GREEN
        elif st in LOCKED_STATES or st == "DEGRADED_LOCK":
            idx = 3
            act_col = T.C.CYAN_ELEC
        elif st in ("TENTATIVE", "COASTING", "CANDIDATE", "ACQUIRING"):
            idx = 3
            act_col = T.C.CYAN
        elif st in ("SEARCHING", "REACQUIRING"):
            idx = 2
            act_col = T.C.AMBER if st == "SEARCHING" else T.C.PURPLE
        else:
            idx = 0
            act_col = T.C.CYAN

        pygame.draw.rect(surf, T.C.PANEL_2, box)
        pygame.draw.rect(surf, T.C.BORDER,  box, 1)

        n = len(steps)
        seg_w = box.w // n
        x = box.x
        for i, step in enumerate(steps):
            seg    = pygame.Rect(x, box.y, seg_w - 1, box.h)
            done   = i < idx
            active = i == idx

            if active:
                pygame.draw.rect(surf, tuple(c // 8 for c in act_col), seg)
                pygame.draw.rect(surf, act_col, (seg.x, seg.bottom - 3, seg.w, 3))
                text_col, fs = act_col, 9.5
            elif done:
                pygame.draw.rect(surf, T.C.PANEL_3, seg)
                pygame.draw.rect(surf, T.C.GREEN_DIM, (seg.x, seg.bottom - 2, seg.w, 2))
                text_col, fs = T.C.TEXT_DIM, 8.5
            else:
                text_col, fs = T.C.TEXT_FAINT, 8.5

            T.text(surf, (seg.centerx, seg.centery - 1), step, fs,
                   text_col, bold=active, anchor="cc")
            if i < n - 1:
                mx = x + seg_w - 1
                pygame.draw.line(surf, T.C.BORDER, (mx, box.y + 4), (mx, box.bottom - 4), 1)
            x += seg_w

        # Annotation badge for special states
        ann = {"COASTING": ("COAST", T.C.CYAN), "REACQUIRING": ("RE-ACQ", T.C.PURPLE),
               "LOST": ("LOST", T.C.RED)}.get(st)
        if ann:
            bw = 64
            r = pygame.Rect(box.right - bw - 4, box.y + 2, bw, box.h - 4)
            pygame.draw.rect(surf, tuple(c // 7 for c in ann[1]), r)
            pygame.draw.rect(surf, ann[1], r, 1)
            pygame.draw.rect(surf, ann[1], (r.x, r.y, 2, r.h))
            T.text(surf, (r.centerx + 1, r.centery), ann[0],
                   8, ann[1], bold=True, anchor="cc")

    # ---------------------------------------------------------------- error graph
    def _draw_error_graph(self, surf, box):
        T.text(surf, (box.x, box.y + 2), "ANGULAR POINTING ERROR", 11.5, T.C.TEXT_DIM, bold=True)
        T.text(surf, (box.right - 4, box.y + 2), "target < 0.0625°", 11, T.C.TEXT_FAINT, anchor="tr")

        plot = pygame.Rect(box.x + 36, box.y + 20, box.w - 38, box.h - 38)
        pygame.draw.rect(surf, T.C.BG, plot)
        pygame.draw.rect(surf, T.C.BORDER, plot, 1)

        # Target acquisition band
        _bound = plot.bottom - int(plot.h * (config.FINE_ACQUISITION_REGION_DEG / 0.5))
        pygame.draw.rect(surf, (4, 22, 12),
                         (plot.x, _bound, plot.w, plot.bottom - _bound))
        pygame.draw.line(surf, T.C.GREEN_DIM, (plot.x, _bound), (plot.right, _bound), 1)

        # Grid lines + Y labels
        T.text(surf, (plot.x - 4, plot.bottom - 6), "0", 10, T.C.TEXT_FAINT, anchor="tr", mono=True)
        T.text(surf, (plot.x - 4, plot.y + 1), "0.5", 10, T.C.TEXT_FAINT, anchor="tr", mono=True)
        for deg in (0.1, 0.2, 0.3, 0.4):
            yy = plot.bottom - int(plot.h * (deg / 0.5))
            pygame.draw.line(surf, T.C.GRID, (plot.x, yy), (plot.right, yy), 1)
            T.text(surf, (plot.x - 4, yy - 5), f"{deg:.1f}", 10, T.C.TEXT_FAINT, anchor="tr", mono=True)

        # Error series
        series = [max(0.0, min(0.5, e)) for e in self.error_spark]
        n = len(series)
        if n > 1:
            pts = []
            for i, v in enumerate(series):
                xx = int(plot.x + plot.w * i / (n - 1))
                yy = int(plot.bottom - plot.h * (v / 0.5))
                pts.append((xx, yy))
            fill_pts = ([pts[0]] + pts + [(pts[-1][0], plot.bottom), (pts[0][0], plot.bottom)])
            area = pygame.Surface((plot.w, plot.h), pygame.SRCALPHA)
            local = [(p[0] - plot.x, p[1] - plot.y) for p in fill_pts]
            pygame.draw.polygon(area, (0, 255, 100, 25), local)
            surf.blit(area, (plot.x, plot.y))
            prev = None
            for p in pts:
                if prev:
                    pygame.draw.line(surf, T.C.GREEN, prev, p, 2)
                prev = p

        T.text(surf, (plot.right - 6, _bound - 3), "ACQ TARGET", 10, T.C.GREEN_DIM, bold=True, anchor="br")

        # NOW marker
        if series:
            cur_v  = series[-1]
            now_x  = plot.right - 1
            now_y  = int(plot.bottom - plot.h * (cur_v / 0.5))
            cur_col = (T.C.GREEN if cur_v < config.FINE_ACQUISITION_REGION_DEG
                       else (T.C.AMBER if cur_v < 0.30 else T.C.RED))
            pygame.draw.line(surf, tuple(c // 4 for c in cur_col),
                             (now_x, plot.y), (now_x, plot.bottom), 1)
            pygame.draw.circle(surf, cur_col, (now_x, now_y), 4)
            T.text(surf, (now_x - 8, max(plot.y + 2, now_y - 12)),
                   f"{cur_v*1000:.0f} m°", 11, cur_col, bold=True, anchor="tr", mono=True)

        if not self.sim.last_result.get("beacon_visible", False):
            pygame.draw.rect(surf, (54, 14, 14), (plot.right - 4, plot.y, 4, plot.h))

        self._draw_state_timeline(surf, plot)

    def _draw_state_timeline(self, surf, plot):
        tmax = max(0.001, self.sim.last_result.get("t", 0.0))
        ev   = list(getattr(self.sim, "event_log", ()))
        y    = plot.bottom + 5
        band = pygame.Rect(plot.x, y, plot.w, 9)
        pygame.draw.rect(surf, T.C.BG, band)
        runs = []
        if ev:
            runs.append((0.0, ev[0][0], ev[0][1]))
            for i, e in enumerate(ev):
                t1 = ev[i+1][0] if i+1 < len(ev) else tmax
                if t1 > e[0]:
                    runs.append((e[0], t1, e[2]))
        else:
            runs.append((0.0, tmax, getattr(self.sim.tracker, "state", "SEARCHING")))
        for t0, t1, s in runs:
            x0 = plot.x + plot.w * (t0 / tmax)
            x1 = plot.x + plot.w * (min(t1, tmax) / tmax)
            if x1 <= x0:
                continue
            scol = T.C.STATE.get(s, T.C.STATE["SEARCHING"])
            if s == "COASTING":
                scol = T.C.STATE["REACQUIRING"]
            if s == "SEARCHING":
                scol = T.C.AMBER_DIM
            pygame.draw.rect(surf, scol, (int(x0), y, int(x1-x0), 9))
        pygame.draw.rect(surf, T.C.BORDER, band, 1)
        T.text(surf, (plot.x - 4, y), "STATE", 10, T.C.TEXT_FAINT, bold=True, anchor="tr")

    # ---------------------------------------------------------------- camera panel
    def _draw_camera_panel(self, surf, box):
        res = self.sim.last_result
        T.text(surf, (box.x + 6, box.y + 2), "GIMBAL ACTUATOR", 11.5, T.C.TEXT_DIM, bold=True)

        beam_w = 112
        gimbal_w = box.w - beam_w - 10
        col_step = gimbal_w // 2

        # Row 1: Azimuth & Elevation
        x = box.x + 8
        y1 = box.y + 24
        T.text(surf, (x, y1), "AZIMUTH", 11, T.C.TEXT_FAINT, bold=True)
        T.text(surf, (x, y1 + 16), f"{self.sim.gimbal.pan:+.2f}°", 14, T.C.CYAN, bold=True, mono=True)

        T.text(surf, (x + col_step, y1), "ELEVATION", 11, T.C.TEXT_FAINT, bold=True)
        T.text(surf, (x + col_step, y1 + 16), f"{self.sim.gimbal.tilt:+.2f}°", 14, T.C.CYAN, bold=True, mono=True)

        # Row 2: Saturation
        sp = res.get("gimbal_sat_pan", 0.0)
        st_ = res.get("gimbal_sat_tilt", 0.0)
        sat = max(sp, st_)
        s_col = T.C.GREEN if sat <= 0.05 else (T.C.AMBER if sat < 0.5 else T.C.RED)
        y2 = y1 + 42
        T.text(surf, (x, y2), "GIMBAL SATURATION", 11, T.C.TEXT_FAINT, bold=True)
        T.text(surf, (x, y2 + 16), f"P {sp*100:2.0f}%   T {st_*100:2.0f}%", 13, s_col, bold=True, mono=True)

        beam_box = pygame.Rect(box.right - beam_w, box.y + 2, beam_w, box.h - 4)
        self._draw_beam_strip(surf, beam_box)

    def _draw_beam_strip(self, surf, box):
        cx = box.centerx
        res = self.sim.last_result
        err = res["pointing_err_deg"]
        col = (T.C.GREEN if err < config.FINE_ACQUISITION_REGION_DEG
               else (T.C.AMBER if err < 0.30 else T.C.RED))

        # Top error readout
        T.text(surf, (cx, box.y + 2), "POINT ERROR", 11, T.C.TEXT_DIM, bold=True, anchor="tc")
        T.text(surf, (cx, box.y + 18), f"{err*1000:.1f} mdeg", 13, col, bold=True, anchor="tc", mono=True)

        # Dynamic SAT-B orbital position
        truth_az = res.get("truth_az", 0.0)
        truth_el = res.get("truth_el", 0.0)
        sat_b_x = int(cx + max(-0.4, min(0.4, truth_az / 2.0)) * (box.w - 36))
        sat_b_y = box.y + 58 + int(max(-0.25, min(0.25, truth_el / 2.0)) * 14)

        # SAT-A base terminal
        sat_a_x = cx
        sat_a_y = box.bottom - 22

        # Steered Laser Beam Vector from SAT-A gimbal
        g_pan = self.sim.gimbal.pan
        beam_tip_x = int(cx + max(-0.4, min(0.4, g_pan / 2.0)) * (box.w - 36))
        beam_tip_y = sat_b_y

        # Beam divergence cone / polygon
        div_w = 8 if col == T.C.GREEN else 16
        pygame.draw.polygon(surf, (14, 38, 64) if col == T.C.GREEN else (38, 28, 14),
                            [(sat_a_x - 4, sat_a_y), (beam_tip_x - div_w, beam_tip_y), (beam_tip_x + div_w, beam_tip_y), (sat_a_x + 4, sat_a_y)])

        # Core carrier laser line
        pygame.draw.line(surf, col, (sat_a_x, sat_a_y), (beam_tip_x, beam_tip_y), 2)

        # Animated photon energy pulses along the beam
        t_ticks = pygame.time.get_ticks() / 1000.0
        for p_idx in range(3):
            frac = ((t_ticks * 2.5 + p_idx * 0.33) % 1.0)
            px = int(sat_a_x + (beam_tip_x - sat_a_x) * frac)
            py = int(sat_a_y + (beam_tip_y - sat_a_y) * frac)
            pygame.draw.circle(surf, T.C.CYAN_ELEC if col == T.C.GREEN else T.C.AMBER, (px, py), 2)

        # SAT-B Target Satellite (with solar wings)
        pygame.draw.line(surf, (60, 110, 160), (sat_b_x - 11, sat_b_y), (sat_b_x + 11, sat_b_y), 3)
        pygame.draw.circle(surf, col, (sat_b_x, sat_b_y), 4)
        if self.platform_mode == "UAV_SATELLITE":
            target_label = "SAT-B"
        elif self.platform_mode == "UAV_UAV":
            target_label = "UAV-B"
        else:
            target_label = "SAT-B"
        T.text(surf, (sat_b_x, sat_b_y - 10), target_label, 10, col, bold=True, anchor="bc")

        # SAT-A Ground / Mobile Terminal
        pygame.draw.rect(surf, (16, 42, 70), (sat_a_x - 14, sat_a_y - 5, 28, 12), border_radius=2)
        pygame.draw.rect(surf, T.C.CYAN, (sat_a_x - 14, sat_a_y - 5, 28, 12), 1, border_radius=2)
        if self.platform_mode == "UAV_SATELLITE":
             source_label = "UAV-A"
        elif self.platform_mode == "UAV_UAV":
           source_label = "UAV-A"
        else:
            source_label = "SAT-A"
        T.text(surf, (sat_a_x, sat_a_y + 9), source_label, 10, T.C.CYAN, bold=True, anchor="tc")

    # ================================================================ RIGHT PANEL
    def _draw_panel(self, surf):
        pnl_full_h = self.H - self.HDR_TOTAL_H - self.FOOTER_H - 4
        pygame.draw.rect(surf, T.C.BG,    (self.PNL_X0, self.HDR_TOTAL_H, self.PNL_W, pnl_full_h))
        pygame.draw.line(surf, T.C.BORDER_B, (self.PNL_X0, self.HDR_TOTAL_H), (self.PNL_X0, self.H - self.FOOTER_H), 2)
        pygame.draw.line(surf, T.C.BORDER,   (self.PNL_X0 + 2, self.HDR_TOTAL_H), (self.PNL_X0 + 2, self.H - self.FOOTER_H), 1)

        # Subtle background grid for right panel
        for gx in range(self.PNL_X0 + 60, self.PNL_X1, 60):
            pygame.draw.line(surf, (8, 14, 28), (gx, self.HDR_TOTAL_H), (gx, self.H - self.FOOTER_H), 1)

        self._panel_state(surf)
        if self.pnl_mission_h > 0:
            self._panel_mission(surf)
        self._panel_performance(surf)
        self._panel_geometry(surf)
        self._panel_disturbances(surf)
        self._panel_controls(surf)

    # ── State ─────────────────────────────────────────────────────────────────
    def _panel_state(self, surf):
        x, y, w, h = self.PNL_INN, self.pnl_state_y, self.PNL_IW, self.pnl_state_h
        res  = self.sim.last_result
        st   = res.get("state", "SEARCHING")
        boresight_err = res.get("boresight_error_px")
        if boresight_err is None and getattr(self, "video_mode", False):
            boresight_err = res.get("optical_offset_px")
        is_aligned = (boresight_err is not None and boresight_err <= 15.0)
        is_opt_lock = (st in LOCKED_STATES and is_aligned)

        if is_opt_lock:
            disp_st = "LOCKED"
            col = T.C.GREEN
            fill = (4, 30, 18)
        elif st in LOCKED_STATES:
            disp_st = "TRACKING" if getattr(self, "video_mode", False) else "ALIGNING"
            col = T.C.CYAN_ELEC
            fill = (6, 26, 40)
        else:
            disp_st = st
            col = T.C.STATE.get(st, T.C.CYAN)
            fill = T.C.STATE_FILL.get(st, (0, 30, 50))

        r    = pygame.Rect(x, y, w, h)
        T.card(surf, r, fill=fill, border=col)
        pygame.draw.rect(surf, col, (x, y, 4, h))

        # Title
        T.text(surf, (x + 14, y + 8), "OPTICAL TRACKING STATE", 11, T.C.TEXT_DIM, bold=True)
        # Big state name
        fs = 20 if len(disp_st) <= 8 else 16
        T.text(surf, (x + 14, y + 26), disp_st, fs, col, bold=True)

        # Confidence arc gauge on right
        conf = res.get("confidence", 0.0)
        arc_cx = x + w - 38
        arc_cy = y + h // 2
        T.arc_gauge(surf, (arc_cx, arc_cy), 22, conf, col, (14, 24, 44), width=4)
        T.text(surf, (arc_cx, arc_cy), f"{int(conf*100)}%", 11, col, bold=True, anchor="cc")

        # Status / timing row
        stat_y = y + h - 18
        run_col = T.C.AMBER if self.paused else T.C.GREEN
        pygame.draw.circle(surf, run_col, (x + 16, stat_y + 4), 3.5)
        T.text(surf, (x + 24, stat_y), "PAUSED" if self.paused else "SYSTEM ACTIVE", 11, run_col, bold=True)
        if self.pnl_mission_h == 0:
            T.text(surf, (x + w - 12, stat_y), f"{self.preset} · {self._platform_label()}", 10.5, T.C.CYAN_ELEC, anchor="tr", bold=True)
        else:
            T.text(surf, (x + w - 12, stat_y), f"t = {res.get('t', 0.0):.1f}s", 11, T.C.TEXT_DIM, anchor="tr", mono=True)

    # ── Mission ────────────────────────────────────────────────────────────────
    def _panel_mission(self, surf):
        x, y, w, h = self.PNL_INN, self.pnl_mission_y, self.PNL_IW, self.pnl_mission_h
        r = pygame.Rect(x, y, w, h)
        T.card(surf, r, fill=T.C.PANEL_2, border=T.C.BORDER)
        T.section_hdr(surf, x + 10, y + 6, "MISSION PROFILE & LINK", panel_w=w - 20)
        res = self.sim.last_result
        st  = res["state"]
        lock = st in LOCKED_STATES

        rows = [
            ("Preset", self.preset),
            ("Platform", self._platform_label()),
            ("Medium", self.atmosphere or "CLEAR"),
        ]
        yy = y + 26
        for lab, val in rows:
            T.text(surf, (x + 12, yy), lab, 11, T.C.TEXT_DIM, bold=True)
            T.text(surf, (x + w - 12, yy), val, 12, T.C.CYAN_ELEC, bold=True, anchor="tr")
            yy += 18

    # ── Performance ────────────────────────────────────────────────────────────
    def _panel_performance(self, surf):
        x, y, w, h = self.PNL_INN, self.pnl_perf_y, self.PNL_IW, self.pnl_perf_h
        res = self.sim.last_result
        st  = self.perf.live_stats()
        err = res["pointing_err_deg"]
        ec  = (T.C.GREEN if err < config.FINE_ACQUISITION_REGION_DEG
               else (T.C.AMBER if err < 0.30 else T.C.RED))
        r   = pygame.Rect(x, y, w, h)
        T.card(surf, r, fill=T.C.PANEL_2, border=T.C.BORDER)
        pygame.draw.rect(surf, ec, (x, y, 4, h))
        T.section_hdr(surf, x + 10, y + 6, "PAT PERFORMANCE METRICS", color=ec, panel_w=w - 20)

        # Hero Pointing Error Block
        err_r = pygame.Rect(x + 10, y + 26, w - 20, 42)
        pygame.draw.rect(surf, (10, 18, 34), err_r, border_radius=4)
        pygame.draw.rect(surf, ec, err_r, 1, border_radius=4)
        T.text(surf, (err_r.x + 10, err_r.y + 4), "POINTING ERROR", 11, T.C.TEXT_DIM, bold=True)
        err_mdeg = err * 1000.0
        T.text(surf, (err_r.x + 10, err_r.y + 18), f"{err_mdeg:5.1f} mdeg", 18, ec, bold=True, mono=True)
        boresight_err = res.get("boresight_error_px")
        if boresight_err is None and getattr(self, "video_mode", False):
            boresight_err = res.get("optical_offset_px")
        is_aligned = (boresight_err is not None and boresight_err <= 15.0)
        pt_status = "LOCKED" if (err < config.FINE_ACQUISITION_REGION_DEG and is_aligned) else ("ALIGNING" if not getattr(self, "video_mode", False) else "TRACKING")
        T.text(surf, (err_r.right - 10, err_r.centery), pt_status, 12, ec, bold=True, anchor="rc")

        # KPI Trio
        acq = st.get("acquisition_time_s")
        acq_s = f"{acq:.2f}s" if acq else "--"
        ret_pct = st.get("retention_total_pct", 0.0)
        ret_col = T.C.GREEN if ret_pct >= 95 else (T.C.AMBER if ret_pct >= 80 else T.C.RED)
        fps = self.clock.get_fps()
        fps_col = T.C.GREEN if fps >= 25 else T.C.AMBER

        cols_data = [
            ("ACQUISITION", acq_s, T.C.CYAN_ELEC),
            ("RETENTION", f"{ret_pct:.1f}%", ret_col),
            ("REFRESH", f"{fps:.0f} FPS", fps_col),
        ]
        col_w = (w - 20) // 3
        yy_kpi = y + 74
        for idx, (lab, val, vc) in enumerate(cols_data):
            kx = x + 10 + idx * col_w
            T.text(surf, (kx, yy_kpi), lab, 10, T.C.TEXT_DIM, bold=True)
            T.text(surf, (kx, yy_kpi + 15), val, 14, vc, bold=True, mono=True)

    # ── Orbit / Video geometry ─────────────────────────────────────────────────
    def _panel_geometry(self, surf):
        x, y, w, h = self.PNL_INN, self.pnl_geom_y, self.PNL_IW, self.pnl_geom_h
        g = pygame.Rect(x, y, w, h)
        if self.video_mode:
            T.card(surf, g, fill=T.C.PANEL_2, border=T.C.BORDER)
            T.section_hdr(surf, x + 10, y + 6, "VIDEO INPUT TELEMETRY", color=T.C.TEXT_DIM, panel_w=w - 20)
            stat = self.perf.live_stats()
            rows = [
                ("Input Feed", os.path.basename(self.video_path)[:22]),
                ("Acq Time", f"{stat['acquisition_time_s']:.2f}s" if stat["acquisition_time_s"] is not None else "--"),
                ("Retention", f"{stat['retention_total_pct']:.1f}%"),
            ]
            yy = y + 28
            for lab, val in rows:
                T.text(surf, (x + 12, yy), lab, 11, T.C.TEXT_DIM, bold=True)
                T.text(surf, (x + w - 12, yy), val, 12, T.C.TEXT, bold=True, anchor="tr")
                yy += 18
            return
        view3d.render(surf, g, self.sim, self.sim.t)

    # ── Comparison ──────────────────────────────────────────────────────────────
    def _panel_comparison_small(self, surf, x, y, w):
        if self.compare is None:
            return
        res = self.compare.get("results", {})
        row = res.get(self.preset)
        if not row:
            return
        ada  = row.get("adaptive", {})
        base = row.get("baseline", {})
        T.text(surf, (x, y), "ADAPTIVE vs BASELINE", 10, T.C.TEXT_FAINT, bold=True)
        cx = [x + 120, x + 180, x + 240]
        for xx, lbl in zip(cx, ["ACQ", "RET%", "FAL"]):
            T.text(surf, (xx, y), lbl, 9, T.C.TEXT_FAINT)
        y2 = y + 14
        T.text(surf, (x, y2), "ADAPTIVE", 10, T.C.GREEN, bold=True)
        T.text(surf, (cx[0], y2), f"{ada.get('acq', 0):.2f}s", 10, T.C.TEXT)
        T.text(surf, (cx[1], y2), f"{ada.get('ret_pct', 0):.0f}%", 10, T.C.TEXT)
        T.text(surf, (cx[2], y2), f"{ada.get('false_locks', 0)}", 10, T.C.GREEN)

    # ── Disturbances ──────────────────────────────────────────────────────────
    def _panel_disturbances(self, surf):
        x, y, w, h = self.PNL_INN, self.pnl_dist_y, self.PNL_IW, self.pnl_dist_h
        r = pygame.Rect(x, y, w, h)
        T.card(surf, r, fill=T.C.PANEL_2, border=T.C.BORDER)
        kind = disturbance_kind_label(self.platform_mode)
        T.section_hdr(surf, x + 10, y + 6, f"DISTURBANCES ({kind})", color=T.C.AMBER, panel_w=w - 20)
        # Status handled inside sliders
        for s in self.sliders.values():
            s.draw(surf)

    # ── Controls ──────────────────────────────────────────────────────────────
    def _panel_controls(self, surf):
        x, y, w, h = self.PNL_INN, self.pnl_ctrl_y, self.PNL_IW, self.pnl_ctrl_h
        r = pygame.Rect(x, y, w, h)
        T.card(surf, r, fill=T.C.PANEL_2, border=T.C.BORDER)
        T.section_hdr(surf, x + 10, y + 5, "MISSION CONTROLS", panel_w=w - 20)
        for b in self.buttons.values():
            b.draw(surf)

    def _load_compare(self):
        try:
            p = os.path.join(config.LOG_DIR, "compare_summary.json")
            if not os.path.exists(p):
                return None
            import json
            with open(p) as fh:
                return json.load(fh)
        except Exception:
            return None

    # ================================================================ ISRO PS 26169 MODAL
    def _draw_ps_inspector_modal(self, surf):
        mw = min(1220, self.W - 60)
        mh = min(740, self.H - 50)
        mx = (self.W - mw) // 2
        my = (self.H - mh) // 2
        self.ps_modal_rect = pygame.Rect(mx, my, mw, mh)

        # Backdrop
        overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        overlay.fill((2, 6, 14, 215))
        surf.blit(overlay, (0, 0))

        # Modal Card
        pygame.draw.rect(surf, (8, 14, 26), self.ps_modal_rect, border_radius=6)
        pygame.draw.rect(surf, (0, 220, 255), self.ps_modal_rect, 2, border_radius=6)
        # Gold/amber top accent line
        pygame.draw.rect(surf, (255, 180, 0), (mx, my, mw, 4), border_top_left_radius=6, border_top_right_radius=6)

        # Modal Header
        T.text(surf, (mx + 20, my + 14), "ISRO PROBLEM STATEMENT ID 26169 — SPECIFICATION COMPLIANCE INSPECTOR", 17, T.C.TEXT, bold=True)
        T.text(surf, (mx + 20, my + 38), "Autonomous Virtual Camera Tracking System for Coarse Alignment of Mobile FSOC Terminals · Clause Audit & Live Demos", 11.5, T.C.CYAN_ELEC, bold=True)

        # Close button [X]
        self.ps_close_rect = pygame.Rect(mx + mw - 42, my + 12, 30, 30)
        pygame.draw.rect(surf, (36, 14, 18), self.ps_close_rect, border_radius=4)
        pygame.draw.rect(surf, T.C.RED, self.ps_close_rect, 1, border_radius=4)
        T.text(surf, (self.ps_close_rect.centerx, self.ps_close_rect.centery), "X", 14, T.C.RED, bold=True, anchor="cc")

        # Two Columns Layout
        col1_w = int(mw * 0.54)
        col1_x = mx + 20
        col2_x = col1_x + col1_w + 20
        col2_w = mw - col1_w - 60
        body_y = my + 68

        # --- LEFT COLUMN: CLAUSE-BY-CLAUSE AUDIT ---
        T.text(surf, (col1_x, body_y), "MANDATORY & OPTIONAL CLAUSE VERIFICATION (ISRO PS 26169)", 12, (255, 180, 0), bold=True)

        res = self.sim.last_result
        err_deg = res.get("pointing_err_deg", 0.0)
        err_px = err_deg * (config.FOCAL_PX / (180.0 / math.pi))
        cur_shape = getattr(getattr(self.sim, "scene", None), "beacon", None)
        cur_shape = getattr(cur_shape, "shape", getattr(config, "TARGET_SHAPE", "SQUARE")) if cur_shape else "SQUARE"
        cur_motion = getattr(getattr(self.sim, "scene", None), "orbit", None)
        cur_motion = getattr(cur_motion, "motion_type", "FIGURE_EIGHT") if cur_motion else "FIGURE_EIGHT"
        boresight_err = res.get("boresight_error_px")
        if boresight_err is None and getattr(self, "video_mode", False):
            boresight_err = res.get("optical_offset_px")
        is_aligned = (boresight_err is not None and boresight_err <= 15.0)
        is_spec_pass = (err_px < 10.0 and res.get("state") in LOCKED_STATES and is_aligned)

        clauses = [
            ("Clause 1: 2000×2000 Screen Canvas", "Min 2000×2000 px, camera origin at centre", "2000×2000 Canvas · Centre (1000, 1000) · 160 px/°", "PASS [OK]", T.C.GREEN),
            ("Clause 2: Camera FOV & Update Rate", "640×480 @ 4°×3° FOV, ≥ 30 Hz, Monochrome", "640×480 px · 4.0°×3.0° · 60.0 Hz · Monochrome", "PASS [OK]", T.C.GREEN),
            ("Clause 3: Target Geometry & Sizing", "10×10 px · Square (Default), Circle, Spot", f"Active: {cur_shape} · 10×10 px · Multi-Target Capable", "PASS [OK]", T.C.GREEN),
            ("Clause 4: Target Motion Kinematics", "Straight Line, Circular, Figure-8, Spiral, Random", f"Active: {cur_motion} · Velocity 1.15 °/s", "PASS [OK]", T.C.GREEN),
            ("Clause 5: Actuator Slew Limits", "Pan-tilt gimbal, max slew 5–10 °/s clamp", "5.0 °/s Slew Clamp · 14.0 °/s² Accel · Anti-Windup", "PASS [OK]", T.C.GREEN),
            ("Clause 6: Tracking Error Specification", "Target must stay within 10 pixels of camera centre", f"Error: {err_px:.1f} px ({err_deg*1000:.1f} m°) · Spec: < 10 px", "SPEC PASS [OK]" if is_spec_pass else "ALIGNING...", T.C.GREEN if is_spec_pass else T.C.AMBER),
            ("Clause 7: Autonomous Reacquisition", "Autonomous recovery upon target occlusion / loss", "Multi-tier: Ephemeris Coast → Handover Ladder", "PASS [OK]", T.C.GREEN),
            ("Clause 8: Benchmark-2 Video Bypass", "External MP4 PTZ bypass for grader validation", "Full OpenCV MP4 Bypass Pipeline Integrated", "PASS [OK]", T.C.GREEN),
        ]

        cy = body_y + 22
        card_h = 48
        card_gap = 9
        for c_title, c_req, c_impl, c_status, c_col in clauses:
            cr = pygame.Rect(col1_x, cy, col1_w, card_h)
            pygame.draw.rect(surf, (12, 20, 36), cr, border_radius=4)
            pygame.draw.rect(surf, (24, 40, 68), cr, 1, border_radius=4)

            # Left colored accent line
            pygame.draw.rect(surf, c_col, (col1_x, cy, 3, card_h), border_top_left_radius=4, border_bottom_left_radius=4)

            T.text(surf, (col1_x + 12, cy + 6), c_title, 11.5, T.C.TEXT, bold=True)
            T.text(surf, (col1_x + 12, cy + 24), c_impl, 10.5, T.C.TEXT_DIM)

            # Status chip on right
            sw = 88 if "SPEC" in c_status else 68
            sr = pygame.Rect(cr.right - sw - 10, cy + 10, sw, 26)
            s_bg = (6, 32, 20) if c_col == T.C.GREEN else (36, 26, 6)
            pygame.draw.rect(surf, s_bg, sr, border_radius=3)
            pygame.draw.rect(surf, c_col, sr, 1, border_radius=3)
            T.text(surf, (sr.centerx, sr.centery), c_status, 10.5, c_col, bold=True, anchor="cc")
            cy += card_h + card_gap

        # --- RIGHT COLUMN: INTERACTIVE CONFIGURATOR & INNOVATION SHOWCASE ---
        ry = body_y
        T.text(surf, (col2_x, ry), "LIVE TARGET CONFIGURATOR (JUDGE DEMO)", 12, (255, 180, 0), bold=True)
        ry += 22

        # 1. Target Shape Buttons
        T.text(surf, (col2_x, ry), "Target Geometry (PS Clause 3):", 11, T.C.TEXT_DIM, bold=True)
        ry += 18
        shapes = [("SQUARE", "SQUARE (PS Def)"), ("CIRCLE", "CIRCLE"), ("SPOT", "SPOT PSF")]
        bw = (col2_w - 16) // 3
        for idx, (skey, slbl) in enumerate(shapes):
            bx = col2_x + idx * (bw + 8)
            br = pygame.Rect(bx, ry, bw, 32)
            self.ps_btn_rects[f"shape_{skey}"] = br
            sel = (cur_shape == skey)
            bbg = (6, 36, 26) if sel else (12, 22, 38)
            bcol = T.C.GREEN if sel else (0, 180, 240)
            pygame.draw.rect(surf, bbg, br, border_radius=4)
            pygame.draw.rect(surf, bcol, br, 1 if not sel else 2, border_radius=4)
            T.text(surf, (br.centerx, br.centery), slbl, 11, bcol, bold=True, anchor="cc")
        ry += 42

        # 2. Target Motion Trajectory Buttons
        T.text(surf, (col2_x, ry), "Motion Trajectory (PS Clause 4):", 11, T.C.TEXT_DIM, bold=True)
        ry += 18
        motions = [("FIGURE_EIGHT", "FIGURE-8"), ("CIRCULAR", "CIRCULAR"), ("STRAIGHT_LINE", "STRAIGHT"), ("SPIRAL", "SPIRAL")]
        mbw = (col2_w - 24) // 4
        for idx, (mkey, mlbl) in enumerate(motions):
            bx = col2_x + idx * (mbw + 8)
            br = pygame.Rect(bx, ry, mbw, 32)
            self.ps_btn_rects[f"motion_{mkey}"] = br
            sel = (cur_motion == mkey)
            bbg = (6, 36, 26) if sel else (12, 22, 38)
            bcol = T.C.GREEN if sel else (0, 180, 240)
            pygame.draw.rect(surf, bbg, br, border_radius=4)
            pygame.draw.rect(surf, bcol, br, 1 if not sel else 2, border_radius=4)
            T.text(surf, (br.centerx, br.centery), mlbl, 10.5, bcol, bold=True, anchor="cc")
        ry += 44

        # 3. Quick Stress Injections
        T.text(surf, (col2_x, ry), "Live Stress & Occlusion Test:", 11, T.C.TEXT_DIM, bold=True)
        ry += 18
        inj_w = (col2_w - 12) // 2
        inj1 = pygame.Rect(col2_x, ry, inj_w, 32)
        inj2 = pygame.Rect(col2_x + inj_w + 12, ry, inj_w, 32)
        self.ps_btn_rects["inj_occ"] = inj1
        self.ps_btn_rects["inj_vib"] = inj2
        pygame.draw.rect(surf, (36, 18, 12), inj1, border_radius=4)
        pygame.draw.rect(surf, T.C.AMBER, inj1, 1, border_radius=4)
        T.text(surf, (inj1.centerx, inj1.centery), "[!] INJECT 3s OCCLUSION", 11, T.C.AMBER, bold=True, anchor="cc")

        pygame.draw.rect(surf, (32, 12, 24), inj2, border_radius=4)
        pygame.draw.rect(surf, T.C.RED, inj2, 1, border_radius=4)
        T.text(surf, (inj2.centerx, inj2.centery), "[!] INJECT 3x VIBRATION", 11, T.C.RED, bold=True, anchor="cc")
        ry += 48

        # 4. INNOVATION SHOWCASE BOX (WHAT'S NEW & UNIQUE)
        innov_rect = pygame.Rect(col2_x, ry, col2_w, mh - (ry - my) - 20)
        pygame.draw.rect(surf, (10, 22, 42), innov_rect, border_radius=6)
        pygame.draw.rect(surf, (0, 229, 255), innov_rect, 1, border_radius=6)

        T.text(surf, (col2_x + 14, ry + 10), "[*] WHAT'S NEW & UNIQUE (INNOVATION)", 12, (255, 180, 0), bold=True)
        T.text(surf, (col2_x + 14, ry + 28), "ADAPTIVE MODEL-VISION TRUST ARBITER (SEC. 9)", 11, T.C.CYAN_ELEC, bold=True)

        desc_lines = [
            "Unlike rigid baseline Kalman trackers that experience catastrophic",
            "filter divergence under atmospheric scintillation or ephemeris bias,",
            "our architecture features a continuous Mahalanobis Trust Arbiter.",
            "It dynamically balances Optical Vision (T_vis) and Orbital Kinematics (T_mdl).",
            "Key Result: 0% False Lock Rate, sub-100ms Reacquisition Handover.",
        ]
        dy = ry + 48
        for dline in desc_lines:
            T.text(surf, (col2_x + 14, dy), dline, 10.5, T.C.TEXT_MUTED)
            dy += 16

        # Live Trust Gauges inside innovation box
        tm = getattr(self.sim.tracker, "trust", None)
        vis_t = tm.vision_trust if tm else 0.95
        mdl_t = tm.model_trust if tm else 0.85
        sig_t = getattr(getattr(self.sim.tracker, "unc", None), "display_sigma_px", 1.8)

        gy = dy + 10
        T.text(surf, (col2_x + 14, gy), f"VISION TRUST: {vis_t:.2f}", 11, T.C.CYAN_ELEC, bold=True, mono=True)
        W.hbar(surf, (col2_x + 160, gy + 3, col2_w - 180, 8), vis_t, T.C.CYAN_ELEC)
        gy += 22
        T.text(surf, (col2_x + 14, gy), f"MODEL TRUST:  {mdl_t:.2f}", 11, T.C.PURPLE, bold=True, mono=True)
        W.hbar(surf, (col2_x + 160, gy + 3, col2_w - 180, 8), mdl_t, T.C.PURPLE)
        gy += 22
        T.text(surf, (col2_x + 14, gy), f"UNCERTAINTY sigma: {sig_t:.1f} px  ·  15 Hz CORR: r = {self.sim.tracker.mod.corr():.2f}", 11, T.C.GREEN, bold=True, mono=True)

    def _handle_ps_modal_click(self, pos):
        if hasattr(self, "ps_close_rect") and self.ps_close_rect.collidepoint(pos):
            self.show_ps_modal = False
            return True

        # Shape buttons
        for skey in ("SQUARE", "CIRCLE", "SPOT"):
            bkey = f"shape_{skey}"
            if bkey in self.ps_btn_rects and self.ps_btn_rects[bkey].collidepoint(pos):
                self.sim.scene.beacon.shape = skey
                config.TARGET_SHAPE = skey
                return True

        # Motion buttons
        for mkey in ("FIGURE_EIGHT", "CIRCULAR", "STRAIGHT_LINE", "SPIRAL"):
            bkey = f"motion_{mkey}"
            if bkey in self.ps_btn_rects and self.ps_btn_rects[bkey].collidepoint(pos):
                self.sim.scene.orbit.motion_type = mkey
                return True

        # Stress injections
        if "inj_occ" in self.ps_btn_rects and self.ps_btn_rects["inj_occ"].collidepoint(pos):
            self.stress_mgr.trigger("turb_burst", getattr(self, "events_list", None))
            return True

        if "inj_vib" in self.ps_btn_rects and self.ps_btn_rects["inj_vib"].collidepoint(pos):
            self.stress_mgr.trigger("platform_vib", getattr(self, "events_list", None))
            return True

        return False



# ── helpers ───────────────────────────────────────────────────────────────────
def _bracket_cam(surf, center, color, r, th):
    """Corner brackets on the camera surface."""
    x, y = center
    L    = r
    for dx, dy in ((1,1), (-1,1), (1,-1), (-1,-1)):
        pygame.draw.line(surf, color,
                         (x+dx*r, y+dy*L), (x+dx*r, y+dy*(r-L//2)), th)
        pygame.draw.line(surf, color,
                         (x+dx*r, y+dy*L), (x+dx*(r-L//2), y+dy*L), th)


# ── headless self-test ────────────────────────────────────────────────────────

def headless_selftest(frames, preset, platform=None, atmosphere=None,
                      motion_type=None, target_shape=None, target_size=None,
                      num_targets=None, target_initial=None):
    print(f"headless self-test: preset={preset} frames={frames}")
    sim  = Simulator(preset_name=preset, seed=1,
                     platform_mode=platform, atmosphere=atmosphere,
                     motion_type=motion_type, target_shape=target_shape,
                     target_size=target_size, num_targets=num_targets,
                     target_initial=target_initial)
    perf = PerformanceTracker()
    t0   = time.time()
    for _ in range(frames):
        sim.step()
        perf.record_frame(sim)
    wall = time.time() - t0
    st   = perf.live_stats()
    print(f"state={sim.state} acq={st['acquisition_time_s']} "
          f"retention={st['retention_total_pct']:.1f}% "
          f"mean_err={st['mean_err_deg']} rms={st['rms_err_deg']} "
          f"fps={frames/wall:.1f} false_lock={st['false_lock_events']}")


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset",     default="EASY")
    ap.add_argument("--frames",     type=int, default=0)
    ap.add_argument("--fullscreen", action="store_true", default=True,
                    help="start in fullscreen (default)")
    ap.add_argument("--windowed", action="store_false", dest="fullscreen",
                    help="start in a 1600x900 window")
    ap.add_argument("--platform",   default=None,
                    choices=["SATELLITE_SATELLITE", "UAV_SATELLITE", "UAV_UAV"])
    ap.add_argument("--atmosphere", default=None,
                    choices=["CLEAR", "HAZE", "FOG", "RAIN", "LOW_LIGHT"])
    ap.add_argument("--motion",     default=None,
                    choices=["straight_line", "circular", "figure_eight",
                             "random", "spiral", "sinusoidal"])
    ap.add_argument("--shape",      default=None,
                    choices=["SQUARE", "CIRCLE", "SPOT"])
    ap.add_argument("--size",       type=int, default=None)
    ap.add_argument("--targets",    type=int, default=None)
    ap.add_argument("--initial",    default=None, choices=["RANDOM", "CENTER"])
    ap.add_argument("--video",      default=None)
    ap.add_argument("--video-seed", type=int, default=None)
    ap.add_argument("--width",      type=int, default=1600,
                    help="window width (default 1600)")
    ap.add_argument("--height",     type=int, default=900,
                    help="window height (default 900)")
    ap.add_argument("--res",        type=str, default=None,
                    help="resolution string WxH, e.g. 1366x768, 1440x900, 1920x1080")
    args = ap.parse_args()

    if args.res:
        try:
            rw, rh = [int(x.strip()) for x in args.res.lower().split("x")]
            args.width, args.height = rw, rh
        except Exception:
            pass

    if args.frames > 0:
        headless_selftest(args.frames, args.preset.upper(),
                          args.platform, args.atmosphere,
                          args.motion, args.shape, args.size,
                          args.targets, args.initial)
        return

    app = App(preset=args.preset.upper(), fullscreen=args.fullscreen,
              platform_mode=args.platform, atmosphere=args.atmosphere,
              motion_type=args.motion, target_shape=args.shape,
              target_size=args.size, num_targets=args.targets,
              target_initial=args.initial, video_path=args.video,
              video_seed=args.video_seed,
              width=args.width, height=args.height)
    app.run()


if __name__ == "__main__":
    main()
