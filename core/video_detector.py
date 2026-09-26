"""
core/video_detector.py
----------------------
Benchmark-2 (evaluator MP4) beacon detector.

Evaluator videos are full-screen (up to 2000x2000), monochrome, carry heavy
salt-and-pepper / Gaussian / Poisson noise and contain a bright square spot of
5-20 px with no 15 Hz modulation and no colour.  This detector relies only on
what the PS guarantees:

  1. Coarse search on an area-downscaled copy (long side <= COARSE_MAX_SIDE),
     so a 2000x2000 frame costs about the same as a ~700x700 one.
  2. Median prefilter (kills salt/pepper) + large-box background subtraction
     (handles haze, fog, low light and illumination gradients).
  3. Matched filter: box filter the size of the expected target.
  4. Robust SNR from median/MAD of the response; local maxima above SNR_MIN.
  5. Full-resolution ROI refinement: half-maximum segmentation of the peak
     component and intensity-weighted sub-pixel centroid.

``ml_score`` is a deterministic physics score (SNR, size plausibility, square
fill ratio): any bright square/spot the PS allows is accepted.  The learned
classifier was trained only on the synthetic glow beacon and scores the PS's
default hard square at ~0.
"""

import math

import cv2
import numpy as np

from core import geometry
from core.detection import Candidate

COARSE_MAX_SIDE = 720
SNR_MIN = 6.0
MAX_CANDIDATES = 6
MIN_TARGET_PX = 5
MAX_TARGET_PX = 20
DEFAULT_TARGET_PX = 10
TRACK_GATE_PX = 60.0


class VideoBeaconDetector:
    def __init__(self, target_px=DEFAULT_TARGET_PX):
        self.target_px = int(max(MIN_TARGET_PX, min(MAX_TARGET_PX, target_px)))
        self._tracks = []
        self._next_id = 1
        self.last_mask = None

    def reset(self):
        self._tracks = []
        self._next_id = 1

    def _coarse(self, grey):
        h, w = grey.shape
        scale = max(1.0, max(h, w) / float(COARSE_MAX_SIDE))
        if scale > 1.0:
            small = cv2.resize(grey, (int(round(w / scale)), int(round(h / scale))),
                               interpolation=cv2.INTER_AREA)
        else:
            small = grey
        small = cv2.medianBlur(small, 3)
        f = small.astype(np.float32)
        bg_k = max(15, int(6 * self.target_px / scale) | 1)
        bg = cv2.blur(f, (bg_k, bg_k))
        k = max(2, int(round(self.target_px / scale)))
        resp = cv2.blur(f - bg, (k, k))
        return scale, resp

    def _refine(self, grey, cx, cy):
        h, w = grey.shape
        r = int(MAX_TARGET_PX * 1.6)
        x0, y0 = max(0, int(cx) - r), max(0, int(cy) - r)
        x1, y1 = min(w, int(cx) + r + 1), min(h, int(cy) + r + 1)
        roi = grey[y0:y1, x0:x1]
        if roi.shape[0] < 3 or roi.shape[1] < 3:
            return None
        roi = cv2.medianBlur(np.ascontiguousarray(roi), 3).astype(np.float32)
        bgv = float(np.median(roi))
        sm = cv2.blur(roi, (3, 3))
        py, px = np.unravel_index(int(np.argmax(sm)), sm.shape)
        peak = float(sm[py, px])
        amp = peak - bgv
        if amp <= 1.0:
            return None
        mask = (sm > bgv + 0.5 * amp).astype(np.uint8)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        lab = labels[py, px]
        if lab == 0:
            return None
        wts = np.clip(roi - bgv, 0, None) * (labels == lab)
        s = float(wts.sum())
        if s < 1e-3:
            return None
        ys, xs = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
        ux = float((wts * xs).sum() / s) + x0
        vy = float((wts * ys).sum() / s) + y0
        area = int(stats[lab, cv2.CC_STAT_AREA])
        bw = int(stats[lab, cv2.CC_STAT_WIDTH])
        bh = int(stats[lab, cv2.CC_STAT_HEIGHT])
        fill = area / float(max(1, bw * bh))
        return ux, vy, area, bw, bh, fill, peak

    def _assign_track(self, cand):
        best, best_d = None, TRACK_GATE_PX
        for tr in self._tracks:
            if tr["claimed"]:
                continue
            d = math.hypot(cand.u - tr["u"], cand.v - tr["v"])
            if d < best_d:
                best, best_d = tr, d
        if best is None:
            best = {"id": self._next_id, "age": 0, "miss": 0}
            self._next_id += 1
            self._tracks.append(best)
        best.update(u=cand.u, v=cand.v, claimed=True, miss=0)
        best["age"] += 1
        cand.track_id = best["id"]
        cand.track_age = best["age"]

    def detect(self, frame_bgr, gimbal_basis, focal_px, cu, cv):
        grey = frame_bgr if frame_bgr.ndim == 2 else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        scale, resp = self._coarse(grey)

        med = float(np.median(resp))
        mad = float(np.median(np.abs(resp - med))) * 1.4826 + 1e-3
        snr_map = (resp - med) / mad
        dil = cv2.dilate(snr_map, np.ones((5, 5), np.uint8))
        peaks = np.argwhere((snr_map >= dil) & (snr_map > SNR_MIN))
        vals = snr_map[peaks[:, 0], peaks[:, 1]] if peaks.shape[0] else np.array([])
        order = np.argsort(-vals)[: MAX_CANDIDATES * 3]
        peaks, vals = peaks[order], vals[order]

        for tr in self._tracks:
            tr["claimed"] = False

        candidates, seen = [], []
        for (py, px), snr in zip(peaks, vals):
            fx, fy = (px + 0.5) * scale - 0.5, (py + 0.5) * scale - 0.5
            if any(math.hypot(fx - sx, fy - sy) < MAX_TARGET_PX for sx, sy in seen):
                continue
            ref = self._refine(grey, fx, fy)
            if ref is None:
                continue
            ux, vy, area, bw, bh, fill, peak = ref
            seen.append((ux, vy))
            side = math.sqrt(max(1, area))
            size_ok = 1.0 if (MIN_TARGET_PX * 0.6) <= side <= (MAX_TARGET_PX * 1.5) else 0.3
            aspect = min(bw, bh) / float(max(1, max(bw, bh)))
            snr_term = 1.0 - math.exp(-float(snr) / 12.0)
            shape_term = 0.5 + 0.25 * min(1.0, fill / 0.7) + 0.25 * aspect
            score = max(0.0, min(1.0, (0.35 + 0.65 * snr_term) * shape_term * size_ok * 1.15))
            cand = Candidate(
                x=ux, y=vy, u=ux, v=vy, area=area, peak=peak,
                snr=float(snr), circularity=float(fill),
                area_norm=float(area) / float(self.target_px ** 2),
                hue_dist=0.0, hue_dist_n=0.0, hue=0.0, ml_score=float(score),
            )
            cand.los_az, cand.los_el = geometry.ray_to_azel(
                ux, vy, focal_px, gimbal_basis, cu=cu, cv=cv)
            candidates.append(cand)
            if len(candidates) >= MAX_CANDIDATES:
                break

        candidates.sort(key=lambda c: (c.ml_score, c.snr), reverse=True)
        for c in candidates:
            self._assign_track(c)
        for tr in self._tracks:
            if not tr["claimed"]:
                tr["miss"] += 1
        self._tracks = [tr for tr in self._tracks if tr["miss"] <= 15]
        return candidates
