"""
core/confidence.py
-------------------
Unified, normalized (0-1) confidence state for the adaptive tracker.

Four separable questions the tracker must answer, plus a single overall value:

  * Identity    - "is this actually our beacon?"   (appearance + modulation)
  * Position    - "how accurate is its position?"  (SNR + centroid stability)
  * Prediction  - "is the predicted motion reliable?" (observation residual vs
                  the model prior, tempered by the disturbance level)
  * Pointing    - "is the gimbal actually pointing where commanded?" (servo
                  attitude error + stabilization residual)
  * Overall     - geometric mean of Identity, Position, Prediction: if any
                  single cue is weak, overall confidence drops.  A product-like
                  combination is deliberate and explainable: confidence in the
                  fused estimate is only as strong as its weakest cue.

Pointing confidence is intentionally kept separate from Overall: a degraded
sensor improves via better tracking, not via the servo; and vice versa.
Nothing in this module touches the control loop -- it is pure estimation that
the TrustManager (core/trust.py) consumes.
"""

import math
import config


def _lpf(old, new, k):
    if old is None:
        return new
    return k * new + (1.0 - k) * old


class ConfidenceState:
    def __init__(self):
        self.identity = 0.0
        self.position = 0.0
        self.prediction = 0.0
        self.pointing = 0.0
        self.overall = 0.0
        self._centroid_rms_px = None
        self._pred_fit = 1.0     # EMA of "prediction agrees with observation"

    # ------------------------------------------------------------------
    def update(self, *, identity_src, snr, centroid_residual_px,
               pred_residual_deg, pred_scale_deg, model_conf, dist_level,
               point_err_deg, point_err_scale_deg, unc_sigma_px=None):
        """Refresh all confidence values from this frame's signals.

        All arguments are raw, un-normalised measurements; normalisation to
        0-1 happens here, with physically-motivated saturations.  `unc_sigma_px`
        (the tracker's INTERNAL position uncertainty, uncapped) folds into the
        position confidence so confidence is *uncertainty-aware*: as the belief
        grows less certain the reported position confidence drops toward zero at
        the credibility/reacquisition line.
        """
        # Identity: appearance (identity_src, 0-1) is the base; the object's
        # own 15 Hz modulation (if known) reinforces it quadratically.
        self.identity = max(0.0, min(1.0, identity_src))

        # Position: SNR drives pixel uncertainty; the running RMS of the
        # centroid jump vs its own smoothing captures flicker/stability.
        self._centroid_rms_px = _lpf(
            self._centroid_rms_px, max(0.0, centroid_residual_px),
            config.TRUST_VISION_W_CENTROID)
        snr_c = max(0.0, min(1.0, snr / (snr + 3.0)))          # snr 20 -> 0.87
        stab_c = max(0.0, min(1.0, 1.0 - self._centroid_rms_px / 6.0))
        self.position = 0.7 * snr_c + 0.3 * stab_c
        # uncertainty-aware penalty: internal sigma past the nominal base pulls
        # position confidence down smoothly as it approaches the credibility line
        if unc_sigma_px is not None and unc_sigma_px > config.UNCERTAINTY_BASE_PX:
            unc_span = max(1e-9, config.REACQUIRE_UNCERTAINTY_PX
                           - config.UNCERTAINTY_BASE_PX)
            unc_frac = min(1.0, max(0.0, (unc_sigma_px - config.UNCERTAINTY_BASE_PX)
                                    / unc_span))
            # full penalty at the REACQUIRE line: a position whose sigma has
            # reached the credibility limit carries no positional trust
            unc_penalty = unc_frac ** 1.5
            self.position = max(0.0, self.position * (1.0 - unc_penalty))

        # Prediction: residual between model prior and observation, scaled by
        # the residual scale and suppressed by the disturbance level.  A target
        # that accelerates, or a corrupted ephemeris, inflates the residual and
        # collapses model trust (the observer then takes over).
        resid_c = max(0.0, min(1.0, 1.0 - pred_residual_deg / max(1e-6, pred_scale_deg)))
        self._pred_fit = _lpf(self._pred_fit, resid_c, config.TRUST_MODEL_W_PRIOR)
        dist_c = 1.0 - max(0.0, min(1.0, dist_level))
        self.prediction = max(0.0, min(1.0, 0.55 * self._pred_fit + 0.45 * (0.5 + 0.5 * model_conf
                                                                           if model_conf else dist_c * 0.5)))

        # Pointing: normalised servo attitude error (cmd vs realized), which
        # already isolates platform disturbance residual.
        self.pointing = max(0.0, min(1.0,
                                     1.0 - point_err_deg / max(1e-6, point_err_scale_deg)))

        # Overall: adaptive fusion of Identity, Position, and Prediction.
        # When the optical sensor clearly observes and confirms the beacon
        # (identity >= 0.50 and position >= 0.45), visual evidence dominates
        # so unmodeled maneuvers or ephemeris divergence cannot collapse lock.
        prod = self.identity * self.position * self.prediction
        geom_mean = prod ** (1.0 / 3.0) if prod > 0 else 0.0
        if self.identity >= 0.50 and self.position >= 0.45:
            vis_cue = max(math.sqrt(self.identity * self.position),
                          0.55 * self.identity + 0.45 * self.position)
            self.overall = max(geom_mean, vis_cue)
        else:
            self.overall = geom_mean

        return self

    def update_pointing(self, point_err_deg, point_err_scale_deg=0.2):
        """Pointing (servo) confidence, updated by the controller after the
        gimbal integrates this frame's command (one-frame lag acceptable)."""
        self.pointing = max(0.0, min(1.0,
                                     1.0 - point_err_deg / max(1e-6,
                                                               point_err_scale_deg)))
        return self.pointing

    # ------------------------------------------------------------------
    def snapshot(self):
        return dict(identity=round(self.identity, 3),
                    position=round(self.position, 3),
                    prediction=round(self.prediction, 3),
                    pointing=round(self.pointing, 3),
                    overall=round(self.overall, 3))