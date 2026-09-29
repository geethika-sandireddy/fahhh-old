
"""
ui/evidence_pages.py
--------------------
Dedicated Target Settings and Run Reports screens plus a live stress overlay.
"""
from __future__ import annotations

import math
import pygame

from ui import theme as T


C = T.C


def _rect_button(surf, rect, label, selected=False, accent=C.CYAN_ELEC, enabled=True, size=12):
    if not enabled:
        bg = (12, 19, 32)
        border = (38, 50, 66)
        txt = C.TEXT_MUTED
    elif selected:
        bg = (0, 42, 62)
        border = accent
        txt = accent
    else:
        bg = (12, 21, 37)
        border = C.BORDER
        txt = C.TEXT_DIM
    pygame.draw.rect(surf, bg, rect, border_radius=4)
    pygame.draw.rect(surf, border, rect, 1, border_radius=4)
    T.text(surf, rect.center, label, size, txt, bold=True, anchor="cc")
    return rect


def _line_chart(surf, rect, samples, keys, labels, colors, title, units=""):
    T.card(surf, rect, fill=C.PANEL_2, border=C.BORDER)
    T.section_title(surf, rect.x + 12, rect.y + 10, title, C.CYAN_ELEC)
    plot = pygame.Rect(rect.x + 38, rect.y + 36, rect.w - 52, rect.h - 56)
    pygame.draw.rect(surf, (7, 14, 27), plot)
    pygame.draw.rect(surf, C.BORDER, plot, 1)
    for frac in (0.0, 0.5, 1.0):
        yy = plot.bottom - int(frac * plot.h)
        pygame.draw.line(surf, (18, 33, 56), (plot.left, yy), (plot.right, yy), 1)

    series = []
    all_vals = []
    for key in keys:
        vals = []
        for s in samples:
            v = s.get(key)
            if v is None:
                vals.append(None)
            else:
                try:
                    fv = float(v)
                except Exception:
                    fv = None
                vals.append(fv)
                if fv is not None and math.isfinite(fv):
                    all_vals.append(fv)
        series.append(vals)

    if all_vals:
        lo, hi = min(all_vals), max(all_vals)
        if abs(hi - lo) < 1e-9:
            pad = max(1.0, abs(hi) * 0.1)
            lo -= pad
            hi += pad
        n = max(1, len(samples) - 1)
        for vals, col in zip(series, colors):
            points = []
            for i, v in enumerate(vals):
                if v is None:
                    continue
                x = plot.left + int((plot.w - 1) * i / n)
                y = plot.bottom - int((plot.h - 1) * ((v - lo) / (hi - lo)))
                points.append((x, y))
            if len(points) >= 2:
                pygame.draw.lines(surf, col, False, points, 2)
        T.text(surf, (plot.left - 6, plot.top), f"{hi:.2f}{units}", 9, C.TEXT_MUTED, anchor="rt", mono=True)
        T.text(surf, (plot.left - 6, plot.bottom), f"{lo:.2f}{units}", 9, C.TEXT_MUTED, anchor="rb", mono=True)
    else:
        T.text(surf, plot.center, "PRESS START RUN TO CAPTURE A TIME-SERIES", 11, C.TEXT_MUTED, bold=True, anchor="cc")

    lx = rect.x + 14
    for label, col in zip(labels, colors):
        pygame.draw.line(surf, col, (lx, rect.bottom - 10), (lx + 18, rect.bottom - 10), 2)
        T.text(surf, (lx + 24, rect.bottom - 15), label, 9.5, C.TEXT_MUTED, bold=True)
        lx += 120


def render_target_settings_page(surf, rect, app):
    x0, y0, w, h = rect.x, rect.y, rect.w, rect.h
    app.target_ui_rects = {}
    T.card(surf, pygame.Rect(x0, y0, w, 52), fill=C.PANEL_2, border=C.BORDER)
    T.section_title(surf, x0 + 16, y0 + 14, "TARGET SETTINGS & TRAJECTORY CONTROL", C.CYAN_ELEC)
    T.text(surf, (x0 + 390, y0 + 15), "CONFIGURE TARGETS WITHOUT TOUCHING THE MAIN MISSION VIEW", 11, C.TEXT_DIM, bold=True)

    scene = getattr(app.sim, "scene", None)
    beacon = getattr(scene, "beacon", None)
    shape = str(getattr(beacon, "shape", "SQUARE")).upper()
    size_x = int(getattr(beacon, "size_px", 10))
    size_y = int(getattr(beacon, "size_py", size_x))
    count = int(getattr(scene, "num_targets", getattr(app, "target_count", 1)))
    primary = int(getattr(app, "primary_target_idx", 0))
    trajectory = str(getattr(app, "current_motion", "straight_line"))

    gap = 14
    left_w = int(w * 0.46)
    right_x = x0 + left_w + gap
    right_w = w - left_w - gap
    top = y0 + 66

    # Target preview + identity
    left = pygame.Rect(x0, top, left_w, h - 66)
    T.card(surf, left, fill=C.PANEL_2, border=C.BORDER)
    T.section_title(surf, left.x + 16, left.y + 12, "ACTIVE TARGET", C.GREEN)
    preview = pygame.Rect(left.x + 16, left.y + 48, left.w - 32, 210)
    pygame.draw.rect(surf, (5, 11, 23), preview)
    pygame.draw.rect(surf, C.BORDER, preview, 1)
    cx, cy = preview.center
    r = max(12, min(45, size_x * 2))
    if shape == "CIRCLE":
        pygame.draw.circle(surf, C.GREEN, (cx, cy), r, 2)
    elif shape == "ELLIPSE":
        pygame.draw.ellipse(surf, C.GREEN, (cx-r-15, cy-r+6, (r+15)*2, (r-6)*2), 2)
    elif shape == "SPOT":
        pygame.draw.circle(surf, C.CYAN_ELEC, (cx, cy), max(4, r//3))
        pygame.draw.circle(surf, C.CYAN_ELEC, (cx, cy), r, 1)
    else:
        pygame.draw.rect(surf, C.GREEN, (cx-r, cy-r, r*2, r*2), 2)
    pygame.draw.line(surf, (22, 40, 62), (cx-70, cy), (cx+70, cy))
    pygame.draw.line(surf, (22, 40, 62), (cx, cy-70), (cx, cy+70))

    info_y = preview.bottom + 20
    info = [
        ("SHAPE", shape),
        ("SIZE", f"{size_x} × {size_y} px"),
        ("TARGET COUNT", str(count)),
        ("PRIMARY", f"TARGET-{primary+1:02d}"),
        ("TRAJECTORY", trajectory.replace("_", " ").upper()),
        ("PLATFORM", getattr(app, "_platform_label", lambda: app.platform_mode)()),
    ]
    for i, (lab, val) in enumerate(info):
        yy = info_y + i * 34
        T.text(surf, (left.x + 18, yy), lab, 10, C.TEXT_MUTED, bold=True)
        T.text(surf, (left.right - 18, yy), str(val), 12, C.TEXT, bold=True, anchor="tr", mono=True)

    # Shape/trajectory controls
    panel = pygame.Rect(right_x, top, right_w, h - 66)
    T.card(surf, panel, fill=C.PANEL_2, border=C.BORDER)
    T.section_title(surf, panel.x + 16, panel.y + 12, "CONFIGURATION", C.AMBER)

    shapes = ["SQUARE", "CIRCLE", "ELLIPSE", "SPOT"]
    y = panel.y + 48
    T.text(surf, (panel.x + 16, y), "TARGET SHAPE", 10, C.TEXT_MUTED, bold=True)
    x = panel.x + 16
    for sh in shapes:
        rr = pygame.Rect(x, y + 22, 108, 32)
        _rect_button(surf, rr, sh, selected=(shape == sh), accent=C.GREEN)
        app.target_ui_rects[f"shape:{sh}"] = rr
        x += 116

    y += 74
    T.text(surf, (panel.x + 16, y), "TRAJECTORY", 10, C.TEXT_MUTED, bold=True)
    trajectories = [
        ("straight_line", "STRAIGHT"),
        ("circular", "CIRCULAR"),
        ("figure_eight", "FIGURE-8"),
        ("random", "RANDOM"),
        ("spiral", "SPIRAL"),
    ]
    x = panel.x + 16
    for key, label in trajectories:
        rr = pygame.Rect(x, y + 22, 118, 32)
        _rect_button(surf, rr, label, selected=(trajectory == key), accent=C.CYAN_ELEC)
        app.target_ui_rects[f"motion:{key}"] = rr
        x += 126
        if x + 118 > panel.right:
            x = panel.x + 16
            y += 40

    y += 78 if x == panel.x + 16 else 42
    T.text(surf, (panel.x + 16, y), "TARGET COUNT", 10, C.TEXT_MUTED, bold=True)
    for j, (lab, key, dx) in enumerate([("-", "count_dec", 0), (str(count), "count_value", 50), ("+", "count_inc", 100)]):
        rr = pygame.Rect(panel.x + 16 + dx, y + 22, 42, 34 if key != "count_value" else 42)
        _rect_button(surf, rr, lab, selected=(key == "count_value"), accent=C.GREEN)
        app.target_ui_rects[key] = rr

    T.text(surf, (panel.x + 210, y), "TARGET SIZE", 10, C.TEXT_MUTED, bold=True)
    for j, (lab, key, dx) in enumerate([("-", "size_dec", 0), (f"{size_x}px", "size_value", 50), ("+", "size_inc", 124)]):
        rr = pygame.Rect(panel.x + 210 + dx, y + 22, 68 if key == "size_value" else 42, 34)
        _rect_button(surf, rr, lab, selected=(key == "size_value"), accent=C.AMBER)
        app.target_ui_rects[key] = rr

    y += 76
    T.text(surf, (panel.x + 16, y), "PRIMARY TARGET / HANDOVER", 10, C.TEXT_MUTED, bold=True)
    rr1 = pygame.Rect(panel.x + 16, y + 22, 72, 34)
    rr2 = pygame.Rect(panel.x + 94, y + 22, 72, 34)
    _rect_button(surf, rr1, "‹ PREV", accent=C.CYAN_ELEC)
    _rect_button(surf, rr2, "NEXT ›", accent=C.CYAN_ELEC)
    app.target_ui_rects["primary_prev"] = rr1
    app.target_ui_rects["primary_next"] = rr2

    # Target list
    list_y = y + 78
    available = getattr(scene, "beacons", []) if scene else []
    for idx, b in enumerate(available[:5]):
        rr = pygame.Rect(panel.x + 16, list_y + idx * 44, panel.w - 32, 36)
        selected = idx == primary
        _rect_button(surf, rr, f"TARGET-{idx+1:02d}   {getattr(b,'shape',shape):<7}   ACTIVE" if selected else f"TARGET-{idx+1:02d}   {getattr(b,'shape',shape):<7}",
                     selected=selected, accent=C.GREEN)
        app.target_ui_rects[f"primary:{idx}"] = rr


def render_live_stress_overlay(surf, rect, sim, stress_mgr):
    """Overlay a judge-readable live response panel on the existing stress page."""
    x0 = rect.right - 455
    y0 = rect.y + 44
    w = 441
    h = 188
    if x0 < rect.x + 20:
        return
    box = pygame.Rect(x0, y0, w, h)
    res = getattr(sim, "last_result", None) or {}
    trk = getattr(sim, "tracker", None)
    unc = getattr(trk, "unc", None)
    confidence = float(res.get("confidence", getattr(getattr(trk, "conf", None), "overall", 0.0)) or 0.0)
    err_px = res.get("boresight_error_px")
    fps = 1.0 / max(1e-6, float(getattr(sim, "dt", 1.0/30.0))) if getattr(sim, "dt", 0) else 0.0
    active = [sc["name"] for sc in getattr(stress_mgr, "scenarios", {}).values() if sc.get("active")]

    T.card(surf, box, fill=(8, 18, 31), border=C.CYAN_ELEC, radius=5)
    T.section_title(surf, box.x + 14, box.y + 10, "LIVE PAT RESPONSE — INJECTION EFFECT", C.CYAN_ELEC)
    labels = [
        ("STATE", str(res.get("state", "SEARCHING"))),
        ("TRACK CONF", f"{confidence*100:.1f}%"),
        ("PAT ERROR", "n/a" if err_px is None else f"{float(err_px):.2f} px"),
        ("FPS", f"{fps:.1f}"),
        ("UNCERTAINTY", f"{float(getattr(unc,'display_sigma_px',0.0) or 0.0):.2f} px"),
        ("GIMBAL", f"P {float(res.get('gimbal_pan',0)):+.2f}° / T {float(res.get('gimbal_tilt',0)):+.2f}°"),
    ]
    for i,(lab,val) in enumerate(labels):
        col = i % 2
        row = i // 2
        xx = box.x + 14 + col * 205
        yy = box.y + 42 + row * 42
        T.text(surf, (xx, yy), lab, 9, C.TEXT_MUTED, bold=True)
        T.text(surf, (xx, yy + 15), val, 12, C.TEXT, bold=True, mono=True)
    stress_text = " + ".join(active) if active else "NO ACTIVE STRESS"
    T.text(surf, (box.x + 14, box.bottom - 20), "ACTIVE:", 9, C.TEXT_MUTED, bold=True)
    T.text(surf, (box.x + 70, box.bottom - 20), stress_text[:54], 10, C.AMBER if active else C.GREEN, bold=True)


def render_run_report_page(surf, rect, app):
    x0, y0, w, h = rect.x, rect.y, rect.w, rect.h
    app.report_ui_rects = {}
    recorder = getattr(app, "run_recorder", None)
    samples = list(getattr(recorder, "samples", [])) if recorder else []
    active = bool(getattr(recorder, "active", False))
    summary = recorder.live_summary() if (recorder and active) else (getattr(recorder, "latest_report", {}) or {}).get("summary", {}) if recorder else {}
    if not summary:
        try:
            ps = app.perf.live_stats()
        except Exception:
            ps = {}
        summary = {
            "fps_mean": ps.get("fps", 0.0),
            "pat_error_mean_px": ps.get("mean_boresight_err_px"),
            "acquisition_time_s": ps.get("acquisition_time_s"),
            "confidence_mean_pct": ps.get("mean_vision_trust_pct") or 0.0,
            "alignment_accuracy_pct": 0.0,
            "lock_retention_pct": ps.get("retention_total_pct", 0.0),
            "reacquisition_count": ps.get("reacquisition_count", 0),
            "max_uncertainty_px": ps.get("max_uncertainty_px"),
        }

    # Header
    header = pygame.Rect(x0, y0, w, 52)
    T.card(surf, header, fill=C.PANEL_2, border=C.BORDER)
    T.section_title(surf, x0 + 16, y0 + 14, "MISSION RUN REPORTS & EVIDENCE", C.CYAN_ELEC)
    status = "CAPTURING" if active else "READY"
    status_col = C.GREEN if active else C.TEXT_DIM
    T.draw_pill_badge(surf, pygame.Rect(x0 + 330, y0 + 12, 104, 26), status, status_col)

    # controls
    btn_y = y0 + 10
    start = pygame.Rect(w + x0 - 370, btn_y, 112, 32)
    stop = pygame.Rect(w + x0 - 250, btn_y, 112, 32)
    openp = pygame.Rect(w + x0 - 130, btn_y, 112, 32)
    _rect_button(surf, start, "START RUN", selected=active, accent=C.GREEN, enabled=not active)
    _rect_button(surf, stop, "STOP & REPORT", accent=C.AMBER, enabled=active)
    _rect_button(surf, openp, "OPEN PRINT", accent=C.CYAN_ELEC, enabled=bool(getattr(recorder, "latest_report", None)))
    app.report_ui_rects.update({"start": start, "stop": stop, "open": openp})

    top = y0 + 64
    kpi_h = 82
    gap = 10
    kw = (w - 3 * gap) // 4
    kpis = [
        ("FPS", f"{float(summary.get('fps_mean',0.0) or 0):.1f}", "runtime"),
        ("PAT ERROR", "n/a" if summary.get("pat_error_mean_px") is None else f"{summary['pat_error_mean_px']:.2f}", "mean px"),
        ("CONFIDENCE", f"{float(summary.get('confidence_mean_pct',0.0) or 0):.1f}", "%"),
        ("ACCURACY", f"{float(summary.get('alignment_accuracy_pct',0.0) or 0):.1f}", "≤15 px"),
    ]
    for i,(lab,val,sub) in enumerate(kpis):
        kr=pygame.Rect(x0+i*(kw+gap),top,kw,kpi_h)
        T.draw_metric_card(surf,kr,lab,val,unit="",sub=sub,color=[C.GREEN,C.CYAN_ELEC,C.PURPLE,C.AMBER][i])

    chart_top = top + kpi_h + 12
    chart_gap = 10
    chart_w = (w - chart_gap) // 2
    chart_h = max(150, int((h - (chart_top-y0) - 208) / 2))
    _line_chart(surf, pygame.Rect(x0, chart_top, chart_w, chart_h), samples,
                ["boresight_error_px"], ["PAT ERROR"], [C.CYAN_ELEC], "PAT PERFORMANCE ERROR", " px")
    _line_chart(surf, pygame.Rect(x0+chart_w+chart_gap, chart_top, chart_w, chart_h), samples,
                ["confidence_pct"], ["CONFIDENCE"], [C.GREEN], "TRACKING CONFIDENCE", " %")
    chart2 = chart_top + chart_h + chart_gap
    _line_chart(surf, pygame.Rect(x0, chart2, chart_w, chart_h), samples,
                ["gimbal_pan_deg","gimbal_tilt_deg"], ["PAN","TILT"], [C.CYAN_ELEC,C.AMBER], "GIMBAL PAN / TILT", "°")
    _line_chart(surf, pygame.Rect(x0+chart_w+chart_gap, chart2, chart_w, chart_h), samples,
                ["disturbance_load_pct","uncertainty_sigma_px"], ["DISTURBANCE","UNCERTAINTY"], [C.RED,C.PURPLE], "DISTURBANCE + UNCERTAINTY", "")

    # Bottom comparison table
    table_y = y0 + h - 192
    table = pygame.Rect(x0, table_y, w, 180)
    T.card(surf, table, fill=C.PANEL_2, border=C.BORDER)
    T.section_title(surf, table.x + 12, table.y + 10, "MODE / RUN COMPARISON — PPT READY", C.AMBER)
    rows = getattr(recorder, "history_rows", lambda: [])()
    headers = ["RUN", "PLATFORM", "SCENARIO", "TRAJECTORY", "FPS", "PAT PX", "ACCURACY", "LOCK"]
    colw = [118,120,90,150,62,75,88,70]
    x = table.x + 12
    for i,hd in enumerate(headers):
        T.text(surf, (x, table.y + 34), hd, 9, C.TEXT_MUTED, bold=True)
        x += colw[i]
    for ridx,row in enumerate(rows[-4:]):
        yy = table.y + 54 + ridx * 27
        x = table.x + 12
        vals = [
            str(row["run_id"])[-12:], str(row["platform"]).replace("_","-"),
            str(row["preset"]), str(row["trajectory"]).replace("_","-"),
            f"{row['fps']:.1f}", "n/a" if row["pat_error_px"] is None else f"{row['pat_error_px']:.2f}",
            f"{row['accuracy']:.1f}%", f"{row['retention']:.1f}%"
        ]
        for i,val in enumerate(vals):
            T.text(surf, (x, yy), val[:22], 10, C.TEXT if ridx == len(rows[-4:])-1 else C.TEXT_DIM, bold=(ridx == len(rows[-4:])-1), mono=(i>=4))
            x += colw[i]
    if not rows:
        T.text(surf, (table.centerx, table.centery + 12), "STOP A RUN TO CREATE THE FIRST COMPARISON ROW", 11, C.TEXT_MUTED, bold=True, anchor="cc")

    ready = ""
    if recorder and recorder.latest_report:
        latest = recorder.latest_report
        ready = f"REPORT READY · {latest.get('pdf_path') or latest.get('html_path')}"
    T.text(surf, (table.x + 12, table.bottom - 17), ready[-150:], 9, C.TEXT_MUTED, mono=True)


def handle_target_settings_click(app, pos):
    rects = getattr(app, "target_ui_rects", {})
    scene = getattr(app.sim, "scene", None)
    if scene is None:
        return False
    for key, rr in rects.items():
        if not rr.collidepoint(pos):
            continue
        if key.startswith("shape:"):
            shape = key.split(":",1)[1]
            scene.set_target_params(shape=shape)
            return True
        if key.startswith("motion:"):
            motion = key.split(":",1)[1]
            app.current_motion = motion
            scene.set_motion_type(motion)
            return True
        if key == "count_dec":
            app._adjust_target_count(-1); return True
        if key == "count_inc":
            app._adjust_target_count(1); return True
        if key == "size_dec":
            beacon = getattr(scene, "beacon", None)
            cur = int(getattr(beacon, "size_px", 10))
            scene.set_target_params(size_px=max(5, cur-1))
            return True
        if key == "size_inc":
            beacon = getattr(scene, "beacon", None)
            cur = int(getattr(beacon, "size_px", 10))
            scene.set_target_params(size_px=min(20, cur+1))
            return True
        if key == "primary_prev":
            app._designate_target((getattr(app,"primary_target_idx",0)-1) % max(1, getattr(scene,"num_targets",1)))
            return True
        if key == "primary_next":
            app._designate_target((getattr(app,"primary_target_idx",0)+1) % max(1, getattr(scene,"num_targets",1)))
            return True
        if key.startswith("primary:"):
            app._designate_target(int(key.split(":")[1]))
            return True
    return False


def handle_report_click(app, pos):
    rects = getattr(app, "report_ui_rects", {})
    recorder = getattr(app, "run_recorder", None)
    if recorder is None:
        return False
    if rects.get("start") and rects["start"].collidepoint(pos):
        return app._start_run()
    if rects.get("stop") and rects["stop"].collidepoint(pos):
        return app._stop_run_and_report()
    if rects.get("open") and rects["open"].collidepoint(pos):
        recorder.open_latest(prefer_pdf=True)
        return True
    return False
