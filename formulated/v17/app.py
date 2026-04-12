import tkinter as tk
from tkinter import ttk
import cv2 as cv
import numpy as np
import threading
import math
import os
import time
from PIL import Image, ImageTk
from datetime import datetime

import log
from constants import *
from utils import rotation_to_euler, load_calib
from measurement_engine import MeasurementEngine

class MeasurementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dual-Pair ArUco Measurement v17")
        self.root.geometry("1600x960")
        self.root.configure(bg=C_BG)

        # ── Measurement vars ──────────────────────────────────────────────────
        self.size_top         = tk.DoubleVar(value=DEFAULT_MARKER_SIZE)
        self.size_bot         = tk.DoubleVar(value=DEFAULT_MARKER_SIZE)
        self.fixed_side       = tk.StringVar(value="Left")
        self.rot_threshold    = tk.DoubleVar(value=12.0)
        self.pitch_threshold  = tk.DoubleVar(value=10.0)
        self.yaw_threshold    = tk.DoubleVar(value=10.0)
        self.use_angle_thresh = tk.BooleanVar(value=False)

        # ── Camera vars ───────────────────────────────────────────────────────
        self.cam_ae         = tk.BooleanVar(value=True)
        self.cam_exposure   = tk.IntVar(value=10000)
        self.cam_gain       = tk.DoubleVar(value=2.0)
        self.cam_awb        = tk.BooleanVar(value=True)
        self.cam_awb_mode   = tk.StringVar(value="Auto")
        self.cam_brightness = tk.DoubleVar(value=0.0)
        self.cam_contrast   = tk.DoubleVar(value=1.0)
        self.cam_saturation = tk.DoubleVar(value=1.0)
        self.cam_sharpness  = tk.DoubleVar(value=1.0)
        self.pc             = None
        self._cam_preview_frame = None

        # ── Movement monitor state ────────────────────────────────────────────
        self.mv_state       = "idle"
        self.mv_init_buf    = {"top": [], "bottom": []}
        self.mv_final_buf   = {"top": [], "bottom": []}
        self.mv_dist_init   = {"top": None, "bottom": None}
        self.mv_dist_final  = {"top": None, "bottom": None}
        self._last_sc       = 0

        # ── Shared data ───────────────────────────────────────────────────────
        def _empty():
            return {
                "A": (0,0,0), "X": (0,0,0), "TR": (0,0,0), "BR": (0,0,0),
                "B": (0,0,0), "C": (0,0,0), "dist": 0.0, "k": 0.0,
                "L_A": (0.0, 0.0), "R_A": (0.0, 0.0), "rot_2d": 0.0,
                "L_z": 0.0, "R_z": 0.0, "p1_px": None, "p2_px": None,
                "L_pitch": 0.0, "L_yaw": 0.0, "R_pitch": 0.0, "R_yaw": 0.0,
                "L_det": False, "R_det": False,
            }

        self.last_data = {
            "top":    _empty(),
            "bottom": _empty(),
            "session_count": 0,
            "lighting": {"status": "no_frame", "mean": 0.0, "std": 0.0},
        }

        self.is_running    = True
        self.current_frame = None
        self.engine        = MeasurementEngine()

        self._style()
        self.setup_ui()
        threading.Thread(target=self.measurement_loop, daemon=True).start()
        self.update_gui_loop()

    def _style(self):
        s = ttk.Style()
        s.theme_use('clam')
        s.configure("TNotebook", background=C_BG, borderwidth=0)
        s.configure("TNotebook.Tab", background=C_PANEL, foreground=C_TEXT_MED, padding=[25, 12], font=F_HEAD, borderwidth=0)
        s.map("TNotebook.Tab", background=[("selected", C_ACCENT)], foreground=[("selected", C_BG)])
        s.configure("TFrame", background=C_BG)
        s.configure("TProgressbar", thickness=24, background=C_ACCENT, troughcolor=C_PANEL, borderwidth=0)

    def setup_ui(self):
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill='both', expand=True, padx=8, pady=8)
        self._build_tab_live()
        self._build_tab_tele()
        self._build_tab_settings()
        self._build_tab_cam()

    def _card(self, parent, title, title_color, **pack_kw):
        f = tk.LabelFrame(parent, text=f"  {title.upper()}  ", font=F_HEAD, fg=title_color, bg=C_PANEL, bd=2, relief="flat", highlightbackground=title_color, highlightthickness=1, padx=20, pady=12)
        f.pack(**pack_kw)
        return f

    def _build_tab_live(self):
        tab = ttk.Frame(self.tabs); self.tabs.add(tab, text=" 📏  Movement "); tab.configure(style="TFrame")
        left = tk.Frame(tab, bg=C_BG); left.pack(side="left", padx=16, pady=16)
        self.canvas = tk.Canvas(left, width=960, height=540, bg="black", highlightthickness=1, highlightbackground=C_PANEL); self.canvas.pack()
        
        warn_f = tk.Frame(left, bg=C_PANEL, bd=1, relief="solid"); warn_f.pack(fill="x", pady=(12, 0))
        self.warn_strip = tk.Text(warn_f, height=6, font=F_HEAD, fg=C_ACCENT, bg=C_PANEL, padx=15, pady=12, bd=0, highlightthickness=0, wrap="word", cursor="arrow", state="disabled"); self.warn_strip.pack(side="left", fill="both", expand=True)
        warn_scroll = tk.Scrollbar(warn_f, orient="vertical", command=self.warn_strip.yview, width=12, bg=C_PANEL, troughcolor=C_BG, bd=0); warn_scroll.pack(side="right", fill="y"); self.warn_strip.config(yscrollcommand=warn_scroll.set)

        right = tk.Frame(tab, bg=C_BG); right.pack(side="right", fill="both", expand=True, padx=12, pady=10)
        for key, title, col in [("top", "UPPER SENSOR", C_TOP), ("bottom", "LOWER SENSOR", C_BOT)]:
            card = self._card(right, title, col, fill="x", pady=6, padx=12)
            dl = tk.Label(card, text="0.000 mm", font=F_DATA, fg=C_GREEN, bg=C_PANEL); dl.pack(pady=(8, 2))
            kl = tk.Label(card, text="INTERSECT RATIO: 0.0000", font=F_SMALL, fg=C_TEXT_MED, bg=C_PANEL); kl.pack(pady=(0, 8))
            if key == "top": self.lbl_dist_top, self.lbl_k_top = dl, kl
            else: self.lbl_dist_bot, self.lbl_k_bot = dl, kl

        stf = tk.Frame(right, bg=C_BG); stf.pack(fill="x", pady=(15, 0))
        self.mv_status_lbl = tk.Label(stf, text="READY FOR INITIAL CAPTURE", font=F_HEAD, fg=C_TEXT_MED, bg=C_BG, wraplength=380, justify="center"); self.mv_status_lbl.pack(pady=(2, 2))
        prog_f = tk.Frame(right, bg=C_BG); prog_f.pack(fill="x", padx=12); self.mv_prog_lbl = tk.Label(prog_f, text="", font=F_HEAD, fg=C_GREEN, bg=C_BG); self.mv_prog_lbl.pack()
        self.mv_prog_bar = ttk.Progressbar(prog_f, length=380, maximum=COLLECT_N, mode="determinate"); self.mv_prog_bar.pack(pady=4)

        btn_f = tk.Frame(right, bg=C_BG); btn_f.pack(pady=10, fill="x", padx=12)
        btn_cfg = dict(font=F_BTN, relief="flat", padx=20, pady=12, bd=0, cursor="hand2")
        def add_hover(btn, normal_bg, hover_bg):
            btn.bind("<Enter>", lambda e: btn.config(bg=hover_bg)); btn.bind("<Leave>", lambda e: btn.config(bg=normal_bg))

        # Reset button set to square proportions (width=5 @ pady=12)
        self.btn_reset = tk.Button(btn_f, text="↺", bg=C_CARD, fg="white", activebackground=C_TEXT_MED, font=F_BTN, relief="flat", width=5, pady=12, bd=0, cursor="hand2", command=self._mv_reset); self.btn_reset.pack(side="right", padx=6); add_hover(self.btn_reset, C_CARD, C_TEXT_MED)
        self.btn_start = tk.Button(btn_f, text="▶ START SESSION", bg=C_GREEN, fg=C_BG, activebackground="#86efac", command=self._mv_start, **btn_cfg); self.btn_start.pack(side="left", expand=True, fill="x", padx=6); add_hover(self.btn_start, C_GREEN, "#86efac")
        self.btn_stop = tk.Button(btn_f, text="■ STOP (SAVE)", bg=C_RED, fg=C_BG, activebackground="#fca5a5", state="disabled", command=self._mv_stop, **btn_cfg); self.btn_stop.pack(side="left", expand=True, fill="x", padx=6); add_hover(self.btn_stop, C_RED, "#fca5a5")

        tk.Frame(right, bg=C_CARD, height=2).pack(fill="x", padx=12, pady=15)
        for key, title, col in [("top", "UPPER", C_TOP), ("bottom", "LOWER", C_BOT)]:
            row = tk.Frame(right, bg=C_PANEL, bd=1, relief="solid"); row.pack(fill="x", padx=12, pady=8)
            tk.Label(row, text=title, font=F_HEAD, fg=col, bg=C_PANEL, width=8).pack(side="left", padx=12, pady=8)
            info = tk.Frame(row, bg=C_PANEL); info.pack(side="left", expand=True)
            init_lbl  = tk.Label(info, text="INIT: —", font=F_BODY, fg=C_TEXT_MED, bg=C_PANEL, anchor="w"); init_lbl.pack(fill="x", padx=8)
            final_lbl = tk.Label(info, text="FINAL: —", font=F_BODY, fg=C_TEXT_MED, bg=C_PANEL, anchor="w"); final_lbl.pack(fill="x", padx=8)
            delta_lbl = tk.Label(row, text="—", font=F_DATA, fg=C_ACCENT, bg=C_PANEL); delta_lbl.pack(side="right", padx=20, pady=12)
            if key == "top": self.mv_init_lbl_top, self.mv_final_lbl_top, self.mv_delta_lbl_top = init_lbl, final_lbl, delta_lbl
            else: self.mv_init_lbl_bot, self.mv_final_lbl_bot, self.mv_delta_lbl_bot = init_lbl, final_lbl, delta_lbl
        tk.Frame(right, bg=C_BG).pack(fill="both", expand=True)

    def _build_tab_tele(self):
        tab = ttk.Frame(self.tabs); self.tabs.add(tab, text=" 🛸  Dual Telemetry "); mtf = tk.Frame(tab, bg=C_BG); mtf.pack(fill="both", expand=True)
        self.tele_vars = {"top": {}, "bottom": {}}
        v_show = ["A", "X", "TR", "BR", "B", "C", "L_ROL", "L_ROT", "L_Z", "L_PITCH", "L_YAW", "R_ROL", "R_ROT", "R_Z", "R_PITCH", "R_YAW"]
        for key, title, col in [("top", "TOP", C_TOP), ("bottom", "BOTTOM", C_BOT)]:
            cf = tk.LabelFrame(mtf, text=f" {title} PAIR TELEMETRY ", font=F_TITLE, fg=col, bg=C_BG, padx=15, pady=15); cf.pack(side="left", fill="both", expand=True, padx=20, pady=20)
            for v_name in v_show:
                row = tk.Frame(cf, bg=C_PANEL, highlightbackground=C_CARD, highlightthickness=1); row.pack(fill="x", padx=10, pady=5)
                tk.Label(row, text=v_name, font=F_HEAD, fg=C_TEXT_MED, bg=C_PANEL).pack(side="left", padx=10, pady=5)
                sv = tk.StringVar(value="—"); tk.Label(row, textvariable=sv, font=F_MONO, fg=C_TEXT_BRT, bg=C_PANEL).pack(side="right", padx=10, pady=5)
                self.tele_vars[key][v_name] = sv

    def _build_tab_settings(self):
        tab = ttk.Frame(self.tabs); self.tabs.add(tab, text=" ⚙  Machine Configuration "); sc = tk.Frame(tab, bg=C_BG); sc.pack(fill="both", expand=True, padx=80, pady=40)
        rf = tk.LabelFrame(sc, text=" Reference Side (Fixed ArUco) ", bg=C_PANEL, fg=C_TEXT_BRT, font=F_HEAD, padx=20, pady=10); rf.pack(fill="x", pady=(0, 12))
        for choice in ["Left", "Right"]: tk.Radiobutton(rf, text=choice, variable=self.fixed_side, value=choice, bg=C_PANEL, fg=C_TEXT_BRT, selectcolor=C_BG, activebackground=C_PANEL, font=F_BODY).pack(side="left", padx=24)

        def create_slider(label, var, color, lo, hi, res=0.1):
            f = tk.LabelFrame(sc, text=f" {label} ", bg=C_PANEL, font=F_HEAD, fg=color, padx=25, pady=12); f.pack(fill="x", pady=10)
            sl = tk.Scale(f, from_=lo, to=hi, resolution=res, orient="horizontal", variable=var, bg=C_PANEL, fg=C_TEXT_BRT, troughcolor=C_BG, highlightthickness=0, length=600, font=F_BODY); sl.pack(side="left", padx=20)
            ent = tk.Entry(f, width=10, bg=C_BG, fg=C_ACCENT, insertbackground=C_TEXT_BRT, bd=0, font=F_MONO); ent.insert(0, f"{var.get():.1f}"); ent.pack(side="left")
            def sync_entry_from_var(*_):
                if self.root.focus_get() != ent: ent.delete(0, tk.END); ent.insert(0, f"{var.get():.1f}"); ent.config(fg=C_ACCENT)
            var.trace_add("write", sync_entry_from_var)
            def commit_change(*_):
                try: val = float(ent.get()); val = max(lo, min(hi, val)); var.set(val); ent.config(fg=C_ACCENT)
                except ValueError: sync_entry_from_var()
                self.root.focus()
            ent.bind("<FocusIn>", lambda e: (ent.config(fg=C_TEXT_BRT), ent.select_range(0, tk.END)))
            ent.bind("<Return>", commit_change); ent.bind("<FocusOut>", commit_change)
            return sl

        create_slider("Upper Pair Marker Size (mm)",  self.size_top, C_TOP,  30, 100)
        create_slider("Bottom Pair Marker Size (mm)", self.size_bot, C_BOT,  30, 100)
        self.rot_threshold_slider = create_slider("Rotation Threshold °  (in-plane / formula switch)", self.rot_threshold, C_TEXT_BRT, 0, 45)
        create_slider("Pitch Threshold °  (forward / backward tilt warning)", self.pitch_threshold, "#8e44ad", 1, 45, res=0.5)
        create_slider("Yaw Threshold °  (left / right tilt  —  'one side in' warning)", self.yaw_threshold, "#16a085", 1, 45, res=0.5)

        tog_f = tk.LabelFrame(sc, text=" Measurement Logic Options ", bg=C_PANEL, font=F_HEAD, fg=C_TEXT_BRT, padx=25, pady=15); tog_f.pack(fill="x", pady=10)
        def _update_tog_label(*_):
            if self.use_angle_thresh.get():
                tog_state_lbl.config(text="ON  — Use Perpendicular Formula (Accurate) below threshold", fg=C_GREEN); self.rot_threshold_slider.config(state="normal")
            else:
                tog_state_lbl.config(text="OFF — Always use Perpendicular Formula (v8 High-Precision)", fg=C_RED); self.rot_threshold_slider.config(state="disabled")
        tog_row = tk.Frame(tog_f, bg=C_PANEL); tog_row.pack(fill="x")
        self._tog_canvas = tk.Canvas(tog_row, width=80, height=40, bg=C_PANEL, highlightthickness=0, cursor="hand2"); self._tog_canvas.pack(side="left", padx=(0, 20))
        tog_state_lbl = tk.Label(tog_row, text="", font=F_HEAD, bg=C_PANEL, anchor="w"); tog_state_lbl.pack(side="left", fill="x", expand=True)
        def _draw_toggle():
            self._tog_canvas.delete("all"); on = self.use_angle_thresh.get(); bg = C_GREEN if on else "#b2bec3"
            self._tog_canvas.create_oval(4, 4, 76, 36, fill=bg, outline=""); cx = 58 if on else 22; self._tog_canvas.create_oval(cx-14, 6, cx+14, 34, fill="white", outline="")
        def _toggle_click(_=None): self.use_angle_thresh.set(not self.use_angle_thresh.get()); _draw_toggle(); _update_tog_label()
        self._tog_canvas.bind("<Button-1>", _toggle_click); _draw_toggle(); _update_tog_label()

    def _build_tab_cam(self):
        tab = ttk.Frame(self.tabs); self.tabs.add(tab, text=" 📷  Camera Controls ")
        cl = tk.Frame(tab, bg=C_BG); cl.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        cr = tk.Frame(tab, bg=C_BG); cr.pack(side="right", fill="y", padx=8, pady=8, ipadx=4)
        tk.Label(cl, text="Live Camera Preview", font=F_TITLE, fg=C_TEXT_BRT, bg=C_BG).pack(pady=(15, 0))
        self.cam_canvas = tk.Canvas(cl, width=800, height=480, bg="black", highlightthickness=2, highlightbackground=C_PANEL); self.cam_canvas.pack(padx=20, pady=20)
        self.lbl_cam_status = tk.Label(cl, text="⏳ Waiting for camera…", font=F_TITLE, fg=C_AMBER, bg=C_BG); self.lbl_cam_status.pack(pady=8)
        self.lbl_lighting_cam = tk.Label(cl, text="", font=F_HEAD, fg=C_TEXT_MED, bg=C_BG); self.lbl_lighting_cam.pack(pady=5)
        tk.Label(cr, text="Sensor Configuration", font=F_TITLE, fg=C_TEXT_BRT, bg=C_BG).pack(pady=(20, 10))

        def cam_row(label, var, lo, hi, res, unit=""):
            rf = tk.LabelFrame(cr, text=f" {label} ", bg=C_PANEL, font=F_HEAD, fg=C_TEXT_BRT, padx=12, pady=8); rf.pack(fill="x", padx=10, pady=5)
            inner = tk.Frame(rf, bg=C_PANEL); inner.pack(fill="x")
            tk.Scale(inner, from_=lo, to=hi, resolution=res, orient="horizontal", variable=var, bg=C_PANEL, fg=C_TEXT_BRT, troughcolor=C_BG, highlightthickness=0, length=300, font=F_SMALL, command=lambda _: self._apply_cam()).pack(side="left")
            tk.Entry(inner, textvariable=var, width=8, bg=C_BG, fg=C_ACCENT, bd=0, font=F_MONO).pack(side="left", padx=8)
            if unit: tk.Label(inner, text=unit, bg=C_PANEL, font=F_SMALL, fg=C_TEXT_MED).pack(side="left")

        ae_f = tk.LabelFrame(cr, text=" Exposure Control ", bg=C_PANEL, font=F_HEAD, fg=C_TEXT_BRT, padx=12, pady=8); ae_f.pack(fill="x", padx=10, pady=5)
        tk.Checkbutton(ae_f, text="Enable Software Auto-Exposure", variable=self.cam_ae, bg=C_PANEL, fg=C_TEXT_BRT, selectcolor=C_BG, activebackground=C_PANEL, font=F_BODY, command=self._apply_cam).pack(anchor="w")
        cam_row("Exposure Time", self.cam_exposure, 100,  66000, 100, "us")

        awb_f = tk.LabelFrame(cr, text=" Color Profile / AWB ", bg=C_PANEL, font=F_HEAD, fg=C_TEXT_BRT, padx=12, pady=8); awb_f.pack(fill="x", padx=10, pady=5)
        tk.Checkbutton(awb_f, text="Hardware Auto White Balance", variable=self.cam_awb, bg=C_PANEL, fg=C_TEXT_BRT, selectcolor=C_BG, activebackground=C_PANEL, font=F_BODY, command=self._apply_cam).pack(anchor="w")
        wbm_row = tk.Frame(awb_f, bg=C_PANEL); wbm_row.pack(fill="x", pady=5)
        tk.Label(wbm_row, text="Mode:", bg=C_PANEL, fg=C_TEXT_MED, font=F_BODY).pack(side="left")
        for mode_name in AWB_MODES: tk.Radiobutton(wbm_row, text=mode_name, variable=self.cam_awb_mode, value=mode_name, bg=C_PANEL, fg=C_TEXT_BRT, selectcolor=C_BG, activebackground=C_PANEL, font=F_SMALL, command=self._apply_cam).pack(side="left", padx=5)

        for l, v, lo, hi, r in [("Brightness",self.cam_brightness,-1,1,0.05), ("Contrast",self.cam_contrast,0,8,0.1), ("Saturation",self.cam_saturation,0,8,0.1), ("Sharpness",self.cam_sharpness,0,8,0.1)]: cam_row(l, v, lo, hi, r)
        tk.Button(cr, text="↺  Restore Factory Defaults", font=F_BTN, bg=C_RED, fg="white", activebackground="#c0392b", relief="flat", padx=15, pady=12, command=self._reset_cam).pack(pady=20, fill="x", padx=10)

    def _mv_start(self):
        if self.mv_state not in ("idle", "done"): return
        self.mv_state = "collecting_init"; self.mv_init_buf = {"top":[], "bottom":[]}; self.btn_start.config(state="disabled"); self.btn_stop.config(state="disabled"); self.mv_prog_bar["value"] = 0; self.mv_status_lbl.config(text=f"Collecting initial distance... (0 / {COLLECT_N})", fg=C_AMBER)

    def _mv_stop(self):
        if self.mv_state != "ready": return
        self.mv_state = "collecting_final"; self.mv_final_buf = {"top":[], "bottom":[]}; self.btn_stop.config(state="disabled"); self.mv_prog_bar["value"] = 0; self.mv_status_lbl.config(text=f"⏳  Collecting final distance…  (0 / {COLLECT_N})", fg="#f39c12")

    def _mv_reset(self):
        self.mv_state = "idle"; self.mv_init_buf = {"top":[], "bottom":[]}; self.mv_final_buf = {"top":[], "bottom":[]}; self.mv_dist_init = {"top":None, "bottom":None}; self.mv_dist_final = {"top":None, "bottom":None}
        self.btn_start.config(state="normal"); self.btn_stop.config(state="disabled"); self.mv_prog_bar["value"] = 0; self.mv_prog_lbl.config(text=""); self.mv_status_lbl.config(text="Press  START  to capture initial gap", fg="#dfe6e9")
        for l in [self.mv_init_lbl_top, self.mv_init_lbl_bot, self.mv_final_lbl_top, self.mv_final_lbl_bot]: l.config(text="Init: —" if "init" in str(l) else "Final: —", fg=C_TEXT_MED)
        self.mv_delta_lbl_top.config(text="—", fg=C_ACCENT); self.mv_delta_lbl_bot.config(text="—", fg=C_ACCENT)

    def _mv_tick(self):
        sc = self.last_data["session_count"]
        if sc == self._last_sc: return
        self._last_sc = sc
        if self.mv_state == "collecting_init":
            for key in ["top", "bottom"]:
                d = self.last_data[key]["dist"]
                if d > 0: self.mv_init_buf[key].append(d)
            n = min(len(self.mv_init_buf["top"]), len(self.mv_init_buf["bottom"]))
            self.mv_prog_bar["value"] = n; self.mv_prog_lbl.config(text=f"Reading {n} / {COLLECT_N}"); self.mv_status_lbl.config(text=f"Collecting initial distance... ({n} / {COLLECT_N})", fg=C_AMBER)
            if n >= COLLECT_N:
                self.mv_dist_init = {k: float(np.mean(self.mv_init_buf[k][:COLLECT_N])) for k in ["top", "bottom"]}
                self.mv_state = "ready"; self.btn_stop.config(state="normal"); self.mv_prog_bar["value"] = 0; self.mv_prog_lbl.config(text=""); self.mv_status_lbl.config(text="✅  Initial captured — move the panel, then press  STOP", fg=C_GREEN)
                self.mv_init_lbl_top.config(text=f"Init: {self.mv_dist_init['top']:.3f} mm"); self.mv_init_lbl_bot.config(text=f"Init: {self.mv_dist_init['bottom']:.3f} mm")
        elif self.mv_state == "collecting_final":
            for key in ["top", "bottom"]:
                d = self.last_data[key]["dist"]
                if d > 0: self.mv_final_buf[key].append(d)
            n = min(len(self.mv_final_buf["top"]), len(self.mv_final_buf["bottom"]))
            self.mv_prog_bar["value"] = n; self.mv_prog_lbl.config(text=f"Reading {n} / {COLLECT_N}"); self.mv_status_lbl.config(text=f"Collecting final distance... ({n} / {COLLECT_N})", fg=C_AMBER)
            if n >= COLLECT_N:
                self.mv_dist_final = {k: float(np.mean(self.mv_final_buf[k][:COLLECT_N])) for k in ["top", "bottom"]}
                self.mv_state = "done"; self.btn_start.config(state="normal"); self.mv_prog_bar["value"] = 0; self.mv_prog_lbl.config(text=""); self.mv_status_lbl.config(text="✅  Measurement complete — press RESET to start over", fg=C_GREEN)
                for k, il, fl, dl in [("top", self.mv_init_lbl_top, self.mv_final_lbl_top, self.mv_delta_lbl_top), ("bottom", self.mv_init_lbl_bot, self.mv_final_lbl_bot, self.mv_delta_lbl_bot)]:
                    di, df = self.mv_dist_init[k], self.mv_dist_final[k]; delta = df - di; il.config(text=f"Init: {di:.3f} mm"); fl.config(text=f"Final: {df:.3f} mm")
                    sign = "+" if delta >= 0 else ""; color = C_RED if delta > 0.5 else (C_GREEN if delta < -0.5 else C_ACCENT); dl.config(text=f"{sign}{delta:.3f} mm", fg=color)

    def _apply_cam(self, *_):
        if not self.pc: return
        c = {"AeEnable":self.cam_ae.get(), "AwbEnable":self.cam_awb.get(), "Brightness":self.cam_brightness.get(), "Contrast":self.cam_contrast.get(), "Saturation":self.cam_saturation.get(), "Sharpness":self.cam_sharpness.get()}
        if not c["AeEnable"]: c["ExposureTime"] = int(self.cam_exposure.get()); c["AnalogueGain"] = float(self.cam_gain.get())
        if not c["AwbEnable"]: c["AwbMode"] = AWB_MODES.get(self.cam_awb_mode.get(), 0)
        try: self.pc.set_controls(c)
        except: pass

    def _reset_cam(self):
        self.cam_ae.set(True); self.cam_exposure.set(10000); self.cam_gain.set(2.0); self.cam_awb.set(True); self.cam_awb_mode.set("Auto"); self.cam_brightness.set(0.0); self.cam_contrast.set(1.0); self.cam_saturation.set(1.0); self.cam_sharpness.set(1.0); self._apply_cam()

    def measurement_loop(self):
        try:
            from picamera2 import Picamera2
            pc = Picamera2(); pc.configure(pc.create_video_configuration(main={"size": RESOLUTION, "format": "RGB888"})); pc.start(); self.pc = pc
        except: return
        log.init_log(); bufs = {"top": [], "bottom": []}; l_s = l_u = time.time() * 1000
        while self.is_running:
            K, dc = load_calib(self.fixed_side.get())
            try: frame = cv.cvtColor(pc.capture_array(), cv.COLOR_RGB2BGR)
            except: continue
            lh, gray = self.engine.analyze_lighting(frame); self.last_data["lighting"] = lh
            top_m, bot_m = self.engine.detect_markers(gray); curr = time.time() * 1000
            
            def proc(marker_list, pk, size):
                if len(marker_list) < 2: self.last_data[pk]["dist"]=0; self.last_data[pk]["L_det"]=False; self.last_data[pk]["R_det"]=False; return
                res = self.engine.get_pose(marker_list[0]["c_raw"] if self.fixed_side.get()=="Left" else marker_list[1]["c_raw"], K, dc, size) # simplified call for demo
                # Accurate porting of v16 loop logic
                marker_list.sort(key=lambda m: m["x"]); left_m, right_m = marker_list[0], marker_list[-1]; is_rf = (self.fixed_side.get() == "Right")
                S_m, T_m = (right_m, left_m) if is_rf else (left_m, right_m)
                S_pts, S_roll, S_z, S_p, S_y = self.engine.get_pose(S_m["c_raw"], K, dc, size)
                T_pts, T_roll, T_z, T_p, T_y = self.engine.get_pose(T_m["c_raw"], K, dc, size)
                Sit, Sib, Sot = self.engine.inner_outer(S_pts, not is_rf); Tit, Tib, _ = self.engine.inner_outer(T_pts, is_rf)
                p1 = tuple(((S_m["c"][1 if not is_rf else 0]+S_m["c"][2 if not is_rf else 3])/2).astype(int))
                p2 = tuple(((T_m["c"][0 if not is_rf else 1]+T_m["c"][3 if not is_rf else 2])/2).astype(int))
                bufs[pk].append({"A":(Sit+Sib)/2.0, "TR":Sit, "BR":Sib, "TL_ref":Sot, "B":Tit, "C":Tib, "X_alt":(Tit+Tib)/2.0, "L_A":(S_roll, self.engine.inplane_rot(S_m["c"])), "R_A":(T_roll, self.engine.inplane_rot(T_m["c"])), "rot":max(self.engine.inplane_rot(S_m["c"]), self.engine.inplane_rot(T_m["c"])), "L_z":S_z, "R_z":T_z, "L_p":S_p, "L_y":S_y, "R_p":T_p, "R_y":T_y, "p1":p1, "p2":p2})
                self.last_data[pk]["L_det"]=True; self.last_data[pk]["R_det"]=True

            if (curr - l_s) >= 100:
                if top_m: proc(top_m, "top", self.size_top.get())
                if bot_m: proc(bot_m, "bottom", self.size_bot.get())
                l_s = curr

            if (curr - l_u) >= 1000:
                for k in ["top", "bottom"]:
                    if bufs[k]:
                        s = bufs[k]; aA = np.mean([x["A"] for x in s], axis=0); aTR = np.mean([x["TR"] for x in s], axis=0); aTL = np.mean([x["TL_ref"] for x in s], axis=0); aBR = np.mean([x["BR"] for x in s], axis=0); aB = np.mean([x["B"] for x in s], axis=0); aC = np.mean([x["C"] for x in s], axis=0); aX_alt = np.mean([x["X_alt"] for x in s], axis=0)
                        v_raw = aTR - aTL; v = v_raw / np.linalg.norm(v_raw) if np.linalg.norm(v_raw)>0 else np.zeros(3); w = aC - aB; u = aB - aA; den = np.dot(v,v)*np.dot(w,w) - np.dot(v,w)**2
                        def _perp():
                            if abs(den)>1e-6: kv_ = np.clip(((np.dot(v,w)*np.dot(u,v)) - (np.dot(v,v)*np.dot(u,w)))/den, 0, 1); ax_ = aB + kv_*w; return ax_, np.linalg.norm(ax_-aA), kv_
                            return aX_alt, np.linalg.norm(aX_alt-aA), 0.5
                        if self.use_angle_thresh.get() and np.mean([x["rot"] for x in s]) > self.rot_threshold.get(): aX, dv, kv = aX_alt, np.linalg.norm(aX_alt-aA), 0.5
                        else: aX, dv, kv = _perp()
                        last = s[-1]; self.last_data[k].update({"A":aA, "X":aX, "TR":aTR, "BR":aBR, "B":aB, "C":aC, "dist":dv, "k":kv, "rot_2d":np.mean([x["rot"] for x in s]), "L_A":np.mean([x["L_A"] for x in s], axis=0), "R_A":np.mean([x["R_A"] for x in s], axis=0), "L_z":np.mean([x["L_z"] for x in s]), "R_z":np.mean([x["R_z"] for x in s]), "L_pitch":np.mean([x["L_p"] for x in s]), "L_yaw":np.mean([x["L_y"] for x in s]), "R_pitch":np.mean([x["R_p"] for x in s]), "R_yaw":np.mean([x["R_y"] for x in s]), "p1_px":last["p1"], "p2_px":last["p2"]})
                        self.last_data["session_count"]+=1; bufs[k].clear()
                        try: log.record(dv, kv, aA, aX, aTR, aBR, aB, aC, aB-aA, aTR-aTL, aC-aB, self.last_data[k]["L_A"], self.last_data[k]["R_A"])
                        except: pass
                        if self.mv_state in ("collecting_init", "ready", "collecting_final"):
                            try: ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]; odir = "captured_images"; os.makedirs(odir, exist_ok=True); cv.imwrite(os.path.join(odir, f"mv_{ts}.jpg"), frame)
                            except: pass
                l_u = curr
            
            for k, color in [("top",(0,165,255)), ("bottom",(255,0,255))]:
                d = self.last_data[k]
                if d["dist"]>0 and d["p1_px"]:
                    X_3d = np.array([d["X"]], dtype=np.float64).reshape(1,1,3)
                    if d["X"][2]>0: p_proj, _ = cv.projectPoints(X_3d, np.zeros(3), np.zeros(3), K, dc); p2 = tuple(p_proj[0].ravel().astype(int))
                    else: p2 = d["p2_px"]
                    cv.line(frame, d["p1_px"], p2, color, 3); cv.circle(frame, p2, 6, (0,255,0), -1); cv.putText(frame, f"{k.upper()}: {d['dist']:.2f}mm", (d["p1_px"][0], d["p1_px"][1]-12), 0, 0.6, color, 2)
            self.current_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB); self._cam_preview_frame = self.current_frame

    def _update_warnings(self):
        p_th, y_th, r_th = self.pitch_threshold.get(), self.yaw_threshold.get(), self.rot_threshold.get(); strip = []
        for k in ["top", "bottom"]:
            d = self.last_data[k]
            if not d["L_det"]: strip.append(f"[{k.upper()}] L-marker not detected")
            if not d["R_det"]: strip.append(f"[{k.upper()}] R-marker not detected")
            if d["rot_2d"] > r_th: strip.append(f"[{k.upper()}] Rotation {d['rot_2d']:.1f}°")
            for m, lbl in [("L", "L-pitch"), ("R", "R-pitch")]:
                v = d.get(f"{m}_pitch", 0)
                if abs(v)>p_th: strip.append(f"[{k.upper()}] {lbl} {v:+.1f}° ({'FWD' if v>0 else 'BWD'})")
            for m, lbl in [("L", "L-yaw"), ("R", "R-yaw")]:
                v = d.get(f"{m}_yaw", 0)
                if abs(v)>y_th: strip.append(f"[{k.upper()}] {lbl} {v:+.1f}° ({'RIGHT' if v>0 else 'LEFT'})")
        li = self.last_data["lighting"]; ls = li["status"]; self.lbl_lighting_cam.config(text=f"LIGHTING PROFILE: {ls.upper()}", fg=C_GREEN if ls=="ok" else C_AMBER)
        if ls == "dark": strip.append(f"CRITICAL: Ambient light too low ({li['mean']:.0f})")
        self.warn_strip.config(state="normal"); self.warn_strip.delete("1.0", tk.END)
        if strip: self.warn_strip.config(fg=C_RED if any("CRITICAL" in x for x in strip) else C_AMBER); self.warn_strip.insert(tk.END, "![!] ACTIVE ALERTS:\n" + "\n".join([f" • {x}" for x in strip]))
        else: self.warn_strip.config(fg=C_GREEN); self.warn_strip.insert(tk.END, "[OK]  SYSTEM OPERATIONAL\n • All marker pairs detected correctly\n • Environmental lighting is nominal\n • All tilt/rotation values within tolerance")
        self.warn_strip.config(state="disabled")

    def update_gui_loop(self):
        if self.current_frame is not None: im = ImageTk.PhotoImage(Image.fromarray(self.current_frame).resize((960, 540))); self.canvas.create_image(0,0, anchor="nw", image=im); self.canvas.img = im
        self.lbl_dist_top.config(text=f"{self.last_data['top']['dist']:.3f} mm"); self.lbl_k_top.config(text=f"INTERSECT RATIO: {self.last_data['top']['k']:.4f}")
        self.lbl_dist_bot.config(text=f"{self.last_data['bottom']['dist']:.3f} mm"); self.lbl_k_bot.config(text=f"INTERSECT RATIO: {self.last_data['bottom']['k']:.4f}")
        for k in ["top", "bottom"]:
            d, tv = self.last_data[k], self.tele_vars[k]; tv["A"].set(f"{d['A'][2]:.2f}"); tv["X"].set(f"{d['X'][2]:.2f}")
            for var in ["TR", "BR", "B", "C"]: v = d[var]; tv[var].set(f"{v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f}")
            lr, rr = d["L_A"], d["R_A"]; tv["L_ROL"].set(f"{lr[0]:.2f}° (roll)"); tv["L_ROT"].set(f"{lr[1]:.2f}° (in-plane)"); tv["L_Z"].set(f"{d['L_z']:.1f}"); tv["L_PITCH"].set(f"{d['L_pitch']:+.2f}"); tv["L_YAW"].set(f"{d['L_yaw']:+.2f}")
            tv["R_ROL"].set(f"{rr[0]:.2f}° (roll)"); tv["R_ROT"].set(f"{rr[1]:.2f}° (in-plane)"); tv["R_Z"].set(f"{d['R_z']:.1f}"); tv["R_PITCH"].set(f"{d['R_pitch']:+.2f}"); tv["R_YAW"].set(f"{d['R_yaw']:+.2f}")
        if self._cam_preview_frame is not None:
            cim = ImageTk.PhotoImage(Image.fromarray(self._cam_preview_frame).resize((800, 480))); self.cam_canvas.create_image(0,0, anchor="nw", image=cim); self.cam_canvas.img=cim
            t_ok, b_ok = self.last_data["top"]["dist"]>0, self.last_data["bottom"]["dist"]>0
            if t_ok and b_ok: self.lbl_cam_status.config(text="STATUS: Both pairs detected", fg=C_GREEN)
            elif t_ok or b_ok: self.lbl_cam_status.config(text=f"STATUS: Only {'TOP' if t_ok else 'BOTTOM'} pair detected", fg=C_AMBER)
            else: self.lbl_cam_status.config(text="STATUS: No ArUco detected — adjust camera settings", fg=C_RED)
        self._mv_tick(); self._update_warnings(); self.root.after(50, self.update_gui_loop)
