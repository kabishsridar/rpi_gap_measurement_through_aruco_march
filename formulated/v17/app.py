import tkinter as tk
from tkinter import ttk
import cv2 as cv
import numpy as np
import threading
import math
import os
import time
from datetime import datetime
from PIL import Image, ImageTk

from constants import *
from utils import rotation_to_euler
from measurement_engine import MeasurementEngine
import log

class MeasurementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dual-Pair ArUco Measurement v17")
        self.root.geometry("1600x960")
        self.root.configure(bg=C_BG)

        self.engine = MeasurementEngine()

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

        # ── Shared data ──
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
            "top": _empty(), "bottom": _empty(), "session_count": 0,
            "lighting": {"status": "no_frame", "mean": 0.0, "std": 0.0},
        }

        self.is_running    = True
        self.current_frame = None

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
        ws = tk.Scrollbar(warn_f, orient="vertical", command=self.warn_strip.yview, width=12, bg=C_PANEL, troughcolor=C_BG, bd=0); ws.pack(side="right", fill="y"); self.warn_strip.config(yscrollcommand=ws.set)

        right = tk.Frame(tab, bg=C_BG); right.pack(side="right", fill="both", expand=True, padx=12, pady=10)
        for key, title, col in [("top", "UPPER SENSOR", C_TOP), ("bottom", "LOWER SENSOR", C_BOT)]:
            card = self._card(right, title, col, fill="x", pady=6, padx=12)
            dl = tk.Label(card, text="0.000 mm", font=F_DATA, fg=C_GREEN, bg=C_PANEL); dl.pack(pady=(8, 2))
            kl = tk.Label(card, text="INTERSECT RATIO: 0.0000", font=F_SMALL, fg=C_TEXT_MED, bg=C_PANEL); kl.pack(pady=(0, 8))
            if key == "top": self.lbl_dist_top, self.lbl_k_top = dl, kl
            else: self.lbl_dist_bot, self.lbl_k_bot = dl, kl

        sf = tk.Frame(right, bg=C_BG); sf.pack(fill="x", pady=(15, 0))
        self.mv_status_lbl = tk.Label(sf, text="READY FOR INITIAL CAPTURE", font=F_HEAD, fg=C_TEXT_MED, bg=C_BG, wraplength=380, justify="center"); self.mv_status_lbl.pack(pady=(2, 2))
        pf = tk.Frame(right, bg=C_BG); pf.pack(fill="x", padx=12); self.mv_prog_lbl = tk.Label(pf, text="", font=F_HEAD, fg=C_GREEN, bg=C_BG); self.mv_prog_lbl.pack()
        self.mv_prog_bar = ttk.Progressbar(pf, length=380, maximum=COLLECT_N, mode="determinate"); self.mv_prog_bar.pack(pady=4)

        bf = tk.Frame(right, bg=C_BG); bf.pack(pady=10, fill="x", padx=12)
        def add_h(b, n, h): b.bind("<Enter>", lambda e: b.config(bg=h)); b.bind("<Leave>", lambda e: b.config(bg=n))
        self.btn_reset = tk.Button(bf, text="↺", bg=C_CARD, fg="white", font=F_BTN, relief="flat", width=5, pady=12, bd=0, cursor="hand2", command=self._mv_reset); self.btn_reset.pack(side="right", padx=6); add_h(self.btn_reset, C_CARD, C_TEXT_MED)
        self.btn_start = tk.Button(bf, text="▶ START SESSION", bg=C_GREEN, fg=C_BG, font=F_BTN, relief="flat", padx=20, pady=12, bd=0, cursor="hand2", command=self._mv_start); self.btn_start.pack(side="left", expand=True, fill="x", padx=6); add_h(self.btn_start, C_GREEN, "#86efac")
        self.btn_stop = tk.Button(bf, text="■ STOP (SAVE)", bg=C_RED, fg=C_BG, font=F_BTN, relief="flat", state="disabled", padx=20, pady=12, bd=0, cursor="hand2", command=self._mv_stop); self.btn_stop.pack(side="left", expand=True, fill="x", padx=6); add_h(self.btn_stop, C_RED, "#fca5a5")

        tk.Frame(right, bg=C_CARD, height=2).pack(fill="x", padx=12, pady=15)
        for key, title, col in [("top", "UPPER", C_TOP), ("bottom", "LOWER", C_BOT)]:
            row = tk.Frame(right, bg=C_PANEL, bd=1, relief="solid"); row.pack(fill="x", padx=12, pady=4)
            dl = tk.Label(row, text="—", font=F_DATA, fg=C_ACCENT, bg=C_PANEL, anchor="e"); dl.pack(side="right", padx=15, pady=8)
            tk.Label(row, text=title, font=F_HEAD, fg=col, bg=C_PANEL, width=8).pack(side="left", padx=12, pady=8)
            info = tk.Frame(row, bg=C_PANEL); info.pack(side="left", fill="x", expand=True)
            il = tk.Label(info, text="INIT: —", font=F_BODY, fg=C_TEXT_MED, bg=C_PANEL, anchor="w"); il.pack(fill="x", padx=8)
            fl = tk.Label(info, text="FINAL: —", font=F_BODY, fg=C_TEXT_MED, bg=C_PANEL, anchor="w"); fl.pack(fill="x", padx=8)
            if key == "top": self.mv_init_lbl_top, self.mv_final_lbl_top, self.mv_delta_lbl_top = il, fl, dl
            else: self.mv_init_lbl_bot, self.mv_final_lbl_bot, self.mv_delta_lbl_bot = il, fl, dl
        tk.Frame(right, bg=C_BG).pack(fill="both", expand=True)

    def _build_tab_tele(self):
        tab = ttk.Frame(self.tabs); self.tabs.add(tab, text=" 🛸  Dual Telemetry ")
        mtf = tk.Frame(tab, bg=C_BG); mtf.pack(fill="both", expand=True)
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
        tab = ttk.Frame(self.tabs); self.tabs.add(tab, text=" ⚙  Machine Configuration ")
        sc = tk.Frame(tab, bg=C_BG); sc.pack(fill="both", expand=True, padx=80, pady=40)
        rf = tk.LabelFrame(sc, text=" Reference Side (Fixed ArUco) ", bg=C_PANEL, fg=C_TEXT_BRT, font=F_HEAD, padx=20, pady=10); rf.pack(fill="x", pady=(0, 12))
        for choice in ["Left", "Right"]:
            tk.Radiobutton(rf, text=choice, variable=self.fixed_side, value=choice, bg=C_PANEL, fg=C_TEXT_BRT, selectcolor=C_BG, activebackground=C_PANEL, font=F_BODY).pack(side="left", padx=24)

        def create_slider(label, var, color, lo, hi, res=0.1):
            f = tk.LabelFrame(sc, text=f" {label} ", bg=C_PANEL, font=F_HEAD, fg=color, padx=25, pady=12); f.pack(fill="x", pady=10)
            sl = tk.Scale(f, from_=lo, to=hi, resolution=res, orient="horizontal", variable=var, bg=C_PANEL, fg=C_TEXT_BRT, troughcolor=C_BG, highlightthickness=0, length=600, font=F_BODY); sl.pack(side="left", padx=20)
            ent = tk.Entry(f, width=10, bg=C_BG, fg=C_ACCENT, insertbackground=C_TEXT_BRT, bd=0, font=F_MONO); ent.insert(0, f"{var.get():.1f}"); ent.pack(side="left")
            def sync(*_):
                if self.root.focus_get() != ent: ent.delete(0, tk.END); ent.insert(0, f"{var.get():.1f}")
            var.trace_add("write", sync)
            def commit(*_):
                try: val = float(ent.get()); var.set(max(lo, min(hi, val))); ent.config(fg=C_ACCENT)
                except: sync()
                self.root.focus()
            ent.bind("<FocusIn>", lambda e: (ent.config(fg=C_TEXT_BRT), ent.select_range(0, tk.END))); ent.bind("<Return>", commit); ent.bind("<FocusOut>", commit)
            return sl

        create_slider("Upper Pair Marker Size (mm)", self.size_top, C_TOP, 30, 100)
        create_slider("Bottom Pair Marker Size (mm)", self.size_bot, C_BOT, 30, 100)
        self.rot_slider = create_slider("Rotation Threshold °", self.rot_threshold, C_TEXT_BRT, 0, 45)
        create_slider("Pitch Threshold °", self.pitch_threshold, "#8e44ad", 1, 45, res=0.5)
        create_slider("Yaw Threshold °", self.yaw_threshold, "#16a085", 1, 45, res=0.5)

        tog_f = tk.LabelFrame(sc, text=" Measurement Logic Options ", bg=C_PANEL, font=F_HEAD, fg=C_TEXT_BRT, padx=25, pady=15); tog_f.pack(fill="x", pady=10)
        def _update(*_):
            on = self.use_angle_thresh.get(); t_lbl.config(text="ON — Use Threshold" if on else "OFF — Always High-Precision", fg=C_GREEN if on else C_RED); self.rot_slider.config(state="normal" if on else "disabled")
            c.delete("all"); bg = C_GREEN if on else "#b2bec3"; c.create_oval(4, 4, 76, 36, fill=bg, outline=""); cx = 58 if on else 22; c.create_oval(cx-14, 6, cx+14, 34, fill="white", outline="")
        tr = tk.Frame(tog_f, bg=C_PANEL); tr.pack(fill="x"); c = tk.Canvas(tr, width=80, height=40, bg=C_PANEL, highlightthickness=0, cursor="hand2"); c.pack(side="left", padx=(0, 20))
        t_lbl = tk.Label(tr, text="", font=F_HEAD, bg=C_PANEL, anchor="w"); t_lbl.pack(side="left", fill="x", expand=True)
        c.bind("<Button-1>", lambda e: (self.use_angle_thresh.set(not self.use_angle_thresh.get()), _update())); _update()

    def _build_tab_cam(self):
        tab = ttk.Frame(self.tabs); self.tabs.add(tab, text=" 📷  Camera Controls ")
        cl = tk.Frame(tab, bg=C_BG); cl.pack(side="left", fill="both", expand=True, padx=8, pady=8); cr = tk.Frame(tab, bg=C_BG); cr.pack(side="right", fill="y", padx=8, pady=8, ipadx=4)
        tk.Label(cl, text="Live Camera Preview", font=F_TITLE, fg=C_TEXT_BRT, bg=C_BG).pack(pady=(15, 0))
        self.cam_canvas = tk.Canvas(cl, width=800, height=480, bg="black", highlightthickness=2, highlightbackground=C_PANEL); self.cam_canvas.pack(padx=20, pady=20)
        self.lbl_cam_status = tk.Label(cl, text="⏳ Waiting for camera…", font=F_TITLE, fg=C_AMBER, bg=C_BG); self.lbl_cam_status.pack(pady=8)
        self.lbl_lighting_cam = tk.Label(cl, text="", font=F_HEAD, fg=C_TEXT_MED, bg=C_BG); self.lbl_lighting_cam.pack(pady=5)
        tk.Label(cr, text="Sensor Configuration", font=F_TITLE, fg=C_TEXT_BRT, bg=C_BG).pack(pady=(20, 10))
        def crw(l, v, lo, hi, r, u=""):
            rf = tk.LabelFrame(cr, text=f" {l} ", bg=C_PANEL, font=F_HEAD, fg=C_TEXT_BRT, padx=12, pady=8); rf.pack(fill="x", padx=10, pady=5); i = tk.Frame(rf, bg=C_PANEL); i.pack(fill="x")
            tk.Scale(i, from_=lo, to=hi, resolution=r, orient="horizontal", variable=v, bg=C_PANEL, fg=C_TEXT_BRT, troughcolor=C_BG, highlightthickness=0, length=300, font=F_SMALL, command=lambda _: self._apply_cam()).pack(side="left")
            tk.Entry(i, textvariable=v, width=8, bg=C_BG, fg=C_ACCENT, bd=0, font=F_MONO).pack(side="left", padx=8)
            if u: tk.Label(i, text=u, bg=C_PANEL, font=F_SMALL, fg=C_TEXT_MED).pack(side="left")
        ae_f = tk.LabelFrame(cr, text=" Exposure Control ", bg=C_PANEL, font=F_HEAD, fg=C_TEXT_BRT, padx=12, pady=8); ae_f.pack(fill="x", padx=10, pady=5); tk.Checkbutton(ae_f, text="Enable Software Auto-Exposure", variable=self.cam_ae, bg=C_PANEL, fg=C_TEXT_BRT, selectcolor=C_BG, activebackground=C_PANEL, font=F_BODY, command=self._apply_cam).pack(anchor="w")
        crw("Exposure Time", self.cam_exposure, 100, 66000, 100, "us"); awb_f = tk.LabelFrame(cr, text=" White Balance ", bg=C_PANEL, font=F_HEAD, fg=C_TEXT_BRT, padx=12, pady=8); awb_f.pack(fill="x", padx=10, pady=5); tk.Checkbutton(awb_f, text="Hardware AWB", variable=self.cam_awb, bg=C_PANEL, fg=C_TEXT_BRT, selectcolor=C_BG, activebackground=C_PANEL, font=F_BODY, command=self._apply_cam).pack(anchor="w")
        wr = tk.Frame(awb_f, bg=C_PANEL); wr.pack(fill="x", pady=5); tk.Label(wr, text="Mode:", bg=C_PANEL, fg=C_TEXT_MED, font=F_BODY).pack(side="left")
        for m in AWB_MODES: tk.Radiobutton(wr, text=m, variable=self.cam_awb_mode, value=m, bg=C_PANEL, fg=C_TEXT_BRT, selectcolor=C_BG, activebackground=C_PANEL, font=F_SMALL, command=self._apply_cam).pack(side="left", padx=5)
        crw("Brightness", self.cam_brightness, -1.0, 1.0, 0.05); crw("Contrast", self.cam_contrast, 0.0, 8.0, 0.1); crw("Saturation", self.cam_saturation, 0.0, 8.0, 0.1); crw("Sharpness", self.cam_sharpness, 0.0, 8.0, 0.1)
        tk.Button(cr, text="↺ Factory Reset", font=F_BTN, bg=C_RED, fg="white", activebackground="#c0392b", relief="flat", padx=15, pady=12, command=self._reset_cam).pack(pady=20, fill="x", padx=10)

    def _mv_start(self):
        if self.mv_state in ("idle", "done"):
            self.mv_state = "collecting_init"; self.mv_init_buf = {"top": [], "bottom": []}; self.btn_start.config(state="disabled"); self.btn_stop.config(state="disabled"); self.mv_prog_bar["value"] = 0; self.mv_status_lbl.config(text=f"Collecting initial... (0 / {COLLECT_N})", fg=C_AMBER)

    def _mv_stop(self):
        if self.mv_state == "ready":
            self.mv_state = "collecting_final"; self.mv_final_buf = {"top": [], "bottom": []}; self.btn_stop.config(state="disabled"); self.mv_prog_bar["value"] = 0; self.mv_status_lbl.config(text=f"Collecting final... (0 / {COLLECT_N})", fg=C_AMBER)

    def _mv_reset(self):
        self.mv_state = "idle"; self.mv_init_buf = {"top": [], "bottom": []}; self.mv_final_buf = {"top": [], "bottom": []}; self.mv_dist_init = {"top": None, "bottom": None}; self.mv_dist_final = {"top": None, "bottom": None}; self.btn_start.config(state="normal"); self.btn_stop.config(state="disabled"); self.mv_prog_bar["value"] = 0; self.mv_prog_lbl.config(text=""); self.mv_status_lbl.config(text="Press START to capture initial gap", fg="#dfe6e9")
        for il, fl, dl in [ (self.mv_init_lbl_top, self.mv_final_lbl_top, self.mv_delta_lbl_top), (self.mv_init_lbl_bot, self.mv_final_lbl_bot, self.mv_delta_lbl_bot) ]: il.config(text="Init: —", fg=C_TEXT_MED); fl.config(text="Final: —", fg=C_TEXT_MED); dl.config(text="—", fg=C_ACCENT)

    def update_gui_loop(self):
        if self.current_frame is not None:
            try: f960 = cv.resize(self.current_frame, (960, 540)); img = ImageTk.PhotoImage(Image.fromarray(cv.cvtColor(f960, cv.COLOR_BGR2RGB))); self.canvas.create_image(0,0, anchor="nw", image=img); self.canvas.image = img
            except: pass
        if self._cam_preview_frame is not None:
            try: f800 = cv.resize(self._cam_preview_frame, (800, 480)); img2 = ImageTk.PhotoImage(Image.fromarray(cv.cvtColor(f800, cv.COLOR_BGR2RGB))); self.cam_canvas.create_image(0,0, anchor="nw", image=img2); self.cam_canvas.image = img2; self.lbl_cam_status.config(text="🟢 Live Feed Active", fg=C_GREEN)
            except: pass
        self._mv_tick(); self._update_telemetry(); self._update_warnings(); self.root.after(30, self.update_gui_loop)

    def _mv_tick(self):
        sc = self.last_data["session_count"]
        if sc == self._last_sc: return
        self._last_sc = sc
        if self.mv_state == "collecting_init":
            for k in ["top", "bottom"]:
                d = self.last_data[k]["dist"]
                if d > 0: self.mv_init_buf[k].append(d)
            n = min(len(self.mv_init_buf["top"]), len(self.mv_init_buf["bottom"])); self.mv_prog_bar["value"] = n; self.mv_prog_lbl.config(text=f"Reading {n} / {COLLECT_N}")
            if n >= COLLECT_N:
                self.mv_dist_init = {"top": float(np.mean(self.mv_init_buf["top"][:COLLECT_N])), "bottom": float(np.mean(self.mv_init_buf["bottom"][:COLLECT_N]))}; self.mv_state = "ready"; self.btn_stop.config(state="normal"); self.mv_prog_bar["value"] = 0; self.mv_prog_lbl.config(text=""); self.mv_status_lbl.config(text="✅ Initial captured — move panel, then STOP", fg=C_GREEN); self.mv_init_lbl_top.config(text=f"Init: {self.mv_dist_init['top']:.3f} mm"); self.mv_init_lbl_bot.config(text=f"Init: {self.mv_dist_init['bottom']:.3f} mm")
        elif self.mv_state == "collecting_final":
            for k in ["top", "bottom"]:
                d = self.last_data[k]["dist"]; if d > 0: self.mv_final_buf[k].append(d)
            n = min(len(self.mv_final_buf["top"]), len(self.mv_final_buf["bottom"])); self.mv_prog_bar["value"] = n; self.mv_prog_lbl.config(text=f"Reading {n} / {COLLECT_N}")
            if n >= COLLECT_N:
                self.mv_dist_final = {"top": float(np.mean(self.mv_final_buf["top"][:COLLECT_N])), "bottom": float(np.mean(self.mv_final_buf["bottom"][:COLLECT_N]))}; self.mv_state = "done"; self.btn_start.config(state="normal"); self.mv_prog_bar["value"] = 0; self.mv_prog_lbl.config(text=""); self.mv_status_lbl.config(text="✅ Measurement complete — press RESET", fg=C_GREEN)
                for key, il, fl, dl in [ ("top", self.mv_init_lbl_top, self.mv_final_lbl_top, self.mv_delta_lbl_top), ("bottom", self.mv_init_lbl_bot, self.mv_final_lbl_bot, self.mv_delta_lbl_bot) ]:
                    di, df = self.mv_dist_init[key], self.mv_dist_final[key]; delta = df - di; il.config(text=f"Init: {di:.3f} mm"); fl.config(text=f"Final: {df:.3f} mm"); sign = "+" if delta >= 0 else ""; color = C_RED if delta > 0.5 else (C_GREEN if delta < -0.5 else C_ACCENT); dl.config(text=f"{sign}{delta:.3f} mm", fg=color)

    def _update_telemetry(self):
        for k in ["top", "bottom"]:
            d = self.last_data[k]; v = self.tele_vars[k]; v["L_ROT"].set(f"{d['rot_2d']:.1f}°"); v["L_Z"].set(f"{d['L_z']:.1f}"); v["L_PITCH"].set(f"{d['L_pitch']:.1f}°"); v["L_YAW"].set(f"{d['L_yaw']:.1f}°"); v["R_PITCH"].set(f"{d['R_pitch']:.1f}°"); v["R_YAW"].set(f"{d['R_yaw']:.1f}°"); v["R_Z"].set(f"{d['R_z']:.1f}")

    def _update_warnings(self):
        p_th, y_th, r_th = self.pitch_threshold.get(), self.yaw_threshold.get(), self.rot_threshold.get(); strip = []
        for k in ["top", "bottom"]:
            d = self.last_data[k]
            if not d["L_det"]: strip.append(f"[{k.upper()}] Left Missing")
            if not d["R_det"]: strip.append(f"[{k.upper()}] Right Missing")
            if d["rot_2d"] > r_th: strip.append(f"[{k.upper()}] Rotation {d['rot_2d']:.1f}°")
            lp, rp = abs(d["L_pitch"]), abs(d["R_pitch"])
            if lp > p_th: strip.append(f"[{k.upper()}] L-Pitch high")
            if rp > p_th: strip.append(f"[{k.upper()}] R-Pitch high")
        self.warn_strip.config(state="normal")
        self.warn_strip.delete("1.0", tk.END)
        if strip:
            self.warn_strip.insert(tk.END, "![!] ALERTS:\n" + "\n".join([f" • {x}" for x in strip]))
        else:
            self.warn_strip.insert(tk.END, "[OK] SYSTEM NOMINAL")
        self.warn_strip.config(state="disabled")

    def load_calib(self):
        f = "camera_params_2.npz" if self.fixed_side.get() == "Right" else "camera_params.npz"
        if os.path.exists(f): 
            d = np.load(f); return d['camera_matrix'], d['dist_coeff']
        return np.eye(3), np.zeros(5)

    def _apply_cam(self, *_):
        if self.pc:
            c = {"AeEnable": self.cam_ae.get(), "AwbEnable": self.cam_awb.get(), "Brightness": self.cam_brightness.get(), "Contrast": self.cam_contrast.get()}; if not self.cam_ae.get(): c["ExposureTime"] = self.cam_exposure.get()
            try: self.pc.set_controls(c)
            except: pass

    def _reset_cam(self): self.cam_ae.set(True); self.cam_exposure.set(10000); self.cam_brightness.set(0.0); self.cam_contrast.set(1.0); self._apply_cam()

    def measurement_loop(self):
        try:
            from picamera2 import Picamera2
            pc = Picamera2(); pc.configure(pc.create_video_configuration(main={"size": RESOLUTION, "format": "RGB888"})); pc.start(); self.pc = pc
        except: return
        detector = cv.aruco.ArucoDetector(cv.aruco.getPredefinedDictionary(ARUCO_DICT), cv.aruco.DetectorParameters())
        log.init_log(); buffers = {"top": [], "bottom": []}; l_s = l_u = time.time()*1000
        while self.is_running:
            K, dist_c = self.load_calib()
            try: frame = cv.cvtColor(pc.capture_array(), cv.COLOR_RGB2BGR)
            except: continue
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY); self.last_data["lighting"] = {"status": "ok", "mean": np.mean(gray), "std": np.std(gray)}
            corners, ids, _ = detector.detectMarkers(gray); curr = time.time()*1000
            if ids is not None and len(ids) >= 1:
                m_data = []
                for i in range(len(ids)):
                    c = corners[i][0]; m_data.append({"c": c, "y": np.mean(c[:, 1]), "x": np.mean(c[:, 0]), "c_raw": corners[i][0]})
                m_data.sort(key=lambda m: m["y"]); top_m, bot_m = m_data[:2], m_data[2:4]
                if len(m_data)==2:
                    dy, dx = abs(m_data[0]["y"]-m_data[1]["y"]), abs(m_data[0]["x"]-m_data[1]["x"])
                    if dx > dy: (top_m, bot_m) = (m_data, []) if (m_data[0]["y"]+m_data[1]["y"])/2 < RESOLUTION[1]/2 else ([], m_data)
                    else: top_m, bot_m = [m_data[0]], [m_data[1]]
                elif len(m_data)==3: top_m, bot_m = m_data[:2], m_data[2:]

                def proc(pair, pk, size):
                    if len(pair) < 2: self.last_data[pk]["dist"] = 0.0; self.last_data[pk]["L_det"] = False; return
                    m1, m2 = sorted(pair, key=lambda x: x["x"]); is_rf = (self.fixed_side.get() == "Right")
                    s_m = m2 if is_rf else m1; t_m = m1 if is_rf else m2
                    
                    def get_p(c):
                        obj = np.array([[-size/2, size/2, 0], [size/2, size/2, 0], [size/2, -size/2, 0], [-size/2, -size/2, 0]], dtype=np.float32)
                        _, rv, tv = cv.solvePnP(obj, c, K, dist_c); R, _ = cv.Rodrigues(rv); p, y, r = rotation_to_euler(R); return np.array([R @ pt + tv.ravel() for pt in obj]), r, float(tv[2,0]), p, y
                    
                    s_pts, s_roll, s_z, s_pitch, s_yaw = get_p(s_m["c_raw"]); t_pts, t_roll, t_z, t_pitch, t_yaw = get_p(t_m["c_raw"])
                    s_irot = self.engine.inplane_rot(s_m["c"]); t_irot = self.engine.inplane_rot(t_m["c"])
                    
                    sit, sib, sot = self.engine.inner_outer(s_pts, not is_rf); tit, tib, _ = self.engine.inner_outer(t_pts, is_rf)
                    px1 = tuple(((m1["c"][1 if not is_rf else 0]+m1["c"][2 if not is_rf else 3])/2).astype(int))
                    px2 = tuple(((m2["c"][0 if not is_rf else 1]+m2["c"][3 if not is_rf else 2])/2).astype(int))
                    
                    buffers[pk].append({"A": (sit+sib)/2, "X_alt": (tit+tib)/2, "TR": sit, "BR": sib, "TL_ref": sot, "B": tit, "C": tib, "L_A": (s_roll, s_irot), "R_A": (t_roll, t_irot), "rot": max(s_irot, t_irot), "L_z": s_z, "R_z": t_z, "L_pitch": s_pitch, "L_yaw": s_yaw, "R_pitch": t_pitch, "R_yaw": t_yaw, "p1_px": px1, "p2_px": px2})
                    self.last_data[pk]["L_det"] = True; self.last_data[pk]["R_det"] = True

                if (curr - l_s) >= 100: proc(top_m, "top", self.size_top.get()); proc(bot_m, "bottom", self.size_bot.get()); l_s = curr

                for key in ["top", "bottom"]:
                    if buffers[key] and (curr - l_u) >= 200:
                        s = buffers[key]; aA = np.mean([x["A"] for x in s], axis=0); aTR = np.mean([x["TR"] for x in s], axis=0); aTL = np.mean([x["TL_ref"] for x in s], axis=0); aC = np.mean([x["C"] for x in s], axis=0); aB = np.mean([x["B"] for x in s], axis=0); aX_alt = np.mean([x["X_alt"] for x in s], axis=0)
                        v_r = aTR-aTL; v = v_r/np.linalg.norm(v_r) if np.linalg.norm(v_r)>0 else np.zeros(3); w = aC-aB; u = aB-aA; den = np.dot(v,v)*np.dot(w,w)-np.dot(v,w)**2
                        def _perp():
                            if abs(den)>1e-6: kv_ = ((np.dot(v,w)*np.dot(u,v))-(np.dot(v,v)*np.dot(u,w)))/den; kv_ = np.clip(kv_, 0, 1); ax_ = aB + kv_*w; return ax_, np.linalg.norm(ax_-aA), kv_
                            return aX_alt, np.linalg.norm(aX_alt-aA), 0.5
                        if self.use_angle_thresh.get() and np.mean([x["rot"] for x in s]) > self.rot_threshold.get(): aX, dv, kv = aX_alt, np.linalg.norm(aX_alt-aA), 0.5
                        else: aX, dv, kv = _perp()
                        l = s[-1]; self.last_data[key].update({"A": aA, "X": aX, "dist": dv, "k": kv, "rot_2d": np.mean([x["rot"] for x in s]), "L_pitch": np.mean([x["L_pitch"] for x in s]), "L_yaw": np.mean([x["L_yaw"] for x in s]), "R_pitch": np.mean([x["R_pitch"] for x in s]), "R_yaw": np.mean([x["R_yaw"] for x in s]), "L_z": np.mean([x["L_z"] for x in s]), "R_z": np.mean([x["R_z"] for x in s]), "p1_px": l["p1_px"], "p2_px": l["p2_px"], "L_A": np.mean([x["L_A"] for x in s], axis=0), "R_A": np.mean([x["R_A"] for x in s], axis=0)})
                        self.last_data["session_count"] += 1; log.record(dv, kv, aA, aX, aTR, np.mean([x["BR"] for x in s], axis=0), aB, aC, aB-aA, aTR-aTL, aC-aB, self.last_data[key]["L_A"], self.last_data[key]["R_A"]); buffers[key].clear()

            for k, color in [("top", (0,165,255)), ("bottom", (255,0,255))]:
                d = self.last_data[k]; if d["dist"]>0 and d["p1_px"]:
                    X_3 = np.array([d["X"]], dtype=np.float64).reshape(1,1,3); p2 = d["p2_px"]
                    if d["X"][2]>0: p_p, _ = cv.projectPoints(X_3, np.zeros(3), np.zeros(3), K, dist_c); p2 = tuple(p_p[0].ravel().astype(int))
                    cv.line(frame, d["p1_px"], p2, color, 3); cv.circle(frame, p2, 6, (0,255,0), -1); cv.putText(frame, f"{k.upper()}: {d['dist']:.2f}mm", (d["p1_px"][0], d["p1_px"][1]-12), 0, 0.6, color, 2)
            self.current_frame = frame; l_u = curr

    def on_close(self): self.is_running = False; self.root.destroy()
