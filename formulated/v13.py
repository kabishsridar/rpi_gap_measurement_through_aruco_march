import tkinter as tk
from tkinter import ttk
import cv2 as cv
import numpy as np
import threading
import math
import os
import time
from PIL import Image, ImageTk
import log

# --- SETTINGS ---
DEFAULT_MARKER_SIZE = 53.8
ARUCO_DICT = cv.aruco.DICT_4X4_50
RESOLUTION = (1280, 720)
COLLECT_N  = 5          # readings to average for initial / final

AWB_MODES = {
    "Auto":        0,
    "Tungsten":    1,
    "Fluorescent": 2,
    "Indoor":      3,
    "Daylight":    4,
    "Cloudy":      5,
}

# Colour palette
C_BG      = "#1e272e"
C_PANEL   = "#2d3436"
C_CARD    = "#ffffff"
C_TOP     = "#e67e22"
C_BOT     = "#9b59b6"
C_GREEN   = "#00b894"
C_RED     = "#d63031"
C_BLUE    = "#0984e3"
C_TEXT    = "#2d3436"
C_MUTED   = "#636e72"


class MeasurementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dual-Pair ArUco Measurement v13")
        self.root.geometry("1600x960")
        self.root.configure(bg=C_BG)

        # ── Measurement vars ──────────────────────────────────────────────
        self.size_top      = tk.DoubleVar(value=DEFAULT_MARKER_SIZE)
        self.size_bot      = tk.DoubleVar(value=DEFAULT_MARKER_SIZE)
        self.fixed_side    = tk.StringVar(value="Left")
        self.rot_threshold = tk.DoubleVar(value=12.0)

        # ── Camera control vars ───────────────────────────────────────────
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

        # ── Movement Monitor state ────────────────────────────────────────
        # States: idle | collecting_init | ready | collecting_final | done
        self.mv_state       = "idle"
        self.mv_init_buf    = {"top": [], "bottom": []}   # readings during START capture
        self.mv_final_buf   = {"top": [], "bottom": []}   # readings during STOP capture
        self.mv_dist_init   = {"top": None, "bottom": None}
        self.mv_dist_final  = {"top": None, "bottom": None}
        self._last_sc       = 0    # track session_count for new-reading detection

        # ── Shared measurement data ───────────────────────────────────────
        temp = {"A": (0,0,0), "X": (0,0,0), "TR": (0,0,0), "BR": (0,0,0),
                "B": (0,0,0), "C": (0,0,0), "dist": 0.0, "k": 0.0,
                "L_A": (0.0, 0.0), "R_A": (0.0, 0.0), "rot_2d": 0.0,
                "L_z": 0.0, "R_z": 0.0, "p1_px": None, "p2_px": None}
        self.last_data = {"top": temp.copy(), "bottom": temp.copy(),
                          "session_count": 0}
        self.is_running    = True
        self.current_frame = None

        self._style()
        self.setup_ui()
        threading.Thread(target=self.measurement_loop, daemon=True).start()
        self.update_gui_loop()

    # ══════════════════════════ STYLE ══════════════════════════════════════
    def _style(self):
        s = ttk.Style()
        s.theme_use('clam')
        s.configure("TNotebook",        background=C_BG,    borderwidth=0)
        s.configure("TNotebook.Tab",    background=C_PANEL, foreground="white",
                    padding=[14, 6],    font=("Helvetica", 10, "bold"))
        s.map("TNotebook.Tab",
              background=[("selected", C_BLUE)],
              foreground=[("selected", "white")])
        s.configure("TFrame", background=C_BG)

    # ══════════════════════════ UI ═════════════════════════════════════════
    def setup_ui(self):
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill='both', expand=True, padx=8, pady=8)

        self._build_tab_live()
        self._build_tab_tele()
        self._build_tab_settings()
        self._build_tab_cam()
        self._build_tab_movement()

    # ── helper: card frame ─────────────────────────────────────────────────
    def _card(self, parent, title, title_color, **pack_kw):
        f = tk.LabelFrame(parent, text=f"  {title}  ",
                          font=("Helvetica", 11, "bold"),
                          fg=title_color, bg=C_CARD,
                          bd=1, relief="solid", padx=10, pady=8)
        f.pack(**pack_kw)
        return f

    # ── TAB 1: LIVE MONITOR ───────────────────────────────────────────────
    def _build_tab_live(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text=" 📽  Live Monitor ")
        tab.configure(style="TFrame")

        left = tk.Frame(tab, bg=C_BG)
        left.pack(side="left", padx=16, pady=16)
        self.canvas = tk.Canvas(left, width=960, height=540, bg="black",
                                highlightthickness=1, highlightbackground=C_MUTED)
        self.canvas.pack()

        right = tk.Frame(tab, bg=C_BG)
        right.pack(side="right", fill="both", expand=True, padx=12, pady=16)

        for key, title, col in [("top", "TOP PAIR", C_TOP),
                                 ("bottom", "BOTTOM PAIR", C_BOT)]:
            card = self._card(right, title, col,
                              fill="x", pady=10, padx=6)
            dl = tk.Label(card, text="0.000 mm",
                          font=("Helvetica", 26, "bold"), fg=C_GREEN, bg=C_CARD)
            dl.pack(pady=(6, 2))
            kl = tk.Label(card, text="k: 0.0000",
                          font=("Helvetica", 11), fg=C_MUTED, bg=C_CARD)
            kl.pack(pady=(0, 4))
            if key == "top":
                self.lbl_dist_top, self.lbl_k_top = dl, kl
            else:
                self.lbl_dist_bot, self.lbl_k_bot = dl, kl

    # ── TAB 2: DUAL TELEMETRY ─────────────────────────────────────────────
    def _build_tab_tele(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text=" 🛸  Dual Telemetry ")
        mtf = tk.Frame(tab, bg="#ecf0f1")
        mtf.pack(fill="both", expand=True)
        self.tele_vars = {"top": {}, "bottom": {}}
        v_show = ["A", "X", "TR", "BR", "B", "C",
                  "L_ROL", "L_ROT", "L_Z", "R_ROL", "R_ROT", "R_Z"]
        for key, title, col in [("top", "TOP", C_TOP), ("bottom", "BOTTOM", C_BOT)]:
            cf = tk.LabelFrame(mtf, text=f" {title} DATA ",
                               font=("Helvetica", 12, "bold"), fg=col, bg="#ecf0f1")
            cf.pack(side="left", fill="both", expand=True, padx=10, pady=10)
            for v_name in v_show:
                row = tk.Frame(cf, bg="white",
                               highlightbackground="#bdc3c7", highlightthickness=1)
                row.pack(fill="x", padx=10, pady=3)
                tk.Label(row, text=v_name, font=("Helvetica", 9),
                         bg="white").pack(side="left", padx=5)
                sv = tk.StringVar(value="—")
                tk.Label(row, textvariable=sv, font=("Courier", 10),
                         bg="white").pack(side="right", padx=5)
                self.tele_vars[key][v_name] = sv

    # ── TAB 3: MACHINE CONFIGURATION ──────────────────────────────────────
    def _build_tab_settings(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text=" ⚙  Machine Configuration ")
        sc = tk.Frame(tab, bg="#ecf0f1")
        sc.pack(fill="both", expand=True, padx=80, pady=40)

        rf = tk.LabelFrame(sc, text=" Reference Side (Fixed ArUco) ", bg="white",
                           font=("Helvetica", 11, "bold"), padx=20, pady=10)
        rf.pack(fill="x", pady=(0, 12))
        for choice in ["Left", "Right"]:
            tk.Radiobutton(rf, text=choice, variable=self.fixed_side,
                           value=choice, bg="white",
                           font=("Helvetica", 12)).pack(side="left", padx=24)

        def create_slider(label, var, color, lo, hi):
            f = tk.LabelFrame(sc, text=f" {label} ", bg="white",
                              font=("Helvetica", 10), fg=color, padx=20, pady=6)
            f.pack(fill="x", pady=6)
            tk.Scale(f, from_=lo, to=hi, resolution=0.1, orient="horizontal",
                     variable=var, bg="white", length=500).pack(side="left", padx=16)
            tk.Entry(f, textvariable=var, width=9,
                     font=("Courier", 10)).pack(side="left")

        create_slider("Upper Pair Marker Size (mm)",  self.size_top, C_TOP,  10, 200)
        create_slider("Bottom Pair Marker Size (mm)", self.size_bot, C_BOT,  10, 200)
        create_slider("Rotation Threshold °  (perpendicular / fallback)",
                      self.rot_threshold, C_TEXT, 0, 45)

    # ── TAB 4: CAMERA CONTROLS ────────────────────────────────────────────
    def _build_tab_cam(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text=" 📷  Camera Controls ")

        cam_left  = tk.Frame(tab, bg="#1a252f")
        cam_left.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        cam_right = tk.Frame(tab, bg="#ecf0f1")
        cam_right.pack(side="right", fill="y", padx=8, pady=8, ipadx=4)

        tk.Label(cam_left, text="Live Camera Preview",
                 font=("Helvetica", 11, "bold"), fg="#ecf0f1",
                 bg="#1a252f").pack(pady=(8, 0))
        self.cam_canvas = tk.Canvas(cam_left, width=800, height=480, bg="black",
                                    highlightthickness=1, highlightbackground=C_MUTED)
        self.cam_canvas.pack(padx=10, pady=10)
        self.lbl_cam_status = tk.Label(cam_left, text="⏳ Waiting for camera…",
                                       font=("Helvetica", 11, "bold"),
                                       fg="#f39c12", bg="#1a252f")
        self.lbl_cam_status.pack(pady=4)

        tk.Label(cam_right, text="Camera Settings",
                 font=("Helvetica", 13, "bold"), fg=C_TEXT,
                 bg="#ecf0f1").pack(pady=(10, 4))

        def cam_row(label, var, lo, hi, res, unit=""):
            rf = tk.LabelFrame(cam_right, text=f" {label} ", bg="white",
                               font=("Helvetica", 9), padx=8, pady=4)
            rf.pack(fill="x", padx=6, pady=3)
            inner = tk.Frame(rf, bg="white"); inner.pack(fill="x")
            tk.Scale(inner, from_=lo, to=hi, resolution=res,
                     orient="horizontal", variable=var, bg="white",
                     length=260, command=lambda _: self._apply_cam()).pack(side="left")
            tk.Entry(inner, textvariable=var, width=7,
                     font=("Courier", 9)).pack(side="left", padx=4)
            if unit:
                tk.Label(inner, text=unit, bg="white",
                         font=("Helvetica", 8), fg=C_MUTED).pack(side="left")

        ae_f = tk.LabelFrame(cam_right, text=" Auto Exposure ", bg="white",
                             font=("Helvetica", 9), padx=8, pady=4)
        ae_f.pack(fill="x", padx=6, pady=3)
        tk.Checkbutton(ae_f, text="Enable Auto Exposure (AE)",
                       variable=self.cam_ae, bg="white",
                       font=("Helvetica", 10),
                       command=self._apply_cam).pack(anchor="w")

        cam_row("Exposure Time", self.cam_exposure, 100,  66000, 100, "µs")
        cam_row("ISO / Gain",    self.cam_gain,     1.0,  16.0,  0.1, "x")

        awb_f = tk.LabelFrame(cam_right, text=" White Balance ", bg="white",
                              font=("Helvetica", 9), padx=8, pady=4)
        awb_f.pack(fill="x", padx=6, pady=3)
        tk.Checkbutton(awb_f, text="Auto White Balance",
                       variable=self.cam_awb, bg="white",
                       font=("Helvetica", 10),
                       command=self._apply_cam).pack(anchor="w")
        wbm_row = tk.Frame(awb_f, bg="white"); wbm_row.pack(fill="x", pady=2)
        tk.Label(wbm_row, text="Mode:", bg="white",
                 font=("Helvetica", 9)).pack(side="left")
        for mode_name in AWB_MODES:
            tk.Radiobutton(wbm_row, text=mode_name, variable=self.cam_awb_mode,
                           value=mode_name, bg="white", font=("Helvetica", 8),
                           command=self._apply_cam).pack(side="left", padx=2)

        cam_row("Brightness", self.cam_brightness, -1.0, 1.0, 0.05)
        cam_row("Contrast",   self.cam_contrast,    0.0, 8.0, 0.1)
        cam_row("Saturation", self.cam_saturation,  0.0, 8.0, 0.1)
        cam_row("Sharpness",  self.cam_sharpness,   0.0, 8.0, 0.1)

        tk.Button(cam_right, text="↺  Reset to Defaults",
                  font=("Helvetica", 10, "bold"), bg=C_RED, fg="white",
                  activebackground="#c0392b", relief="flat", padx=10, pady=6,
                  command=self._reset_cam).pack(pady=10, fill="x", padx=6)

    # ── TAB 5: MOVEMENT MONITOR ───────────────────────────────────────────
    def _build_tab_movement(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text=" 📏  Movement Monitor ")

        # ── Outer layout: top = controls, bottom = results ─────────────────
        top_bar = tk.Frame(tab, bg=C_BG)
        top_bar.pack(fill="x", padx=20, pady=(18, 0))

        result_area = tk.Frame(tab, bg=C_BG)
        result_area.pack(fill="both", expand=True, padx=20, pady=12)

        # ── Status label ───────────────────────────────────────────────────
        self.mv_status_lbl = tk.Label(
            top_bar, text="Press  START  to capture initial gap distance",
            font=("Helvetica", 13), fg="#dfe6e9", bg=C_BG)
        self.mv_status_lbl.pack(side="left", padx=(0, 30))

        # ── Progress bar (hidden until collecting) ─────────────────────────
        prog_frame = tk.Frame(top_bar, bg=C_BG)
        prog_frame.pack(side="left", padx=(0, 30))
        self.mv_prog_lbl = tk.Label(prog_frame, text="",
                                    font=("Helvetica", 11), fg=C_GREEN, bg=C_BG)
        self.mv_prog_lbl.pack()
        self.mv_prog_bar = ttk.Progressbar(prog_frame, length=180, maximum=COLLECT_N,
                                           mode="determinate")
        self.mv_prog_bar.pack(pady=2)

        # ── START / STOP / RESET buttons ───────────────────────────────────
        btn_frame = tk.Frame(top_bar, bg=C_BG)
        btn_frame.pack(side="right")

        btn_cfg = dict(font=("Helvetica", 14, "bold"), relief="flat",
                       padx=24, pady=10, bd=0, cursor="hand2")

        self.btn_start = tk.Button(
            btn_frame, text="▶  START", bg=C_GREEN, fg="white",
            activebackground="#00cec9",
            command=self._mv_start, **btn_cfg)
        self.btn_start.pack(side="left", padx=8)

        self.btn_stop = tk.Button(
            btn_frame, text="■  STOP", bg=C_RED, fg="white",
            activebackground="#ff7675",
            state="disabled",
            command=self._mv_stop, **btn_cfg)
        self.btn_stop.pack(side="left", padx=8)

        self.btn_reset = tk.Button(
            btn_frame, text="↺  RESET", bg=C_MUTED, fg="white",
            activebackground="#b2bec3",
            command=self._mv_reset, **btn_cfg)
        self.btn_reset.pack(side="left", padx=8)

        # ── Separator ──────────────────────────────────────────────────────
        sep = tk.Frame(tab, bg=C_MUTED, height=1)
        sep.pack(fill="x", padx=20, pady=6)

        # ── Result cards: TOP and BOTTOM side-by-side ─────────────────────
        for key, title, col in [("top", "TOP PAIR", C_TOP),
                                 ("bottom", "BOTTOM PAIR", C_BOT)]:
            card = tk.LabelFrame(result_area, text=f"  {title}  ",
                                 font=("Helvetica", 12, "bold"),
                                 fg=col, bg=C_CARD, bd=1, relief="solid",
                                 padx=20, pady=14)
            card.pack(side="left", fill="both", expand=True, padx=12, pady=4)

            # Row labels
            def make_row(parent, label, fg_col=C_MUTED, font_sz=11):
                row = tk.Frame(parent, bg=C_CARD)
                row.pack(fill="x", pady=3)
                tk.Label(row, text=label, font=("Helvetica", font_sz),
                         fg=fg_col, bg=C_CARD, anchor="w",
                         width=18).pack(side="left")
                val = tk.Label(row, text="—", font=("Helvetica", font_sz, "bold"),
                               fg=C_TEXT, bg=C_CARD, anchor="e")
                val.pack(side="right")
                return val

            init_lbl  = make_row(card, "Initial Distance:")
            final_lbl = make_row(card, "Final Distance:")

            # Big delta display
            tk.Frame(card, bg="#dfe6e9", height=1).pack(fill="x", pady=8)
            tk.Label(card, text="Distance Moved",
                     font=("Helvetica", 12), fg=C_MUTED, bg=C_CARD).pack()
            delta_lbl = tk.Label(card, text="—",
                                 font=("Helvetica", 40, "bold"),
                                 fg=C_BLUE, bg=C_CARD)
            delta_lbl.pack(pady=(4, 6))

            if key == "top":
                self.mv_init_lbl_top  = init_lbl
                self.mv_final_lbl_top = final_lbl
                self.mv_delta_lbl_top = delta_lbl
            else:
                self.mv_init_lbl_bot  = init_lbl
                self.mv_final_lbl_bot = final_lbl
                self.mv_delta_lbl_bot = delta_lbl

    # ══════════════════════ MOVEMENT MONITOR LOGIC ═════════════════════════
    def _mv_start(self):
        """Begin collecting 5 readings for initial distance."""
        if self.mv_state not in ("idle", "done"):
            return
        self.mv_state    = "collecting_init"
        self.mv_init_buf = {"top": [], "bottom": []}
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="disabled")
        self.mv_prog_bar["value"] = 0
        self.mv_status_lbl.config(
            text=f"⏳  Collecting initial distance…  (0 / {COLLECT_N})",
            fg="#f39c12")

    def _mv_stop(self):
        """Begin collecting 5 readings for final distance."""
        if self.mv_state != "ready":
            return
        self.mv_state     = "collecting_final"
        self.mv_final_buf = {"top": [], "bottom": []}
        self.btn_stop.config(state="disabled")
        self.mv_prog_bar["value"] = 0
        self.mv_status_lbl.config(
            text=f"⏳  Collecting final distance…  (0 / {COLLECT_N})",
            fg="#f39c12")

    def _mv_reset(self):
        """Reset everything back to idle."""
        self.mv_state      = "idle"
        self.mv_init_buf   = {"top": [], "bottom": []}
        self.mv_final_buf  = {"top": [], "bottom": []}
        self.mv_dist_init  = {"top": None, "bottom": None}
        self.mv_dist_final = {"top": None, "bottom": None}
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.mv_prog_bar["value"] = 0
        self.mv_prog_lbl.config(text="")
        self.mv_status_lbl.config(
            text="Press  START  to capture initial gap distance", fg="#dfe6e9")
        for lbl in [self.mv_init_lbl_top,  self.mv_init_lbl_bot,
                    self.mv_final_lbl_top, self.mv_final_lbl_bot]:
            lbl.config(text="—", fg=C_TEXT)
        for lbl in [self.mv_delta_lbl_top, self.mv_delta_lbl_bot]:
            lbl.config(text="—", fg=C_BLUE)

    def _mv_tick(self):
        """
        Called in update_gui_loop whenever a NEW measurement arrives.
        Drives the state machine: buffers readings, computes averages,
        transitions states, and updates all Movement Monitor labels.
        """
        sc = self.last_data["session_count"]
        if sc == self._last_sc:
            return           # no new measurement this tick
        self._last_sc = sc

        if self.mv_state == "collecting_init":
            for key in ["top", "bottom"]:
                d = self.last_data[key]["dist"]
                if d > 0:
                    self.mv_init_buf[key].append(d)

            n = min(len(self.mv_init_buf["top"]),
                    len(self.mv_init_buf["bottom"]))
            self.mv_prog_bar["value"] = n
            self.mv_prog_lbl.config(text=f"Reading {n} / {COLLECT_N}")
            self.mv_status_lbl.config(
                text=f"⏳  Collecting initial distance…  ({n} / {COLLECT_N})",
                fg="#f39c12")

            if n >= COLLECT_N:
                self.mv_dist_init = {
                    "top":    float(np.mean(self.mv_init_buf["top"][:COLLECT_N])),
                    "bottom": float(np.mean(self.mv_init_buf["bottom"][:COLLECT_N])),
                }
                self.mv_state = "ready"
                self.btn_stop.config(state="normal")
                self.mv_prog_bar["value"] = 0
                self.mv_prog_lbl.config(text="")
                self.mv_status_lbl.config(
                    text="✅  Initial captured — move the panel, then press  STOP",
                    fg=C_GREEN)
                # Update init labels
                self.mv_init_lbl_top.config(
                    text=f"{self.mv_dist_init['top']:.3f} mm", fg=C_TEXT)
                self.mv_init_lbl_bot.config(
                    text=f"{self.mv_dist_init['bottom']:.3f} mm", fg=C_TEXT)

        elif self.mv_state == "collecting_final":
            for key in ["top", "bottom"]:
                d = self.last_data[key]["dist"]
                if d > 0:
                    self.mv_final_buf[key].append(d)

            n = min(len(self.mv_final_buf["top"]),
                    len(self.mv_final_buf["bottom"]))
            self.mv_prog_bar["value"] = n
            self.mv_prog_lbl.config(text=f"Reading {n} / {COLLECT_N}")
            self.mv_status_lbl.config(
                text=f"⏳  Collecting final distance…  ({n} / {COLLECT_N})",
                fg="#f39c12")

            if n >= COLLECT_N:
                self.mv_dist_final = {
                    "top":    float(np.mean(self.mv_final_buf["top"][:COLLECT_N])),
                    "bottom": float(np.mean(self.mv_final_buf["bottom"][:COLLECT_N])),
                }
                self.mv_state = "done"
                self.btn_start.config(state="normal")   # allow re-capture
                self.mv_prog_bar["value"] = 0
                self.mv_prog_lbl.config(text="")
                self.mv_status_lbl.config(
                    text="✅  Measurement complete — press RESET to start over",
                    fg=C_GREEN)

                # Update final labels and deltas
                for key, init_lbl, final_lbl, delta_lbl in [
                    ("top",    self.mv_init_lbl_top,
                               self.mv_final_lbl_top, self.mv_delta_lbl_top),
                    ("bottom", self.mv_init_lbl_bot,
                               self.mv_final_lbl_bot, self.mv_delta_lbl_bot),
                ]:
                    di = self.mv_dist_init[key]
                    df = self.mv_dist_final[key]
                    delta = df - di
                    init_lbl.config(text=f"{di:.3f} mm",    fg=C_TEXT)
                    final_lbl.config(text=f"{df:.3f} mm",   fg=C_TEXT)
                    sign  = "+" if delta >= 0 else ""
                    color = C_RED if delta > 0.5 else (C_GREEN if delta < -0.5 else C_BLUE)
                    delta_lbl.config(text=f"{sign}{delta:.3f} mm", fg=color)

    # ══════════════════════ CAMERA CONTROL HELPERS ═════════════════════════
    def _apply_cam(self, *_):
        if self.pc is None:
            return
        controls = {}
        ae_on = self.cam_ae.get()
        controls["AeEnable"] = ae_on
        if not ae_on:
            controls["ExposureTime"] = int(self.cam_exposure.get())
            controls["AnalogueGain"] = float(self.cam_gain.get())
        awb_on = self.cam_awb.get()
        controls["AwbEnable"] = awb_on
        if not awb_on:
            controls["AwbMode"] = AWB_MODES.get(self.cam_awb_mode.get(), 0)
        controls["Brightness"]  = float(self.cam_brightness.get())
        controls["Contrast"]    = float(self.cam_contrast.get())
        controls["Saturation"]  = float(self.cam_saturation.get())
        controls["Sharpness"]   = float(self.cam_sharpness.get())
        try:
            self.pc.set_controls(controls)
        except Exception:
            pass

    def _reset_cam(self):
        self.cam_ae.set(True);        self.cam_exposure.set(10000)
        self.cam_gain.set(2.0);       self.cam_awb.set(True)
        self.cam_awb_mode.set("Auto");self.cam_brightness.set(0.0)
        self.cam_contrast.set(1.0);   self.cam_saturation.set(1.0)
        self.cam_sharpness.set(1.0);  self._apply_cam()

    # ══════════════════════════ CALIBRATION ════════════════════════════════
    def load_calib(self):
        f = "camera_params_2.npz" if self.fixed_side.get() == "Right" else "camera_params.npz"
        if os.path.exists(f):
            d = np.load(f)
            return d['camera_matrix'], d['dist_coeff']
        return (np.array([[1280,0,640],[0,1280,360],[0,0,1]], dtype=np.float32),
                np.zeros(5))

    # ══════════════════════════ MEASUREMENT ════════════════════════════════
    # ── All geometry / formula code is IDENTICAL to v12 / v11 ─────────────
    def measurement_loop(self):
        try:
            from picamera2 import Picamera2
            pc = Picamera2()
            pc.configure(pc.create_video_configuration(
                main={"size": RESOLUTION, "format": "RGB888"}))
            pc.start()
            self.pc = pc
        except:
            return

        detector = cv.aruco.ArucoDetector(
            cv.aruco.getPredefinedDictionary(ARUCO_DICT),
            cv.aruco.DetectorParameters())
        log.init_log()
        buffers = {"top": [], "bottom": []}
        l_s = l_u = time.time() * 1000

        while self.is_running:
            K, dist_c = self.load_calib()
            try:
                frame = cv.cvtColor(pc.capture_array(), cv.COLOR_RGB2BGR)
            except:
                continue

            corners_raw, ids, _ = detector.detectMarkers(
                cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
            curr = time.time() * 1000

            if ids is not None and len(ids) >= 1:
                m_data = []
                for i in range(len(ids)):
                    raw = corners_raw[i][0]
                    idx_x = np.argsort(raw[:, 0])
                    lp = raw[idx_x[:2]];  rp = raw[idx_x[2:]]
                    tl = lp[np.argmin(lp[:, 1])];  bl = lp[np.argmax(lp[:, 1])]
                    tr = rp[np.argmin(rp[:, 1])];  br = rp[np.argmax(rp[:, 1])]
                    cy = (tl[1]+tr[1]+br[1]+bl[1]) / 4.0
                    cx = (tl[0]+tr[0]+br[0]+bl[0]) / 4.0
                    m_data.append({"c":     np.array([tl, tr, br, bl], dtype=np.float32),
                                   "c_raw": raw, "y": cy, "x": cx})

                m_data.sort(key=lambda m: m["y"])
                top_m = m_data[:2]
                bot_m = m_data[2:4] if len(m_data) >= 4 else []
                if len(m_data) == 2:
                    dy_s = abs(m_data[0]["y"] - m_data[1]["y"])
                    dx_s = abs(m_data[0]["x"] - m_data[1]["x"])
                    if dx_s > dy_s:
                        avg_y = (m_data[0]["y"] + m_data[1]["y"]) / 2
                        top_m = m_data if avg_y < RESOLUTION[1]/2 else []
                        bot_m = [] if avg_y < RESOLUTION[1]/2 else m_data
                    else:
                        top_m = [m_data[0]]; bot_m = [m_data[1]]
                elif len(m_data) == 3:
                    top_m = m_data[:2]; bot_m = m_data[2:]

                def proc(marker_list, key, size):
                    if len(marker_list) < 2:
                        self.last_data[key]["dist"] = 0.0
                        self.last_data[key]["p1_px"] = None
                        return
                    marker_list.sort(key=lambda m: m["x"])
                    left_m  = marker_list[0];  right_m = marker_list[-1]
                    is_rf   = (self.fixed_side.get() == "Right")
                    S_m = right_m if is_rf else left_m
                    T_m = left_m  if is_rf else right_m
                    h = size / 2.0
                    obj = np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]],
                                   dtype=np.float32)

                    def get_pose(c_raw):
                        _, rv, tv = cv.solvePnP(obj, c_raw, K, dist_c)
                        R, _ = cv.Rodrigues(rv)
                        pts3d = np.array([R @ pt + tv.ravel() for pt in obj])
                        return pts3d, math.degrees(math.atan2(R[1,0], R[0,0])), float(tv[2])

                    def inplane_rot(c_sorted):
                        return math.degrees(math.atan2(
                            c_sorted[1,1]-c_sorted[0,1],
                            c_sorted[1,0]-c_sorted[0,0]))

                    S_pts, S_roll, S_z = get_pose(S_m["c_raw"])
                    T_pts, T_roll, T_z = get_pose(T_m["c_raw"])
                    S_irot = abs(inplane_rot(S_m["c"]))
                    T_irot = abs(inplane_rot(T_m["c"]))

                    def inner_outer(pts, inner_is_right):
                        order = pts[:, 0].argsort()
                        inner = pts[order[2:]] if inner_is_right else pts[order[:2]]
                        outer = pts[order[:2]] if inner_is_right else pts[order[2:]]
                        inner = inner[inner[:, 1].argsort()]
                        outer = outer[outer[:, 1].argsort()]
                        return inner[0], inner[1], outer[0]

                    S_inner_top, S_inner_bot, S_outer_top = inner_outer(S_pts, not is_rf)
                    A = (S_inner_top + S_inner_bot) / 2.0
                    T_inner_top, T_inner_bot, _ = inner_outer(T_pts, is_rf)
                    B = T_inner_top;  C = T_inner_bot;  X_alt = (B + C) / 2.0

                    if not is_rf:
                        p_src_2d = tuple(((S_m["c"][1]+S_m["c"][2])/2).astype(int))
                        p_tgt_2d = tuple(((T_m["c"][0]+T_m["c"][3])/2).astype(int))
                    else:
                        p_src_2d = tuple(((S_m["c"][0]+S_m["c"][3])/2).astype(int))
                        p_tgt_2d = tuple(((T_m["c"][1]+T_m["c"][2])/2).astype(int))

                    buffers[key].append({
                        "A": A, "X_alt": X_alt,
                        "TR": S_inner_top, "BR": S_inner_bot, "TL_ref": S_outer_top,
                        "B": B, "C": C,
                        "L_A": (S_roll, S_irot), "R_A": (T_roll, T_irot),
                        "rot": max(S_irot, T_irot),
                        "L_z": S_z, "R_z": T_z,
                        "p1_px": p_src_2d, "p2_px": p_tgt_2d,
                    })

                if (curr - l_s) >= 100:
                    proc(top_m, "top",    self.size_top.get())
                    proc(bot_m, "bottom", self.size_bot.get())
                    l_s = curr

                if (curr - l_u) >= 500:
                    for key in ["top", "bottom"]:
                        if buffers[key]:
                            s  = buffers[key]
                            aA     = np.mean([x["A"]      for x in s], axis=0)
                            aTR    = np.mean([x["TR"]     for x in s], axis=0)
                            aTL    = np.mean([x["TL_ref"] for x in s], axis=0)
                            aBR    = np.mean([x["BR"]     for x in s], axis=0)
                            aB     = np.mean([x["B"]      for x in s], axis=0)
                            aC     = np.mean([x["C"]      for x in s], axis=0)
                            aX_alt = np.mean([x["X_alt"]  for x in s], axis=0)
                            aL     = np.mean([x["L_A"]    for x in s], axis=0)
                            aR     = np.mean([x["R_A"]    for x in s], axis=0)
                            avg_rot = float(np.mean([x["rot"] for x in s]))
                            avg_Lz  = float(np.mean([x["L_z"] for x in s]))
                            avg_Rz  = float(np.mean([x["R_z"] for x in s]))
                            p1_px = tuple(np.mean([x["p1_px"] for x in s],
                                                  axis=0).astype(int))
                            p2_px = tuple(np.mean([x["p2_px"] for x in s],
                                                  axis=0).astype(int))
                            thresh = self.rot_threshold.get()

                            if avg_rot > thresh:
                                aX = aX_alt; dv = np.linalg.norm(aX - aA); kv = 0.5
                            else:
                                v_raw = aTR - aTL
                                v_len = np.linalg.norm(v_raw)
                                v = v_raw / v_len if v_len > 0 else np.zeros(3)
                                w = aC - aB;  u = aB - aA
                                den = (np.dot(v,v)*np.dot(w,w)) - (np.dot(v,w)**2)
                                if abs(den) > 1e-6:
                                    kv = ((np.dot(v,w)*np.dot(u,v)) -
                                          (np.dot(v,v)*np.dot(u,w))) / den
                                    kv = float(np.clip(kv, 0.0, 1.0))
                                    aX = aB + kv * w
                                    dv = np.linalg.norm(aX - aA)
                                else:
                                    aX = aX_alt; dv = np.linalg.norm(aX - aA); kv = 0.5

                            self.last_data[key].update({
                                "A": aA, "X": aX, "TR": aTR, "BR": aBR,
                                "B": aB, "C": aC, "dist": dv, "k": kv,
                                "rot_2d": avg_rot, "L_A": aL, "R_A": aR,
                                "L_z": avg_Lz, "R_z": avg_Rz,
                                "p1_px": p1_px, "p2_px": p2_px
                            })
                            self.last_data["session_count"] += 1
                            buffers[key].clear()

                            # ── Logging ────────────────────────────────────
                            # CSV + DB: always recorded on every 500ms cycle
                            v_vec = aTR - aTL
                            u_vec = aA  # placeholder (A point used as U)
                            w_vec = aC - aB
                            try:
                                log.record(
                                    dv, kv, aA, aX, aTR, aBR, aB, aC,
                                    u_vec, v_vec, w_vec, aL, aR
                                )
                            except Exception:
                                pass

                            # Image: saved throughout the full movement process
                            # (from START press → collecting_init → ready → collecting_final)
                            if self.mv_state in ("collecting_init", "ready", "collecting_final"):
                                try:
                                    log.save_image(frame)
                                except Exception:
                                    pass
                    l_u = curr

            else:
                self.last_data["top"]["dist"]    = 0.0
                self.last_data["bottom"]["dist"] = 0.0

            # ── Draw measurement lines ─────────────────────────────────────
            for key, color in [("top", (0, 165, 255)), ("bottom", (255, 0, 255))]:
                d = self.last_data[key]
                if (d["dist"] > 0
                        and d.get("p1_px") is not None
                        and d.get("p2_px") is not None):
                    p1 = d["p1_px"]
                    X_3d = np.array([d["X"]], dtype=np.float64).reshape(1, 1, 3)
                    if d["X"][2] > 0:
                        p_proj, _ = cv.projectPoints(X_3d, np.zeros(3),
                                                     np.zeros(3), K, dist_c)
                        p2 = tuple(p_proj[0].ravel().astype(int))
                    else:
                        p2 = d["p2_px"]
                    cv.line(frame, p1, p2, color, 3)
                    cv.circle(frame, p2, 6, (0, 255, 0), -1)
                    cv.putText(frame, f"{key.upper()}: {d['dist']:.2f}mm",
                               (p1[0], p1[1]-12), 0, 0.6, color, 2)
                    max_rot = max(d["L_A"][1], d["R_A"][1])
                    if max_rot > self.rot_threshold.get():
                        cv.rectangle(frame,
                                     (p1[0]-90, p1[1]-85), (p1[0]+60, p1[1]-50),
                                     (0, 0, 255), -1)
                        cv.putText(frame, f"ROT:{max_rot:.1f}°",
                                   (p1[0]-85, p1[1]-60), 0, 0.55, (0,255,255), 2)

            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            self.current_frame      = rgb
            self._cam_preview_frame = rgb

    # ══════════════════════════ GUI REFRESH ════════════════════════════════
    def update_gui_loop(self):
        # ── Tab 1: Live Monitor ──
        if self.current_frame is not None:
            img = ImageTk.PhotoImage(
                image=Image.fromarray(self.current_frame).resize((960, 540)))
            self.canvas.create_image(0, 0, anchor="nw", image=img)
            self.canvas.img = img

        self.lbl_dist_top.config(text=f"{self.last_data['top']['dist']:.3f} mm")
        self.lbl_k_top.config(text=f"k: {self.last_data['top']['k']:.4f}")
        self.lbl_dist_bot.config(text=f"{self.last_data['bottom']['dist']:.3f} mm")
        self.lbl_k_bot.config(text=f"k: {self.last_data['bottom']['k']:.4f}")

        # ── Tab 2: Telemetry ──
        for key in ["top", "bottom"]:
            d = self.last_data[key]
            for var in ["A", "X", "TR", "BR", "B", "C"]:
                v = d[var]
                self.tele_vars[key][var].set(f"{v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f}")
            self.tele_vars[key]["L_ROL"].set(f"{d['L_A'][0]:.2f}° (roll)")
            self.tele_vars[key]["L_ROT"].set(f"{d['L_A'][1]:.2f}° (in-plane)")
            self.tele_vars[key]["L_Z"].set(f"{d['L_z']:.1f} mm")
            self.tele_vars[key]["R_ROL"].set(f"{d['R_A'][0]:.2f}° (roll)")
            self.tele_vars[key]["R_ROT"].set(f"{d['R_A'][1]:.2f}° (in-plane)")
            self.tele_vars[key]["R_Z"].set(f"{d['R_z']:.1f} mm")

        # ── Tab 4: Camera Controls ──
        if self._cam_preview_frame is not None:
            cam_img = ImageTk.PhotoImage(
                image=Image.fromarray(self._cam_preview_frame).resize((800, 480)))
            self.cam_canvas.create_image(0, 0, anchor="nw", image=cam_img)
            self.cam_canvas.img = cam_img
            top_ok = self.last_data["top"]["dist"]    > 0
            bot_ok = self.last_data["bottom"]["dist"] > 0
            if top_ok and bot_ok:
                self.lbl_cam_status.config(text="✅ Both pairs detected", fg=C_GREEN)
            elif top_ok or bot_ok:
                self.lbl_cam_status.config(
                    text=f"⚠️  Only {'TOP' if top_ok else 'BOTTOM'} pair detected",
                    fg="#f39c12")
            else:
                self.lbl_cam_status.config(
                    text="❌ No ArUco detected — adjust camera settings", fg=C_RED)

        # ── Tab 5: Movement Monitor ──
        self._mv_tick()

        self.root.after(50, self.update_gui_loop)


if __name__ == "__main__":
    root = tk.Tk()
    app = MeasurementApp(root)
    root.mainloop()
