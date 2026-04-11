"""
Dual-Pair ArUco Gap Measurement — v15
Professional dark UI with real-time tilt/lighting warnings.
"""
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

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_MARKER_SIZE = 53.8
ARUCO_DICT          = cv.aruco.DICT_4X4_50
RESOLUTION          = (1280, 720)
COLLECT_N           = 5

LIGHT_LOW    = 40     # mean brightness below → too dark
LIGHT_HIGH   = 215    # mean brightness above → overexposed
CONTRAST_MIN = 18     # std-dev below → flat / low-contrast

AWB_MODES = {
    "Auto": 0, "Tungsten": 1, "Fluorescent": 2,
    "Indoor": 3, "Daylight": 4, "Cloudy": 5,
}

# ── Professional dark colour palette ──────────────────────────────────────────
C_BG      = "#0d1117"    # window background
C_SURF    = "#161b22"    # card / panel surface
C_SURF2   = "#1c2128"    # row / inner surface
C_BORDER  = "#30363d"    # subtle borders
C_TEXT    = "#e6edf3"    # primary text
C_MUTED   = "#8b949e"    # secondary / label text
C_TOP     = "#f0883e"    # TOP pair accent (orange)
C_BOT     = "#a371f7"    # BOT pair accent (purple)
C_GREEN   = "#3fb950"    # success / healthy
C_RED     = "#f85149"    # error / missing
C_AMBER   = "#d29922"    # warning / caution
C_BLUE    = "#58a6ff"    # info / delta
C_TEAL    = "#39d3bb"    # heading accent
C_WHITE   = "#ffffff"

# Status chip colours  {"state": (bg, fg, label)}
CHIP = {
    "ok":    (C_GREEN, C_BG,   "  OK  "),
    "warn":  (C_AMBER, C_BG,   " WARN "),
    "error": (C_RED,   C_WHITE, "ERROR "),
    "wait":  (C_MUTED, C_BG,   "  ──  "),
}


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def rotation_to_euler(R):
    """
    Decompose 3×3 OpenCV rotation matrix → (pitch, yaw, roll) degrees.
      pitch – X-axis: top of marker tilts toward / away from camera
      yaw   – Y-axis: left/right edge closer ('one side in' scenario)
      roll  – Z-axis: in-plane spin
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll  = math.degrees(math.atan2( R[2, 1],  R[2, 2]))
        pitch = math.degrees(math.atan2(-R[2, 0],  sy))
        yaw   = math.degrees(math.atan2( R[1, 0],  R[0, 0]))
    else:
        roll  = math.degrees(math.atan2(-R[1, 2],  R[1, 1]))
        pitch = math.degrees(math.atan2(-R[2, 0],  sy))
        yaw   = 0.0
    return pitch, yaw, roll


# ══════════════════════════════════════════════════════════════════════════════
#  Application
# ══════════════════════════════════════════════════════════════════════════════

class MeasurementApp:

    # ── init ──────────────────────────────────────────────────────────────────
    def __init__(self, root):
        self.root = root
        self.root.title("ArUco Gap Measurement  │  v15")
        self.root.geometry("1720x960")
        self.root.configure(bg=C_BG)
        self.root.minsize(1400, 800)

        # Measurement vars
        self.size_top         = tk.DoubleVar(value=DEFAULT_MARKER_SIZE)
        self.size_bot         = tk.DoubleVar(value=DEFAULT_MARKER_SIZE)
        self.fixed_side       = tk.StringVar(value="Left")
        self.rot_threshold    = tk.DoubleVar(value=12.0)
        self.pitch_threshold  = tk.DoubleVar(value=10.0)
        self.yaw_threshold    = tk.DoubleVar(value=10.0)
        self.use_angle_thresh = tk.BooleanVar(value=True)

        # Camera vars
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

        # Movement state
        self.mv_state       = "idle"
        self.mv_init_buf    = {"top": [], "bottom": []}
        self.mv_final_buf   = {"top": [], "bottom": []}
        self.mv_dist_init   = {"top": None, "bottom": None}
        self.mv_dist_final  = {"top": None, "bottom": None}
        self._last_sc       = 0

        # Shared measurement data
        def _empty():
            return {
                "A": (0,0,0), "X": (0,0,0), "TR": (0,0,0), "BR": (0,0,0),
                "B": (0,0,0), "C": (0,0,0), "dist": 0.0, "k": 0.0,
                "L_A": (0.0, 0.0), "R_A": (0.0, 0.0), "rot_2d": 0.0,
                "L_z": 0.0, "R_z": 0.0, "p1_px": None, "p2_px": None,
                "L_pitch": 0.0, "L_yaw": 0.0,
                "R_pitch": 0.0, "R_yaw": 0.0,
                "L_det": False, "R_det": False,
            }

        self.last_data = {
            "top": _empty(), "bottom": _empty(),
            "session_count": 0,
            "lighting": {"status": "no_frame", "mean": 0.0, "std": 0.0},
        }

        self.is_running    = True
        self.current_frame = None

        self._style()
        self.setup_ui()
        threading.Thread(target=self.measurement_loop, daemon=True).start()
        self.update_gui_loop()

    # ══════════════════════════════════════════════════════════════════════════
    #  Style
    # ══════════════════════════════════════════════════════════════════════════
    def _style(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TNotebook",
                    background=C_BG, borderwidth=0, tabmargins=[0, 0, 0, 0])
        s.configure("TNotebook.Tab",
                    background=C_SURF, foreground=C_MUTED,
                    padding=[18, 9], font=("Helvetica", 10, "bold"),
                    borderwidth=0)
        s.map("TNotebook.Tab",
              background=[("selected", C_SURF2)],
              foreground=[("selected", C_TEXT)])
        s.configure("TFrame", background=C_BG)
        s.configure("Dark.Horizontal.TProgressbar",
                    troughcolor=C_SURF, background=C_GREEN,
                    lightcolor=C_GREEN, darkcolor=C_GREEN,
                    bordercolor=C_BORDER, thickness=6)

    # ══════════════════════════════════════════════════════════════════════════
    #  UI wiring — 5 tabs
    # ══════════════════════════════════════════════════════════════════════════
    def setup_ui(self):
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True, padx=0, pady=0)
        self._build_tab_live()
        self._build_tab_warnings()
        self._build_tab_tele()
        self._build_tab_settings()
        self._build_tab_cam()

    # ── Low-level UI factories ────────────────────────────────────────────────
    def _section(self, parent, title, accent=C_TEAL, **pack_kw):
        """A titled surface card."""
        outer = tk.Frame(parent, bg=C_BORDER, padx=1, pady=1)
        outer.pack(**pack_kw)
        inner = tk.Frame(outer, bg=C_SURF, padx=14, pady=10)
        inner.pack(fill="both", expand=True)
        if title:
            hdr = tk.Frame(inner, bg=C_SURF)
            hdr.pack(fill="x", pady=(0, 8))
            tk.Label(hdr, text=title.upper(),
                     font=("Helvetica", 8, "bold"), fg=accent,
                     bg=C_SURF, letterSpacing=2).pack(side="left")
            tk.Frame(hdr, bg=C_BORDER, height=1).pack(
                side="left", fill="x", expand=True, padx=(8, 0), pady=6)
        return inner

    def _h_divider(self, parent):
        tk.Frame(parent, bg=C_BORDER, height=1).pack(fill="x", pady=8)

    def _label(self, parent, text, font=("Helvetica", 10), fg=C_TEXT, bg=C_SURF, **kw):
        return tk.Label(parent, text=text, font=font, fg=fg, bg=bg, **kw)

    def _chip_row(self, parent, description):
        """
        Returns (chip_label, value_label).
        chip_label background/text updated via _set_chip().
        """
        row = tk.Frame(parent, bg=C_SURF2, pady=0)
        row.pack(fill="x", pady=2)
        # Left colour strip
        strip = tk.Frame(row, bg=C_MUTED, width=4)
        strip.pack(side="left", fill="y")
        # Chip
        chip = tk.Label(row, text=" ──  ", font=("Courier", 9, "bold"),
                        fg=C_BG, bg=C_MUTED, padx=6, pady=5, width=7)
        chip.pack(side="left", padx=(6, 0))
        # Description
        tk.Label(row, text=description, font=("Helvetica", 10),
                 fg=C_TEXT, bg=C_SURF2, anchor="w").pack(
            side="left", fill="x", expand=True, padx=10)
        # Value
        val = tk.Label(row, text="—", font=("Courier", 10, "bold"),
                       fg=C_MUTED, bg=C_SURF2, width=20, anchor="e")
        val.pack(side="right", padx=10, pady=5)
        # Keep strip reference on chip for colour sync
        chip._strip = strip
        return chip, val

    def _set_chip(self, chip, val, state, value_text=""):
        bg, fg, label = CHIP[state]
        chip.config(bg=bg, fg=fg, text=label)
        chip._strip.config(bg=bg)
        col = bg if state != "wait" else C_MUTED
        val.config(text=value_text, fg=col)

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 1 — MOVEMENT
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_live(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="  📏  Movement  ")
        tab.configure(style="TFrame")

        # ── Left: camera pane ─────────────────────────────────────────────────
        cam_pane = tk.Frame(tab, bg=C_BG)
        cam_pane.pack(side="left", fill="y", padx=(14, 8), pady=14)

        # Canvas with coloured border
        cam_border = tk.Frame(cam_pane, bg=C_BORDER, padx=1, pady=1)
        cam_border.pack()
        self.canvas = tk.Canvas(cam_border, width=960, height=540,
                                bg="#000000", highlightthickness=0)
        self.canvas.pack()

        # Warning strip — full-width below camera
        self.warn_strip = tk.Label(
            cam_pane, text="  Waiting for camera…",
            font=("Helvetica", 10, "bold"),
            fg=C_MUTED, bg=C_SURF, anchor="w",
            padx=12, pady=6)
        self.warn_strip.pack(fill="x", pady=(5, 0))

        # ── Right: control panel ──────────────────────────────────────────────
        ctrl = tk.Frame(tab, bg=C_BG)
        ctrl.pack(side="right", fill="both", expand=True, padx=(0, 14), pady=14)

        # Distance cards
        for key, label, accent in [("top", "TOP PAIR", C_TOP),
                                    ("bottom", "BOTTOM PAIR", C_BOT)]:
            card = self._section(ctrl, label, accent=accent, fill="x", pady=(0, 6))
            dist_lbl = tk.Label(card, text="─ ─ ─", font=("Helvetica", 34, "bold"),
                                fg=accent, bg=C_SURF, anchor="center")
            dist_lbl.pack(fill="x")
            k_lbl = tk.Label(card, text="k: ─ ─ ─ ─",
                             font=("Courier", 10), fg=C_MUTED, bg=C_SURF, anchor="center")
            k_lbl.pack()
            tk.Frame(card, bg=C_BORDER, height=1).pack(fill="x", pady=(6, 4))
            pair_status = tk.Label(card, text="● Waiting for data…",
                                   font=("Helvetica", 9), fg=C_MUTED, bg=C_SURF, anchor="w")
            pair_status.pack(fill="x")
            if key == "top":
                self.lbl_dist_top   = dist_lbl
                self.lbl_k_top      = k_lbl
                self.lbl_status_top = pair_status
            else:
                self.lbl_dist_bot   = dist_lbl
                self.lbl_k_bot      = k_lbl
                self.lbl_status_bot = pair_status

        # Movement status + progress
        st_card = self._section(ctrl, "Capture State", fill="x", pady=(0, 6))
        self.mv_status_lbl = tk.Label(
            st_card, text="Press  START  to capture the initial gap",
            font=("Helvetica", 11), fg=C_MUTED, bg=C_SURF,
            wraplength=310, justify="center", anchor="center")
        self.mv_status_lbl.pack(fill="x", pady=(0, 6))
        prog_row = tk.Frame(st_card, bg=C_SURF); prog_row.pack(fill="x")
        self.mv_prog_lbl = tk.Label(prog_row, text="",
                                    font=("Helvetica", 9), fg=C_TEAL, bg=C_SURF)
        self.mv_prog_lbl.pack(side="right")
        self.mv_prog_bar = ttk.Progressbar(
            prog_row, length=200, maximum=COLLECT_N, mode="determinate",
            style="Dark.Horizontal.TProgressbar")
        self.mv_prog_bar.pack(side="left", fill="x", expand=True)

        # Buttons
        btn_card = self._section(ctrl, "", fill="x", pady=(0, 6))
        btn_r = tk.Frame(btn_card, bg=C_SURF); btn_r.pack(fill="x")
        _bc = dict(font=("Helvetica", 12, "bold"), relief="flat",
                   bd=0, cursor="hand2", pady=10)
        self.btn_start = tk.Button(btn_r, text="▶  START",
                                   bg=C_GREEN, fg=C_BG, activebackground="#2ea043",
                                   command=self._mv_start, **_bc)
        self.btn_start.pack(side="left", expand=True, fill="x", padx=(0, 4))
        self.btn_stop  = tk.Button(btn_r, text="■  STOP",
                                   bg=C_RED, fg=C_WHITE, activebackground="#da3633",
                                   state="disabled",
                                   command=self._mv_stop,  **_bc)
        self.btn_stop.pack(side="left", expand=True, fill="x", padx=(0, 4))
        self.btn_reset = tk.Button(btn_r, text="↺",
                                   bg=C_SURF2, fg=C_MUTED, activebackground=C_BORDER,
                                   command=self._mv_reset, **_bc)
        self.btn_reset.pack(side="left", padx=0)

        # Distance moved section
        delta_card = self._section(ctrl, "Distance Moved", fill="x", pady=(0, 6))
        for key, label, accent in [("top", "TOP", C_TOP), ("bottom", "BOT", C_BOT)]:
            drow = tk.Frame(delta_card, bg=C_SURF2, pady=0)
            drow.pack(fill="x", pady=2)
            tk.Frame(drow, bg=accent, width=4).pack(side="left", fill="y")
            tk.Label(drow, text=label, font=("Helvetica", 9, "bold"),
                     fg=accent, bg=C_SURF2, width=5).pack(side="left", padx=(8, 4), pady=6)
            sub = tk.Frame(drow, bg=C_SURF2); sub.pack(side="left", fill="x", expand=True)
            il = tk.Label(sub, text="Init:  —", font=("Helvetica", 9),
                          fg=C_MUTED, bg=C_SURF2, anchor="w")
            il.pack(fill="x", padx=2)
            fl = tk.Label(sub, text="Final: —", font=("Helvetica", 9),
                          fg=C_MUTED, bg=C_SURF2, anchor="w")
            fl.pack(fill="x", padx=2)
            dl = tk.Label(drow, text="—", font=("Helvetica", 22, "bold"),
                          fg=C_BLUE, bg=C_SURF2)
            dl.pack(side="right", padx=12, pady=6)
            if key == "top":
                self.mv_init_lbl_top  = il
                self.mv_final_lbl_top = fl
                self.mv_delta_lbl_top = dl
            else:
                self.mv_init_lbl_bot  = il
                self.mv_final_lbl_bot = fl
                self.mv_delta_lbl_bot = dl

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 2 — WARNINGS DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_warnings(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="  ⚠  Warnings  ")
        tab.configure(style="TFrame")

        root_f = tk.Frame(tab, bg=C_BG)
        root_f.pack(fill="both", expand=True, padx=16, pady=16)

        # ── Top bar: overall status + tilt threshold controls ─────────────────
        top_bar = tk.Frame(root_f, bg=C_BG)
        top_bar.pack(fill="x", pady=(0, 12))

        # Overall health pill
        self.overall_frame = tk.Frame(top_bar, bg=C_MUTED, padx=14, pady=8)
        self.overall_frame.pack(side="left")
        self.overall_icon = tk.Label(self.overall_frame, text="⬤",
                                     font=("Helvetica", 18), fg=C_MUTED, bg=C_MUTED)
        self.overall_icon.pack(side="left", padx=(0, 8))
        self.overall_lbl  = tk.Label(self.overall_frame, text="No data yet",
                                     font=("Helvetica", 12, "bold"),
                                     fg=C_BG, bg=C_MUTED, anchor="w")
        self.overall_lbl.pack(side="left")

        # Threshold controls (right side of top bar)
        thr_f = tk.Frame(top_bar, bg=C_BG)
        thr_f.pack(side="right")

        def _thr_ctrl(parent, label, var, accent):
            f = tk.Frame(parent, bg=C_SURF, padx=10, pady=6)
            f.pack(side="left", padx=(8, 0))
            tk.Label(f, text=label, font=("Helvetica", 8, "bold"),
                     fg=accent, bg=C_SURF).pack(anchor="w")
            row = tk.Frame(f, bg=C_SURF); row.pack(fill="x")
            sl = tk.Scale(row, from_=1, to=45, resolution=0.5, orient="horizontal",
                          variable=var, bg=C_SURF, fg=C_TEXT, troughcolor=C_BG,
                          activebackground=accent, highlightthickness=0,
                          showvalue=False, length=140)
            sl.pack(side="left")
            tk.Label(row, textvariable=var, font=("Courier", 10, "bold"),
                     fg=accent, bg=C_SURF, width=4).pack(side="left", padx=(4, 0))
            tk.Label(row, text="°", font=("Helvetica", 10),
                     fg=C_MUTED, bg=C_SURF).pack(side="left")

        _thr_ctrl(thr_f, "PITCH THRESHOLD", self.pitch_threshold, C_AMBER)
        _thr_ctrl(thr_f, "YAW THRESHOLD",   self.yaw_threshold,   C_TOP)

        # ── Pair columns ──────────────────────────────────────────────────────
        cols = tk.Frame(root_f, bg=C_BG)
        cols.pack(fill="both", expand=True)

        self._warn_refs = {}

        for i, (pair_key, label, accent) in enumerate(
                [("top", "TOP PAIR", C_TOP), ("bottom", "BOTTOM PAIR", C_BOT)]):
            col = self._section(cols, label, accent=accent,
                                side="left", fill="both", expand=True,
                                padx=(0, 8) if i == 0 else 0)

            refs = {}
            defs = [
                ("L_det",   "Left Marker Detected",               "Detection"),
                ("R_det",   "Right Marker Detected",              "Detection"),
                ("rot",     "In-Plane Rotation",                  "Rotation"),
                ("L_pitch", "Left Marker — Pitch  (fwd / bwd)",   "Tilt"),
                ("R_pitch", "Right Marker — Pitch  (fwd / bwd)",  "Tilt"),
                ("L_yaw",   "Left Marker — Yaw  (left / right)",  "Tilt"),
                ("R_yaw",   "Right Marker — Yaw  (left / right)", "Tilt"),
            ]
            current_group = None
            for ref_key, desc, group in defs:
                if group != current_group:
                    current_group = group
                    tk.Label(col, text=group.upper(),
                             font=("Helvetica", 7, "bold"), fg=C_MUTED, bg=C_SURF,
                             anchor="w").pack(fill="x", padx=2, pady=(6, 1))
                chip, val = self._chip_row(col, desc)
                refs[ref_key] = (chip, val)

            # Euler angle display panel
            tk.Frame(col, bg=C_BORDER, height=1).pack(fill="x", pady=(10, 6))
            tk.Label(col, text="LIVE ANGLES (°)",
                     font=("Helvetica", 7, "bold"), fg=C_MUTED, bg=C_SURF, anchor="w").pack(fill="x")
            ang_grid = tk.Frame(col, bg=C_SURF); ang_grid.pack(fill="x", pady=4)
            ang_vars = {}
            for ci, (ak, albl, acol) in enumerate([
                ("L_pitch", "L-Pitch", C_AMBER), ("L_yaw",  "L-Yaw",   C_TOP),
                ("L_roll",  "L-Roll",  C_MUTED), ("R_pitch", "R-Pitch", C_AMBER),
                ("R_yaw",   "R-Yaw",   C_TOP),   ("R_roll",  "R-Roll",  C_MUTED),
            ]):
                cell = tk.Frame(ang_grid, bg=C_SURF2, padx=6, pady=6)
                cell.grid(row=0, column=ci, padx=2, pady=0, sticky="nsew")
                ang_grid.columnconfigure(ci, weight=1)
                tk.Label(cell, text=albl, font=("Helvetica", 7, "bold"),
                         fg=acol, bg=C_SURF2).pack()
                sv = tk.StringVar(value="—")
                tk.Label(cell, textvariable=sv, font=("Courier", 12, "bold"),
                         fg=C_TEXT, bg=C_SURF2).pack()
                ang_vars[ak] = sv
            refs["_ang"] = ang_vars
            self._warn_refs[pair_key] = refs

        # ── Lighting section (full width, below columns) ──────────────────────
        light_sec = self._section(root_f, "Lighting", accent=C_BLUE,
                                  fill="x", pady=(10, 0))
        self._light_brt_chip, self._light_brt_val = \
            self._chip_row(light_sec, "Frame Brightness  (target: 40 – 215)")
        self._light_ctr_chip, self._light_ctr_val = \
            self._chip_row(light_sec, "Frame Contrast — Std Dev  (target: > 18)")

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 3 — DUAL TELEMETRY
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_tele(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="  🛸  Telemetry  ")
        tab.configure(style="TFrame")

        outer = tk.Frame(tab, bg=C_BG)
        outer.pack(fill="both", expand=True, padx=16, pady=16)

        self.tele_vars = {"top": {}, "bottom": {}}
        v_show = [
            ("A",       "Point A  (source mid-edge)"),
            ("X",       "Point X  (closest on target)"),
            ("TR",      "TR  (inner-top corner)"),
            ("BR",      "BR  (inner-bot corner)"),
            ("B",       "Point B  (target top)"),
            ("C",       "Point C  (target bot)"),
            ("L_ROL",   "Left Roll  (in-plane)"),
            ("L_ROT",   "Left Rotation  (2-D)"),
            ("L_Z",     "Left Z depth"),
            ("L_PITCH", "Left Pitch  (fwd/bwd)"),
            ("L_YAW",   "Left Yaw  (left/right)"),
            ("R_ROL",   "Right Roll  (in-plane)"),
            ("R_ROT",   "Right Rotation  (2-D)"),
            ("R_Z",     "Right Z depth"),
            ("R_PITCH", "Right Pitch  (fwd/bwd)"),
            ("R_YAW",   "Right Yaw  (left/right)"),
        ]
        for pair_key, title, accent in [("top", "TOP PAIR", C_TOP),
                                         ("bottom", "BOTTOM PAIR", C_BOT)]:
            sec = self._section(outer, title, accent=accent,
                                side="left", fill="both", expand=True,
                                padx=(0, 8) if pair_key == "top" else 0)
            for v_key, v_label in v_show:
                row = tk.Frame(sec, bg=C_SURF2)
                row.pack(fill="x", pady=1)
                tk.Label(row, text=v_label, font=("Helvetica", 9),
                         fg=C_MUTED, bg=C_SURF2, anchor="w", width=30).pack(
                    side="left", padx=8, pady=4)
                sv = tk.StringVar(value="—")
                tk.Label(row, textvariable=sv, font=("Courier", 10, "bold"),
                         fg=C_TEXT, bg=C_SURF2, anchor="e").pack(
                    side="right", padx=8)
                self.tele_vars[pair_key][v_key] = sv

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 4 — MACHINE CONFIGURATION
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_settings(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="  ⚙  Configuration  ")
        tab.configure(style="TFrame")

        outer = tk.Frame(tab, bg=C_BG)
        outer.pack(fill="both", expand=True, padx=60, pady=30)

        # Reference side
        ref_sec = self._section(outer, "Reference Side  (Fixed Marker)", fill="x", pady=(0, 10))
        rb_row = tk.Frame(ref_sec, bg=C_SURF); rb_row.pack(fill="x")
        for choice in ["Left", "Right"]:
            tk.Radiobutton(rb_row, text=f"  {choice}  ", variable=self.fixed_side,
                           value=choice, bg=C_SURF, fg=C_TEXT,
                           selectcolor=C_BLUE, activebackground=C_SURF,
                           activeforeground=C_TEXT, indicatoron=0,
                           font=("Helvetica", 11, "bold"),
                           relief="flat", padx=20, pady=8).pack(side="left", padx=(0, 8))

        # Slider helper
        def _slider_block(parent, title, desc, var, accent, lo, hi, res=0.1):
            sec = self._section(parent, title, accent=accent, fill="x", pady=(0, 10))
            tk.Label(sec, text=desc, font=("Helvetica", 9), fg=C_MUTED,
                     bg=C_SURF, anchor="w").pack(fill="x", pady=(0, 6))
            row = tk.Frame(sec, bg=C_SURF); row.pack(fill="x")
            sl = tk.Scale(row, from_=lo, to=hi, resolution=res, orient="horizontal",
                          variable=var, bg=C_SURF, fg=C_TEXT, troughcolor=C_BG,
                          activebackground=accent, highlightthickness=0,
                          showvalue=False, length=480)
            sl.pack(side="left")
            val_lbl = tk.Label(row, textvariable=var, font=("Courier", 12, "bold"),
                               fg=accent, bg=C_SURF, width=7)
            val_lbl.pack(side="left", padx=(8, 4))
            tk.Label(row, text="mm" if lo >= 10 else "°",
                     font=("Helvetica", 10), fg=C_MUTED, bg=C_SURF).pack(side="left")
            ent = tk.Entry(row, textvariable=var, width=8,
                           font=("Courier", 10), bg=C_SURF2, fg=C_TEXT,
                           insertbackground=C_TEXT, relief="flat",
                           highlightbackground=C_BORDER, highlightthickness=1)
            ent.pack(side="right", padx=(8, 0))
            return sl

        _slider_block(outer, "Upper Pair — Marker Size",
                      "Physical side length of the TOP pair ArUco markers.",
                      self.size_top, C_TOP, 10, 200)
        _slider_block(outer, "Bottom Pair — Marker Size",
                      "Physical side length of the BOTTOM pair ArUco markers.",
                      self.size_bot, C_BOT, 10, 200)
        self.rot_threshold_slider = _slider_block(
            outer, "Rotation Threshold",
            "In-plane 2-D rotation limit before formula falls back to midpoint.",
            self.rot_threshold, C_MUTED, 0, 45)
        _slider_block(outer, "Pitch Threshold",
                      "Maximum forward/backward tilt (X-axis rotation) before warning.",
                      self.pitch_threshold, C_AMBER, 1, 45, res=0.5)
        _slider_block(outer, "Yaw Threshold",
                      "Maximum left/right tilt (Y-axis rotation) before warning — "
                      "catches the 'one side in, one side out' condition.",
                      self.yaw_threshold, C_TOP, 1, 45, res=0.5)

        # Angle-threshold mode toggle
        tog_sec = self._section(outer, "Angle Threshold Mode", fill="x", pady=(0, 0))
        tog_row = tk.Frame(tog_sec, bg=C_SURF); tog_row.pack(fill="x")
        self._tog_canvas = tk.Canvas(tog_row, width=52, height=26, bg=C_SURF,
                                     highlightthickness=0, cursor="hand2")
        self._tog_canvas.pack(side="left", padx=(0, 14))
        self._tog_lbl = tk.Label(tog_row, text="", font=("Helvetica", 10),
                                 bg=C_SURF, fg=C_TEXT, anchor="w")
        self._tog_lbl.pack(side="left", fill="x", expand=True)

        def _draw_toggle():
            self._tog_canvas.delete("all")
            on = self.use_angle_thresh.get()
            bg = C_GREEN if on else C_MUTED
            self._tog_canvas.create_oval(1, 1, 51, 25, fill=bg, outline="")
            cx = 37 if on else 15
            self._tog_canvas.create_oval(cx-11, 3, cx+11, 23, fill=C_WHITE, outline="")

        def _update_tog(*_):
            on = self.use_angle_thresh.get()
            self._tog_lbl.config(
                text="ON — perpendicular formula up to threshold, midpoint above" if on
                     else "OFF — always perpendicular formula (v8 behaviour)",
                fg=C_GREEN if on else C_RED)
            self.rot_threshold_slider.config(state="normal" if on else "disabled")

        def _toggle(_=None):
            self.use_angle_thresh.set(not self.use_angle_thresh.get())
            _draw_toggle(); _update_tog()

        self._tog_canvas.bind("<Button-1>", _toggle)
        _draw_toggle(); _update_tog()

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 5 — CAMERA CONTROLS
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_cam(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="  📷  Camera  ")
        tab.configure(style="TFrame")

        main = tk.Frame(tab, bg=C_BG)
        main.pack(fill="both", expand=True, padx=14, pady=14)

        # ── Left: preview ─────────────────────────────────────────────────────
        pv = self._section(main, "Live Preview", side="left",
                           fill="both", expand=True, padx=(0, 8), pady=0)
        border = tk.Frame(pv, bg=C_BORDER, padx=1, pady=1)
        border.pack(expand=True)
        self.cam_canvas = tk.Canvas(border, width=800, height=480,
                                    bg="black", highlightthickness=0)
        self.cam_canvas.pack()
        self.lbl_cam_status = tk.Label(
            pv, text="⏳  Waiting for camera…",
            font=("Helvetica", 11, "bold"), fg=C_AMBER, bg=C_SURF)
        self.lbl_cam_status.pack(pady=(8, 2), anchor="center")
        self.lbl_lighting_cam = tk.Label(pv, text="",
                                         font=("Helvetica", 9), fg=C_MUTED, bg=C_SURF)
        self.lbl_lighting_cam.pack(pady=(0, 4), anchor="center")

        # ── Right: controls ───────────────────────────────────────────────────
        ctl = tk.Frame(main, bg=C_BG)
        ctl.pack(side="right", fill="y")

        def _cam_row(parent, label, var, lo, hi, res, unit=""):
            sec = self._section(parent, label, fill="x", pady=(0, 6))
            row = tk.Frame(sec, bg=C_SURF); row.pack(fill="x")
            tk.Scale(row, from_=lo, to=hi, resolution=res,
                     orient="horizontal", variable=var,
                     bg=C_SURF, fg=C_TEXT, troughcolor=C_BG,
                     activebackground=C_BLUE, highlightthickness=0,
                     showvalue=False, length=210,
                     command=lambda _: self._apply_cam()).pack(side="left")
            tk.Label(row, textvariable=var, font=("Courier", 9, "bold"),
                     fg=C_BLUE, bg=C_SURF, width=7).pack(side="left", padx=(4, 2))
            if unit:
                tk.Label(row, text=unit, font=("Helvetica", 8),
                         fg=C_MUTED, bg=C_SURF).pack(side="left")

        ae_sec = self._section(ctl, "Exposure", fill="x", pady=(0, 6))
        tk.Checkbutton(ae_sec, text="Auto Exposure (AE)",
                       variable=self.cam_ae, bg=C_SURF, fg=C_TEXT,
                       selectcolor=C_SURF2, activebackground=C_SURF,
                       activeforeground=C_TEXT, font=("Helvetica", 10),
                       command=self._apply_cam).pack(anchor="w")
        _cam_row(ctl, "Exposure Time", self.cam_exposure, 100,  66000, 100, "µs")
        _cam_row(ctl, "ISO / Gain",    self.cam_gain,     1.0,  16.0,  0.1, "×")

        awb_sec = self._section(ctl, "White Balance", fill="x", pady=(0, 6))
        tk.Checkbutton(awb_sec, text="Auto White Balance",
                       variable=self.cam_awb, bg=C_SURF, fg=C_TEXT,
                       selectcolor=C_SURF2, activebackground=C_SURF,
                       activeforeground=C_TEXT, font=("Helvetica", 10),
                       command=self._apply_cam).pack(anchor="w")
        wbr = tk.Frame(awb_sec, bg=C_SURF); wbr.pack(fill="x", pady=(4, 0))
        tk.Label(wbr, text="Mode:", font=("Helvetica", 9),
                 fg=C_MUTED, bg=C_SURF).pack(side="left")
        for mode_name in AWB_MODES:
            tk.Radiobutton(wbr, text=mode_name, variable=self.cam_awb_mode,
                           value=mode_name, bg=C_SURF, fg=C_TEXT,
                           selectcolor=C_SURF2, activebackground=C_SURF,
                           font=("Helvetica", 8),
                           command=self._apply_cam).pack(side="left", padx=2)

        _cam_row(ctl, "Brightness", self.cam_brightness, -1.0, 1.0, 0.05)
        _cam_row(ctl, "Contrast",   self.cam_contrast,    0.0, 8.0, 0.1)
        _cam_row(ctl, "Saturation", self.cam_saturation,  0.0, 8.0, 0.1)
        _cam_row(ctl, "Sharpness",  self.cam_sharpness,   0.0, 8.0, 0.1)

        tk.Button(ctl, text="↺  Reset All to Defaults",
                  font=("Helvetica", 10, "bold"), bg=C_RED, fg=C_WHITE,
                  activebackground="#da3633", relief="flat", pady=9, cursor="hand2",
                  command=self._reset_cam).pack(fill="x", pady=(6, 0))

    # ══════════════════════════════════════════════════════════════════════════
    #  Movement monitor logic
    # ══════════════════════════════════════════════════════════════════════════
    def _mv_start(self):
        if self.mv_state not in ("idle", "done"):
            return
        self.mv_state    = "collecting_init"
        self.mv_init_buf = {"top": [], "bottom": []}
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="disabled")
        self.mv_prog_bar["value"] = 0
        self.mv_status_lbl.config(
            text=f"⏳  Collecting initial distance…  0 / {COLLECT_N}", fg=C_AMBER)

    def _mv_stop(self):
        if self.mv_state != "ready":
            return
        self.mv_state     = "collecting_final"
        self.mv_final_buf = {"top": [], "bottom": []}
        self.btn_stop.config(state="disabled")
        self.mv_prog_bar["value"] = 0
        self.mv_status_lbl.config(
            text=f"⏳  Collecting final distance…  0 / {COLLECT_N}", fg=C_AMBER)

    def _mv_reset(self):
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
            text="Press  START  to capture the initial gap", fg=C_MUTED)
        for lbl in (self.mv_init_lbl_top, self.mv_init_lbl_bot):
            lbl.config(text="Init:  —", fg=C_MUTED)
        for lbl in (self.mv_final_lbl_top, self.mv_final_lbl_bot):
            lbl.config(text="Final: —", fg=C_MUTED)
        for lbl in (self.mv_delta_lbl_top, self.mv_delta_lbl_bot):
            lbl.config(text="—", fg=C_BLUE)

    def _mv_tick(self):
        sc = self.last_data["session_count"]
        if sc == self._last_sc:
            return
        self._last_sc = sc

        if self.mv_state == "collecting_init":
            for key in ["top", "bottom"]:
                d = self.last_data[key]["dist"]
                if d > 0:
                    self.mv_init_buf[key].append(d)
            n = min(len(self.mv_init_buf["top"]), len(self.mv_init_buf["bottom"]))
            self.mv_prog_bar["value"] = n
            self.mv_prog_lbl.config(text=f"{n} / {COLLECT_N}")
            self.mv_status_lbl.config(
                text=f"⏳  Collecting initial…  {n} / {COLLECT_N}", fg=C_AMBER)
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
                self.mv_init_lbl_top.config(
                    text=f"Init:  {self.mv_dist_init['top']:.3f} mm", fg=C_TEXT)
                self.mv_init_lbl_bot.config(
                    text=f"Init:  {self.mv_dist_init['bottom']:.3f} mm", fg=C_TEXT)

        elif self.mv_state == "collecting_final":
            for key in ["top", "bottom"]:
                d = self.last_data[key]["dist"]
                if d > 0:
                    self.mv_final_buf[key].append(d)
            n = min(len(self.mv_final_buf["top"]), len(self.mv_final_buf["bottom"]))
            self.mv_prog_bar["value"] = n
            self.mv_prog_lbl.config(text=f"{n} / {COLLECT_N}")
            self.mv_status_lbl.config(
                text=f"⏳  Collecting final…  {n} / {COLLECT_N}", fg=C_AMBER)
            if n >= COLLECT_N:
                self.mv_dist_final = {
                    "top":    float(np.mean(self.mv_final_buf["top"][:COLLECT_N])),
                    "bottom": float(np.mean(self.mv_final_buf["bottom"][:COLLECT_N])),
                }
                self.mv_state = "done"
                self.btn_start.config(state="normal")
                self.mv_prog_bar["value"] = 0
                self.mv_prog_lbl.config(text="")
                self.mv_status_lbl.config(
                    text="✅  Measurement complete — press ↺ to start over", fg=C_GREEN)
                for key, il, fl, dl in [
                    ("top",    self.mv_init_lbl_top,  self.mv_final_lbl_top,  self.mv_delta_lbl_top),
                    ("bottom", self.mv_init_lbl_bot,  self.mv_final_lbl_bot,  self.mv_delta_lbl_bot),
                ]:
                    di = self.mv_dist_init[key]; df = self.mv_dist_final[key]
                    delta = df - di
                    il.config(text=f"Init:  {di:.3f} mm", fg=C_TEXT)
                    fl.config(text=f"Final: {df:.3f} mm", fg=C_TEXT)
                    sign  = "+" if delta >= 0 else ""
                    col   = C_RED if delta > 0.5 else (C_GREEN if delta < -0.5 else C_BLUE)
                    dl.config(text=f"{sign}{delta:.3f} mm", fg=col)

    # ══════════════════════════════════════════════════════════════════════════
    #  Camera helpers
    # ══════════════════════════════════════════════════════════════════════════
    def _apply_cam(self, *_):
        if self.pc is None:
            return
        controls = {}
        ae = self.cam_ae.get(); controls["AeEnable"] = ae
        if not ae:
            controls["ExposureTime"] = int(self.cam_exposure.get())
            controls["AnalogueGain"] = float(self.cam_gain.get())
        awb = self.cam_awb.get(); controls["AwbEnable"] = awb
        if not awb:
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
        self.cam_ae.set(True);  self.cam_exposure.set(10000)
        self.cam_gain.set(2.0); self.cam_awb.set(True)
        self.cam_awb_mode.set("Auto"); self.cam_brightness.set(0.0)
        self.cam_contrast.set(1.0);    self.cam_saturation.set(1.0)
        self.cam_sharpness.set(1.0);   self._apply_cam()

    # ══════════════════════════════════════════════════════════════════════════
    #  Calibration
    # ══════════════════════════════════════════════════════════════════════════
    def load_calib(self):
        f = "camera_params_2.npz" if self.fixed_side.get() == "Right" else "camera_params.npz"
        if os.path.exists(f):
            d = np.load(f)
            return d['camera_matrix'], d['dist_coeff']
        return (np.array([[1280, 0, 640], [0, 1280, 360], [0, 0, 1]], dtype=np.float32),
                np.zeros(5))

    # ══════════════════════════════════════════════════════════════════════════
    #  Background measurement loop
    # ══════════════════════════════════════════════════════════════════════════
    def measurement_loop(self):
        try:
            from picamera2 import Picamera2
            pc = Picamera2()
            pc.configure(pc.create_video_configuration(
                main={"size": RESOLUTION, "format": "RGB888"}))
            pc.start()
            self.pc = pc
        except Exception:
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
            except Exception:
                continue

            # Lighting check (every frame)
            gray_full = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            mean_b = float(np.mean(gray_full))
            std_b  = float(np.std(gray_full))
            self.last_data["lighting"] = {
                "status": ("dark"   if mean_b < LIGHT_LOW  else
                           "bright" if mean_b > LIGHT_HIGH else
                           "flat"   if std_b  < CONTRAST_MIN else "ok"),
                "mean": mean_b, "std": std_b,
            }

            corners_raw, ids, _ = detector.detectMarkers(gray_full)
            curr = time.time() * 1000

            if ids is not None and len(ids) >= 1:
                m_data = []
                for i in range(len(ids)):
                    raw  = corners_raw[i][0]
                    ix   = np.argsort(raw[:, 0])
                    lp   = raw[ix[:2]]; rp = raw[ix[2:]]
                    tl   = lp[np.argmin(lp[:, 1])]; bl = lp[np.argmax(lp[:, 1])]
                    tr   = rp[np.argmin(rp[:, 1])]; br = rp[np.argmax(rp[:, 1])]
                    cy   = (tl[1]+tr[1]+br[1]+bl[1]) / 4.0
                    cx   = (tl[0]+tr[0]+br[0]+bl[0]) / 4.0
                    m_data.append({"c": np.array([tl, tr, br, bl], dtype=np.float32),
                                   "c_raw": raw, "y": cy, "x": cx})

                m_data.sort(key=lambda m: m["y"])
                top_m = m_data[:2]; bot_m = m_data[2:4] if len(m_data) >= 4 else []
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

                def proc(marker_list, pair_key, size):
                    if len(marker_list) < 2:
                        self.last_data[pair_key].update(
                            {"dist": 0.0, "p1_px": None, "L_det": False, "R_det": False})
                        return

                    marker_list.sort(key=lambda m: m["x"])
                    left_m = marker_list[0]; right_m = marker_list[-1]
                    is_rf  = (self.fixed_side.get() == "Right")
                    S_m = right_m if is_rf else left_m
                    T_m = left_m  if is_rf else right_m
                    h   = size / 2.0
                    obj = np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]], dtype=np.float32)

                    def get_pose(c_raw):
                        _, rv, tv = cv.solvePnP(obj, c_raw, K, dist_c)
                        R, _ = cv.Rodrigues(rv)
                        pts3d = np.array([R @ pt + tv.ravel() for pt in obj])
                        roll_r = math.degrees(math.atan2(R[1,0], R[0,0]))
                        pitch, yaw, _ = rotation_to_euler(R)
                        return pts3d, roll_r, float(tv[2]), pitch, yaw

                    def irot(cs):
                        return abs(math.degrees(math.atan2(
                            cs[1,1]-cs[0,1], cs[1,0]-cs[0,0])))

                    S_pts, S_roll, S_z, S_pitch, S_yaw = get_pose(S_m["c_raw"])
                    T_pts, T_roll, T_z, T_pitch, T_yaw = get_pose(T_m["c_raw"])
                    S_ir = irot(S_m["c"]); T_ir = irot(T_m["c"])

                    def io(pts, inner_right):
                        o = pts[:,0].argsort()
                        i_ = pts[o[2:]] if inner_right else pts[o[:2]]
                        ou = pts[o[:2]] if inner_right else pts[o[2:]]
                        return i_[i_[:,1].argsort()][0], i_[i_[:,1].argsort()][1], \
                               ou[ou[:,1].argsort()][0]

                    Si_t, Si_b, So_t = io(S_pts, not is_rf)
                    A = (Si_t + Si_b) / 2.0
                    Ti_t, Ti_b, _ = io(T_pts, is_rf)
                    Bv = Ti_t; Cv = Ti_b; X_alt = (Bv + Cv) / 2.0

                    p1 = tuple(((S_m["c"][1]+S_m["c"][2])/2).astype(int)) if not is_rf \
                         else tuple(((S_m["c"][0]+S_m["c"][3])/2).astype(int))
                    p2 = tuple(((T_m["c"][0]+T_m["c"][3])/2).astype(int)) if not is_rf \
                         else tuple(((T_m["c"][1]+T_m["c"][2])/2).astype(int))

                    buffers[pair_key].append({
                        "A": A, "X_alt": X_alt,
                        "TR": Si_t, "BR": Si_b, "TL_ref": So_t,
                        "B": Bv, "C": Cv,
                        "L_A": (S_roll, S_ir), "R_A": (T_roll, T_ir),
                        "rot": max(S_ir, T_ir),
                        "L_z": S_z, "R_z": T_z,
                        "L_pitch": S_pitch, "L_yaw": S_yaw,
                        "R_pitch": T_pitch, "R_yaw": T_yaw,
                        "p1_px": p1, "p2_px": p2,
                    })
                    self.last_data[pair_key]["L_det"] = True
                    self.last_data[pair_key]["R_det"] = True

                if (curr - l_s) >= 100:
                    proc(top_m, "top",    self.size_top.get())
                    proc(bot_m, "bottom", self.size_bot.get())
                    l_s = curr

                if (curr - l_u) >= 1000:
                    for key in ["top", "bottom"]:
                        if buffers[key]:
                            s = buffers[key]
                            aA  = np.mean([x["A"]      for x in s], axis=0)
                            aTR = np.mean([x["TR"]     for x in s], axis=0)
                            aTL = np.mean([x["TL_ref"] for x in s], axis=0)
                            aBR = np.mean([x["BR"]     for x in s], axis=0)
                            aB  = np.mean([x["B"]      for x in s], axis=0)
                            aC  = np.mean([x["C"]      for x in s], axis=0)
                            aXa = np.mean([x["X_alt"]  for x in s], axis=0)
                            aL  = np.mean([x["L_A"]    for x in s], axis=0)
                            aR  = np.mean([x["R_A"]    for x in s], axis=0)
                            avg_rot    = float(np.mean([x["rot"]     for x in s]))
                            avg_Lz     = float(np.mean([x["L_z"]     for x in s]))
                            avg_Rz     = float(np.mean([x["R_z"]     for x in s]))
                            avg_Lp  = float(np.mean([x["L_pitch"] for x in s]))
                            avg_Ly  = float(np.mean([x["L_yaw"]   for x in s]))
                            avg_Rp  = float(np.mean([x["R_pitch"] for x in s]))
                            avg_Ry  = float(np.mean([x["R_yaw"]   for x in s]))
                            p1_px = tuple(np.mean([x["p1_px"] for x in s],axis=0).astype(int))
                            p2_px = tuple(np.mean([x["p2_px"] for x in s],axis=0).astype(int))

                            v_r = aTR - aTL; vl = np.linalg.norm(v_r)
                            v = v_r / vl if vl > 0 else np.zeros(3)
                            w = aC - aB; u = aB - aA
                            den = np.dot(v,v)*np.dot(w,w) - np.dot(v,w)**2

                            def _perp():
                                if abs(den) > 1e-6:
                                    kv_ = (np.dot(v,w)*np.dot(u,v) -
                                           np.dot(v,v)*np.dot(u,w)) / den
                                    kv_ = float(np.clip(kv_, 0.0, 1.0))
                                    aX_ = aB + kv_ * w
                                    return aX_, np.linalg.norm(aX_ - aA), kv_
                                return aXa, np.linalg.norm(aXa - aA), 0.5

                            if self.use_angle_thresh.get():
                                if avg_rot > self.rot_threshold.get():
                                    aX = aXa; dv = np.linalg.norm(aX-aA); kv = 0.5
                                else:
                                    aX, dv, kv = _perp()
                            else:
                                aX, dv, kv = _perp()

                            self.last_data[key].update({
                                "A": aA, "X": aX, "TR": aTR, "BR": aBR,
                                "B": aB, "C": aC, "dist": dv, "k": kv,
                                "rot_2d": avg_rot, "L_A": aL, "R_A": aR,
                                "L_z": avg_Lz, "R_z": avg_Rz,
                                "L_pitch": avg_Lp, "L_yaw": avg_Ly,
                                "R_pitch": avg_Rp, "R_yaw": avg_Ry,
                                "p1_px": p1_px, "p2_px": p2_px,
                            })
                            self.last_data["session_count"] += 1
                            buffers[key].clear()

                            uv = aA; vv = aTR-aTL; wv = aC-aB
                            try:
                                log.record(dv, kv, aA, aX, aTR, aBR, aB, aC, uv, vv, wv, aL, aR)
                            except Exception:
                                pass
                            if self.mv_state in ("collecting_init","ready","collecting_final"):
                                try:
                                    from datetime import datetime
                                    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
                                    d = "captured_images"; os.makedirs(d, exist_ok=True)
                                    cv.imwrite(os.path.join(d, f"mv_{ts}.jpg"), frame)
                                except Exception:
                                    pass
                    l_u = curr
            else:
                for key in ["top", "bottom"]:
                    self.last_data[key].update({"dist": 0.0, "L_det": False, "R_det": False})

            # ── Overlay on frame ──────────────────────────────────────────────
            for key, col in [("top", (55, 135, 240)), ("bottom", (163, 113, 247))]:
                d = self.last_data[key]
                if d["dist"] > 0 and d.get("p1_px") and d.get("p2_px"):
                    p1 = d["p1_px"]
                    X3d = np.array([d["X"]], dtype=np.float64).reshape(1,1,3)
                    if d["X"][2] > 0:
                        pp, _ = cv.projectPoints(X3d, np.zeros(3), np.zeros(3), K, dist_c)
                        p2 = tuple(pp[0].ravel().astype(int))
                    else:
                        p2 = d["p2_px"]
                    cv.line(frame, p1, p2, col, 3)
                    cv.circle(frame, p2, 7, (60, 230, 120), -1)
                    cv.putText(frame, f"{key.upper()}: {d['dist']:.2f} mm",
                               (p1[0], p1[1]-14), cv.FONT_HERSHEY_SIMPLEX, 0.62, col, 2)

                    # Warning badge
                    p_th = self.pitch_threshold.get()
                    y_th = self.yaw_threshold.get()
                    r_th = self.rot_threshold.get()
                    wl = []
                    mr = max(d["L_A"][1], d["R_A"][1])
                    if mr > r_th: wl.append(f"ROT {mr:.1f}d")
                    lp = d.get("L_pitch", 0.0)
                    if abs(lp) > p_th: wl.append(f"L-P {lp:+.1f}d {'F' if lp>0 else 'B'}")
                    rp = d.get("R_pitch", 0.0)
                    if abs(rp) > p_th: wl.append(f"R-P {rp:+.1f}d {'F' if rp>0 else 'B'}")
                    ly = d.get("L_yaw", 0.0)
                    if abs(ly) > y_th: wl.append(f"L-Y {ly:+.1f}d {'R' if ly>0 else 'L'}")
                    ry = d.get("R_yaw", 0.0)
                    if abs(ry) > y_th: wl.append(f"R-Y {ry:+.1f}d {'R' if ry>0 else 'L'}")
                    if wl:
                        bx1 = max(0, p1[0]-4); by1 = max(0, p1[1]-30-22*len(wl))
                        bx2 = bx1 + 210; by2 = by1 + 22*len(wl) + 8
                        cv.rectangle(frame, (bx1,by1), (bx2,by2), (15,15,15), -1)
                        cv.rectangle(frame, (bx1,by1), (bx2,by2), (60,60,200), 1)
                        for li, ln in enumerate(wl):
                            cv.putText(frame, ln, (bx1+5, by1+17+li*22),
                                       cv.FONT_HERSHEY_SIMPLEX, 0.45, (80,210,255), 1)

            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            self.current_frame = rgb; self._cam_preview_frame = rgb

    # ══════════════════════════════════════════════════════════════════════════
    #  Warning dashboard update  (GUI thread)
    # ══════════════════════════════════════════════════════════════════════════
    def _update_warnings(self):
        p_th = self.pitch_threshold.get()
        y_th = self.yaw_threshold.get()
        r_th = self.rot_threshold.get()
        all_ok = True
        any_error = False
        strip_parts = []

        for pair_key in ["top", "bottom"]:
            d   = self.last_data[pair_key]
            ref = self._warn_refs[pair_key]

            def _row(rk, state, value):
                chip, val = ref[rk]
                self._set_chip(chip, val, state, value)

            # Detection
            l_det = d.get("L_det", False)
            r_det = d.get("R_det", False)
            _row("L_det", "ok" if l_det else "error",
                 "Detected" if l_det else "NOT FOUND")
            _row("R_det", "ok" if r_det else "error",
                 "Detected" if r_det else "NOT FOUND")
            if not l_det:
                strip_parts.append(f"[{pair_key.upper()}] L-marker missing")
                any_error = True; all_ok = False
            if not r_det:
                strip_parts.append(f"[{pair_key.upper()}] R-marker missing")
                any_error = True; all_ok = False

            # In-plane rotation
            rot = d.get("rot_2d", 0.0)
            rot_ok = rot <= r_th
            _row("rot", "ok" if rot_ok else "warn", f"{rot:.1f}°")
            if not rot_ok:
                strip_parts.append(f"[{pair_key.upper()}] Rotation {rot:.1f}°")
                all_ok = False

            # Pitch
            lp = d.get("L_pitch", 0.0)
            rp = d.get("R_pitch", 0.0)
            ld = "FWD" if lp > 0 else "BWD"
            rd = "FWD" if rp > 0 else "BWD"
            _row("L_pitch", "ok" if abs(lp) <= p_th else "error",
                 f"{lp:+.1f}°  {ld}" if abs(lp) > p_th else f"{lp:+.1f}°")
            _row("R_pitch", "ok" if abs(rp) <= p_th else "error",
                 f"{rp:+.1f}°  {rd}" if abs(rp) > p_th else f"{rp:+.1f}°")
            if abs(lp) > p_th:
                strip_parts.append(f"[{pair_key.upper()}] L-pitch {lp:+.1f}° {ld}")
                any_error = True; all_ok = False
            if abs(rp) > p_th:
                strip_parts.append(f"[{pair_key.upper()}] R-pitch {rp:+.1f}° {rd}")
                any_error = True; all_ok = False

            # Yaw
            ly = d.get("L_yaw", 0.0)
            ry = d.get("R_yaw", 0.0)
            lyd = "RIGHT" if ly > 0 else "LEFT"
            ryd = "RIGHT" if ry > 0 else "LEFT"
            _row("L_yaw", "ok" if abs(ly) <= y_th else "error",
                 f"{ly:+.1f}°  {lyd}" if abs(ly) > y_th else f"{ly:+.1f}°")
            _row("R_yaw", "ok" if abs(ry) <= y_th else "error",
                 f"{ry:+.1f}°  {ryd}" if abs(ry) > y_th else f"{ry:+.1f}°")
            if abs(ly) > y_th:
                strip_parts.append(f"[{pair_key.upper()}] L-yaw {ly:+.1f}° {lyd}")
                any_error = True; all_ok = False
            if abs(ry) > y_th:
                strip_parts.append(f"[{pair_key.upper()}] R-yaw {ry:+.1f}° {ryd}")
                any_error = True; all_ok = False

            # Angle cells
            ang = ref["_ang"]
            ang["L_pitch"].set(f"{lp:+.1f}")
            ang["L_yaw"].set(f"{ly:+.1f}")
            ang["L_roll"].set(f"{d['L_A'][0]:+.1f}")
            ang["R_pitch"].set(f"{rp:+.1f}")
            ang["R_yaw"].set(f"{ry:+.1f}")
            ang["R_roll"].set(f"{d['R_A'][0]:+.1f}")

            # Per-pair status in Movement tab
            st_lbl = self.lbl_status_top if pair_key == "top" else self.lbl_status_bot
            if not (l_det and r_det):
                st_lbl.config(text="● Marker(s) not detected", fg=C_RED)
            elif not rot_ok or abs(lp) > p_th or abs(rp) > p_th \
                    or abs(ly) > y_th or abs(ry) > y_th:
                st_lbl.config(text="● Warning — check Warnings tab", fg=C_AMBER)
            else:
                st_lbl.config(text="● All sensors healthy", fg=C_GREEN)

        # Lighting
        li = self.last_data.get("lighting", {})
        ls = li.get("status", "no_frame")
        mb = li.get("mean", 0.0); sb = li.get("std", 0.0)
        if ls == "ok":
            self._set_chip(self._light_brt_chip, self._light_brt_val,
                           "ok", f"{mb:.0f} / 255")
            self._set_chip(self._light_ctr_chip, self._light_ctr_val,
                           "ok", f"std {sb:.0f}")
            light_txt = f"💡 OK  (brightness {mb:.0f},  contrast {sb:.0f})"
            light_col = C_GREEN
        elif ls == "dark":
            self._set_chip(self._light_brt_chip, self._light_brt_val,
                           "error", f"{mb:.0f} / 255  TOO DARK")
            self._set_chip(self._light_ctr_chip, self._light_ctr_val, "wait", "—")
            light_txt = f"💡 TOO DARK  (brightness {mb:.0f})"
            light_col = C_RED
            strip_parts.append("Lighting too dark"); all_ok = False; any_error = True
        elif ls == "bright":
            self._set_chip(self._light_brt_chip, self._light_brt_val,
                           "warn", f"{mb:.0f} / 255  OVEREXPOSED")
            self._set_chip(self._light_ctr_chip, self._light_ctr_val, "wait", "—")
            light_txt = f"💡 OVEREXPOSED  (brightness {mb:.0f})"
            light_col = C_AMBER
            strip_parts.append("Lighting overexposed"); all_ok = False
        elif ls == "flat":
            self._set_chip(self._light_brt_chip, self._light_brt_val,
                           "ok", f"{mb:.0f} / 255")
            self._set_chip(self._light_ctr_chip, self._light_ctr_val,
                           "warn", f"std {sb:.0f}  LOW CONTRAST")
            light_txt = f"💡 LOW CONTRAST  (std {sb:.0f})"
            light_col = C_AMBER
            strip_parts.append("Lighting low contrast"); all_ok = False
        else:
            self._set_chip(self._light_brt_chip, self._light_brt_val, "wait", "—")
            self._set_chip(self._light_ctr_chip, self._light_ctr_val, "wait", "—")
            light_txt = "💡 Waiting…"; light_col = C_MUTED

        self.lbl_lighting_cam.config(text=light_txt, fg=light_col)

        # Overall health pill in Warnings tab
        if any_error:
            ov_bg, ov_txt = C_RED,   "⬤  ERRORS DETECTED — check rows below"
        elif not all_ok:
            ov_bg, ov_txt = C_AMBER, "⬤  WARNINGS ACTIVE"
        else:
            ov_bg, ov_txt = C_GREEN, "⬤  ALL SYSTEMS HEALTHY"
        self.overall_frame.config(bg=ov_bg)
        self.overall_icon.config(bg=ov_bg, fg=ov_bg)
        self.overall_lbl.config(text=ov_txt, bg=ov_bg,
                                fg=C_BG if not any_error else C_WHITE)

        # Warning strip (Movement tab, below camera)
        if strip_parts:
            self.warn_strip.config(
                text="  ⚠  " + "   │   ".join(strip_parts),
                fg=C_WHITE, bg="#3d0b0b" if any_error else "#3d2b00")
        else:
            self.warn_strip.config(
                text="  ✅  All markers detected and within thresholds  —  Lighting OK",
                fg=C_GREEN, bg="#0d2318")

    # ══════════════════════════════════════════════════════════════════════════
    #  GUI refresh (50 ms)
    # ══════════════════════════════════════════════════════════════════════════
    def update_gui_loop(self):
        # Movement tab: camera
        if self.current_frame is not None:
            img = ImageTk.PhotoImage(
                image=Image.fromarray(self.current_frame).resize((960, 540)))
            self.canvas.create_image(0, 0, anchor="nw", image=img)
            self.canvas.img = img

        # Distance labels
        for key, dl, kl in [("top",    self.lbl_dist_top, self.lbl_k_top),
                              ("bottom", self.lbl_dist_bot, self.lbl_k_bot)]:
            d = self.last_data[key]
            dist = d["dist"]
            dl.config(text=f"{dist:.3f} mm" if dist > 0 else "─ ─ ─")
            kl.config(text=f"k: {d['k']:.4f}" if dist > 0 else "k: ─ ─ ─ ─")

        # Telemetry tab
        for key in ["top", "bottom"]:
            d  = self.last_data[key]
            tv = self.tele_vars[key]
            for v in ["A", "X", "TR", "BR", "B", "C"]:
                vv = d[v]; tv[v].set(f"{vv[0]:.1f},  {vv[1]:.1f},  {vv[2]:.1f}")
            lr = d.get("L_A", (0.0, 0.0)); rr = d.get("R_A", (0.0, 0.0))
            tv["L_ROL"].set(f"{lr[0]:+.2f}°")
            tv["L_ROT"].set(f"{lr[1]:.2f}° (2-D)")
            tv["L_Z"].set(f"{d['L_z']:.1f} mm")
            tv["L_PITCH"].set(f"{d.get('L_pitch', 0.0):+.2f}°")
            tv["L_YAW"].set(f"{d.get('L_yaw',   0.0):+.2f}°")
            tv["R_ROL"].set(f"{rr[0]:+.2f}°")
            tv["R_ROT"].set(f"{rr[1]:.2f}° (2-D)")
            tv["R_Z"].set(f"{d['R_z']:.1f} mm")
            tv["R_PITCH"].set(f"{d.get('R_pitch', 0.0):+.2f}°")
            tv["R_YAW"].set(f"{d.get('R_yaw',   0.0):+.2f}°")

        # Camera tab: preview + status
        if self._cam_preview_frame is not None:
            ci = ImageTk.PhotoImage(
                image=Image.fromarray(self._cam_preview_frame).resize((800, 480)))
            self.cam_canvas.create_image(0, 0, anchor="nw", image=ci)
            self.cam_canvas.img = ci
            top_ok = self.last_data["top"]["dist"]    > 0
            bot_ok = self.last_data["bottom"]["dist"] > 0
            if top_ok and bot_ok:
                self.lbl_cam_status.config(text="✅  Both pairs detected", fg=C_GREEN)
            elif top_ok or bot_ok:
                who = "TOP" if top_ok else "BOTTOM"
                self.lbl_cam_status.config(
                    text=f"⚠  Only {who} pair detected", fg=C_AMBER)
            else:
                self.lbl_cam_status.config(
                    text="❌  No ArUco markers detected", fg=C_RED)

        self._mv_tick()
        self._update_warnings()

        self.root.after(50, self.update_gui_loop)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = MeasurementApp(root)
    root.mainloop()
