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
RESOLUTION  = (1280, 720)
COLLECT_N   = 5          # readings to average for initial / final

AWB_MODES = {
    "Auto":        0,
    "Tungsten":    1,
    "Fluorescent": 2,
    "Indoor":      3,
    "Daylight":    4,
    "Cloudy":      5,
}

# ── Colour palette ──────────────────────────────────────────────────────────
C_BG      = "#0f1923"
C_PANEL   = "#1a2636"
C_CARD    = "#1e2d40"
C_CARD_LT = "#253347"
C_TOP     = "#e67e22"
C_BOT     = "#9b59b6"
C_GREEN   = "#00e676"
C_RED     = "#ff1744"
C_YELLOW  = "#ffd600"
C_ORANGE  = "#ff6d00"
C_BLUE    = "#40c4ff"
C_TEAL    = "#00e5ff"
C_TEXT    = "#ecf0f1"
C_MUTED   = "#607d8b"
C_WHITE   = "#ffffff"

# ── Warning severity colours ─────────────────────────────────────────────────
W_OK      = C_GREEN
W_WARN    = C_YELLOW
W_ERROR   = C_RED


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def rotation_to_euler(R):
    """
    Decompose a 3×3 rotation matrix into (pitch, yaw, roll) in degrees.

    Convention used (OpenCV camera frame):
      • pitch  – rotation around X-axis  →  forward/backward tilt
      • yaw    – rotation around Y-axis  →  left/right tilt
      • roll   – rotation around Z-axis  →  in-plane spin

    Returns (pitch_deg, yaw_deg, roll_deg)
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll  = math.degrees(math.atan2(R[2, 1], R[2, 2]))
        pitch = math.degrees(math.atan2(-R[2, 0], sy))
        yaw   = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    else:
        roll  = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
        pitch = math.degrees(math.atan2(-R[2, 0], sy))
        yaw   = 0.0

    return pitch, yaw, roll


def inplane_rot_deg(c_sorted):
    """In-plane 2-D rotation from the top edge of sorted corners."""
    return math.degrees(math.atan2(
        c_sorted[1, 1] - c_sorted[0, 1],
        c_sorted[1, 0] - c_sorted[0, 0]))


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Application
# ═══════════════════════════════════════════════════════════════════════════════

class MeasurementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dual-Pair ArUco Measurement v15")
        self.root.geometry("1680x980")
        self.root.configure(bg=C_BG)

        # ── Measurement vars ──────────────────────────────────────────────────
        self.size_top          = tk.DoubleVar(value=DEFAULT_MARKER_SIZE)
        self.size_bot          = tk.DoubleVar(value=DEFAULT_MARKER_SIZE)
        self.fixed_side        = tk.StringVar(value="Left")
        self.rot_threshold     = tk.DoubleVar(value=12.0)
        self.tilt_threshold    = tk.DoubleVar(value=10.0)   # NEW: pitch / yaw tilt warning
        self.use_angle_thresh  = tk.BooleanVar(value=True)

        # ── Camera control vars ───────────────────────────────────────────────
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

        # ── Movement Monitor state ────────────────────────────────────────────
        self.mv_state       = "idle"
        self.mv_init_buf    = {"top": [], "bottom": []}
        self.mv_final_buf   = {"top": [], "bottom": []}
        self.mv_dist_init   = {"top": None, "bottom": None}
        self.mv_dist_final  = {"top": None, "bottom": None}
        self._last_sc       = 0

        # ── Shared measurement data ───────────────────────────────────────────
        # Added per-marker pitch/yaw/roll for tilt warnings
        def _empty_pair():
            return {
                "A": (0,0,0), "X": (0,0,0), "TR": (0,0,0), "BR": (0,0,0),
                "B": (0,0,0), "C": (0,0,0), "dist": 0.0, "k": 0.0,
                "L_A": (0.0, 0.0), "R_A": (0.0, 0.0), "rot_2d": 0.0,
                "L_z": 0.0, "R_z": 0.0, "p1_px": None, "p2_px": None,
                # NEW: euler angles per marker (pitch, yaw, roll)
                "L_pitch": 0.0, "L_yaw": 0.0, "L_roll": 0.0,
                "R_pitch": 0.0, "R_yaw": 0.0, "R_roll": 0.0,
                # NEW: detection flags
                "L_detected": False, "R_detected": False,
            }

        self.last_data = {
            "top":    _empty_pair(),
            "bottom": _empty_pair(),
            "session_count": 0,
        }

        # ── Warning state (computed each GUI tick) ────────────────────────────
        # List of {"label", "colour", "icon"} dicts
        self._warnings = []

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
        s.configure("TNotebook", background=C_BG, borderwidth=0)
        s.configure("TNotebook.Tab", background=C_PANEL, foreground=C_TEXT,
                    padding=[14, 6], font=("Helvetica", 10, "bold"))
        s.map("TNotebook.Tab",
              background=[("selected", "#1565c0")],
              foreground=[("selected", C_WHITE)])
        s.configure("TFrame", background=C_BG)
        # Progressbar
        s.configure("green.Horizontal.TProgressbar",
                     troughcolor=C_CARD, background=C_GREEN,
                     lightcolor=C_GREEN, darkcolor=C_GREEN, bordercolor=C_CARD)

    # ══════════════════════════════════════════════════════════════════════════
    #  UI wiring
    # ══════════════════════════════════════════════════════════════════════════
    def setup_ui(self):
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True, padx=8, pady=8)
        self._build_tab_live()
        self._build_tab_warnings()
        self._build_tab_tele()
        self._build_tab_settings()
        self._build_tab_cam()

    # ── Generic helpers ───────────────────────────────────────────────────────
    def _dark_card(self, parent, title, title_color, **pack_kw):
        f = tk.LabelFrame(parent, text=f"  {title}  ",
                          font=("Helvetica", 10, "bold"),
                          fg=title_color, bg=C_CARD,
                          bd=1, relief="solid", padx=8, pady=6)
        f.pack(**pack_kw)
        return f

    def _label(self, parent, text, font=("Helvetica", 10), fg=C_TEXT, bg=C_BG, **kw):
        lbl = tk.Label(parent, text=text, font=font, fg=fg, bg=bg, **kw)
        return lbl

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 1 — MOVEMENT (live feed + movement monitor)
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_live(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="  📏  Movement  ")
        tab.configure(style="TFrame")

        # ── Left: camera feed ─────────────────────────────────────────────────
        left = tk.Frame(tab, bg=C_BG)
        left.pack(side="left", padx=12, pady=12)

        self.canvas = tk.Canvas(left, width=960, height=540, bg="#050d17",
                                highlightthickness=2, highlightbackground=C_BLUE)
        self.canvas.pack()

        # Mini warning strip below canvas
        self.warn_strip = tk.Label(
            left,
            text="",
            font=("Helvetica", 10, "bold"),
            fg=C_BG, bg=C_CARD,
            wraplength=960, justify="left", anchor="w",
            padx=8, pady=4)
        self.warn_strip.pack(fill="x", pady=(4, 0))

        # ── Right: live distances → buttons → results ─────────────────────────
        right = tk.Frame(tab, bg=C_BG)
        right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Distance cards
        for key, title, col in [("top", "TOP PAIR", C_TOP),
                                 ("bottom", "BOTTOM PAIR", C_BOT)]:
            card = self._dark_card(right, title, col, fill="x", pady=4, padx=4)
            dl = tk.Label(card, text="0.000 mm",
                          font=("Helvetica", 26, "bold"), fg=C_GREEN, bg=C_CARD)
            dl.pack(pady=(4, 0))
            kl = tk.Label(card, text="k: 0.0000",
                          font=("Helvetica", 10), fg=C_MUTED, bg=C_CARD)
            kl.pack(pady=(0, 4))
            if key == "top":
                self.lbl_dist_top, self.lbl_k_top = dl, kl
            else:
                self.lbl_dist_bot, self.lbl_k_bot = dl, kl

        # Status + progress
        self.mv_status_lbl = tk.Label(
            right, text="Press  START  to capture initial gap",
            font=("Helvetica", 11), fg=C_TEXT, bg=C_BG,
            wraplength=270, justify="center")
        self.mv_status_lbl.pack(pady=(10, 2))

        prog_f = tk.Frame(right, bg=C_BG); prog_f.pack(fill="x", padx=6)
        self.mv_prog_lbl = tk.Label(prog_f, text="",
                                    font=("Helvetica", 10), fg=C_GREEN, bg=C_BG)
        self.mv_prog_lbl.pack()
        self.mv_prog_bar = ttk.Progressbar(
            prog_f, length=240, maximum=COLLECT_N, mode="determinate",
            style="green.Horizontal.TProgressbar")
        self.mv_prog_bar.pack(pady=2)

        # Buttons
        btn_f = tk.Frame(right, bg=C_BG); btn_f.pack(pady=8, fill="x", padx=6)
        btn_cfg = dict(font=("Helvetica", 13, "bold"), relief="flat",
                       padx=14, pady=8, bd=0, cursor="hand2")

        self.btn_start = tk.Button(btn_f, text="▶ START", bg="#00897b", fg=C_WHITE,
                                   activebackground="#26a69a",
                                   command=self._mv_start, **btn_cfg)
        self.btn_start.pack(side="left", expand=True, fill="x", padx=2)

        self.btn_stop = tk.Button(btn_f, text="■ STOP", bg="#c62828", fg=C_WHITE,
                                  activebackground="#ef5350",
                                  state="disabled",
                                  command=self._mv_stop, **btn_cfg)
        self.btn_stop.pack(side="left", expand=True, fill="x", padx=2)

        self.btn_reset = tk.Button(btn_f, text="↺", bg=C_MUTED, fg=C_WHITE,
                                   activebackground="#90a4ae",
                                   command=self._mv_reset, **btn_cfg)
        self.btn_reset.pack(side="left", padx=2)

        # Divider
        tk.Frame(right, bg=C_MUTED, height=1).pack(fill="x", padx=6, pady=8)
        tk.Label(right, text="Distance Moved", font=("Helvetica", 11, "bold"),
                 fg=C_TEXT, bg=C_BG).pack()

        # Delta rows
        for key, title, col in [("top", "TOP", C_TOP), ("bottom", "BOT", C_BOT)]:
            row = tk.Frame(right, bg=C_CARD, bd=0, relief="flat",
                           highlightbackground=C_MUTED, highlightthickness=1)
            row.pack(fill="x", padx=6, pady=4)
            tk.Label(row, text=title, font=("Helvetica", 10, "bold"),
                     fg=col, bg=C_CARD, width=5).pack(side="left", padx=6, pady=6)

            info = tk.Frame(row, bg=C_CARD); info.pack(side="left", expand=True)
            init_lbl = tk.Label(info, text="Init: —", font=("Helvetica", 9),
                                fg=C_MUTED, bg=C_CARD, anchor="w")
            init_lbl.pack(fill="x", padx=4)
            final_lbl = tk.Label(info, text="Final: —", font=("Helvetica", 9),
                                 fg=C_MUTED, bg=C_CARD, anchor="w")
            final_lbl.pack(fill="x", padx=4)

            delta_lbl = tk.Label(row, text="—", font=("Helvetica", 22, "bold"),
                                 fg=C_BLUE, bg=C_CARD)
            delta_lbl.pack(side="right", padx=10, pady=6)

            if key == "top":
                self.mv_init_lbl_top  = init_lbl
                self.mv_final_lbl_top = final_lbl
                self.mv_delta_lbl_top = delta_lbl
            else:
                self.mv_init_lbl_bot  = init_lbl
                self.mv_final_lbl_bot = final_lbl
                self.mv_delta_lbl_bot = delta_lbl

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 2 — WARNINGS DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_warnings(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="  ⚠  Warnings  ")
        tab.configure(style="TFrame")

        outer = tk.Frame(tab, bg=C_BG)
        outer.pack(fill="both", expand=True, padx=20, pady=20)

        # Header
        hdr = tk.Frame(outer, bg=C_BG)
        hdr.pack(fill="x", pady=(0, 14))
        tk.Label(hdr, text="ArUco Health Monitor",
                 font=("Helvetica", 18, "bold"), fg=C_TEAL, bg=C_BG).pack(side="left")
        self.warn_overall_icon = tk.Label(
            hdr, text="●", font=("Helvetica", 22), fg=C_MUTED, bg=C_BG)
        self.warn_overall_icon.pack(side="right", padx=6)
        self.warn_overall_lbl = tk.Label(
            hdr, text="No data yet", font=("Helvetica", 12, "bold"),
            fg=C_MUTED, bg=C_BG)
        self.warn_overall_lbl.pack(side="right")

        # Tilt threshold control — placed at top of warnings tab
        thr_card = tk.LabelFrame(outer, text="  Tilt Threshold (°)  ",
                                 font=("Helvetica", 10, "bold"), fg=C_TEAL,
                                 bg=C_CARD, bd=1, relief="solid", padx=12, pady=8)
        thr_card.pack(fill="x", pady=(0, 14))

        thr_inner = tk.Frame(thr_card, bg=C_CARD); thr_inner.pack(fill="x")
        tk.Label(thr_inner,
                 text=(
                     "Tilt threshold sets the maximum allowed pitch (forward / backward)\n"
                     "and yaw (left / right) angle for each marker. "
                     "Exceeding it triggers a warning."
                 ),
                 font=("Helvetica", 9), fg=C_MUTED, bg=C_CARD,
                 justify="left").pack(anchor="w", pady=(0, 6))

        thr_row = tk.Frame(thr_card, bg=C_CARD); thr_row.pack(fill="x")
        self.tilt_slider = tk.Scale(
            thr_row, from_=1, to=45, resolution=0.5,
            orient="horizontal", variable=self.tilt_threshold,
            bg=C_CARD, fg=C_TEXT, troughcolor=C_BG,
            activebackground=C_TEAL, length=500,
            highlightthickness=0)
        self.tilt_slider.pack(side="left", padx=(0, 10))
        self.tilt_val_lbl = tk.Label(
            thr_row,
            textvariable=self.tilt_threshold,
            font=("Courier", 13, "bold"), fg=C_TEAL, bg=C_CARD, width=5)
        self.tilt_val_lbl.pack(side="left")
        tk.Label(thr_row, text="°", font=("Helvetica", 13), fg=C_TEXT,
                 bg=C_CARD).pack(side="left")

        # Per-pair warning panels
        pairs_frame = tk.Frame(outer, bg=C_BG)
        pairs_frame.pack(fill="both", expand=True)

        self._warn_rows = {}
        for key, title, col in [("top", "TOP PAIR", C_TOP), ("bottom", "BOTTOM PAIR", C_BOT)]:
            col_frame = tk.LabelFrame(
                pairs_frame, text=f"  {title}  ",
                font=("Helvetica", 11, "bold"), fg=col, bg=C_CARD,
                bd=1, relief="solid", padx=10, pady=10)
            col_frame.pack(side="left", fill="both", expand=True, padx=8)

            rows = {}

            # Each warning type: key → (label_text, icon_label, status_label)
            warn_defs = [
                ("detect_L",  "Left Marker Detected"),
                ("detect_R",  "Right Marker Detected"),
                ("rot",       "In-Plane Rotation"),
                ("pitch_L",   "Left Marker Pitch  (fwd/bwd tilt)"),
                ("pitch_R",   "Right Marker Pitch  (fwd/bwd tilt)"),
                ("yaw_L",     "Left Marker Yaw  (left/right tilt)"),
                ("yaw_R",     "Right Marker Yaw  (left/right tilt)"),
                ("depth_diff","Depth Difference (L-R Z)"),
            ]

            for w_key, w_label in warn_defs:
                row_f = tk.Frame(col_frame, bg=C_CARD_LT,
                                 highlightbackground=C_BG, highlightthickness=1)
                row_f.pack(fill="x", pady=3)

                icon = tk.Label(row_f, text="●", font=("Helvetica", 14),
                                fg=C_MUTED, bg=C_CARD_LT, width=2)
                icon.pack(side="left", padx=(6, 2), pady=4)

                tk.Label(row_f, text=w_label, font=("Helvetica", 10),
                         fg=C_TEXT, bg=C_CARD_LT, anchor="w").pack(
                    side="left", padx=4, fill="x", expand=True)

                val_lbl = tk.Label(row_f, text="—",
                                   font=("Courier", 10, "bold"),
                                   fg=C_MUTED, bg=C_CARD_LT, width=18, anchor="e")
                val_lbl.pack(side="right", padx=8, pady=4)

                rows[w_key] = (icon, val_lbl)

            # Angle display section
            ang_f = tk.Frame(col_frame, bg=C_CARD); ang_f.pack(fill="x", pady=(8, 0))
            tk.Label(ang_f, text="Euler Angles (°)",
                     font=("Helvetica", 9, "bold"), fg=C_MUTED, bg=C_CARD).pack(anchor="w")

            ang_grid = tk.Frame(ang_f, bg=C_CARD); ang_grid.pack(fill="x")
            ang_vars = {}
            for col_idx, (a_key, a_label, a_color) in enumerate([
                ("L_pitch", "L-Pitch", C_ORANGE),
                ("L_yaw",   "L-Yaw",   C_BLUE),
                ("L_roll",  "L-Roll",  C_TEAL),
                ("R_pitch", "R-Pitch", C_ORANGE),
                ("R_yaw",   "R-Yaw",   C_BLUE),
                ("R_roll",  "R-Roll",  C_TEAL),
            ]):
                cell = tk.Frame(ang_grid, bg=C_BG, padx=4, pady=4)
                cell.grid(row=0, column=col_idx, padx=2, pady=2, sticky="nsew")
                ang_grid.columnconfigure(col_idx, weight=1)
                tk.Label(cell, text=a_label, font=("Helvetica", 8),
                         fg=a_color, bg=C_BG).pack()
                sv = tk.StringVar(value="—")
                tk.Label(cell, textvariable=sv, font=("Courier", 11, "bold"),
                         fg=C_WHITE, bg=C_BG).pack()
                ang_vars[a_key] = sv

            rows["_ang_vars"] = ang_vars
            self._warn_rows[key] = rows

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 3 — DUAL TELEMETRY
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_tele(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="  🛸  Dual Telemetry  ")
        mtf = tk.Frame(tab, bg=C_PANEL); mtf.pack(fill="both", expand=True)

        self.tele_vars = {"top": {}, "bottom": {}}
        v_show = ["A", "X", "TR", "BR", "B", "C",
                  "L_ROL", "L_ROT", "L_Z", "L_PITCH", "L_YAW",
                  "R_ROL", "R_ROT", "R_Z", "R_PITCH", "R_YAW"]

        for key, title, col in [("top", "TOP", C_TOP), ("bottom", "BOTTOM", C_BOT)]:
            cf = tk.LabelFrame(mtf, text=f"  {title} DATA  ",
                               font=("Helvetica", 12, "bold"), fg=col, bg=C_PANEL)
            cf.pack(side="left", fill="both", expand=True, padx=10, pady=10)

            for v_name in v_show:
                row = tk.Frame(cf, bg=C_CARD,
                               highlightbackground=C_BG, highlightthickness=1)
                row.pack(fill="x", padx=10, pady=2)
                tk.Label(row, text=v_name, font=("Helvetica", 9),
                         fg=C_MUTED, bg=C_CARD, width=10, anchor="w").pack(
                    side="left", padx=5)
                sv = tk.StringVar(value="—")
                tk.Label(row, textvariable=sv, font=("Courier", 10),
                         fg=C_TEXT, bg=C_CARD).pack(side="right", padx=5)
                self.tele_vars[key][v_name] = sv

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 4 — MACHINE CONFIGURATION
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_settings(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="  ⚙  Machine Config  ")
        sc = tk.Frame(tab, bg=C_PANEL); sc.pack(fill="both", expand=True, padx=60, pady=30)

        rf = tk.LabelFrame(sc, text="  Reference Side (Fixed ArUco)  ", bg=C_CARD,
                           fg=C_TEXT, font=("Helvetica", 11, "bold"), padx=20, pady=10)
        rf.pack(fill="x", pady=(0, 12))
        for choice in ["Left", "Right"]:
            tk.Radiobutton(rf, text=choice, variable=self.fixed_side,
                           value=choice, bg=C_CARD, fg=C_TEXT, selectcolor=C_BG,
                           activebackground=C_CARD, activeforeground=C_TEXT,
                           font=("Helvetica", 12)).pack(side="left", padx=24)

        def create_slider(label, var, color, lo, hi, res=0.1):
            f = tk.LabelFrame(sc, text=f"  {label}  ", bg=C_CARD,
                              font=("Helvetica", 10), fg=color, padx=20, pady=6)
            f.pack(fill="x", pady=6)
            sl = tk.Scale(f, from_=lo, to=hi, resolution=res, orient="horizontal",
                          variable=var, bg=C_CARD, fg=C_TEXT,
                          troughcolor=C_BG, activebackground=color,
                          highlightthickness=0, length=560)
            sl.pack(side="left", padx=16)
            tk.Entry(f, textvariable=var, width=9, font=("Courier", 10),
                     bg=C_BG, fg=C_TEXT, insertbackground=C_TEXT).pack(side="left")
            return sl

        create_slider("Upper Pair Marker Size (mm)",   self.size_top,          C_TOP,  10, 200)
        create_slider("Bottom Pair Marker Size (mm)",  self.size_bot,          C_BOT,  10, 200)
        self.rot_threshold_slider = create_slider(
            "Rotation Threshold °  (perpendicular / fallback)",
            self.rot_threshold, C_ORANGE, 0, 45)

        # NOTE: Tilt threshold also lives in the Warnings tab for quick access,
        # but we mirror it here too as a second slider linked to the same variable.
        create_slider(
            "Tilt Threshold °  (pitch / yaw warning)",
            self.tilt_threshold, C_TEAL, 1, 45, res=0.5)

        # ── Angle Threshold Toggle ────────────────────────────────────────────
        tog_f = tk.LabelFrame(sc, text="  Angle Threshold Mode  ", bg=C_CARD,
                              font=("Helvetica", 10), fg=C_TEXT, padx=20, pady=10)
        tog_f.pack(fill="x", pady=6)

        def _update_tog_label(*_):
            if self.use_angle_thresh.get():
                tog_state_lbl.config(
                    text="ON  — Perpendicular formula up to threshold, midpoint above",
                    fg=C_GREEN)
                self.rot_threshold_slider.config(state="normal")
            else:
                tog_state_lbl.config(
                    text="OFF — Always use perpendicular formula (v8 behaviour)",
                    fg=C_RED)
                self.rot_threshold_slider.config(state="disabled")

        tog_row = tk.Frame(tog_f, bg=C_CARD); tog_row.pack(fill="x")
        self._tog_canvas = tk.Canvas(tog_row, width=56, height=28, bg=C_CARD,
                                     highlightthickness=0, cursor="hand2")
        self._tog_canvas.pack(side="left", padx=(0, 12))
        tog_state_lbl = tk.Label(tog_row, text="", font=("Helvetica", 10),
                                 bg=C_CARD, fg=C_TEXT, anchor="w")
        tog_state_lbl.pack(side="left", fill="x", expand=True)

        def _draw_toggle():
            self._tog_canvas.delete("all")
            on = self.use_angle_thresh.get()
            bg = C_GREEN if on else C_MUTED
            self._tog_canvas.create_oval(2, 2, 54, 26, fill=bg, outline="")
            cx = 38 if on else 18
            self._tog_canvas.create_oval(cx-12, 4, cx+12, 24, fill=C_WHITE, outline="")

        def _toggle_click(_=None):
            self.use_angle_thresh.set(not self.use_angle_thresh.get())
            _draw_toggle(); _update_tog_label()

        self._tog_canvas.bind("<Button-1>", _toggle_click)
        _draw_toggle(); _update_tog_label()

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 5 — CAMERA CONTROLS
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_cam(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="  📷  Camera  ")

        cam_left  = tk.Frame(tab, bg="#0a1520")
        cam_left.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        cam_right = tk.Frame(tab, bg=C_PANEL)
        cam_right.pack(side="right", fill="y", padx=8, pady=8, ipadx=4)

        tk.Label(cam_left, text="Live Camera Preview",
                 font=("Helvetica", 11, "bold"), fg=C_TEXT, bg="#0a1520").pack(pady=(8, 0))
        self.cam_canvas = tk.Canvas(cam_left, width=800, height=480, bg="black",
                                    highlightthickness=2, highlightbackground=C_BLUE)
        self.cam_canvas.pack(padx=10, pady=10)
        self.lbl_cam_status = tk.Label(cam_left, text="⏳ Waiting for camera…",
                                       font=("Helvetica", 11, "bold"),
                                       fg="#f39c12", bg="#0a1520")
        self.lbl_cam_status.pack(pady=4)

        tk.Label(cam_right, text="Camera Settings",
                 font=("Helvetica", 13, "bold"), fg=C_TEXT,
                 bg=C_PANEL).pack(pady=(10, 4))

        def cam_row(label, var, lo, hi, res, unit=""):
            rf = tk.LabelFrame(cam_right, text=f"  {label}  ", bg=C_CARD,
                               font=("Helvetica", 9), fg=C_MUTED, padx=8, pady=4)
            rf.pack(fill="x", padx=6, pady=3)
            inner = tk.Frame(rf, bg=C_CARD); inner.pack(fill="x")
            tk.Scale(inner, from_=lo, to=hi, resolution=res,
                     orient="horizontal", variable=var, bg=C_CARD, fg=C_TEXT,
                     troughcolor=C_BG, highlightthickness=0,
                     length=240, command=lambda _: self._apply_cam()).pack(side="left")
            tk.Entry(inner, textvariable=var, width=7, font=("Courier", 9),
                     bg=C_BG, fg=C_TEXT, insertbackground=C_TEXT).pack(side="left", padx=4)
            if unit:
                tk.Label(inner, text=unit, bg=C_CARD, fg=C_MUTED,
                         font=("Helvetica", 8)).pack(side="left")

        ae_f = tk.LabelFrame(cam_right, text="  Auto Exposure  ", bg=C_CARD,
                             font=("Helvetica", 9), fg=C_MUTED, padx=8, pady=4)
        ae_f.pack(fill="x", padx=6, pady=3)
        tk.Checkbutton(ae_f, text="Enable Auto Exposure (AE)", variable=self.cam_ae,
                       bg=C_CARD, fg=C_TEXT, selectcolor=C_BG,
                       activebackground=C_CARD, activeforeground=C_TEXT,
                       font=("Helvetica", 10), command=self._apply_cam).pack(anchor="w")

        cam_row("Exposure Time", self.cam_exposure, 100,  66000, 100, "µs")
        cam_row("ISO / Gain",    self.cam_gain,     1.0,  16.0,  0.1, "x")

        awb_f = tk.LabelFrame(cam_right, text="  White Balance  ", bg=C_CARD,
                              font=("Helvetica", 9), fg=C_MUTED, padx=8, pady=4)
        awb_f.pack(fill="x", padx=6, pady=3)
        tk.Checkbutton(awb_f, text="Auto White Balance", variable=self.cam_awb,
                       bg=C_CARD, fg=C_TEXT, selectcolor=C_BG,
                       activebackground=C_CARD, activeforeground=C_TEXT,
                       font=("Helvetica", 10), command=self._apply_cam).pack(anchor="w")
        wbm_row = tk.Frame(awb_f, bg=C_CARD); wbm_row.pack(fill="x", pady=2)
        tk.Label(wbm_row, text="Mode:", bg=C_CARD, fg=C_TEXT,
                 font=("Helvetica", 9)).pack(side="left")
        for mode_name in AWB_MODES:
            tk.Radiobutton(wbm_row, text=mode_name, variable=self.cam_awb_mode,
                           value=mode_name, bg=C_CARD, fg=C_TEXT, selectcolor=C_BG,
                           activebackground=C_CARD, activeforeground=C_TEXT,
                           font=("Helvetica", 8),
                           command=self._apply_cam).pack(side="left", padx=2)

        cam_row("Brightness", self.cam_brightness, -1.0, 1.0, 0.05)
        cam_row("Contrast",   self.cam_contrast,    0.0, 8.0, 0.1)
        cam_row("Saturation", self.cam_saturation,  0.0, 8.0, 0.1)
        cam_row("Sharpness",  self.cam_sharpness,   0.0, 8.0, 0.1)

        tk.Button(cam_right, text="↺  Reset to Defaults",
                  font=("Helvetica", 10, "bold"), bg=C_RED, fg=C_WHITE,
                  activebackground="#c62828", relief="flat", padx=10, pady=6,
                  command=self._reset_cam).pack(pady=10, fill="x", padx=6)

    # ══════════════════════════════════════════════════════════════════════════
    #  Movement Monitor
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
            text=f"⏳  Collecting initial distance…  (0 / {COLLECT_N})",
            fg=C_YELLOW)

    def _mv_stop(self):
        if self.mv_state != "ready":
            return
        self.mv_state     = "collecting_final"
        self.mv_final_buf = {"top": [], "bottom": []}
        self.btn_stop.config(state="disabled")
        self.mv_prog_bar["value"] = 0
        self.mv_status_lbl.config(
            text=f"⏳  Collecting final distance…  (0 / {COLLECT_N})",
            fg=C_YELLOW)

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
            text="Press  START  to capture initial gap", fg=C_TEXT)
        self.mv_init_lbl_top.config(text="Init: —",   fg=C_MUTED)
        self.mv_init_lbl_bot.config(text="Init: —",   fg=C_MUTED)
        self.mv_final_lbl_top.config(text="Final: —", fg=C_MUTED)
        self.mv_final_lbl_bot.config(text="Final: —", fg=C_MUTED)
        self.mv_delta_lbl_top.config(text="—", fg=C_BLUE)
        self.mv_delta_lbl_bot.config(text="—", fg=C_BLUE)

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
            self.mv_prog_lbl.config(text=f"Reading {n} / {COLLECT_N}")
            self.mv_status_lbl.config(
                text=f"⏳  Collecting initial distance…  ({n} / {COLLECT_N})",
                fg=C_YELLOW)
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
                    text=f"Init: {self.mv_dist_init['top']:.3f} mm", fg=C_TEXT)
                self.mv_init_lbl_bot.config(
                    text=f"Init: {self.mv_dist_init['bottom']:.3f} mm", fg=C_TEXT)

        elif self.mv_state == "collecting_final":
            for key in ["top", "bottom"]:
                d = self.last_data[key]["dist"]
                if d > 0:
                    self.mv_final_buf[key].append(d)
            n = min(len(self.mv_final_buf["top"]), len(self.mv_final_buf["bottom"]))
            self.mv_prog_bar["value"] = n
            self.mv_prog_lbl.config(text=f"Reading {n} / {COLLECT_N}")
            self.mv_status_lbl.config(
                text=f"⏳  Collecting final distance…  ({n} / {COLLECT_N})",
                fg=C_YELLOW)
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
                    text="✅  Measurement complete — press RESET to start over",
                    fg=C_GREEN)
                for key, init_lbl, final_lbl, delta_lbl in [
                    ("top",    self.mv_init_lbl_top, self.mv_final_lbl_top, self.mv_delta_lbl_top),
                    ("bottom", self.mv_init_lbl_bot, self.mv_final_lbl_bot, self.mv_delta_lbl_bot),
                ]:
                    di = self.mv_dist_init[key]
                    df = self.mv_dist_final[key]
                    delta = df - di
                    init_lbl.config(text=f"Init: {di:.3f} mm",  fg=C_TEXT)
                    final_lbl.config(text=f"Final: {df:.3f} mm", fg=C_TEXT)
                    sign  = "+" if delta >= 0 else ""
                    color = C_RED if delta > 0.5 else (C_GREEN if delta < -0.5 else C_BLUE)
                    delta_lbl.config(text=f"{sign}{delta:.3f} mm", fg=color)

    # ══════════════════════════════════════════════════════════════════════════
    #  Camera control helpers
    # ══════════════════════════════════════════════════════════════════════════
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
        self.cam_awb_mode.set("Auto"); self.cam_brightness.set(0.0)
        self.cam_contrast.set(1.0);   self.cam_saturation.set(1.0)
        self.cam_sharpness.set(1.0);  self._apply_cam()

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
    #  Measurement loop (runs in background thread)
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

            corners_raw, ids, _ = detector.detectMarkers(
                cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
            curr = time.time() * 1000

            if ids is not None and len(ids) >= 1:
                m_data = []
                for i in range(len(ids)):
                    raw = corners_raw[i][0]
                    idx_x = np.argsort(raw[:, 0])
                    lp = raw[idx_x[:2]]; rp = raw[idx_x[2:]]
                    tl = lp[np.argmin(lp[:, 1])]; bl = lp[np.argmax(lp[:, 1])]
                    tr = rp[np.argmin(rp[:, 1])]; br = rp[np.argmax(rp[:, 1])]
                    cy = (tl[1]+tr[1]+br[1]+bl[1]) / 4.0
                    cx = (tl[0]+tr[0]+br[0]+bl[0]) / 4.0
                    m_data.append({"c": np.array([tl, tr, br, bl], dtype=np.float32),
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

                # ── proc: geometry + euler angles ─────────────────────────────
                def proc(marker_list, key, size):
                    if len(marker_list) < 2:
                        self.last_data[key]["dist"]       = 0.0
                        self.last_data[key]["p1_px"]      = None
                        self.last_data[key]["L_detected"] = False
                        self.last_data[key]["R_detected"] = False
                        return

                    marker_list.sort(key=lambda m: m["x"])
                    left_m  = marker_list[0]; right_m = marker_list[-1]
                    is_rf   = (self.fixed_side.get() == "Right")
                    S_m = right_m if is_rf else left_m
                    T_m = left_m  if is_rf else right_m
                    h = size / 2.0
                    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]],
                                   dtype=np.float32)

                    def get_pose(c_raw):
                        _, rv, tv = cv.solvePnP(obj, c_raw, K, dist_c)
                        R, _ = cv.Rodrigues(rv)
                        pts3d = np.array([R @ pt + tv.ravel() for pt in obj])
                        pitch, yaw, roll = rotation_to_euler(R)
                        irot  = abs(inplane_rot_deg(
                            np.array([c_raw[np.argsort(c_raw[:, 0])[:2]],
                                      c_raw[np.argsort(c_raw[:, 0])[2:]]]).reshape(-1, 2)))
                        return pts3d, roll, float(tv[2]), pitch, yaw

                    def get_irot(c_sorted):
                        return abs(inplane_rot_deg(c_sorted))

                    S_pts, S_roll, S_z, S_pitch, S_yaw = get_pose(S_m["c_raw"])
                    T_pts, T_roll, T_z, T_pitch, T_yaw = get_pose(T_m["c_raw"])
                    S_irot = get_irot(S_m["c"])
                    T_irot = get_irot(T_m["c"])

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
                    B = T_inner_top; C_ = T_inner_bot; X_alt = (B + C_) / 2.0

                    if not is_rf:
                        p_src_2d = tuple(((S_m["c"][1]+S_m["c"][2])/2).astype(int))
                        p_tgt_2d = tuple(((T_m["c"][0]+T_m["c"][3])/2).astype(int))
                    else:
                        p_src_2d = tuple(((S_m["c"][0]+S_m["c"][3])/2).astype(int))
                        p_tgt_2d = tuple(((T_m["c"][1]+T_m["c"][2])/2).astype(int))

                    buffers[key].append({
                        "A": A, "X_alt": X_alt,
                        "TR": S_inner_top, "BR": S_inner_bot, "TL_ref": S_outer_top,
                        "B": B, "C": C_,
                        "L_A":    (S_roll, S_irot),
                        "R_A":    (T_roll, T_irot),
                        "rot":    max(S_irot, T_irot),
                        "L_z":    S_z, "R_z": T_z,
                        "L_pitch": S_pitch, "L_yaw": S_yaw,
                        "R_pitch": T_pitch, "R_yaw": T_yaw,
                        "p1_px":  p_src_2d, "p2_px": p_tgt_2d,
                    })
                    self.last_data[key]["L_detected"] = True
                    self.last_data[key]["R_detected"] = True

                if (curr - l_s) >= 100:
                    proc(top_m, "top",    self.size_top.get())
                    proc(bot_m, "bottom", self.size_bot.get())
                    l_s = curr

                # ── 1s averaging window ───────────────────────────────────────
                if (curr - l_u) >= 1000:
                    for key in ["top", "bottom"]:
                        if buffers[key]:
                            s = buffers[key]
                            aA     = np.mean([x["A"]       for x in s], axis=0)
                            aTR    = np.mean([x["TR"]      for x in s], axis=0)
                            aTL    = np.mean([x["TL_ref"]  for x in s], axis=0)
                            aBR    = np.mean([x["BR"]      for x in s], axis=0)
                            aB     = np.mean([x["B"]       for x in s], axis=0)
                            aC     = np.mean([x["C"]       for x in s], axis=0)
                            aX_alt = np.mean([x["X_alt"]   for x in s], axis=0)
                            aL     = np.mean([x["L_A"]     for x in s], axis=0)
                            aR     = np.mean([x["R_A"]     for x in s], axis=0)
                            avg_rot   = float(np.mean([x["rot"]     for x in s]))
                            avg_Lz    = float(np.mean([x["L_z"]     for x in s]))
                            avg_Rz    = float(np.mean([x["R_z"]     for x in s]))
                            avg_Lpitch = float(np.mean([x["L_pitch"] for x in s]))
                            avg_Lyaw   = float(np.mean([x["L_yaw"]  for x in s]))
                            avg_Rpitch = float(np.mean([x["R_pitch"] for x in s]))
                            avg_Ryaw   = float(np.mean([x["R_yaw"]  for x in s]))
                            p1_px = tuple(np.mean([x["p1_px"] for x in s],
                                                  axis=0).astype(int))
                            p2_px = tuple(np.mean([x["p2_px"] for x in s],
                                                  axis=0).astype(int))

                            # ── Distance formula ──────────────────────────────
                            v_raw = aTR - aTL
                            v_len = np.linalg.norm(v_raw)
                            v = v_raw / v_len if v_len > 0 else np.zeros(3)
                            w = aC - aB; u = aB - aA
                            den = (np.dot(v,v)*np.dot(w,w)) - (np.dot(v,w)**2)

                            def _perp():
                                if abs(den) > 1e-6:
                                    kv_ = ((np.dot(v,w)*np.dot(u,v)) -
                                           (np.dot(v,v)*np.dot(u,w))) / den
                                    kv_ = float(np.clip(kv_, 0.0, 1.0))
                                    aX_ = aB + kv_ * w
                                    return aX_, np.linalg.norm(aX_ - aA), kv_
                                return aX_alt, np.linalg.norm(aX_alt - aA), 0.5

                            if self.use_angle_thresh.get():
                                thresh = self.rot_threshold.get()
                                if avg_rot > thresh:
                                    aX = aX_alt
                                    dv = np.linalg.norm(aX - aA)
                                    kv = 0.5
                                else:
                                    aX, dv, kv = _perp()
                            else:
                                aX, dv, kv = _perp()

                            self.last_data[key].update({
                                "A": aA, "X": aX, "TR": aTR, "BR": aBR,
                                "B": aB, "C": aC, "dist": dv, "k": kv,
                                "rot_2d": avg_rot, "L_A": aL, "R_A": aR,
                                "L_z": avg_Lz, "R_z": avg_Rz,
                                "L_pitch": avg_Lpitch, "L_yaw": avg_Lyaw,
                                "R_pitch": avg_Rpitch, "R_yaw": avg_Ryaw,
                                "p1_px": p1_px, "p2_px": p2_px,
                            })
                            self.last_data["session_count"] += 1
                            buffers[key].clear()

                            # CSV / DB log
                            v_vec = aTR - aTL; u_vec = aA; w_vec = aC - aB
                            try:
                                log.record(dv, kv, aA, aX, aTR, aBR, aB, aC,
                                           u_vec, v_vec, w_vec, aL, aR)
                            except Exception:
                                pass

                            # Image during movement capture
                            if self.mv_state in ("collecting_init", "ready",
                                                 "collecting_final"):
                                try:
                                    from datetime import datetime
                                    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
                                    img_dir = "captured_images"
                                    os.makedirs(img_dir, exist_ok=True)
                                    cv.imwrite(os.path.join(img_dir, f"mv_{ts}.jpg"), frame)
                                except Exception:
                                    pass
                    l_u = curr

            else:
                # No markers detected
                for key in ["top", "bottom"]:
                    self.last_data[key]["dist"]       = 0.0
                    self.last_data[key]["L_detected"] = False
                    self.last_data[key]["R_detected"] = False

            # ── Draw overlay on frame ─────────────────────────────────────────
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

                    # ── Tilt / rotation overlays on frame ────────────────────
                    tilt_th = self.tilt_threshold.get()
                    warnlines = []
                    max_rot = max(d["L_A"][1], d["R_A"][1])
                    if max_rot > self.rot_threshold.get():
                        warnlines.append(f"ROT:{max_rot:.1f}°")
                    if abs(d.get("L_pitch", 0)) > tilt_th:
                        dir_ = "FWD" if d["L_pitch"] > 0 else "BWD"
                        warnlines.append(f"L-PITCH:{d['L_pitch']:.1f}°({dir_})")
                    if abs(d.get("R_pitch", 0)) > tilt_th:
                        dir_ = "FWD" if d["R_pitch"] > 0 else "BWD"
                        warnlines.append(f"R-PITCH:{d['R_pitch']:.1f}°({dir_})")
                    if abs(d.get("L_yaw", 0)) > tilt_th:
                        dir_ = "RIGHT" if d["L_yaw"] > 0 else "LEFT"
                        warnlines.append(f"L-YAW:{d['L_yaw']:.1f}°({dir_})")
                    if abs(d.get("R_yaw", 0)) > tilt_th:
                        dir_ = "RIGHT" if d["R_yaw"] > 0 else "LEFT"
                        warnlines.append(f"R-YAW:{d['R_yaw']:.1f}°({dir_})")

                    if warnlines:
                        box_x1 = p1[0] - 5
                        box_y1 = p1[1] - 90
                        box_x2 = p1[0] + 200
                        box_y2 = p1[1] - 50 + 20 * len(warnlines)
                        cv.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2),
                                     (0, 0, 0), -1)
                        cv.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2),
                                     (0, 0, 220), 2)
                        for li, wl in enumerate(warnlines):
                            cv.putText(frame, wl,
                                       (box_x1 + 4, box_y1 + 18 + li * 20),
                                       cv.FONT_HERSHEY_SIMPLEX, 0.48,
                                       (0, 220, 255), 1)

            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            self.current_frame      = rgb
            self._cam_preview_frame = rgb

    # ══════════════════════════════════════════════════════════════════════════
    #  Warning computation (GUI thread)
    # ══════════════════════════════════════════════════════════════════════════
    def _compute_warnings(self):
        """
        Build and push all warning rows to the Warnings tab and the strip label.
        Called every GUI tick.
        """
        tilt_th = self.tilt_threshold.get()
        rot_th  = self.rot_threshold.get()
        any_error = False
        any_warn  = False
        strip_parts = []

        for key in ["top", "bottom"]:
            d = self.last_data[key]
            rows = self._warn_rows[key]
            ang  = rows["_ang_vars"]

            # ── Helper to set icon + value label ────────────────────────────
            def set_row(w_key, colour, text):
                icon_lbl, val_lbl = rows[w_key]
                icon_lbl.config(fg=colour)
                val_lbl.config(text=text, fg=colour)

            # 1. Detection
            l_det = d.get("L_detected", False)
            r_det = d.get("R_detected", False)
            set_row("detect_L",
                    W_OK if l_det else W_ERROR,
                    "✔ OK" if l_det else "✘ NOT FOUND")
            set_row("detect_R",
                    W_OK if r_det else W_ERROR,
                    "✔ OK" if r_det else "✘ NOT FOUND")
            if not l_det:
                strip_parts.append(f"[{key.upper()}] Left marker not detected")
                any_error = True
            if not r_det:
                strip_parts.append(f"[{key.upper()}] Right marker not detected")
                any_error = True

            # 2. In-plane rotation
            rot = d.get("rot_2d", 0.0)
            rot_ok = rot <= rot_th
            set_row("rot",
                    W_OK if rot_ok else W_WARN,
                    f"{rot:.1f}° {'✔' if rot_ok else '⚠ ROTATED'}")
            if not rot_ok:
                strip_parts.append(f"[{key.upper()}] In-plane rotation {rot:.1f}° > {rot_th}°")
                any_warn = True

            # 3. Left marker pitch (forward / backward tilt)
            lp = d.get("L_pitch", 0.0)
            lp_abs = abs(lp)
            lp_dir = "fwd" if lp > 0 else "bwd"
            lp_ok  = lp_abs <= tilt_th
            set_row("pitch_L",
                    W_OK if lp_ok else W_ERROR,
                    f"{lp:.1f}° {'✔' if lp_ok else ('⚠ TILTED ' + lp_dir.upper())}")
            if not lp_ok:
                strip_parts.append(
                    f"[{key.upper()}] Left pitch {lp:.1f}° ({lp_dir}) > {tilt_th}°")
                any_error = True

            # 4. Right marker pitch
            rp = d.get("R_pitch", 0.0)
            rp_abs = abs(rp)
            rp_dir = "fwd" if rp > 0 else "bwd"
            rp_ok  = rp_abs <= tilt_th
            set_row("pitch_R",
                    W_OK if rp_ok else W_ERROR,
                    f"{rp:.1f}° {'✔' if rp_ok else ('⚠ TILTED ' + rp_dir.upper())}")
            if not rp_ok:
                strip_parts.append(
                    f"[{key.upper()}] Right pitch {rp:.1f}° ({rp_dir}) > {tilt_th}°")
                any_error = True

            # 5. Left marker yaw (left / right tilt)
            ly = d.get("L_yaw", 0.0)
            ly_abs = abs(ly)
            ly_dir = "right" if ly > 0 else "left"
            ly_ok  = ly_abs <= tilt_th
            set_row("yaw_L",
                    W_OK if ly_ok else W_ERROR,
                    f"{ly:.1f}° {'✔' if ly_ok else ('⚠ TILTED ' + ly_dir.upper())}")
            if not ly_ok:
                strip_parts.append(
                    f"[{key.upper()}] Left yaw {ly:.1f}° ({ly_dir}) > {tilt_th}°")
                any_error = True

            # 6. Right marker yaw
            ry = d.get("R_yaw", 0.0)
            ry_abs = abs(ry)
            ry_dir = "right" if ry > 0 else "left"
            ry_ok  = ry_abs <= tilt_th
            set_row("yaw_R",
                    W_OK if ry_ok else W_ERROR,
                    f"{ry:.1f}° {'✔' if ry_ok else ('⚠ TILTED ' + ry_dir.upper())}")
            if not ry_ok:
                strip_parts.append(
                    f"[{key.upper()}] Right yaw {ry:.1f}° ({ry_dir}) > {tilt_th}°")
                any_error = True

            # 7. Depth difference L-R
            lz = d.get("L_z", 0.0); rz = d.get("R_z", 0.0)
            dz = abs(lz - rz)
            dz_ok = dz <= 20.0  # fixed 20 mm depth tolerance
            set_row("depth_diff",
                    W_OK if dz_ok else W_WARN,
                    f"ΔZ={dz:.1f}mm {'✔' if dz_ok else '⚠ CHECK'}")
            if not dz_ok:
                strip_parts.append(f"[{key.upper()}] Depth diff {dz:.1f}mm > 20mm")
                any_warn = True

            # ── Angle value cells ────────────────────────────────────────────
            lr = d.get("L_A", (0.0, 0.0))
            rr = d.get("R_A", (0.0, 0.0))
            ang["L_pitch"].set(f"{lp:+.1f}")
            ang["L_yaw"].set(f"{ly:+.1f}")
            ang["L_roll"].set(f"{lr[0]:+.1f}")
            ang["R_pitch"].set(f"{rp:+.1f}")
            ang["R_yaw"].set(f"{ry:+.1f}")
            ang["R_roll"].set(f"{rr[0]:+.1f}")

        # ── Overall health indicator ─────────────────────────────────────────
        if any_error:
            ov_col  = W_ERROR
            ov_text = "⛔  Issues detected"
        elif any_warn:
            ov_col  = W_WARN
            ov_text = "⚠  Warnings active"
        else:
            ov_col  = W_OK
            ov_text = "✅  All markers healthy"

        self.warn_overall_icon.config(fg=ov_col)
        self.warn_overall_lbl.config(text=ov_text, fg=ov_col)

        # ── Strip label under camera ─────────────────────────────────────────
        if strip_parts:
            self.warn_strip.config(
                text="  ⚠  " + "   |   ".join(strip_parts),
                bg="#3e0a0a", fg=W_ERROR)
        else:
            self.warn_strip.config(
                text="  ✅  All markers OK — no warnings",
                bg="#0a2e1a", fg=W_OK)

    # ══════════════════════════════════════════════════════════════════════════
    #  GUI refresh loop (runs in main thread, every 50 ms)
    # ══════════════════════════════════════════════════════════════════════════
    def update_gui_loop(self):
        # Tab 1: live feed
        if self.current_frame is not None:
            img = ImageTk.PhotoImage(
                image=Image.fromarray(self.current_frame).resize((960, 540)))
            self.canvas.create_image(0, 0, anchor="nw", image=img)
            self.canvas.img = img

        self.lbl_dist_top.config(text=f"{self.last_data['top']['dist']:.3f} mm")
        self.lbl_k_top.config(text=f"k: {self.last_data['top']['k']:.4f}")
        self.lbl_dist_bot.config(text=f"{self.last_data['bottom']['dist']:.3f} mm")
        self.lbl_k_bot.config(text=f"k: {self.last_data['bottom']['k']:.4f}")

        # Tab 3: telemetry
        for key in ["top", "bottom"]:
            d = self.last_data[key]
            for var in ["A", "X", "TR", "BR", "B", "C"]:
                v = d[var]
                self.tele_vars[key][var].set(f"{v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f}")
            lr = d.get("L_A", (0.0, 0.0))
            rr = d.get("R_A", (0.0, 0.0))
            self.tele_vars[key]["L_ROL"].set(f"{lr[0]:.2f}° (roll)")
            self.tele_vars[key]["L_ROT"].set(f"{lr[1]:.2f}° (in-plane)")
            self.tele_vars[key]["L_Z"].set(f"{d['L_z']:.1f} mm")
            self.tele_vars[key]["L_PITCH"].set(f"{d.get('L_pitch', 0.0):.2f}°")
            self.tele_vars[key]["L_YAW"].set(f"{d.get('L_yaw', 0.0):.2f}°")
            self.tele_vars[key]["R_ROL"].set(f"{rr[0]:.2f}° (roll)")
            self.tele_vars[key]["R_ROT"].set(f"{rr[1]:.2f}° (in-plane)")
            self.tele_vars[key]["R_Z"].set(f"{d['R_z']:.1f} mm")
            self.tele_vars[key]["R_PITCH"].set(f"{d.get('R_pitch', 0.0):.2f}°")
            self.tele_vars[key]["R_YAW"].set(f"{d.get('R_yaw', 0.0):.2f}°")

        # Tab 5: camera preview + detection status
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
                    fg=C_YELLOW)
            else:
                self.lbl_cam_status.config(
                    text="❌ No ArUco detected — adjust camera settings", fg=C_RED)

        # Movement monitor tick
        self._mv_tick()

        # Warning dashboard
        self._compute_warnings()

        self.root.after(50, self.update_gui_loop)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app = MeasurementApp(root)
    root.mainloop()
