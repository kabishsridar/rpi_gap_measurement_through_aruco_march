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
COLLECT_N           = 5          # readings to average for initial / final

# Lighting health thresholds (pixel intensity 0-255)
LIGHT_LOW    = 40     # mean brightness below → too dark
LIGHT_HIGH   = 215    # mean brightness above → overexposed
CONTRAST_MIN = 18     # std-dev below → flat / foggy / blown-out

AWB_MODES = {
    "Auto":        0,
    "Tungsten":    1,
    "Fluorescent": 2,
    "Indoor":      3,
    "Daylight":    4,
    "Cloudy":      5,
}

# ── Control Room Aesthetic Palette ───────────────────────────────────────────
C_BG      = "#0f172a"  # Deep slate/navy
C_PANEL   = "#1e293b"  # Rich navy card background
C_CARD    = "#334155"  # Lighter slate for nested cards
C_ACCENT   = "#38bdf8"  # Professional cyan blue
C_TOP      = "#fb923c"  # Vivid orange
C_BOT      = "#c084fc"  # Vivid purple
C_GREEN    = "#4ade80"  # Vibrant green
C_RED      = "#f87171"  # Vibrant red
C_TEXT_BRT = "#f8fafc"  # Off-white
C_TEXT_MED = "#94a3b8"  # Muted blue-gray
C_AMBER    = "#fbbf24"  # Warning amber

# ── Typography ───────────────────────────────────────────────────────────────
F_TITLE = ("Inter", 16, "bold")      # Modern font fallback
F_HEAD  = ("Inter", 13, "bold")
F_BODY  = ("Inter", 12)
F_SMALL = ("Inter", 10)
F_DATA  = ("Inter", 36, "bold")      # Huge, legible numbers
F_MONO  = ("Consolas", 13)
F_BTN   = ("Inter", 14, "bold")


# ══════════════════════════════════════════════════════════════════════════════
#  Euler-angle helper
# ══════════════════════════════════════════════════════════════════════════════
def rotation_to_euler(R):
    """
    Decompose an OpenCV 3×3 rotation matrix into (pitch, yaw, roll) degrees.

    Camera-frame XYZ convention
    ──────────────────────────────────────────────────────────────────────────
    pitch  – rotation around X-axis: marker top tilts toward / away from lens
    yaw    – rotation around Y-axis: left edge closer / right edge closer
              → "one side in, one side out" scenario shows as large |yaw|
    roll   – rotation around Z-axis: in-plane spin (already tracked as rot_2d)
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll  = math.degrees(math.atan2( R[2, 1],  R[2, 2]))
        pitch = math.degrees(math.atan2(-R[2, 0],  sy))
        yaw   = math.degrees(math.atan2( R[1, 0],  R[0, 0]))
    else:                               # gimbal-lock fallback
        roll  = math.degrees(math.atan2(-R[1, 2],  R[1, 1]))
        pitch = math.degrees(math.atan2(-R[2, 0],  sy))
        yaw   = 0.0
    return pitch, yaw, roll


# ══════════════════════════════════════════════════════════════════════════════
#  Application
# ══════════════════════════════════════════════════════════════════════════════
class MeasurementApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Dual-Pair ArUco Measurement v15")
        self.root.geometry("1600x960")
        self.root.configure(bg=C_BG)

        # ── Measurement vars ──────────────────────────────────────────────────
        self.size_top         = tk.DoubleVar(value=DEFAULT_MARKER_SIZE)
        self.size_bot         = tk.DoubleVar(value=DEFAULT_MARKER_SIZE)
        self.fixed_side       = tk.StringVar(value="Left")
        self.rot_threshold    = tk.DoubleVar(value=12.0)
        self.pitch_threshold  = tk.DoubleVar(value=10.0)   # NEW
        self.yaw_threshold    = tk.DoubleVar(value=10.0)   # NEW
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

        # ── Shared data (bg thread writes, GUI thread reads) ──────────────────
        def _empty():
            return {
                "A": (0,0,0), "X": (0,0,0), "TR": (0,0,0), "BR": (0,0,0),
                "B": (0,0,0), "C": (0,0,0), "dist": 0.0, "k": 0.0,
                "L_A": (0.0, 0.0), "R_A": (0.0, 0.0), "rot_2d": 0.0,
                "L_z": 0.0, "R_z": 0.0, "p1_px": None, "p2_px": None,
                # Euler angles (new)
                "L_pitch": 0.0, "L_yaw": 0.0,
                "R_pitch": 0.0, "R_yaw": 0.0,
                # Detection status (new)
                "L_det": False, "R_det": False,
            }

        self.last_data = {
            "top":    _empty(),
            "bottom": _empty(),
            "session_count": 0,
            # Lighting (written by bg thread every frame)
            "lighting": {"status": "no_frame", "mean": 0.0, "std": 0.0},
        }

        self.is_running    = True
        self.current_frame = None

        self._style()
        self.setup_ui()
        threading.Thread(target=self.measurement_loop, daemon=True).start()
        self.update_gui_loop()

    # ══════════════════════════════════════════════════════════════════════════
    #  Style  (v14-identical)
    # ══════════════════════════════════════════════════════════════════════════
    def _style(self):
        s = ttk.Style()
        s.theme_use('clam')
        s.configure("TNotebook", 
                    background=C_BG, 
                    borderwidth=0)
        s.configure("TNotebook.Tab", 
                    background=C_PANEL, 
                    foreground=C_TEXT_MED,
                    padding=[25, 12], 
                    font=F_HEAD, 
                    borderwidth=0)
        s.map("TNotebook.Tab",
              background=[("selected", C_ACCENT)],
              foreground=[("selected", C_BG)])
        s.configure("TFrame", background=C_BG)
        s.configure("TProgressbar", 
                    thickness=24, 
                    background=C_ACCENT, 
                    troughcolor=C_PANEL,
                    borderwidth=0)

    # ══════════════════════════════════════════════════════════════════════════
    #  UI wiring  — 4 tabs, same as v14
    # ══════════════════════════════════════════════════════════════════════════
    def setup_ui(self):
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill='both', expand=True, padx=8, pady=8)
        self._build_tab_live()
        self._build_tab_tele()
        self._build_tab_settings()
        self._build_tab_cam()

    # ── helper: white card frame (v14-identical) ──────────────────────────────
    def _card(self, parent, title, title_color, **pack_kw):
        f = tk.LabelFrame(parent, text=f"  {title.upper()}  ",
                          font=F_HEAD,
                          fg=title_color, bg=C_PANEL,
                          bd=2, relief="flat", highlightbackground=title_color, 
                          highlightthickness=1, padx=20, pady=20)
        f.pack(**pack_kw)
        return f

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 1: MOVEMENT  (v14 layout + warning panel + strip)
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_live(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text=" 📏  Movement ")
        tab.configure(style="TFrame")

        # ── Left: camera feed ─────────────────────────────────────────────────
        left = tk.Frame(tab, bg=C_BG)
        left.pack(side="left", padx=16, pady=16)

        self.canvas = tk.Canvas(left, width=960, height=540, bg="black",
                                highlightthickness=1, highlightbackground=C_PANEL)
        self.canvas.pack()

        # Warning box directly below the canvas (multi-line list)
        self.warn_strip = tk.Label(
            left,
            text="  READY  —  SYSTEM INITIALIZING",
            font=F_HEAD,
            fg=C_ACCENT, bg=C_PANEL,
            anchor="nw", padx=20, pady=15,
            bd=1, relief="solid",
            wraplength=920, justify="left",
            height=6)  # Box height for roughly 6 lines
        self.warn_strip.pack(fill="x", pady=(12, 0))

        # ── Right panel ───────────────────────────────────────────────────────
        right = tk.Frame(tab, bg=C_BG)
        right.pack(side="right", fill="both", expand=True, padx=12, pady=10)

        # Distance cards
        for key, title, col in [("top", "UPPER SENSOR", C_TOP),
                                 ("bottom", "LOWER SENSOR", C_BOT)]:
            card = self._card(right, title, col, fill="x", pady=10, padx=12)
            dl = tk.Label(card, text="0.000 mm",
                          font=F_DATA, fg=C_GREEN, bg=C_PANEL)
            dl.pack(pady=(12, 4))
            kl = tk.Label(card, text="INTERSECT RATIO: 0.0000",
                          font=F_SMALL, fg=C_TEXT_MED, bg=C_PANEL)
            kl.pack(pady=(0, 12))
            if key == "top":
                self.lbl_dist_top, self.lbl_k_top = dl, kl
            else:
                self.lbl_dist_bot, self.lbl_k_bot = dl, kl

        # Status + progress
        stf = tk.Frame(right, bg=C_BG)
        stf.pack(fill="x", pady=(15, 0))
        self.mv_status_lbl = tk.Label(
            stf, text="READY FOR INITIAL CAPTURE",
            font=F_HEAD, fg=C_TEXT_MED, bg=C_BG,
            wraplength=380, justify="center")
        self.mv_status_lbl.pack(pady=(5, 5))

        prog_f = tk.Frame(right, bg=C_BG)
        prog_f.pack(fill="x", padx=12)
        self.mv_prog_lbl = tk.Label(prog_f, text="",
                                    font=F_HEAD, fg=C_GREEN, bg=C_BG)
        self.mv_prog_lbl.pack()
        self.mv_prog_bar = ttk.Progressbar(prog_f, length=380, maximum=COLLECT_N,
                                           mode="determinate")
        self.mv_prog_bar.pack(pady=8)

        # Buttons
        btn_f = tk.Frame(right, bg=C_BG)
        btn_f.pack(pady=20, fill="x", padx=12)
        btn_cfg = dict(font=F_BTN, relief="flat",
                       padx=24, pady=16, bd=0, cursor="hand2")

        def add_hover(btn, normal_bg, hover_bg):
            btn.bind("<Enter>", lambda e: btn.config(bg=hover_bg))
            btn.bind("<Leave>", lambda e: btn.config(bg=normal_bg))

        self.btn_start = tk.Button(
            btn_f, text="▶ START SESSION", bg=C_GREEN, fg=C_BG,
            activebackground="#86efac",
            command=self._mv_start, **btn_cfg)
        self.btn_start.pack(side="left", expand=True, fill="x", padx=6)
        add_hover(self.btn_start, C_GREEN, "#86efac")

        self.btn_stop = tk.Button(
            btn_f, text="■ STOP (SAVE)", bg=C_RED, fg=C_BG,
            activebackground="#fca5a5", state="disabled",
            command=self._mv_stop, **btn_cfg)
        self.btn_stop.pack(side="left", expand=True, fill="x", padx=6)
        add_hover(self.btn_stop, C_RED, "#fca5a5")

        self.btn_reset = tk.Button(
            btn_f, text="↺", bg=C_CARD, fg="white",
            activebackground=C_TEXT_MED,
            command=self._mv_reset, **btn_cfg)
        self.btn_reset.pack(side="left", padx=6)
        add_hover(self.btn_reset, C_CARD, C_TEXT_MED)

        # Delta rows
        tk.Frame(right, bg=C_CARD, height=2).pack(fill="x", padx=12, pady=15)
        
        for key, title, col in [("top", "UPPER", C_TOP), ("bottom", "LOWER", C_BOT)]:
            row = tk.Frame(right, bg=C_PANEL, bd=1, relief="solid")
            row.pack(fill="x", padx=12, pady=8)
            tk.Label(row, text=title, font=F_HEAD,
                     fg=col, bg=C_PANEL, width=8).pack(side="left", padx=12, pady=12)
            info = tk.Frame(row, bg=C_PANEL); info.pack(side="left", expand=True)
            init_lbl  = tk.Label(info, text="INIT: —",
                                 font=F_BODY, fg=C_TEXT_MED, bg=C_PANEL, anchor="w")
            init_lbl.pack(fill="x", padx=8)
            final_lbl = tk.Label(info, text="FINAL: —",
                                 font=F_BODY, fg=C_TEXT_MED, bg=C_PANEL, anchor="w")
            final_lbl.pack(fill="x", padx=8)
            delta_lbl = tk.Label(row, text="—",
                                 font=F_DATA, fg=C_ACCENT, bg=C_PANEL)
            delta_lbl.pack(side="right", padx=20, pady=12)
            if key == "top":
                self.mv_init_lbl_top  = init_lbl
                self.mv_final_lbl_top = final_lbl
                self.mv_delta_lbl_top = delta_lbl
            else:
                self.mv_init_lbl_bot  = init_lbl
                self.mv_final_lbl_bot = final_lbl
                self.mv_delta_lbl_bot = delta_lbl

        # Warnings Dashboard
        warn_f = tk.LabelFrame(right, text="  SENSORY DIAGNOSTICS  ",
                               font=F_HEAD,
                               fg=C_RED, bg=C_PANEL, bd=0, padx=15, pady=15)
        warn_f.pack(fill="x", padx=12, pady=(20, 10))

        # Modern header
        hdr = tk.Frame(warn_f, bg=C_PANEL)
        hdr.pack(fill="x")
        cols = ["PAIR", "L-DET", "R-DET", "ROTATION", "PITCH", "YAW"]
        for col_i, txt in enumerate(cols):
            w = 8 if col_i == 0 else 10
            tk.Label(hdr, text=txt, font=F_SMALL,
                     fg=C_TEXT_MED, bg=C_PANEL, width=w,
                     anchor="center").grid(row=0, column=col_i, padx=2)

        self.warn_dots = {}
        for r_i, (key, title, col) in enumerate(
                [("top", "UPPER", C_TOP), ("bottom", "LOWER", C_BOT)], start=1):
            dots = {}
            tk.Label(hdr, text=title, font=F_HEAD,
                     fg=col, bg=C_PANEL, width=8, anchor="w").grid(
                row=r_i, column=0, padx=(0, 6), pady=6)
            for c_i, field in enumerate(["L_det", "R_det", "rot", "pitch", "yaw"], start=1):
                dot = tk.Label(hdr, text="■", font=("Inter", 20),
                               fg=C_CARD, bg=C_PANEL, width=8, anchor="center")
                dot.grid(row=r_i, column=c_i, padx=2, pady=6)
                dots[field] = dot
            self.warn_dots[key] = dots

        # Lighting "Chips"
        light_f = tk.Frame(warn_f, bg=C_PANEL)
        light_f.pack(fill="x", pady=(15, 0))
        
        def _make_chip(parent, label):
            f = tk.Frame(parent, bg=C_CARD, padx=8, pady=4)
            f.pack(side="left", padx=5)
            tk.Label(f, text=label, font=F_SMALL, fg=C_TEXT_MED, bg=C_CARD).pack(side="left")
            val = tk.Label(f, text="—", font=F_HEAD, fg=C_TEXT_BRT, bg=C_CARD)
            val.pack(side="left", padx=(5, 0))
            return f, val

        tk.Label(light_f, text="LIGHTING:", font=F_HEAD, fg=C_TEXT_BRT, bg=C_PANEL).pack(side="left", padx=(0, 10))
        self._light_brt_chip, self._light_brt_val = _make_chip(light_f, "BRT")
        self._light_ctr_chip, self._light_ctr_val = _make_chip(light_f, "CTR")
        
        self.lbl_lighting_mv = tk.Label(light_f, text="SCANNING...", font=F_HEAD, fg=C_ACCENT, bg=C_PANEL)
        self.lbl_lighting_mv.pack(side="right")

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 2: DUAL TELEMETRY  (v14 + L_PITCH, L_YAW, R_PITCH, R_YAW rows)
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_tele(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text=" 🛸  Dual Telemetry ")
        mtf = tk.Frame(tab, bg=C_BG)
        mtf.pack(fill="both", expand=True)
        self.tele_vars = {"top": {}, "bottom": {}}
        v_show = ["A", "X", "TR", "BR", "B", "C",
                  "L_ROL", "L_ROT", "L_Z", "L_PITCH", "L_YAW",
                  "R_ROL", "R_ROT", "R_Z", "R_PITCH", "R_YAW"]
        for key, title, col in [("top", "TOP", C_TOP), ("bottom", "BOTTOM", C_BOT)]:
            cf = tk.LabelFrame(mtf, text=f" {title} PAIR TELEMETRY ",
                               font=F_TITLE, fg=col, bg=C_BG, padx=15, pady=15)
            cf.pack(side="left", fill="both", expand=True, padx=20, pady=20)
            for v_name in v_show:
                row = tk.Frame(cf, bg=C_PANEL,
                               highlightbackground=C_CARD, highlightthickness=1)
                row.pack(fill="x", padx=10, pady=5)
                tk.Label(row, text=v_name, font=F_HEAD, fg=C_TEXT_MED,
                         bg=C_PANEL).pack(side="left", padx=10, pady=5)
                sv = tk.StringVar(value="—")
                tk.Label(row, textvariable=sv, font=F_MONO, fg=C_TEXT_BRT,
                         bg=C_PANEL).pack(side="right", padx=10, pady=5)
                self.tele_vars[key][v_name] = sv

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 3: MACHINE CONFIGURATION
    #  v14-identical + Pitch Threshold + Yaw Threshold sliders
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_settings(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text=" ⚙  Machine Configuration ")
        sc = tk.Frame(tab, bg=C_BG)
        sc.pack(fill="both", expand=True, padx=80, pady=40)

        rf = tk.LabelFrame(sc, text=" Reference Side (Fixed ArUco) ", bg=C_PANEL,
                           fg=C_TEXT_BRT, font=F_HEAD, padx=20, pady=10)
        rf.pack(fill="x", pady=(0, 12))
        for choice in ["Left", "Right"]:
            tk.Radiobutton(rf, text=choice, variable=self.fixed_side,
                           value=choice, bg=C_PANEL, fg=C_TEXT_BRT,
                           selectcolor=C_BG, activebackground=C_PANEL,
                           font=F_BODY).pack(side="left", padx=24)

        def create_slider(label, var, color, lo, hi, res=0.1):
            f = tk.LabelFrame(sc, text=f" {label} ", bg=C_PANEL,
                              font=F_HEAD, fg=color, padx=25, pady=12)
            f.pack(fill="x", pady=10)
            sl = tk.Scale(f, from_=lo, to=hi, resolution=res, orient="horizontal",
                          variable=var, bg=C_PANEL, fg=C_TEXT_BRT, troughcolor=C_BG,
                          highlightthickness=0, length=600, font=F_BODY)
            sl.pack(side="left", padx=20)
            tk.Entry(f, textvariable=var, width=10, bg=C_BG, fg=C_ACCENT,
                     insertbackground=C_TEXT_BRT, bd=0, font=F_MONO).pack(side="left")
            return sl

        create_slider("Upper Pair Marker Size (mm)",  self.size_top, C_TOP,  10, 200)
        create_slider("Bottom Pair Marker Size (mm)", self.size_bot, C_BOT,  10, 200)

        self.rot_threshold_slider = create_slider(
            "Rotation Threshold °  (in-plane / formula switch)",
            self.rot_threshold, C_TEXT_BRT, 0, 45)

        create_slider(
            "Pitch Threshold °  (forward / backward tilt warning)",
            self.pitch_threshold, "#8e44ad", 1, 45, res=0.5)

        create_slider(
            "Yaw Threshold °  (left / right tilt  —  'one side in' warning)",
            self.yaw_threshold, "#16a085", 1, 45, res=0.5)

        # ── Angle-threshold toggle
        tog_f = tk.LabelFrame(sc, text=" Measurement Logic Options ", bg=C_PANEL,
                              font=F_HEAD, fg=C_TEXT_BRT, padx=25, pady=15)
        tog_f.pack(fill="x", pady=10)

        def _update_tog_label(*_):
            if self.use_angle_thresh.get():
                tog_state_lbl.config(
                    text="ON  — Use Perpendicular Formula (Accurate) below threshold",
                    fg=C_GREEN)
                self.rot_threshold_slider.config(state="normal")
            else:
                tog_state_lbl.config(
                    text="OFF — Always use Perpendicular Formula (v8 High-Precision)",
                    fg=C_RED)
                self.rot_threshold_slider.config(state="disabled")

        tog_row = tk.Frame(tog_f, bg=C_PANEL); tog_row.pack(fill="x")
        self._tog_canvas = tk.Canvas(tog_row, width=80, height=40, bg=C_PANEL,
                                     highlightthickness=0, cursor="hand2")
        self._tog_canvas.pack(side="left", padx=(0, 20))
        tog_state_lbl = tk.Label(tog_row, text="",
                                 font=F_HEAD, bg=C_PANEL, anchor="w")
        tog_state_lbl.pack(side="left", fill="x", expand=True)

        def _draw_toggle():
            self._tog_canvas.delete("all")
            on = self.use_angle_thresh.get()
            bg = C_GREEN if on else "#b2bec3"
            self._tog_canvas.create_oval(4, 4, 76, 36, fill=bg, outline="")
            cx = 58 if on else 22
            self._tog_canvas.create_oval(cx-14, 6, cx+14, 34, fill="white", outline="")

        def _toggle_click(_=None):
            self.use_angle_thresh.set(not self.use_angle_thresh.get())
            _draw_toggle(); _update_tog_label()

        self._tog_canvas.bind("<Button-1>", _toggle_click)
        _draw_toggle(); _update_tog_label()

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 4: CAMERA CONTROLS  (v14-identical + lighting label)
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_cam(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text=" 📷  Camera Controls ")

        cam_left  = tk.Frame(tab, bg=C_BG)
        cam_left.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        cam_right = tk.Frame(tab, bg=C_BG)
        cam_right.pack(side="right", fill="y", padx=8, pady=8, ipadx=4)

        tk.Label(cam_left, text="Live Camera Preview",
                 font=F_TITLE, fg=C_TEXT_BRT,
                 bg=C_BG).pack(pady=(15, 0))
        self.cam_canvas = tk.Canvas(cam_left, width=800, height=480, bg="black",
                                    highlightthickness=2, highlightbackground=C_PANEL)
        self.cam_canvas.pack(padx=20, pady=20)
        self.lbl_cam_status = tk.Label(cam_left, text="⏳ Waiting for camera…",
                                       font=F_TITLE,
                                       fg=C_AMBER, bg=C_BG)
        self.lbl_cam_status.pack(pady=8)
        # Lighting status in camera tab
        self.lbl_lighting_cam = tk.Label(cam_left, text="",
                                         font=F_HEAD,
                                         fg=C_TEXT_MED, bg=C_BG)
        self.lbl_lighting_cam.pack(pady=5)

        tk.Label(cam_right, text="Sensor Configuration",
                 font=F_TITLE, fg=C_TEXT_BRT,
                 bg=C_BG).pack(pady=(20, 10))

        def cam_row(label, var, lo, hi, res, unit=""):
            rf = tk.LabelFrame(cam_right, text=f" {label} ", bg=C_PANEL,
                               font=F_HEAD, fg=C_TEXT_BRT, padx=12, pady=8)
            rf.pack(fill="x", padx=10, pady=5)
            inner = tk.Frame(rf, bg=C_PANEL); inner.pack(fill="x")
            tk.Scale(inner, from_=lo, to=hi, resolution=res,
                     orient="horizontal", variable=var, bg=C_PANEL, fg=C_TEXT_BRT,
                     troughcolor=C_BG, highlightthickness=0,
                     length=300, font=F_SMALL, command=lambda _: self._apply_cam()).pack(side="left")
            tk.Entry(inner, textvariable=var, width=8, bg=C_BG, fg=C_ACCENT,
                     bd=0, font=F_MONO).pack(side="left", padx=8)
            if unit:
                tk.Label(inner, text=unit, bg=C_PANEL,
                         font=F_SMALL, fg=C_TEXT_MED).pack(side="left")

        ae_f = tk.LabelFrame(cam_right, text=" Exposure Control ", bg=C_PANEL,
                             font=F_HEAD, fg=C_TEXT_BRT, padx=12, pady=8)
        ae_f.pack(fill="x", padx=10, pady=5)
        tk.Checkbutton(ae_f, text="Enable Software Auto-Exposure",
                       variable=self.cam_ae, bg=C_PANEL, fg=C_TEXT_BRT,
                       selectcolor=C_BG, activebackground=C_PANEL,
                       font=F_BODY,
                       command=self._apply_cam).pack(anchor="w")

        cam_row("Exposure Time", self.cam_exposure, 100,  66000, 100, "us")
        cam_row("ISO / Gain",    self.cam_gain,     1.0,  16.0,  0.1, "x")

        awb_f = tk.LabelFrame(cam_right, text=" Color Profile / AWB ", bg=C_PANEL,
                             font=F_HEAD, fg=C_TEXT_BRT, padx=12, pady=8)
        awb_f.pack(fill="x", padx=10, pady=5)
        tk.Checkbutton(awb_f, text="Hardware Auto White Balance",
                       variable=self.cam_awb, bg=C_PANEL, fg=C_TEXT_BRT,
                       selectcolor=C_BG, activebackground=C_PANEL,
                       font=F_BODY,
                       command=self._apply_cam).pack(anchor="w")
        wbm_row = tk.Frame(awb_f, bg=C_PANEL); wbm_row.pack(fill="x", pady=5)
        tk.Label(wbm_row, text="Mode:", bg=C_PANEL, fg=C_TEXT_MED,
                 font=F_BODY).pack(side="left")
        for mode_name in AWB_MODES:
            tk.Radiobutton(wbm_row, text=mode_name, variable=self.cam_awb_mode,
                           value=mode_name, bg=C_PANEL, fg=C_TEXT_BRT,
                           selectcolor=C_BG, activebackground=C_PANEL, font=F_SMALL,
                           command=self._apply_cam).pack(side="left", padx=5)

        cam_row("Brightness", self.cam_brightness, -1.0, 1.0, 0.05)
        cam_row("Contrast",   self.cam_contrast,    0.0, 8.0, 0.1)
        cam_row("Saturation", self.cam_saturation,  0.0, 8.0, 0.1)
        cam_row("Sharpness",  self.cam_sharpness,   0.0, 8.0, 0.1)

        tk.Button(cam_right, text="↺  Restore Factory Defaults",
                  font=F_BTN, bg=C_RED, fg="white",
                  activebackground="#c0392b", relief="flat", padx=15, pady=12,
                  command=self._reset_cam).pack(pady=20, fill="x", padx=10)

    # ══════════════════════════════════════════════════════════════════════════
    #  Movement monitor logic  (v14-identical)
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
            text=f"Collecting initial distance... (0 / {COLLECT_N})",
            fg=C_AMBER)

    def _mv_stop(self):
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
            text="Press  START  to capture initial gap", fg="#dfe6e9")
        self.mv_init_lbl_top.config(text="Init: —",   fg=C_TEXT_MED)
        self.mv_init_lbl_bot.config(text="Init: —",   fg=C_TEXT_MED)
        self.mv_final_lbl_top.config(text="Final: —", fg=C_TEXT_MED)
        self.mv_final_lbl_bot.config(text="Final: —", fg=C_TEXT_MED)
        self.mv_delta_lbl_top.config(text="—", fg=C_ACCENT)
        self.mv_delta_lbl_bot.config(text="—", fg=C_ACCENT)

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
                text=f"Collecting initial distance... ({n} / {COLLECT_N})",
                fg=C_AMBER)
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
                    text=f"Init: {self.mv_dist_init['top']:.3f} mm", fg=C_TEXT_MED)
                self.mv_init_lbl_bot.config(
                    text=f"Init: {self.mv_dist_init['bottom']:.3f} mm", fg=C_TEXT_MED)

        elif self.mv_state == "collecting_final":
            for key in ["top", "bottom"]:
                d = self.last_data[key]["dist"]
                if d > 0:
                    self.mv_final_buf[key].append(d)
            n = min(len(self.mv_final_buf["top"]), len(self.mv_final_buf["bottom"]))
            self.mv_prog_bar["value"] = n
            self.mv_prog_lbl.config(text=f"Reading {n} / {COLLECT_N}")
            self.mv_status_lbl.config(
                text=f"Collecting final distance... ({n} / {COLLECT_N})",
                fg=C_AMBER)
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
                for key, il, fl, dl in [
                    ("top",    self.mv_init_lbl_top,  self.mv_final_lbl_top,  self.mv_delta_lbl_top),
                    ("bottom", self.mv_init_lbl_bot,  self.mv_final_lbl_bot,  self.mv_delta_lbl_bot),
                ]:
                    di = self.mv_dist_init[key]; df = self.mv_dist_final[key]
                    delta = df - di
                    il.config(text=f"Init: {di:.3f} mm",   fg=C_TEXT_MED)
                    fl.config(text=f"Final: {df:.3f} mm",  fg=C_TEXT_MED)
                    sign  = "+" if delta >= 0 else ""
                    color = C_RED if delta > 0.5 else (C_GREEN if delta < -0.5 else C_ACCENT)
                    dl.config(text=f"{sign}{delta:.3f} mm", fg=color)

    # ══════════════════════════════════════════════════════════════════════════
    #  Camera control helpers  (v14-identical)
    # ══════════════════════════════════════════════════════════════════════════
    def _apply_cam(self, *_):
        if self.pc is None:
            return
        controls = {}
        ae_on = self.cam_ae.get(); controls["AeEnable"] = ae_on
        if not ae_on:
            controls["ExposureTime"] = int(self.cam_exposure.get())
            controls["AnalogueGain"] = float(self.cam_gain.get())
        awb_on = self.cam_awb.get(); controls["AwbEnable"] = awb_on
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
    #  Calibration  (v14-identical)
    # ══════════════════════════════════════════════════════════════════════════
    def load_calib(self):
        f = "camera_params_2.npz" if self.fixed_side.get() == "Right" else "camera_params.npz"
        if os.path.exists(f):
            d = np.load(f)
            return d['camera_matrix'], d['dist_coeff']
        return (np.array([[1280, 0, 640], [0, 1280, 360], [0, 0, 1]], dtype=np.float32),
                np.zeros(5))

    # ══════════════════════════════════════════════════════════════════════════
    #  Measurement loop  (background thread)
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

            # ── Lighting analysis (every frame, very cheap) ───────────────────
            gray_full = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            mean_b = float(np.mean(gray_full))
            std_b  = float(np.std(gray_full))
            if mean_b < LIGHT_LOW:
                light_status = "dark"
            elif mean_b > LIGHT_HIGH:
                light_status = "bright"
            elif std_b < CONTRAST_MIN:
                light_status = "flat"
            else:
                light_status = "ok"
            self.last_data["lighting"] = {
                "status": light_status,
                "mean":   mean_b,
                "std":    std_b,
            }

            # ── ArUco detection ───────────────────────────────────────────────
            corners_raw, ids, _ = detector.detectMarkers(gray_full)
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

                # ── proc: pose + euler angles ─────────────────────────────────
                def proc(marker_list, pair_key, size):
                    if len(marker_list) < 2:
                        self.last_data[pair_key]["dist"]  = 0.0
                        self.last_data[pair_key]["p1_px"] = None
                        self.last_data[pair_key]["L_det"] = False
                        self.last_data[pair_key]["R_det"] = False
                        return

                    marker_list.sort(key=lambda m: m["x"])
                    left_m  = marker_list[0];  right_m = marker_list[-1]
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
                        # roll estimate matching v14
                        roll_r = math.degrees(math.atan2(R[1, 0], R[0, 0]))
                        # full euler decomposition
                        pitch, yaw, _ = rotation_to_euler(R)
                        return pts3d, roll_r, float(tv[2, 0]), pitch, yaw

                    def inplane_rot(c_sorted):
                        return abs(math.degrees(math.atan2(
                            c_sorted[1, 1] - c_sorted[0, 1],
                            c_sorted[1, 0] - c_sorted[0, 0])))

                    S_pts, S_roll, S_z, S_pitch, S_yaw = get_pose(S_m["c_raw"])
                    T_pts, T_roll, T_z, T_pitch, T_yaw = get_pose(T_m["c_raw"])
                    S_irot = inplane_rot(S_m["c"])
                    T_irot = inplane_rot(T_m["c"])

                    def inner_outer(pts, inner_is_right):
                        order = pts[:, 0].argsort()
                        inner = pts[order[2:]] if inner_is_right else pts[order[:2]]
                        outer = pts[order[:2]] if inner_is_right else pts[order[2:]]
                        inner = inner[inner[:, 1].argsort()]
                        outer = outer[outer[:, 1].argsort()]
                        return inner[0], inner[1], outer[0]

                    S_inner_top, S_inner_bot, S_outer_top = inner_outer(S_pts, not is_rf)
                    A  = (S_inner_top + S_inner_bot) / 2.0
                    T_inner_top, T_inner_bot, _ = inner_outer(T_pts, is_rf)
                    Bv = T_inner_top; Cv = T_inner_bot; X_alt = (Bv + Cv) / 2.0

                    if not is_rf:
                        p_src_2d = tuple(((S_m["c"][1]+S_m["c"][2])/2).astype(int))
                        p_tgt_2d = tuple(((T_m["c"][0]+T_m["c"][3])/2).astype(int))
                    else:
                        p_src_2d = tuple(((S_m["c"][0]+S_m["c"][3])/2).astype(int))
                        p_tgt_2d = tuple(((T_m["c"][1]+T_m["c"][2])/2).astype(int))

                    buffers[pair_key].append({
                        "A": A, "X_alt": X_alt,
                        "TR": S_inner_top, "BR": S_inner_bot, "TL_ref": S_outer_top,
                        "B": Bv, "C": Cv,
                        "L_A": (S_roll, S_irot), "R_A": (T_roll, T_irot),
                        "rot": max(S_irot, T_irot),
                        "L_z": S_z, "R_z": T_z,
                        "L_pitch": S_pitch, "L_yaw": S_yaw,
                        "R_pitch": T_pitch, "R_yaw": T_yaw,
                        "p1_px": p_src_2d, "p2_px": p_tgt_2d,
                    })
                    self.last_data[pair_key]["L_det"] = True
                    self.last_data[pair_key]["R_det"] = True

                if (curr - l_s) >= 100:
                    proc(top_m, "top",    self.size_top.get())
                    proc(bot_m, "bottom", self.size_bot.get())
                    l_s = curr

                # ── 1-second averaging window (v14-identical + new fields) ──
                if (curr - l_u) >= 1000:
                    for key in ["top", "bottom"]:
                        if buffers[key]:
                            s = buffers[key]
                            aA      = np.mean([x["A"]       for x in s], axis=0)
                            aTR     = np.mean([x["TR"]      for x in s], axis=0)
                            aTL     = np.mean([x["TL_ref"]  for x in s], axis=0)
                            aBR     = np.mean([x["BR"]      for x in s], axis=0)
                            aB      = np.mean([x["B"]       for x in s], axis=0)
                            aC      = np.mean([x["C"]       for x in s], axis=0)
                            aX_alt  = np.mean([x["X_alt"]   for x in s], axis=0)
                            aL      = np.mean([x["L_A"]     for x in s], axis=0)
                            aR      = np.mean([x["R_A"]     for x in s], axis=0)
                            avg_rot    = float(np.mean([x["rot"]     for x in s]))
                            avg_Lz     = float(np.mean([x["L_z"]     for x in s]))
                            avg_Rz     = float(np.mean([x["R_z"]     for x in s]))
                            avg_Lpitch = float(np.mean([x["L_pitch"] for x in s]))
                            avg_Lyaw   = float(np.mean([x["L_yaw"]   for x in s]))
                            avg_Rpitch = float(np.mean([x["R_pitch"] for x in s]))
                            avg_Ryaw   = float(np.mean([x["R_yaw"]   for x in s]))
                            p1_px = tuple(np.mean([x["p1_px"] for x in s],
                                                  axis=0).astype(int))
                            p2_px = tuple(np.mean([x["p2_px"] for x in s],
                                                  axis=0).astype(int))

                            # ── Distance formula (v14-identical) ─────────────
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

                            # CSV + DB (v14-identical)
                            v_vec = aTR - aTL; u_vec = aA; w_vec = aC - aB
                            try:
                                log.record(dv, kv, aA, aX, aTR, aBR, aB, aC,
                                           u_vec, v_vec, w_vec, aL, aR)
                            except Exception:
                                pass

                            # Image capture during movement phase
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
                # No markers at all
                for key in ["top", "bottom"]:
                    self.last_data[key]["dist"]  = 0.0
                    self.last_data[key]["L_det"] = False
                    self.last_data[key]["R_det"] = False

            # ── Draw measurement lines + warning overlays on frame ────────────
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

                    # Warning badge lines
                    p_th = self.pitch_threshold.get()
                    y_th = self.yaw_threshold.get()
                    r_th = self.rot_threshold.get()
                    wlines = []
                    max_rot = max(d["L_A"][1], d["R_A"][1])
                    if max_rot > r_th:
                        wlines.append(f"ROT {max_rot:.1f}deg")
                    lp = d.get("L_pitch", 0.0)
                    if abs(lp) > p_th:
                        wlines.append(f"L-PITCH {lp:+.1f}d {'FWD' if lp>0 else 'BWD'}")
                    rp = d.get("R_pitch", 0.0)
                    if abs(rp) > p_th:
                        wlines.append(f"R-PITCH {rp:+.1f}d {'FWD' if rp>0 else 'BWD'}")
                    ly = d.get("L_yaw", 0.0)
                    if abs(ly) > y_th:
                        wlines.append(f"L-YAW {ly:+.1f}d {'R' if ly>0 else 'L'}")
                    ry = d.get("R_yaw", 0.0)
                    if abs(ry) > y_th:
                        wlines.append(f"R-YAW {ry:+.1f}d {'R' if ry>0 else 'L'}")
                    if wlines:
                        # Simplified summary box above the ArUco marker
                        bx1 = max(0, p1[0] - 5)
                        by1 = max(0, p1[1] - 45)
                        cv.rectangle(frame, (bx1, by1), (bx1 + 140, by1 + 28), (20, 20, 20), -1)
                        cv.rectangle(frame, (bx1, by1), (bx1 + 140, by1 + 28), (0, 30, 180), 2)
                        cv.putText(frame, f"ISSUES: {len(wlines)}",
                                   (bx1 + 8, by1 + 20),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.5,
                                   (0, 220, 255), 2)

            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            self.current_frame      = rgb
            self._cam_preview_frame = rgb

    # ══════════════════════════════════════════════════════════════════════════
    #  Warning widget update  (GUI thread, called every 50 ms)
    # ══════════════════════════════════════════════════════════════════════════
    def _update_warnings(self):
        p_th = self.pitch_threshold.get()
        y_th = self.yaw_threshold.get()
        r_th = self.rot_threshold.get()
        strip_parts = []

        for key in ["top", "bottom"]:
            d    = self.last_data[key]
            dots = self.warn_dots[key]

            # Detection
            l_det = d.get("L_det", False)
            r_det = d.get("R_det", False)
            dots["L_det"].config(fg=C_GREEN if l_det else C_RED)
            dots["R_det"].config(fg=C_GREEN if r_det else C_RED)
            if not l_det:
                strip_parts.append(f"[{key.upper()}] L-marker not detected")
            if not r_det:
                strip_parts.append(f"[{key.upper()}] R-marker not detected")

            # In-plane rotation
            rot    = d.get("rot_2d", 0.0)
            rot_ok = rot <= r_th
            dots["rot"].config(fg=C_GREEN if rot_ok else C_AMBER)
            if not rot_ok:
                strip_parts.append(f"[{key.upper()}] Rotation {rot:.1f}°")

            # Pitch (forward / backward tilt)
            lp     = d.get("L_pitch", 0.0)
            rp     = d.get("R_pitch", 0.0)
            p_ok   = abs(lp) <= p_th and abs(rp) <= p_th
            dots["pitch"].config(fg=C_GREEN if p_ok else C_RED)
            if abs(lp) > p_th:
                strip_parts.append(
                    f"[{key.upper()}] L-pitch {lp:+.1f}° ({'FWD' if lp>0 else 'BWD'})")
            if abs(rp) > p_th:
                strip_parts.append(
                    f"[{key.upper()}] R-pitch {rp:+.1f}° ({'FWD' if rp>0 else 'BWD'})")

            # Yaw (left / right tilt — "one side in")
            ly   = d.get("L_yaw", 0.0)
            ry   = d.get("R_yaw", 0.0)
            y_ok = abs(ly) <= y_th and abs(ry) <= y_th
            dots["yaw"].config(fg=C_GREEN if y_ok else C_RED)
            if abs(ly) > y_th:
                strip_parts.append(
                    f"[{key.upper()}] L-yaw {ly:+.1f}° ({'RIGHT' if ly>0 else 'LEFT'})")
            if abs(ry) > y_th:
                strip_parts.append(
                    f"[{key.upper()}] R-yaw {ry:+.1f}° ({'RIGHT' if ry>0 else 'LEFT'})")

        # Lighting
        li   = self.last_data.get("lighting", {})
        ls   = li.get("status", "no_frame")
        mean_b = li.get("mean", 0.0)
        std_b  = li.get("std",  0.0)

        # Update chips
        self._light_brt_val.config(text=f"{mean_b:.0f}")
        self._light_ctr_val.config(text=f"{std_b:.0f}")
        
        brt_ok = ls not in ["dark", "bright"]
        self._light_brt_chip.config(bg=C_GREEN if brt_ok else C_RED)
        self._light_ctr_chip.config(bg=C_GREEN if ls != "flat" else C_AMBER)
        
        if ls == "ok":
            light_txt = "NOMINAL"
            light_col = C_GREEN
        elif ls == "dark":
            light_txt = "UNDEREXPOSED"
            light_col = C_RED
            strip_parts.append(f"CRITICAL: Ambient light too low ({mean_b:.0f})")
        elif ls == "bright":
            light_txt = "OVEREXPOSED"
            light_col = C_AMBER
            strip_parts.append(f"WARNING: Light exposure high ({mean_b:.0f})")
        elif ls == "flat":
            light_txt = "LOW CONTRAST"
            light_col = C_AMBER
            strip_parts.append(f"WARNING: Image contrast degraded ({std_b:.0f})")
        else:
            light_txt = "INIT..."; light_col = C_ACCENT

        self.lbl_lighting_mv.config(text=light_txt, fg=light_col)
        self.lbl_lighting_cam.config(text=f"LIGHTING PROFILE: {light_txt}", fg=light_col)

        # Warning box update (as a list)
        if strip_parts:
            is_crit = any("CRITICAL" in x for x in strip_parts)
            list_text = "![!] ACTIVE ALERTS:\n" + "\n".join([f" • {x}" for x in strip_parts])
            self.warn_strip.config(
                text=list_text,
                fg=C_RED if is_crit else C_AMBER, bg=C_PANEL)
        else:
            self.warn_strip.config(
                text="[OK]  SYSTEM OPERATIONAL\n • All marker pairs detected correctly\n • Environmental lighting is nominal\n • All tilt/rotation values within tolerance",
                fg=C_GREEN, bg=C_PANEL)

    # ══════════════════════════════════════════════════════════════════════════
    #  GUI refresh loop  (main thread, 50 ms)
    # ══════════════════════════════════════════════════════════════════════════
    def update_gui_loop(self):
        # Tab 1: live canvas
        if self.current_frame is not None:
            img = ImageTk.PhotoImage(
                image=Image.fromarray(self.current_frame).resize((960, 540)))
            self.canvas.create_image(0, 0, anchor="nw", image=img)
            self.canvas.img = img

        self.lbl_dist_top.config(text=f"{self.last_data['top']['dist']:.3f} mm")
        self.lbl_k_top.config(text=f"INTERSECT RATIO: {self.last_data['top']['k']:.4f}")
        self.lbl_dist_bot.config(text=f"{self.last_data['bottom']['dist']:.3f} mm")
        self.lbl_k_bot.config(text=f"INTERSECT RATIO: {self.last_data['bottom']['k']:.4f}")

        # Tab 2: telemetry
        for key in ["top", "bottom"]:
            d, tv = self.last_data[key], self.tele_vars[key]
            # Use Z-component only for A and X as requested
            tv["A"].set(f"{d['A'][2]:.2f}")
            tv["X"].set(f"{d['X'][2]:.2f}")
            
            # TR, BR, B, C can stay as triplets or simplified? User didn't specify, but I'll keep them as triplets for now unless asked.
            for var in ["TR", "BR", "B", "C"]:
                v = d[var]
                tv[var].set(f"{v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f}")

            lr, rr = d.get("L_A", (0.0, 0.0)), d.get("R_A", (0.0, 0.0))
            tv["L_ROL"].set(f"{lr[0]:.2f}° (roll)")
            tv["L_ROT"].set(f"{lr[1]:.2f}° (in-plane)")
            tv["L_Z"].set(f"{d['L_z']:.1f}")
            tv["L_PITCH"].set(f"{d.get('L_pitch', 0.0):+.2f}")
            tv["L_YAW"].set(f"{d.get('L_yaw',   0.0):+.2f}")
            
            tv["R_ROL"].set(f"{rr[0]:.2f}° (roll)")
            tv["R_ROT"].set(f"{rr[1]:.2f}° (in-plane)")
            tv["R_Z"].set(f"{d['R_z']:.1f}")
            tv["R_PITCH"].set(f"{d.get('R_pitch', 0.0):+.2f}")
            tv["R_YAW"].set(f"{d.get('R_yaw',   0.0):+.2f}")

        # Tab 4: camera preview + detection status
        if self._cam_preview_frame is not None:
            cam_img = ImageTk.PhotoImage(
                image=Image.fromarray(self._cam_preview_frame).resize((800, 480)))
            self.cam_canvas.create_image(0, 0, anchor="nw", image=cam_img)
            self.cam_canvas.img = cam_img
            top_ok = self.last_data["top"]["dist"]    > 0
            bot_ok = self.last_data["bottom"]["dist"] > 0
            if top_ok and bot_ok:
                self.lbl_cam_status.config(text="STATUS: Both pairs detected", fg=C_GREEN)
            elif top_ok or bot_ok:
                self.lbl_cam_status.config(
                    text=f"STATUS: Only {'TOP' if top_ok else 'BOTTOM'} pair detected",
                    fg=C_AMBER)
            else:
                self.lbl_cam_status.config(
                    text="STATUS: No ArUco detected — adjust camera settings", fg=C_RED)

        # Movement monitor tick
        self._mv_tick()

        # Warning panel + strip update
        self._update_warnings()

        self.root.after(50, self.update_gui_loop)


if __name__ == "__main__":
    root = tk.Tk()
    app = MeasurementApp(root)
    root.mainloop()
