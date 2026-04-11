"""
Dual-Pair ArUco Gap Measurement — v15 (Industrial Professional Edition)
Robust v14/v15 math + Industrial High-Visibility UI.
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

LIGHT_LOW    = 40     
LIGHT_HIGH   = 215    
CONTRAST_MIN = 18     

AWB_MODES = {
    "Auto": 0, "Tungsten": 1, "Fluorescent": 2,
    "Indoor": 3, "Daylight": 4, "Cloudy": 5,
}

# ── Industrial high-contrast dark palette ──────────────────────────────────────
C_BG      = "#02040a"    # deep midnight black
C_SURF    = "#0d1117"    # standard panel background
C_SURF2   = "#161b22"    # highlighted rows
C_BORDER  = "#30363d"    # medium gray border
C_TEXT    = "#f0f6fc"    # clean white text
C_MUTED   = "#8b949e"    # steel gray text
C_TOP     = "#ff9b4f"    # vibrant orange
C_BOT     = "#bc8cff"    # soft lavender
C_GREEN   = "#39d353"    # industrial green
C_RED     = "#ff443a"    # emergency red
C_AMBER   = "#e3b341"    # caution amber
C_BLUE    = "#58a6ff"    # info blue
C_TEAL    = "#1fdbd1"    # cyan accent
C_WHITE   = "#ffffff"

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
# ══════════════════════════════════════════════════════════════════════════

class MeasurementApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Dual ArUco Precision Measurement  │  INDUSTRIAL v15")
        self.root.geometry("1820x960")
        self.root.configure(bg=C_BG)

        # Measurement variables
        self.size_top         = tk.DoubleVar(value=DEFAULT_MARKER_SIZE)
        self.size_bot         = tk.DoubleVar(value=DEFAULT_MARKER_SIZE)
        self.fixed_side       = tk.StringVar(value="Left")
        self.rot_threshold    = tk.DoubleVar(value=15.0)
        self.pitch_threshold  = tk.DoubleVar(value=20.0)
        self.yaw_threshold    = tk.DoubleVar(value=20.0)
        self.use_angle_thresh = tk.BooleanVar(value=False)

        # Camera variables
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

        # Data structure
        def _empty():
            return {
                "A": (0,0,0), "X": (0,0,0), "dist": 0.0, "k": 0.0,
                "TR": (0,0,0), "BR": (0,0,0), "B": (0,0,0), "C": (0,0,0),
                "rot_2d": 0.0, "L_z": 0.0, "R_z": 0.0,
                "L_pitch": 0.0, "L_yaw": 0.0, "L_A": (0.0, 0.0),
                "R_pitch": 0.0, "R_yaw": 0.0, "R_A": (0.0, 0.0),
                "L_det": False, "R_det": False,
                "p1_px": None, "p2_px": None
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

    def _style(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TNotebook", background=C_BG, borderwidth=0)
        s.configure("TNotebook.Tab", background=C_SURF, foreground=C_MUTED,
                    padding=[24, 12], font=("Helvetica", 12, "bold"))
        s.map("TNotebook.Tab", background=[("selected", C_SURF2)], foreground=[("selected", C_TEXT)])
        s.configure("TFrame", background=C_BG)
        s.configure("Industrial.Horizontal.TProgressbar", troughcolor=C_SURF,
                    background=C_GREEN, bordercolor=C_BORDER, thickness=12)

    def setup_ui(self):
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True)
        self._build_tab_live()
        self._build_tab_warnings()
        self._build_tab_tele()
        self._build_tab_settings()
        self._build_tab_cam()

    def _section(self, parent, title, accent=C_TEAL, **pack_kw):
        outer = tk.Frame(parent, bg=C_BORDER, padx=1, pady=1)
        outer.pack(**pack_kw)
        inner = tk.Frame(outer, bg=C_SURF, padx=16, pady=16)
        inner.pack(fill="both", expand=True)
        if title:
            hdr = tk.Frame(inner, bg=C_SURF)
            hdr.pack(fill="x", pady=(0, 12))
            tk.Label(hdr, text=title.upper(), font=("Helvetica", 11, "bold"),
                     fg=accent, bg=C_SURF).pack(side="left")
            tk.Frame(hdr, bg=C_BORDER, height=2).pack(side="left", fill="x", expand=True, padx=(12, 0))
        return inner

    def _chip_row(self, parent, description):
        row = tk.Frame(parent, bg=C_SURF2, pady=4)
        row.pack(fill="x", pady=4)
        strip = tk.Frame(row, bg=C_MUTED, width=6)
        strip.pack(side="left", fill="y")
        chip = tk.Label(row, text=" ──  ", font=("Courier", 14, "bold"),
                        fg=C_BG, bg=C_MUTED, padx=10, pady=8, width=7)
        chip.pack(side="left", padx=(10, 0))
        tk.Label(row, text=description, font=("Helvetica", 14),
                 fg=C_TEXT, bg=C_SURF2, anchor="w").pack(side="left", fill="x", expand=True, padx=15)
        val = tk.Label(row, text="—", font=("Courier", 16, "bold"),
                       fg=C_MUTED, bg=C_SURF2, width=20, anchor="e")
        val.pack(side="right", padx=15)
        chip._strip = strip
        return chip, val

    def _set_chip(self, chip, val, state, value_text=""):
        bg, fg, label = CHIP[state]
        chip.config(bg=bg, fg=fg, text=label)
        chip._strip.config(bg=bg)
        col = bg if state != "wait" else C_MUTED
        val.config(text=value_text, fg=col)

    def _build_tab_live(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="  📊  Real-Time Gap  ")
        lp = tk.Frame(tab, bg=C_BG)
        lp.pack(side="left", fill="both", expand=True, padx=20, pady=20)
        self.canvas = tk.Canvas(lp, width=1080, height=608, bg="#000000", highlightthickness=2, highlightbackground=C_BORDER)
        self.canvas.pack(fill="both", expand=True)
        self.warn_strip = tk.Label(lp, text="System Standby", font=("Helvetica", 16, "bold"),
                                   fg=C_MUTED, bg=C_SURF, pady=15, padx=20, anchor="w")
        self.warn_strip.pack(fill="x", pady=(10, 0))
        rp = tk.Frame(tab, bg=C_BG, width=450)
        rp.pack(side="right", fill="y", padx=(0, 20), pady=20)
        rp.pack_propagate(False)
        for key, label, accent in [("top", "UPPER GAP", C_TOP), ("bottom", "LOWER GAP", C_BOT)]:
            card = self._section(rp, label, accent=accent, fill="x", pady=(0, 15))
            dl = tk.Label(card, text="00.000", font=("Helvetica", 72, "bold"), fg=accent, bg=C_SURF)
            dl.pack()
            kl = tk.Label(card, text="k-factor: —", font=("Courier", 14), fg=C_MUTED, bg=C_SURF)
            kl.pack()
            sl = tk.Label(card, text="● SENSORS STABLE", font=("Helvetica", 12, "bold"), fg=C_GREEN, bg=C_SURF, pady=5)
            sl.pack()
            if key == "top": self.lbl_dist_top, self.lbl_k_top, self.lbl_status_top = dl, kl, sl
            else: self.lbl_dist_bot, self.lbl_k_bot, self.lbl_status_bot = dl, kl, sl
        cp = self._section(rp, "Sequencer", fill="x", pady=(0, 15))
        self.mv_status_lbl = tk.Label(cp, text="Ready for Measurement", font=("Helvetica", 14, "bold"), fg=C_TEAL, bg=C_SURF)
        self.mv_status_lbl.pack(pady=5)
        self.mv_prog_bar = ttk.Progressbar(cp, mode="determinate", maximum=COLLECT_N, style="Industrial.Horizontal.TProgressbar")
        self.mv_prog_bar.pack(fill="x", pady=10)
        br = tk.Frame(cp, bg=C_SURF)
        br.pack(fill="x", pady=5)
        btn_style = {"font": ("Helvetica", 16, "bold"), "relief": "flat", "pady": 15}
        self.btn_start = tk.Button(br, text="START GAP", bg=C_GREEN, fg=C_BG, command=self._mv_start, **btn_style)
        self.btn_start.pack(side="left", expand=True, fill="x", padx=(0, 5))
        self.btn_stop = tk.Button(br, text="FINAL GAP", bg=C_RED, fg=C_WHITE, state="disabled", command=self._mv_stop, **btn_style)
        self.btn_stop.pack(side="left", expand=True, fill="x")
        dr = self._section(rp, "Movement Delta", fill="x")
        for key, label, accent in [("top", "UPPER", C_TOP), ("bottom", "LOWER", C_BOT)]:
            row = tk.Frame(dr, bg=C_SURF2, pady=10)
            row.pack(fill="x", pady=5)
            tk.Label(row, text=label, font=("Helvetica", 12, "bold"), fg=accent, bg=C_SURF2, width=8).pack(side="left")
            delta_val = tk.Label(row, text="—", font=("Helvetica", 32, "bold"), fg=C_BLUE, bg=C_SURF2)
            delta_val.pack(side="right", padx=15)
            if key == "top": self.mv_delta_lbl_top = delta_val
            else: self.mv_delta_lbl_bot = delta_val

    def _build_tab_warnings(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="  ⚠  Diagnostics  ")
        main = tk.Frame(tab, bg=C_BG, padx=30, pady=20)
        main.pack(fill="both", expand=True)
        hh = tk.Frame(main, bg=C_BG); hh.pack(fill="x", pady=(0, 20))
        self.overall_frame = tk.Frame(hh, bg=C_MUTED, padx=30, pady=15); self.overall_frame.pack(side="left")
        self.overall_lbl = tk.Label(self.overall_frame, text="SYSTEM INITIALIZING", font=("Helvetica", 20, "bold"), fg=C_BG, bg=C_MUTED); self.overall_lbl.pack()
        tc = tk.Frame(hh, bg=C_BG); tc.pack(side="right")
        def _slider(parent, lbl, var, col):
            f = tk.Frame(parent, bg=C_SURF, padx=15, pady=10); f.pack(side="left", padx=10)
            tk.Label(f, text=lbl, font=("Helvetica", 10, "bold"), fg=col, bg=C_SURF).pack(anchor="w")
            r = tk.Frame(f, bg=C_SURF); r.pack()
            tk.Scale(r, from_=0, to=180, orient="horizontal", variable=var, bg=C_SURF, fg=C_TEXT, length=250, highlightthickness=0).pack(side="left")
            tk.Label(r, textvariable=var, font=("Courier", 14, "bold"), fg=col, bg=C_SURF, width=4).pack(side="left")
        _slider(tc, "PITCH LIMIT", self.pitch_threshold, C_AMBER); _slider(tc, "YAW LIMIT", self.yaw_threshold, C_TOP)
        dg = tk.Frame(main, bg=C_BG); dg.pack(fill="both", expand=True)
        self._warn_refs = {}
        for i, (key, label, accent) in enumerate([("top", "UPPER SENSORS", C_TOP), ("bottom", "LOWER SENSORS", C_BOT)]):
            sec = self._section(dg, label, accent=accent, side="left", fill="both", expand=True, padx=(0, 20) if i==0 else 0)
            refs = {}
            for k2, desc in [("L_det", "LEFT Detected"), ("R_det", "RIGHT Detected"), ("rot", "In-Plane Rotation"), 
                             ("L_pitch", "L-Pitch (Tilt)"), ("R_pitch", "R-Pitch (Tilt)"),
                             ("L_yaw", "L-Yaw (Tilt)"), ("R_yaw", "R-Yaw (Tilt)")]:
                chip, val = self._chip_row(sec, desc); refs[k2] = (chip, val)
            am = tk.Frame(sec, bg=C_SURF2, pady=10); am.pack(fill="x", pady=15)
            tk.Label(am, text="LIVE SENSOR ANGLES", font=("Helvetica", 10, "bold"), fg=C_MUTED, bg=C_SURF2).pack()
            av = {}
            for ak, al, ac in [("L_pitch", "L-P", C_AMBER), ("L_yaw", "L-Y", C_TOP), ("L_roll", "L-R", C_MUTED),
                               ("R_pitch", "R-P", C_AMBER), ("R_yaw", "R-Y", C_TOP), ("R_roll", "R-R", C_MUTED)]:
                c = tk.Frame(am, bg=C_SURF2); c.pack(side="left", expand=True)
                tk.Label(c, text=al, font=("Helvetica", 9, "bold"), fg=ac, bg=C_SURF2).pack()
                s = tk.StringVar(value="—")
                tk.Label(c, textvariable=s, font=("Courier", 16, "bold"), fg=C_TEXT, bg=C_SURF2).pack(); av[ak] = s
            refs["_ang"] = av; self._warn_refs[key] = refs
        ls = self._section(main, "Environmental Conditions", accent=C_BLUE, fill="x", pady=(20, 0))
        self._light_brt_chip, self._light_brt_val = self._chip_row(ls, "Brightness Level")
        self._light_ctr_chip, self._light_ctr_val = self._chip_row(ls, "Contrast Stability")

    def _build_tab_tele(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="  🛸  3D Metrics  ")
        m = tk.Frame(tab, bg=C_BG, padx=40, pady=20); m.pack(fill="both", expand=True)
        self.tele_vars = {"top": {}, "bottom": {}}
        v = [("A", "Source Center"), ("X", "Gap Intersect"), ("L_PITCH", "L-Pitch"), ("L_YAW", "L-Yaw"), ("R_PITCH", "R-Pitch"), ("R_YAW", "R-Yaw"), ("L_Z", "L-Depth"), ("R_Z", "R-Depth")]
        for key, lab, acc in [("top", "UPPER TRIANGULATION", C_TOP), ("bottom", "LOWER TRIANGULATION", C_BOT)]:
            s = self._section(m, lab, accent=acc, side="left", fill="both", expand=True, padx=(0, 20) if key=="top" else 0)
            for vk, vl in v:
                r = tk.Frame(s, bg=C_SURF2, pady=5); r.pack(fill="x", pady=2)
                tk.Label(r, text=vl, font=("Helvetica", 14), fg=C_MUTED, bg=C_SURF2, width=20, anchor="w").pack(side="left", padx=15)
                sv = tk.StringVar(value="—")
                tk.Label(r, textvariable=sv, font=("Courier", 14, "bold"), fg=C_TEXT, bg=C_SURF2).pack(side="right", padx=15); self.tele_vars[key][vk] = sv

    def _build_tab_settings(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="  ⚙  Parameters  ")
        f = tk.Frame(tab, bg=C_BG, padx=100, pady=50); f.pack(fill="both", expand=True)
        rs = self._section(f, "Reference Orientation", fill="x", pady=20); rb = tk.Frame(rs, bg=C_SURF); rb.pack()
        for c in ["Left", "Right"]:
            tk.Radiobutton(rb, text=c, variable=self.fixed_side, value=c, bg=C_SURF, fg=C_TEXT, selectcolor=C_BLUE, indicatoron=0, font=("Helvetica", 20, "bold"), padx=60, pady=20, relief="flat").pack(side="left", padx=10)
        def _sb(title, var, acc, lo, hi):
            s = self._section(f, title, accent=acc, fill="x", pady=10); r = tk.Frame(s, bg=C_SURF); r.pack(fill="x")
            tk.Scale(r, from_=lo, to=hi, orient="horizontal", variable=var, bg=C_SURF, fg=C_TEXT, length=600, highlightthickness=0).pack(side="left")
            tk.Entry(r, textvariable=var, width=10, font=("Courier", 18), bg=C_SURF2, fg=C_TEXT, relief="flat").pack(side="right")
        _sb("Upper Marker Size (mm)", self.size_top, C_TOP, 10, 200)
        _sb("Lower Marker Size (mm)", self.size_bot, C_BOT, 10, 200)
        _sb("Software Filter Threshold (deg)", self.rot_threshold, C_MUTED, 0, 180)

    def _build_tab_cam(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="  📷  Optics  ")
        m = tk.Frame(tab, bg=C_BG, padx=20, pady=20); m.pack(fill="both", expand=True)
        pv = self._section(m, "Optical Stream", side="left", fill="both", expand=True, padx=(0, 20))
        self.cam_canvas = tk.Canvas(pv, width=800, height=480, bg="black", highlightthickness=0); self.cam_canvas.pack(expand=True)
        self.lbl_cam_status = tk.Label(pv, text="DETECTION MODE ACTIVE", font=("Helvetica", 14, "bold"), fg=C_GREEN, bg=C_SURF); self.lbl_cam_status.pack(pady=10)
        ctl = tk.Frame(m, bg=C_BG); ctl.pack(side="right", fill="y")
        def _cr(l, v, lo, hi):
            s = self._section(ctl, l, fill="x", pady=5)
            tk.Scale(s, from_=lo, to=hi, orient="horizontal", variable=v, bg=C_SURF, fg=C_BLUE, length=300, command=lambda _: self._apply_cam()).pack()
        _cr("Exposure Time (µs)", self.cam_exposure, 100, 66000)
        _cr("Sensor Gain", self.cam_gain, 1.0, 16.0)
        _cr("Contrast", self.cam_contrast, 0.0, 8.0)
        _cr("Sharpness", self.cam_sharpness, 0.0, 8.0)
        tk.Button(ctl, text="RESET OPTICS", font=("Helvetica", 12, "bold"), bg=C_RED, fg=C_WHITE, command=self._reset_cam, pady=10).pack(fill="x", pady=20)

    # ══════════════════════════════════════════════════════════════════════════
    #  Logic (Restored from Robust Version)
    # ══════════════════════════════════════════════════════════════════════════
    def _mv_start(self):
        self.mv_state = "collecting_init"; self.mv_init_buf = {"top":[], "bottom":[]}
        self.btn_start.config(state="disabled"); self.mv_status_lbl.config(text="DATA ACQUISITION (INIT)", fg=C_AMBER)

    def _mv_stop(self):
        self.mv_state = "collecting_final"; self.mv_final_buf = {"top":[], "bottom":[]}
        self.btn_stop.config(state="disabled"); self.mv_status_lbl.config(text="DATA ACQUISITION (FINAL)", fg=C_AMBER)

    def _apply_cam(self, *_):
        if self.pc is None: return
        try:
            c = {"AeEnable": self.cam_ae.get(), "Brightness": self.cam_brightness.get(), "Contrast": self.cam_contrast.get(),
                 "Saturation": self.cam_saturation.get(), "Sharpness": self.cam_sharpness.get()}
            if not self.cam_ae.get():
                c["ExposureTime"] = int(self.cam_exposure.get()); c["AnalogueGain"] = float(self.cam_gain.get())
            if not self.cam_awb.get():
                c["AwbEnable"] = False; c["AwbMode"] = AWB_MODES.get(self.cam_awb_mode.get(), 0)
            else: c["AwbEnable"] = True
            self.pc.set_controls(c)
        except: pass

    def _reset_cam(self):
        self.cam_ae.set(True); self.cam_exposure.set(10000); self.cam_gain.set(2.0)
        self.cam_awb.set(True); self.cam_brightness.set(0.0); self.cam_contrast.set(1.0)
        self.cam_saturation.set(1.0); self.cam_sharpness.set(1.0); self._apply_cam()

    def load_calib(self):
        f = "camera_params_2.npz" if self.fixed_side.get() == "Right" else "camera_params.npz"
        if os.path.exists(f): d = np.load(f); return d['camera_matrix'], d['dist_coeff']
        return np.eye(3), np.zeros(5)

    def measurement_loop(self):
        try:
            from picamera2 import Picamera2
            pc = Picamera2(); pc.configure(pc.create_video_configuration(main={"size": RESOLUTION, "format": "RGB888"})); pc.start(); self.pc = pc
        except: return
        det = cv.aruco.ArucoDetector(cv.aruco.getPredefinedDictionary(ARUCO_DICT), cv.aruco.DetectorParameters())
        log.init_log(); buf = {"top":[], "bottom":[]}; l_u = time.time()*1000; l_s = l_u

        while self.is_running:
            K, dc = self.load_calib()
            try: frame = cv.cvtColor(pc.capture_array(), cv.COLOR_RGB2BGR)
            except: continue
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            m_b, s_b = np.mean(gray), np.std(gray)
            self.last_data["lighting"] = {"status": "ok" if 40<m_b<215 and s_b>18 else "bad", "mean": m_b, "std": s_b}
            corners_raw, ids, _ = det.detectMarkers(gray)
            curr = time.time()*1000

            if ids is not None and len(ids) >= 1:
                m_data = []
                for i in range(len(ids)):
                    raw = corners_raw[i][0]; ix = np.argsort(raw[:, 0]); lp, rp = raw[ix[:2]], raw[ix[2:]]
                    tl = lp[np.argmin(lp[:, 1])]; bl = lp[np.argmax(lp[:, 1])]
                    tr = rp[np.argmin(rp[:, 1])]; br = rp[np.argmax(rp[:, 1])]
                    m_data.append({"c": np.array([tl, tr, br, bl], dtype=np.float32), "c_raw": raw, "y": np.mean(raw[:,1]), "x": np.mean(raw[:,0])})
                m_data.sort(key=lambda m: m["y"])
                sets = {"top": m_data[:2], "bottom": m_data[2:4] if len(m_data) >= 4 else []}

                def _proc(ml, pk, sz):
                    if len(ml) < 2: self.last_data[pk].update({"dist": 0.0, "p1_px": None, "L_det": False, "R_det": False}); return
                    ml.sort(key=lambda m: m["x"])
                    is_r = (self.fixed_side.get() == "Right")
                    Sm, Tm = (ml[1], ml[0]) if is_r else (ml[0], ml[1])
                    h = sz / 2.0; obj = np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]], dtype=np.float32)

                    def _gp(c_raw):
                        _, rv, tv = cv.solvePnP(obj, c_raw, K, dc); R, _ = cv.Rodrigues(rv)
                        p, y, rl = rotation_to_euler(R); roll_img = math.degrees(math.atan2(R[1,0], R[0,0]))
                        return np.array([R @ pt + tv.ravel() for pt in obj]), roll_img, float(tv[2]), p, y

                    Spts, Sr, Sz, Sp, Sy = _gp(Sm["c_raw"]); Tpts, Tr, Tz, Tp, Ty = _gp(Tm["c_raw"])
                    irt = lambda cs: abs(math.degrees(math.atan2(cs[1,1]-cs[0,1], cs[1,0]-cs[0,0])))
                    Sir, Tir = irt(Sm["c"]), irt(Tm["c"])

                    def _io(pts, inner_right):
                        o = pts[:,0].argsort(); i_ = pts[o[2:]] if inner_right else pts[o[:2]]
                        ou = pts[o[:2]] if inner_right else pts[o[2:]]
                        return i_[i_[:,1].argsort()][0], i_[i_[:,1].argsort()][1], ou[ou[:,1].argsort()][0]

                    St, Sb, So = _io(Spts, not is_r); Ta, Tc, _ = _io(Tpts, is_r)
                    A = (St + Sb) / 2.0; Bv = Ta; Cv = Tc; X_alt = (Bv + Cv) / 2.0
                    p1 = tuple(((Sm["c"][1]+Sm["c"][2])/2).astype(int)) if not is_r else tuple(((Sm["c"][0]+Sm["c"][3])/2).astype(int))
                    p2 = tuple(((Tm["c"][0]+Tm["c"][3])/2).astype(int)) if not is_r else tuple(((Tm["c"][1]+Tm["c"][2])/2).astype(int))

                    buf[pk].append({
                        "A": A, "X_alt": X_alt, "TR": St, "BR": Sb, "TL_ref": So, "B": Bv, "C": Cv,
                        "L_A": (Sr, Sir), "R_A": (Tr, Tir), "rot": max(Sir, Tir), "L_z": Sz, "R_z": Tz,
                        "L_p": Sp, "L_y": Sy, "R_p": Tp, "R_y": Ty, "p1_px": p1, "p2_px": p2
                    })
                    self.last_data[pk]["L_det"] = True; self.last_data[pk]["R_det"] = True

                if (curr - l_s) >= 100:
                    _proc(sets["top"], "top", self.size_top.get()); _proc(sets["bottom"], "bottom", self.size_bot.get())
                    l_s = curr

                if (curr - l_u) >= 1000:
                    for k in ["top", "bottom"]:
                        if buf[k]:
                            s = buf[k]; aA = np.mean([x["A"] for x in s], axis=0); aTR = np.mean([x["TR"] for x in s], axis=0)
                            aTL = np.mean([x["TL_ref"] for x in s], axis=0); aB = np.mean([x["B"] for x in s], axis=0)
                            aC = np.mean([x["C"] for x in s], axis=0); aXa = np.mean([x["X_alt"] for x in s], axis=0)
                            aL = np.mean([x["L_A"] for x in s], axis=0); aR = np.mean([x["R_A"] for x in s], axis=0)
                            av_rot = float(np.mean([x["rot"] for x in s])); av_Lz = float(np.mean([x["L_z"] for x in s]))
                            av_Rz = float(np.mean([x["R_z"] for x in s])); av_Lp = float(np.mean([x["L_p"] for x in s]))
                            av_Ly = float(np.mean([x["L_y"] for x in s])); av_Rp = float(np.mean([x["R_p"] for x in s]))
                            av_Ry = float(np.mean([x["R_y"] for x in s]))
                            p1_px = tuple(np.mean([x["p1_px"] for x in s], axis=0).astype(int))
                            p2_px = tuple(np.mean([x["p2_px"] for x in s], axis=0).astype(int))
                            
                            v = (aTR-aTL)/np.linalg.norm(aTR-aTL); w = aC-aB; u = aB-aA
                            den = np.dot(v,v)*np.dot(w,w) - np.dot(v,w)**2
                            if self.use_angle_thresh.get() and av_rot > self.rot_threshold.get(): aX, dv, kv = aXa, np.linalg.norm(aXa-aA), 0.5
                            else:
                                if abs(den) > 1e-6:
                                    kv = np.clip((np.dot(v,w)*np.dot(u,v)-np.dot(v,v)*np.dot(u,w))/den, 0, 1)
                                    aX = aB + kv*w; dv = np.linalg.norm(aX-aA)
                                else: aX, dv, kv = aXa, np.linalg.norm(aXa-aA), 0.5
                            
                            self.last_data[k].update({
                                "A": aA, "X": aX, "TR": aTR, "B": aB, "C": aC, "dist": dv, "k": kv,
                                "rot_2d": av_rot, "L_A": aL, "R_A": aR, "L_z": av_Lz, "R_z": av_Rz,
                                "L_pitch": av_Lp, "L_yaw": av_Ly, "R_pitch": av_Rp, "R_yaw": av_Ry,
                                "p1_px": p1_px, "p2_px": p2_px
                            })
                            self.last_data["session_count"] += 1; buf[k] = []
                            try: log.record(dv, kv, aA, aX, aTR, aTR, aB, aC, aA, (aTR-aTL), (aC-aB), aL, aR)
                            except: pass
                    l_u = curr
            else:
                for k in ["top", "bottom"]: self.last_data[k].update({"dist": 0.0, "L_det": False, "R_det": False})

            for k, col in [("top",(55,135,240)), ("bottom",(163,113,247))]:
                d = self.last_data[k]
                if d["dist"] > 0 and d["p1_px"]:
                    cv.line(frame, d["p1_px"], d["p2_px"], col, 4)
                    cv.putText(frame, f"{d['dist']:.2f}", (d["p1_px"][0], d["p1_px"][1]-15), 0, 0.9, col, 2)
            self.current_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB); self._cam_preview_frame = self.current_frame

    def update_gui_loop(self):
        if self.current_frame is not None:
            im = ImageTk.PhotoImage(image=Image.fromarray(self.current_frame).resize((1080, 608)))
            self.canvas.create_image(0, 0, anchor="nw", image=im); self.canvas.img = im
        for k, dl, kl, sl in [("top", self.lbl_dist_top, self.lbl_k_top, self.lbl_status_top), ("bottom", self.lbl_dist_bot, self.lbl_k_bot, self.lbl_status_bot)]:
            d = self.last_data[k]; dist = d["dist"]
            dl.config(text=f"{dist:.3f}" if dist>0 else "00.000")
            kl.config(text=f"k: {d['k']:.4f}")
            p_ok = d["L_det"] and d["R_det"] and abs(d["L_pitch"])<self.pitch_threshold.get() and abs(d["L_yaw"])<self.yaw_threshold.get()
            sl.config(text="● DATA STABLE" if p_ok else "● SENSOR ERROR", fg=C_GREEN if p_ok else C_RED)
        for pk in ["top", "bottom"]:
            d, r = self.last_data[pk], self._warn_refs[pk]
            self._set_chip(*r["L_det"], "ok" if d["L_det"] else "error", "ONLINE" if d["L_det"] else "OFFLINE")
            self._set_chip(*r["R_det"], "ok" if d["R_det"] else "error", "ONLINE" if d["R_det"] else "OFFLINE")
            self._set_chip(*r["rot"], "ok" if d["rot_2d"]<self.rot_threshold.get() else "warn", f"{d['rot_2d']:.1f}°")
            self._set_chip(*r["L_pitch"], "ok" if abs(d["L_pitch"])<self.pitch_threshold.get() else "error", f"{d['L_pitch']:.1f}°")
            self._set_chip(*r["L_yaw"], "ok" if abs(d["L_yaw"])<self.yaw_threshold.get() else "error", f"{d['L_yaw']:.1f}°")
            self._set_chip(*r["R_pitch"], "ok" if abs(d["R_pitch"])<self.pitch_threshold.get() else "error", f"{d['R_pitch']:.1f}°")
            self._set_chip(*r["R_yaw"], "ok" if abs(d["R_yaw"])<self.yaw_threshold.get() else "error", f"{d['R_yaw']:.1f}°")
            ang = r["_ang"]; ang["L_pitch"].set(f"{d['L_pitch']:+.1f}"); ang["L_yaw"].set(f"{d['L_yaw']:+.1f}"); ang["L_roll"].set(f"{d['L_A'][0]:+.1f}")
            ang["R_pitch"].set(f"{d['R_pitch']:+.1f}"); ang["R_yaw"].set(f"{d['R_yaw']:+.1f}"); ang["R_roll"].set(f"{d['R_A'][0]:+.1f}")
        li = self.last_data["lighting"]
        self._set_chip(self._light_brt_chip, self._light_brt_val, "ok" if li["status"]=="ok" else "error", f"{li['mean']:.0f}")
        self._set_chip(self._light_ctr_chip, self._light_ctr_val, "ok" if li["status"]=="ok" else "warn", f"{li['std']:.0f}")
        for pk in ["top", "bottom"]:
            d, tv = self.last_data[pk], self.tele_vars[pk]; tv["A"].set(f"{d['A'][2]:.2f}"); tv["X"].set(f"{d['X'][2]:.2f}")
            tv["L_PITCH"].set(f"{d['L_pitch']:+.2f}"); tv["L_YAW"].set(f"{d['L_yaw']:+.2f}"); tv["R_PITCH"].set(f"{d['R_pitch']:+.2f}"); tv["R_YAW"].set(f"{d['R_yaw']:+.2f}")
            tv["L_Z"].set(f"{d['L_z']:.1f}"); tv["R_Z"].set(f"{d['R_z']:.1f}")
        
        # Movement monitoring logic
        sc = self.last_data["session_count"]
        if sc != self._last_sc:
            self._last_sc = sc
            if self.mv_state == "collecting_init":
                for k in ["top", "bottom"]:
                    if self.last_data[k]["dist"] > 0: self.mv_init_buf[k].append(self.last_data[k]["dist"])
                n = min(len(self.mv_init_buf["top"]), len(self.mv_init_buf["bottom"]))
                self.mv_prog_bar["value"] = n
                if n >= COLLECT_N:
                    self.mv_dist_init = {"top": np.mean(self.mv_init_buf["top"]), "bottom": np.mean(self.mv_init_buf["bottom"])}
                    self.mv_state = "ready"; self.btn_stop.config(state="normal"); self.mv_status_lbl.config(text="PHASE 1 COMPLETE", fg=C_GREEN)
            elif self.mv_state == "collecting_final":
                for k in ["top", "bottom"]:
                    if self.last_data[k]["dist"] > 0: self.mv_final_buf[k].append(self.last_data[k]["dist"])
                n = min(len(self.mv_final_buf["top"]), len(self.mv_final_buf["bottom"]))
                self.mv_prog_bar["value"] = n
                if n >= COLLECT_N:
                    df = {"top": np.mean(self.mv_final_buf["top"]), "bottom": np.mean(self.mv_final_buf["bottom"])}
                    self.mv_state = "done"; self.mv_status_lbl.config(text="MEASUREMENT COMPLETE", fg=C_GREEN)
                    dt = df["top"] - self.mv_dist_init["top"]; db = df["bottom"] - self.mv_dist_init["bottom"]
                    self.mv_delta_lbl_top.config(text=f"{dt:+.3f}"); self.mv_delta_lbl_bot.config(text=f"{db:+.3f}")

        self.root.after(50, self.update_gui_loop)

if __name__ == "__main__":
    root = tk.Tk(); app = MeasurementApp(root); root.mainloop()
