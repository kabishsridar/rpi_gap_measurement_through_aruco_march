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

# ── v14 colour palette (Compact UI) ───────────────────────────────────────────
C_BG    = "#1e272e"
C_PANEL = "#2d3436"
C_CARD  = "#ffffff"
C_TOP   = "#e67e22"
C_BOT   = "#9b59b6"
C_GREEN = "#00b894"
C_RED   = "#d63031"
C_BLUE  = "#0984e3"
C_TEXT  = "#2d3436"
C_MUTED = "#636e72"
C_AMBER = "#f39c12"


# ══════════════════════════════════════════════════════════════════════════════
#  Euler-angle helper
# ══════════════════════════════════════════════════════════════════════════════
def rotation_to_euler(R):
    """
    Decompose an OpenCV 3×3 rotation matrix into (pitch, yaw, roll) degrees.
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
        self.root.title("Dual-Pair ArUco Measurement v15 (Compact)")
        self.root.geometry("1600x960")
        self.root.configure(bg=C_BG)

        # ── Measurement vars ──────────────────────────────────────────────────
        self.size_top         = tk.DoubleVar(value=DEFAULT_MARKER_SIZE)
        self.size_bot         = tk.DoubleVar(value=DEFAULT_MARKER_SIZE)
        self.fixed_side       = tk.StringVar(value="Left")
        self.rot_threshold    = tk.DoubleVar(value=12.0)
        self.pitch_threshold  = tk.DoubleVar(value=10.0)
        self.yaw_threshold    = tk.DoubleVar(value=10.0)
        self.use_angle_thresh = tk.BooleanVar(value=True)

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

        self._style()
        self.setup_ui()
        threading.Thread(target=self.measurement_loop, daemon=True).start()
        self.update_gui_loop()

    def _style(self):
        s = ttk.Style()
        s.theme_use('clam')
        s.configure("TNotebook",       background=C_BG,    borderwidth=0)
        s.configure("TNotebook.Tab",   background=C_PANEL, foreground="white",
                    padding=[14, 6],   font=("Helvetica", 10, "bold"))
        s.map("TNotebook.Tab",
              background=[("selected", C_BLUE)],
              foreground=[("selected", "white")])
        s.configure("TFrame", background=C_BG)

    def setup_ui(self):
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill='both', expand=True, padx=8, pady=8)
        self._build_tab_live()
        self._build_tab_tele()
        self._build_tab_settings()
        self._build_tab_cam()

    def _card(self, parent, title, title_color, **pack_kw):
        f = tk.LabelFrame(parent, text=f"  {title}  ",
                          font=("Helvetica", 11, "bold"),
                          fg=title_color, bg=C_CARD,
                          bd=1, relief="solid", padx=10, pady=8)
        f.pack(**pack_kw)
        return f

    def _build_tab_live(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text=" 📏  Movement ")
        tab.configure(style="TFrame")

        left = tk.Frame(tab, bg=C_BG)
        left.pack(side="left", padx=16, pady=16)

        self.canvas = tk.Canvas(left, width=960, height=540, bg="black",
                                highlightthickness=1, highlightbackground=C_MUTED)
        self.canvas.pack()

        self.warn_strip = tk.Label(
            left, text="  ⏳  Waiting for camera data…",
            font=("Helvetica", 10, "bold"),
            fg=C_MUTED, bg=C_CARD,
            anchor="w", padx=8, pady=3,
            wraplength=955, justify="left")
        self.warn_strip.pack(fill="x", pady=(4, 0))

        right = tk.Frame(tab, bg=C_BG)
        right.pack(side="right", fill="both", expand=True, padx=12, pady=10)

        for key, title, col in [("top", "TOP PAIR", C_TOP),
                                 ("bottom", "BOTTOM PAIR", C_BOT)]:
            card = self._card(right, title, col, fill="x", pady=6, padx=6)
            dl = tk.Label(card, text="0.000 mm",
                          font=("Helvetica", 24, "bold"), fg=C_GREEN, bg=C_CARD)
            dl.pack(pady=(4, 0))
            kl = tk.Label(card, text="k: 0.0000",
                          font=("Helvetica", 10), fg=C_MUTED, bg=C_CARD)
            kl.pack(pady=(0, 4))
            if key == "top":
                self.lbl_dist_top, self.lbl_k_top = dl, kl
            else:
                self.lbl_dist_bot, self.lbl_k_bot = dl, kl

        self.mv_status_lbl = tk.Label(
            right, text="Press  START  to capture initial gap",
            font=("Helvetica", 11), fg="#dfe6e9", bg=C_BG,
            wraplength=260, justify="center")
        self.mv_status_lbl.pack(pady=(10, 2))

        prog_f = tk.Frame(right, bg=C_BG)
        prog_f.pack(fill="x", padx=6)
        self.mv_prog_lbl = tk.Label(prog_f, text="",
                                    font=("Helvetica", 10), fg=C_GREEN, bg=C_BG)
        self.mv_prog_lbl.pack()
        self.mv_prog_bar = ttk.Progressbar(prog_f, length=240, maximum=COLLECT_N,
                                           mode="determinate")
        self.mv_prog_bar.pack(pady=2)

        btn_f = tk.Frame(right, bg=C_BG)
        btn_f.pack(pady=8, fill="x", padx=6)
        btn_cfg = dict(font=("Helvetica", 13, "bold"), relief="flat",
                       padx=14, pady=8, bd=0, cursor="hand2")

        self.btn_start = tk.Button(
            btn_f, text="▶ START", bg=C_GREEN, fg="white",
            activebackground="#00cec9",
            command=self._mv_start, **btn_cfg)
        self.btn_start.pack(side="left", expand=True, fill="x", padx=2)

        self.btn_stop = tk.Button(
            btn_f, text="■ STOP", bg=C_RED, fg="white",
            activebackground="#ff7675", state="disabled",
            command=self._mv_stop, **btn_cfg)
        self.btn_stop.pack(side="left", expand=True, fill="x", padx=2)

        self.btn_reset = tk.Button(
            btn_f, text="↺", bg=C_MUTED, fg="white",
            activebackground="#b2bec3",
            command=self._mv_reset, **btn_cfg)
        self.btn_reset.pack(side="left", padx=2)

        tk.Frame(right, bg=C_MUTED, height=1).pack(fill="x", padx=6, pady=8)
        tk.Label(right, text="Distance Moved",
                 font=("Helvetica", 11, "bold"), fg="#dfe6e9",
                 bg=C_BG).pack()

        for key, title, col in [("top", "TOP", C_TOP), ("bottom", "BOT", C_BOT)]:
            row = tk.Frame(right, bg=C_CARD, bd=1, relief="solid")
            row.pack(fill="x", padx=6, pady=4)
            tk.Label(row, text=title, font=("Helvetica", 10, "bold"),
                     fg=col, bg=C_CARD, width=5).pack(side="left", padx=6, pady=6)
            info = tk.Frame(row, bg=C_CARD); info.pack(side="left", expand=True)
            init_lbl  = tk.Label(info, text="Init: —",
                                 font=("Helvetica", 9), fg=C_MUTED, bg=C_CARD, anchor="w")
            init_lbl.pack(fill="x", padx=4)
            final_lbl = tk.Label(info, text="Final: —",
                                 font=("Helvetica", 9), fg=C_MUTED, bg=C_CARD, anchor="w")
            final_lbl.pack(fill="x", padx=4)
            delta_lbl = tk.Label(row, text="—",
                                 font=("Helvetica", 22, "bold"), fg=C_BLUE, bg=C_CARD)
            delta_lbl.pack(side="right", padx=10, pady=6)
            if key == "top":
                self.mv_init_lbl_top  = init_lbl
                self.mv_final_lbl_top = final_lbl
                self.mv_delta_lbl_top = delta_lbl
            else:
                self.mv_init_lbl_bot  = init_lbl
                self.mv_final_lbl_bot = final_lbl
                self.mv_delta_lbl_bot = delta_lbl

        tk.Frame(right, bg=C_MUTED, height=1).pack(fill="x", padx=6, pady=(8, 4))
        warn_f = tk.LabelFrame(right, text="  ⚠  Warnings  ",
                               font=("Helvetica", 10, "bold"),
                               fg=C_RED, bg=C_CARD, bd=1, relief="solid",
                               padx=6, pady=6)
        warn_f.pack(fill="x", padx=6)

        hdr = tk.Frame(warn_f, bg=C_CARD)
        hdr.pack(fill="x")
        for col_i, txt in enumerate(["", "L-Det", "R-Det", " Rot ", "Pitch", " Yaw "]):
            w = 4 if col_i == 0 else 6
            tk.Label(hdr, text=txt, font=("Helvetica", 7, "bold"),
                     fg=C_MUTED, bg=C_CARD, width=w,
                     anchor="center").grid(row=0, column=col_i, padx=1)

        self.warn_dots = {}
        for r_i, (key, title, col) in enumerate(
                [("top", "TOP", C_TOP), ("bottom", "BOT", C_BOT)], start=1):
            dots = {}
            tk.Label(hdr, text=title, font=("Helvetica", 8, "bold"),
                     fg=col, bg=C_CARD, width=4, anchor="w").grid(
                row=r_i, column=0, padx=(0, 2), pady=2)
            for c_i, field in enumerate(["L_det", "R_det", "rot", "pitch", "yaw"], start=1):
                dot = tk.Label(hdr, text="●", font=("Helvetica", 14),
                               fg=C_MUTED, bg=C_CARD, width=6, anchor="center")
                dot.grid(row=r_i, column=c_i, padx=1, pady=2)
                dots[field] = dot
            self.warn_dots[key] = dots

        tk.Label(warn_f, text="●=OK  ●=warn  ●=error     Pitch=fwd/bwd  Yaw=left/right",
                 font=("Helvetica", 7), fg=C_MUTED, bg=C_CARD).pack(anchor="w", pady=(2, 0))

        light_row = tk.Frame(warn_f, bg=C_CARD)
        light_row.pack(fill="x", pady=(4, 0))
        tk.Label(light_row, text="💡 Lighting:", font=("Helvetica", 9, "bold"),
                 fg=C_TEXT, bg=C_CARD).pack(side="left")
        self.lbl_lighting_mv = tk.Label(light_row, text="—", font=("Helvetica", 9, "bold"),
                                         fg=C_MUTED, bg=C_CARD)
        self.lbl_lighting_mv.pack(side="left", padx=6)

    def _build_tab_tele(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text=" 🛸  Dual Telemetry ")
        mtf = tk.Frame(tab, bg="#ecf0f1")
        mtf.pack(fill="both", expand=True)
        self.tele_vars = {"top": {}, "bottom": {}}
        v_show = ["A", "X", "TR", "BR", "B", "C",
                  "L_ROL", "L_ROT", "L_Z", "L_PITCH", "L_YAW",
                  "R_ROL", "R_ROT", "R_Z", "R_PITCH", "R_YAW"]
        for key, title, col in [("top", "TOP", C_TOP), ("bottom", "BOTTOM", C_BOT)]:
            cf = tk.LabelFrame(mtf, text=f" {title} DATA ",
                               font=("Helvetica", 12, "bold"), fg=col, bg="#ecf0f1")
            cf.pack(side="left", fill="both", expand=True, padx=10, pady=10)
            for v_name in v_show:
                row = tk.Frame(cf, bg="white",
                               highlightbackground="#bdc3c7", highlightthickness=1)
                row.pack(fill="x", padx=10, pady=3)
                tk.Label(row, text=v_name, font=("Helvetica", 9), bg="white").pack(side="left", padx=5)
                sv = tk.StringVar(value="—")
                tk.Label(row, textvariable=sv, font=("Courier", 10), bg="white").pack(side="right", padx=5)
                self.tele_vars[key][v_name] = sv

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
                           value=choice, bg="white", font=("Helvetica", 12)).pack(side="left", padx=24)

        def create_slider(label, var, color, lo, hi, res=0.1):
            f = tk.LabelFrame(sc, text=f" {label} ", bg="white", font=("Helvetica", 10), fg=color, padx=20, pady=6)
            f.pack(fill="x", pady=6)
            sl = tk.Scale(f, from_=lo, to=hi, resolution=res, orient="horizontal", variable=var, bg="white", length=500)
            sl.pack(side="left", padx=16)
            tk.Entry(f, textvariable=var, width=9, font=("Courier", 10)).pack(side="left")
            return sl

        create_slider("Upper Pair Marker Size (mm)",  self.size_top, C_TOP,  10, 200)
        create_slider("Bottom Pair Marker Size (mm)", self.size_bot, C_BOT,  10, 200)
        self.rot_threshold_slider = create_slider("Rotation Threshold °", self.rot_threshold, C_TEXT, 0, 45)
        create_slider("Pitch Threshold °", self.pitch_threshold, "#8e44ad", 1, 45, res=0.5)
        create_slider("Yaw Threshold °", self.yaw_threshold, "#16a085", 1, 45, res=0.5)

        tog_f = tk.LabelFrame(sc, text=" Angle Threshold Mode ", bg="white", font=("Helvetica", 10), fg="#2c3e50", padx=20, pady=10)
        tog_f.pack(fill="x", pady=6)

        def _update_tog_label(*_):
            if self.use_angle_thresh.get():
                tog_state_lbl.config(text="ON — Perpendicular formula below threshold", fg=C_GREEN)
                self.rot_threshold_slider.config(state="normal")
            else:
                tog_state_lbl.config(text="OFF — Always use perpendicular formula", fg=C_RED)
                self.rot_threshold_slider.config(state="disabled")

        tog_row = tk.Frame(tog_f, bg="white"); tog_row.pack(fill="x")
        self._tog_canvas = tk.Canvas(tog_row, width=56, height=28, bg="white", highlightthickness=0, cursor="hand2")
        self._tog_canvas.pack(side="left", padx=(0, 12))
        tog_state_lbl = tk.Label(tog_row, text="", font=("Helvetica", 10), bg="white", anchor="w")
        tog_state_lbl.pack(side="left", fill="x", expand=True)

        def _draw_toggle():
            self._tog_canvas.delete("all")
            on = self.use_angle_thresh.get()
            bg = C_GREEN if on else "#b2bec3"
            self._tog_canvas.create_oval(2, 2, 54, 26, fill=bg, outline="")
            cx = 38 if on else 18
            self._tog_canvas.create_oval(cx-12, 4, cx+12, 24, fill="white", outline="")
        def _toggle_click(_=None):
            self.use_angle_thresh.set(not self.use_angle_thresh.get())
            _draw_toggle(); _update_tog_label()
        self._tog_canvas.bind("<Button-1>", _toggle_click)
        _draw_toggle(); _update_tog_label()

    def _build_tab_cam(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text=" 📷  Camera Controls ")
        cam_left  = tk.Frame(tab, bg="#1a252f"); cam_left.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        cam_right = tk.Frame(tab, bg="#ecf0f1"); cam_right.pack(side="right", fill="y", padx=8, pady=8, ipadx=4)
        tk.Label(cam_left, text="Live Preview", font=("Helvetica", 11, "bold"), fg="#ecf0f1", bg="#1a252f").pack(pady=(8, 0))
        self.cam_canvas = tk.Canvas(cam_left, width=800, height=480, bg="black", highlightthickness=1, highlightbackground=C_MUTED)
        self.cam_canvas.pack(padx=10, pady=10)
        self.lbl_cam_status = tk.Label(cam_left, text="⏳ Waiting…", font=("Helvetica", 11, "bold"), fg="#f39c12", bg="#1a252f")
        self.lbl_cam_status.pack(pady=4)
        self.lbl_lighting_cam = tk.Label(cam_left, text="", font=("Helvetica", 10, "bold"), fg=C_MUTED, bg="#1a252f")
        self.lbl_lighting_cam.pack(pady=2)

        def cam_row(label, var, lo, hi, res, unit=""):
            rf = tk.LabelFrame(cam_right, text=label, bg="white", font=("Helvetica", 9), padx=8, pady=4); rf.pack(fill="x", padx=6, pady=3)
            tk.Scale(rf, from_=lo, to=hi, resolution=res, orient="horizontal", variable=var, bg="white", length=260, command=lambda _: self._apply_cam()).pack(side="left")
            tk.Entry(rf, textvariable=var, width=7, font=("Courier", 9)).pack(side="left", padx=4)

        tk.Checkbutton(cam_right, text="Auto Exposure", variable=self.cam_ae, bg="#ecf0f1", command=self._apply_cam).pack(anchor="w", padx=10)
        cam_row("Exposure", self.cam_exposure, 100,  66000, 100, "µs")
        cam_row("Gain",     self.cam_gain,     1.0,  16.0,  0.1, "x")
        tk.Checkbutton(cam_right, text="Auto White Balance", variable=self.cam_awb, bg="#ecf0f1", command=self._apply_cam).pack(anchor="w", padx=10)
        cam_row("Contrast",   self.cam_contrast,    0.0, 8.0, 0.1)
        tk.Button(cam_right, text="Reset Defaults", bg=C_RED, fg="white", command=self._reset_cam).pack(pady=10, fill="x", padx=6)

    def _mv_start(self):
        if self.mv_state in ("idle", "done"):
            self.mv_state = "collecting_init"; self.mv_init_buf = {"top":[],"bottom":[]}
            self.btn_start.config(state="disabled"); self.mv_prog_bar["value"] = 0

    def _mv_stop(self):
        if self.mv_state == "ready":
            self.mv_state = "collecting_final"; self.mv_final_buf = {"top":[],"bottom":[]}
            self.btn_stop.config(state="disabled"); self.mv_prog_bar["value"] = 0

    def _mv_reset(self):
        self.mv_state = "idle"; self._mv_reset_ui()

    def _mv_reset_ui(self):
        self.btn_start.config(state="normal"); self.btn_stop.config(state="disabled")
        self.mv_prog_bar["value"] = 0; self.mv_prog_lbl.config(text="")
        self.mv_status_lbl.config(text="Press START to capture initial gap", fg="#dfe6e9")

    def _mv_tick(self):
        sc = self.last_data["session_count"]
        if sc == self._last_sc: return
        self._last_sc = sc
        if self.mv_state == "collecting_init":
            for k in ["top", "bottom"]:
                d = self.last_data[k]["dist"]
                if d > 0: self.mv_init_buf[k].append(d)
            n = min(len(self.mv_init_buf["top"]), len(self.mv_init_buf["bottom"]))
            self.mv_prog_bar["value"] = n
            if n >= COLLECT_N:
                self.mv_dist_init = {"top": float(np.mean(self.mv_init_buf["top"])), "bottom": float(np.mean(self.mv_init_buf["bottom"]))}
                self.mv_state = "ready"; self.btn_stop.config(state="normal")
                self.mv_status_lbl.config(text="✅ Initial captured — press STOP", fg=C_GREEN)
        elif self.mv_state == "collecting_final":
            for k in ["top", "bottom"]:
                d = self.last_data[k]["dist"]
                if d > 0: self.mv_final_buf[k].append(d)
            n = min(len(self.mv_final_buf["top"]), len(self.mv_final_buf["bottom"]))
            self.mv_prog_bar["value"] = n
            if n >= COLLECT_N:
                self.mv_dist_final = {"top": float(np.mean(self.mv_final_buf["top"])), "bottom": float(np.mean(self.mv_final_buf["bottom"]))}
                self.mv_state = "done"; self.btn_start.config(state="normal")
                self.mv_status_lbl.config(text="✅ Complete", fg=C_GREEN)
                for k, il, fl, dl in [("top", self.mv_init_lbl_top, self.mv_final_lbl_top, self.mv_delta_lbl_top), ("bottom", self.mv_init_lbl_bot, self.mv_final_lbl_bot, self.mv_delta_lbl_bot)]:
                    di, df = self.mv_dist_init[k], self.mv_dist_final[k]; delta = df - di
                    il.config(text=f"Init: {di:.3f}"); fl.config(text=f"Final: {df:.3f}"); dl.config(text=f"{delta:+.3f}")

    def _apply_cam(self):
        if self.pc:
            self.pc.set_controls({"AeEnable":self.cam_ae.get(),"AwbEnable":self.cam_awb.get(),"Contrast":float(self.cam_contrast.get())})

    def _reset_cam(self):
        self.cam_ae.set(True); self.cam_awb.set(True); self._apply_cam()

    def load_calib(self):
        f = "camera_params_2.npz" if self.fixed_side.get() == "Right" else "camera_params.npz"
        if os.path.exists(f): d = np.load(f); return d['camera_matrix'], d['dist_coeff']
        return np.eye(3), np.zeros(5)

    def measurement_loop(self):
        try: from picamera2 import Picamera2; pc = Picamera2(); pc.configure(pc.create_video_configuration(main={"size": RESOLUTION, "format": "RGB888"})); pc.start(); self.pc = pc
        except: return
        detector = cv.aruco.ArucoDetector(cv.aruco.getPredefinedDictionary(ARUCO_DICT), cv.aruco.DetectorParameters()); log.init_log()
        buffers = {"top": [], "bottom": []}; l_s = l_u = time.time() * 1000
        while self.is_running:
            K, dist_c = self.load_calib()
            try: frame = cv.cvtColor(pc.capture_array(), cv.COLOR_RGB2BGR)
            except: continue
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            m_b, s_b = float(np.mean(gray)), float(np.std(gray))
            ls = "dark" if m_b < LIGHT_LOW else ("bright" if m_b > LIGHT_HIGH else ("flat" if s_b < CONTRAST_MIN else "ok"))
            self.last_data["lighting"] = {"status":ls, "mean":m_b, "std":s_b}
            corners, ids, _ = detector.detectMarkers(gray); curr = time.time() * 1000
            if ids is not None:
                m_data = []
                for i in range(len(ids)):
                    raw = corners[i][0]; idx = np.argsort(raw[:, 0]); lp, rp = raw[idx[:2]], raw[idx[2:]]
                    tl = lp[np.argmin(lp[:, 1])]; bl = lp[np.argmax(lp[:, 1])]; tr = rp[np.argmin(rp[:, 1])]; br = rp[np.argmax(rp[:, 1])]
                    m_data.append({"c": np.array([tl, tr, br, bl], dtype=np.float32), "c_raw": raw, "y": (tl[1]+br[1])/2, "x": (tl[0]+br[0])/2})
                m_data.sort(key=lambda m: m["y"]); top_m = m_data[:2]; bot_m = m_data[2:4] if len(m_data) >= 4 else []
                def proc(marker_list, pair_key, size):
                    if len(marker_list) < 2: self.last_data[pair_key].update({"dist":0.0,"L_det":False,"R_det":False}); return
                    marker_list.sort(key=lambda m: m["x"]); is_rf = (self.fixed_side.get() == "Right")
                    S_m, T_m = (marker_list[-1], marker_list[0]) if is_rf else (marker_list[0], marker_list[-1])
                    h = size / 2.0; obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)
                    def get_p(c):
                        _, rv, tv = cv.solvePnP(obj, c, K, dist_c); R, _ = cv.Rodrigues(rv)
                        pitch, yaw, roll = rotation_to_euler(R)
                        return np.array([R @ pt + tv.ravel() for pt in obj]), roll, float(tv[2]), pitch, yaw
                    S_pts, S_r, S_z, S_p, S_y = get_p(S_m["c_raw"]); T_pts, T_r, T_z, T_p, T_y = get_p(T_m["c_raw"])
                    S_ir = abs(math.degrees(math.atan2(S_m["c"][1,1]-S_m["c"][0,1], S_m["c"][1,0]-S_m["c"][0,0])))
                    T_ir = abs(math.degrees(math.atan2(T_m["c"][1,1]-T_m["c"][0,1], T_m["c"][1,0]-T_m["c"][0,0])))
                    def ino(pts, inr): order = pts[:,0].argsort(); inner = pts[order[2:]] if inr else pts[order[:2]]; inner = inner[inner[:,1].argsort()]; return inner[0], inner[1], pts[order[:2 if inr else 2]][0]
                    St, Sb, So = ino(S_pts, not is_rf); Tt, Tb, _ = ino(T_pts, is_rf); A = (St+Sb)/2.0; B, C = Tt, Tb
                    buffers[pair_key].append({"A":A,"B":B,"C":C,"TR":St,"TL_ref":So,"L_A":(S_r,S_ir),"R_A":(T_r,T_ir),"rot":max(S_ir,T_ir),"L_z":S_z,"R_z":T_z,"L_p":S_p,"L_y":S_y,"R_p":T_p,"R_y":T_y,"p1_px":tuple(S_m["c"][1 if not is_rf else 0].astype(int)),"p2_px":tuple(T_m["c"][0 if not is_rf else 1].astype(int))})
                    self.last_data[pair_key].update({"L_det":True,"R_det":True})
                if (curr-l_s) >= 100: proc(top_m, "top", self.size_top.get()); proc(bot_m, "bottom", self.size_bot.get()); l_s = curr
                if (curr-l_u) >= 1000:
                    for k in ["top", "bottom"]:
                        if buffers[k]:
                            s = buffers[k]; aA = np.mean([x["A"] for x in s], axis=0); aB = np.mean([x["B"] for x in s], axis=0); aC = np.mean([x["C"] for x in s], axis=0); aTR = np.mean([x["TR"] for x in s], axis=0); aTL = np.mean([x["TL_ref"] for x in s], axis=0)
                            v = (aTR-aTL)/np.linalg.norm(aTR-aTL); w, u = aC-aB, aB-aA; den = np.dot(v,v)*np.dot(w,w)-np.dot(v,w)**2
                            if abs(den)>1e-6: kv = float(np.clip(((np.dot(v,w)*np.dot(u,v))-(np.dot(v,v)*np.dot(u,w)))/den,0,1)); aX=aB+kv*w; dv=np.linalg.norm(aX-aA)
                            else: aX=(aB+aC)/2; dv=np.linalg.norm(aX-aA); kv=0.5
                            self.last_data[k].update({"A":aA,"X":aX,"dist":dv,"k":kv,"rot_2d":np.mean([x["rot"] for x in s]),"L_pitch":np.mean([x["L_p"] for x in s]),"L_yaw":np.mean([x["L_y"] for x in s]),"R_pitch":np.mean([x["R_p"] for x in s]),"R_yaw":np.mean([x["R_y"] for x in s]),"L_z":np.mean([x["L_z"] for x in s]),"R_z":np.mean([x["R_z"] for x in s]),"p1_px":s[-1]["p1_px"],"p2_px":s[-1]["p2_px"]}); self.last_data["session_count"]+=1; buffers[k].clear()
                    l_u = curr
            else:
                for k in ["top", "bottom"]: self.last_data[k].update({"dist":0.0,"L_det":False,"R_det":False})
            self.current_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    def _update_warnings(self):
        pt, yt, rt, sp = self.pitch_threshold.get(), self.yaw_threshold.get(), self.rot_threshold.get(), []
        for k in ["top", "bottom"]:
            d, dots = self.last_data[k], self.warn_dots[k]
            dots["L_det"].config(fg=C_GREEN if d["L_det"] else C_RED); dots["R_det"].config(fg=C_GREEN if d["R_det"] else C_RED)
            rot_ok = d.get("rot_2d",0) <= rt; dots["rot"].config(fg=C_GREEN if rot_ok else C_AMBER)
            p_ok = abs(d.get("L_pitch",0))<=pt and abs(d.get("R_pitch",0))<=pt; dots["pitch"].config(fg=C_GREEN if p_ok else C_RED)
            y_ok = abs(d.get("L_yaw",0))<=yt and abs(d.get("R_yaw",0))<=yt; dots["yaw"].config(fg=C_GREEN if y_ok else C_RED)
            if not d["L_det"]: sp.append(f"[{k}] L-Missing")
            if not d["R_det"]: sp.append(f"[{k}] R-Missing")
        li = self.last_data["lighting"]; ls = li["status"]
        self.lbl_lighting_mv.config(text=ls.upper(), fg=C_GREEN if ls=="ok" else C_RED)
        self.warn_strip.config(text="  ⚠  " + " | ".join(sp) if sp else "  ✅  All OK", fg=C_RED if sp else C_GREEN)

    def update_gui_loop(self):
        if self.current_frame is not None:
            img = ImageTk.PhotoImage(image=Image.fromarray(self.current_frame).resize((960, 540))); self.canvas.create_image(0, 0, anchor="nw", image=img); self.canvas.img = img
        for k, ld, lk in [("top", self.lbl_dist_top, self.lbl_k_top), ("bottom", self.lbl_dist_bot, self.lbl_k_bot)]:
            d = self.last_data[k]; ld.config(text=f"{d['dist']:.3f} mm"); lk.config(text=f"k: {d['k']:.4f}")
            tv = self.tele_vars[k]; tv["A"].set(f"{d['A'][2]:.2f}"); tv["X"].set(f"{d['X'][2]:.2f}"); tv["L_PITCH"].set(f"{d['L_pitch']:+.2f}"); tv["L_YAW"].set(f"{d['L_yaw']:+.2f}"); tv["R_PITCH"].set(f"{d['R_pitch']:+.2f}"); tv["R_YAW"].set(f"{d['R_yaw']:+.2f}"); tv["L_Z"].set(f"{d['L_z']:.1f}"); tv["R_Z"].set(f"{d['R_z']:.1f}")
        self._mv_tick(); self._update_warnings(); self.root.after(50, self.update_gui_loop)

if __name__ == "__main__":
    root = tk.Tk(); app = MeasurementApp(root); root.mainloop()
