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

AWB_MODES = {
    "Auto":        0,
    "Tungsten":    1,
    "Fluorescent": 2,
    "Indoor":      3,
    "Daylight":    4,
    "Cloudy":      5,
}


class MeasurementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dual-Pair ArUco Measurement v11")
        self.root.geometry("1500x950")
        self.root.configure(bg="#2c3e50")

        # ── Measurement vars ──────────────────────────────────────────────
        self.size_top      = tk.DoubleVar(value=DEFAULT_MARKER_SIZE)
        self.size_bot      = tk.DoubleVar(value=DEFAULT_MARKER_SIZE)
        self.fixed_side    = tk.StringVar(value="Left")
        self.rot_threshold = tk.DoubleVar(value=12.0)

        # ── Camera control vars ───────────────────────────────────────────
        self.cam_ae         = tk.BooleanVar(value=True)   # Auto-exposure ON/OFF
        self.cam_exposure   = tk.IntVar(value=10000)      # µs  (100 – 66000)
        self.cam_gain       = tk.DoubleVar(value=2.0)     # AnalogueGain / ISO
        self.cam_awb        = tk.BooleanVar(value=True)   # Auto WB ON/OFF
        self.cam_awb_mode   = tk.StringVar(value="Auto")  # WB preset name
        self.cam_brightness = tk.DoubleVar(value=0.0)     # -1.0 – 1.0
        self.cam_contrast   = tk.DoubleVar(value=1.0)     # 0.0 – 32.0
        self.cam_saturation = tk.DoubleVar(value=1.0)     # 0.0 – 32.0
        self.cam_sharpness  = tk.DoubleVar(value=1.0)     # 0.0 – 16.0

        # Shared picamera2 handle (set by measurement_loop once started)
        self.pc = None
        self._cam_preview_frame = None   # latest frame for camera-controls tab

        temp = {"A": (0,0,0), "X": (0,0,0), "TR": (0,0,0), "BR": (0,0,0),
                "B": (0,0,0), "C": (0,0,0), "dist": 0.0, "k": 0.0,
                "L_A": (0.0, 0.0), "R_A": (0.0, 0.0), "rot_2d": 0.0,
                "L_z": 0.0, "R_z": 0.0, "p1_px": None, "p2_px": None}
        self.last_data = {"top": temp.copy(), "bottom": temp.copy(),
                          "session_count": 0}
        self.is_running    = True
        self.current_frame = None

        ttk.Style().theme_use('clam')
        self.setup_ui()
        threading.Thread(target=self.measurement_loop, daemon=True).start()
        self.update_gui_loop()

    # ═══════════════════════════════ UI ═══════════════════════════════════
    def setup_ui(self):
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill='both', expand=True, padx=10, pady=10)

        # ── TAB 1: LIVE MONITOR ──────────────────────────────────────────
        self.tab_live = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_live, text=" 📽 Live Monitor ")
        self.canvas = tk.Canvas(self.tab_live, width=960, height=540, bg="black")
        self.canvas.pack(side="left", padx=20, pady=20)
        st = tk.Frame(self.tab_live, bg="#ecf0f1")
        st.pack(side="right", fill="both", expand=True, padx=10, pady=20)

        for key, title, color in [("top",    "TOP PAIR",    "#e67e22"),
                                   ("bottom", "BOTTOM PAIR", "#9b59b6")]:
            f = tk.LabelFrame(st, text=f" {title} ",
                              font=("Helvetica", 12, "bold"), bg="#ecf0f1", fg=color)
            f.pack(fill="x", pady=10, padx=10)
            dl = tk.Label(f, text="0.000 mm",
                          font=("Helvetica", 24, "bold"), fg="#27ae60", bg="#ecf0f1")
            dl.pack(pady=10)
            kl = tk.Label(f, text="k: 0.0000", font=("Helvetica", 11), bg="#ecf0f1")
            kl.pack()
            if key == "top":
                self.lbl_dist_top, self.lbl_k_top = dl, kl
            else:
                self.lbl_dist_bot, self.lbl_k_bot = dl, kl

        # ── TAB 2: DUAL TELEMETRY ────────────────────────────────────────
        self.tab_tele = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_tele, text=" 🛸 Dual Telemetry ")
        mtf = tk.Frame(self.tab_tele, bg="#ecf0f1")
        mtf.pack(fill="both", expand=True)
        self.tele_vars = {"top": {}, "bottom": {}}
        v_show = ["A", "X", "TR", "BR", "B", "C",
                  "L_ROL", "L_ROT", "L_Z", "R_ROL", "R_ROT", "R_Z"]
        for key, title, color in [("top", "TOP", "#e67e22"),
                                   ("bottom", "BOTTOM", "#9b59b6")]:
            cf = tk.LabelFrame(mtf, text=f" {title} DATA ",
                               font=("Helvetica", 12, "bold"), fg=color, bg="#ecf0f1")
            cf.pack(side="left", fill="both", expand=True, padx=10, pady=10)
            for v_name in v_show:
                f = tk.Frame(cf, bg="white",
                             highlightbackground="#bdc3c7", highlightthickness=1)
                f.pack(fill="x", padx=10, pady=3)
                tk.Label(f, text=v_name, font=("Helvetica", 9),
                         bg="white").pack(side="left", padx=5)
                sv = tk.StringVar(value="—")
                tk.Label(f, textvariable=sv, font=("Courier", 10),
                         bg="white").pack(side="right", padx=5)
                self.tele_vars[key][v_name] = sv

        # ── TAB 3: MACHINE CONFIGURATION ─────────────────────────────────
        self.tab_settings = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_settings, text=" ⚙ Machine Configuration ")
        sc = tk.Frame(self.tab_settings, bg="#ecf0f1")
        sc.pack(fill="both", expand=True, padx=100, pady=50)
        rf = tk.LabelFrame(sc, text=" Reference Side (Fixed ArUco) ", bg="white",
                           font=("Helvetica", 11, "bold"), padx=20, pady=10)
        rf.pack(fill="x", pady=(0, 10))
        for choice in ["Left", "Right"]:
            tk.Radiobutton(rf, text=choice, variable=self.fixed_side,
                           value=choice, bg="white",
                           font=("Helvetica", 11)).pack(side="left", padx=20)

        def create_slider(parent, label, var, color, lo, hi):
            f = tk.LabelFrame(parent, text=f" {label} ", bg="white",
                              font=("Helvetica", 10), fg=color, padx=20, pady=5)
            f.pack(fill="x", pady=5)
            tk.Scale(f, from_=lo, to=hi, resolution=0.1, orient="horizontal",
                     variable=var, bg="white", length=400).pack(side="left", padx=20)
            tk.Entry(f, textvariable=var, width=8).pack(side="left")

        create_slider(sc, "Upper Pair Marker Size (mm)",  self.size_top, "#e67e22", 10, 200)
        create_slider(sc, "Bottom Pair Marker Size (mm)", self.size_bot, "#9b59b6", 10, 200)
        create_slider(sc, "Rotation Threshold °  (fallback trigger)",
                      self.rot_threshold, "#34495e", 0, 45)

        # ── TAB 4: CAMERA CONTROLS ───────────────────────────────────────
        self.tab_cam = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_cam, text=" 📷 Camera Controls ")

        # Split: left = preview, right = sliders
        cam_left  = tk.Frame(self.tab_cam, bg="#1a252f")
        cam_left.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        cam_right = tk.Frame(self.tab_cam, bg="#ecf0f1")
        cam_right.pack(side="right", fill="y", padx=10, pady=10, ipadx=5)

        # Live preview canvas (same feed, shown here independently)
        tk.Label(cam_left, text="Live Camera Preview",
                 font=("Helvetica", 11, "bold"), fg="#ecf0f1",
                 bg="#1a252f").pack(pady=(8, 0))
        self.cam_canvas = tk.Canvas(cam_left, width=800, height=480, bg="black")
        self.cam_canvas.pack(padx=10, pady=10)

        # ── Detection status badge ──
        self.lbl_cam_status = tk.Label(cam_left, text="⏳ Waiting for camera…",
                                       font=("Helvetica", 11, "bold"),
                                       fg="#f39c12", bg="#1a252f")
        self.lbl_cam_status.pack(pady=4)

        # ── Slider panel ─────────────────────────────────────────────────
        tk.Label(cam_right, text="Camera Settings",
                 font=("Helvetica", 13, "bold"), fg="#2c3e50",
                 bg="#ecf0f1").pack(pady=(10, 4))

        def cam_row(parent, label, var, lo, hi, res, unit=""):
            """One slider row with label + entry."""
            rf = tk.LabelFrame(parent, text=f" {label} ", bg="white",
                               font=("Helvetica", 9), padx=8, pady=4)
            rf.pack(fill="x", padx=6, pady=3)
            inner = tk.Frame(rf, bg="white"); inner.pack(fill="x")
            sc = tk.Scale(inner, from_=lo, to=hi, resolution=res,
                          orient="horizontal", variable=var, bg="white",
                          length=260, command=lambda _: self._apply_cam())
            sc.pack(side="left")
            tk.Entry(inner, textvariable=var, width=7,
                     font=("Courier", 9)).pack(side="left", padx=4)
            if unit:
                tk.Label(inner, text=unit, bg="white",
                         font=("Helvetica", 8), fg="#7f8c8d").pack(side="left")

        # Auto-exposure toggle
        ae_f = tk.LabelFrame(cam_right, text=" Auto Exposure ", bg="white",
                             font=("Helvetica", 9), padx=8, pady=4)
        ae_f.pack(fill="x", padx=6, pady=3)
        tk.Checkbutton(ae_f, text="Enable Auto Exposure (AE)",
                       variable=self.cam_ae, bg="white",
                       font=("Helvetica", 10),
                       command=self._apply_cam).pack(anchor="w")

        cam_row(cam_right, "Exposure Time",  self.cam_exposure,  100,  66000, 100, "µs")
        cam_row(cam_right, "ISO / Gain",     self.cam_gain,      1.0,  16.0,  0.1, "x")

        # Auto WB toggle + mode selector
        awb_f = tk.LabelFrame(cam_right, text=" White Balance ", bg="white",
                              font=("Helvetica", 9), padx=8, pady=4)
        awb_f.pack(fill="x", padx=6, pady=3)
        tk.Checkbutton(awb_f, text="Auto White Balance",
                       variable=self.cam_awb, bg="white",
                       font=("Helvetica", 10),
                       command=self._apply_cam).pack(anchor="w")
        wbm_row = tk.Frame(awb_f, bg="white"); wbm_row.pack(fill="x", pady=2)
        tk.Label(wbm_row, text="WB Mode:", bg="white",
                 font=("Helvetica", 9)).pack(side="left")
        for mode_name in AWB_MODES:
            tk.Radiobutton(wbm_row, text=mode_name, variable=self.cam_awb_mode,
                           value=mode_name, bg="white", font=("Helvetica", 8),
                           command=self._apply_cam).pack(side="left", padx=2)

        cam_row(cam_right, "Brightness", self.cam_brightness, -1.0, 1.0,  0.05)
        cam_row(cam_right, "Contrast",   self.cam_contrast,    0.0, 8.0,  0.1)
        cam_row(cam_right, "Saturation", self.cam_saturation,  0.0, 8.0,  0.1)
        cam_row(cam_right, "Sharpness",  self.cam_sharpness,   0.0, 8.0,  0.1)

        # Reset button
        tk.Button(cam_right, text="↺  Reset to Defaults",
                  font=("Helvetica", 10, "bold"), bg="#e74c3c", fg="white",
                  activebackground="#c0392b", relief="flat", padx=10, pady=6,
                  command=self._reset_cam).pack(pady=10, fill="x", padx=6)

    # ══════════════════════ CAMERA CONTROL HELPERS ═════════════════════════
    def _apply_cam(self, *_):
        """Push current slider values to Picamera2. Called on every slider move."""
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
            mode_val = AWB_MODES.get(self.cam_awb_mode.get(), 0)
            controls["AwbMode"] = mode_val

        controls["Brightness"]  = float(self.cam_brightness.get())
        controls["Contrast"]    = float(self.cam_contrast.get())
        controls["Saturation"]  = float(self.cam_saturation.get())
        controls["Sharpness"]   = float(self.cam_sharpness.get())

        try:
            self.pc.set_controls(controls)
        except Exception:
            pass

    def _reset_cam(self):
        """Reset all camera vars to defaults and re-apply."""
        self.cam_ae.set(True)
        self.cam_exposure.set(10000)
        self.cam_gain.set(2.0)
        self.cam_awb.set(True)
        self.cam_awb_mode.set("Auto")
        self.cam_brightness.set(0.0)
        self.cam_contrast.set(1.0)
        self.cam_saturation.set(1.0)
        self.cam_sharpness.set(1.0)
        self._apply_cam()

    # ══════════════════════════ CALIBRATION ════════════════════════════════
    def load_calib(self):
        f = "camera_params_2.npz" if self.fixed_side.get() == "Right" else "camera_params.npz"
        if os.path.exists(f):
            d = np.load(f)
            return d['camera_matrix'], d['dist_coeff']
        return (np.array([[1280,0,640],[0,1280,360],[0,0,1]], dtype=np.float32),
                np.zeros(5))

    # ══════════════════════════ MEASUREMENT ════════════════════════════════
    def measurement_loop(self):
        try:
            from picamera2 import Picamera2
            pc = Picamera2()
            pc.configure(pc.create_video_configuration(
                main={"size": RESOLUTION, "format": "RGB888"}))
            pc.start()
            self.pc = pc          # share handle for camera controls tab
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

                # ── Build marker list ──────────────────────────────────────
                m_data = []
                for i in range(len(ids)):
                    raw = corners_raw[i][0]
                    idx_x = np.argsort(raw[:, 0])
                    lp = raw[idx_x[:2]];  rp = raw[idx_x[2:]]
                    tl = lp[np.argmin(lp[:, 1])];  bl = lp[np.argmax(lp[:, 1])]
                    tr = rp[np.argmin(rp[:, 1])];  br = rp[np.argmax(rp[:, 1])]
                    cy = (tl[1]+tr[1]+br[1]+bl[1]) / 4.0
                    cx = (tl[0]+tr[0]+br[0]+bl[0]) / 4.0
                    m_data.append({
                        "c":     np.array([tl, tr, br, bl], dtype=np.float32),
                        "c_raw": raw,
                        "y": cy, "x": cx})

                # ── Y-rank pairing ─────────────────────────────────────────
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

                # ── Process one pair ───────────────────────────────────────
                def proc(marker_list, key, size):
                    if len(marker_list) < 2:
                        self.last_data[key]["dist"] = 0.0
                        self.last_data[key]["p1_px"] = None
                        return

                    marker_list.sort(key=lambda m: m["x"])
                    left_m  = marker_list[0]
                    right_m = marker_list[-1]
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
                        roll    = math.degrees(math.atan2(R[1,0], R[0,0]))
                        z_depth = float(tv[2])
                        return pts3d, roll, z_depth

                    def inplane_rot(c_sorted):
                        dy_e = c_sorted[1,1] - c_sorted[0,1]
                        dx_e = c_sorted[1,0] - c_sorted[0,0]
                        return math.degrees(math.atan2(dy_e, dx_e))

                    S_pts, S_roll, S_z = get_pose(S_m["c_raw"])
                    T_pts, T_roll, T_z = get_pose(T_m["c_raw"])
                    S_irot = abs(inplane_rot(S_m["c"]))
                    T_irot = abs(inplane_rot(T_m["c"]))

                    def inner_outer(pts, inner_is_right):
                        order = pts[:, 0].argsort()
                        if inner_is_right:
                            inner = pts[order[2:]]; outer = pts[order[:2]]
                        else:
                            inner = pts[order[:2]]; outer = pts[order[2:]]
                        inner = inner[inner[:, 1].argsort()]
                        outer = outer[outer[:, 1].argsort()]
                        return inner[0], inner[1], outer[0]

                    S_inner_top, S_inner_bot, S_outer_top = inner_outer(S_pts, not is_rf)
                    A = (S_inner_top + S_inner_bot) / 2.0

                    T_inner_top, T_inner_bot, _ = inner_outer(T_pts, is_rf)
                    B = T_inner_top;  C = T_inner_bot
                    X_alt = (B + C) / 2.0

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
                            s = buffers[key]
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
                    l_u = curr

            else:
                self.last_data["top"]["dist"]    = 0.0
                self.last_data["bottom"]["dist"] = 0.0

            # ── Draw lines ────────────────────────────────────────────────
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
                        cv.putText(frame, f"ROT: {max_rot:.1f}deg",
                                   (p1[0]-85, p1[1]-60), 0, 0.55, (0, 255, 255), 2)

            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            self.current_frame      = rgb
            self._cam_preview_frame = rgb   # also feed camera controls tab

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

        # ── Tab 4: Camera Controls preview ──
        if self._cam_preview_frame is not None:
            cam_img = ImageTk.PhotoImage(
                image=Image.fromarray(self._cam_preview_frame).resize((800, 480)))
            self.cam_canvas.create_image(0, 0, anchor="nw", image=cam_img)
            self.cam_canvas.img = cam_img

            # Update detection status badge
            top_ok = self.last_data["top"]["dist"] > 0
            bot_ok = self.last_data["bottom"]["dist"] > 0
            if top_ok and bot_ok:
                self.lbl_cam_status.config(
                    text="✅ Both pairs detected", fg="#2ecc71")
            elif top_ok or bot_ok:
                pair = "TOP" if top_ok else "BOTTOM"
                self.lbl_cam_status.config(
                    text=f"⚠️  Only {pair} pair detected", fg="#f39c12")
            else:
                self.lbl_cam_status.config(
                    text="❌ No ArUco detected — adjust camera settings",
                    fg="#e74c3c")

        self.root.after(50, self.update_gui_loop)


if __name__ == "__main__":
    root = tk.Tk()
    app = MeasurementApp(root)
    root.mainloop()
