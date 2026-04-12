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
import log

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_MARKER_SIZE = 53.8
ARUCO_DICT          = cv.aruco.DICT_4X4_50
RESOLUTION          = (1280, 720)
COLLECT_N           = 5

# Control Room Aesthetic Palette
C_BG      = "#0f172a"
C_PANEL   = "#1e293b"
C_CARD    = "#334155"
C_ACCENT   = "#38bdf8"
C_TOP      = "#fb923c"
C_BOT      = "#c084fc"
C_GREEN    = "#4ade80"
C_RED      = "#f87171"
C_TEXT_BRT = "#f8fafc"
C_TEXT_MED = "#94a3b8"
C_AMBER    = "#fbbf24"

# Typography
F_TITLE = ("Inter", 16, "bold")
F_HEAD  = ("Inter", 13, "bold")
F_BODY  = ("Inter", 12)
F_SMALL = ("Inter", 10)
F_DATA  = ("Inter", 28, "bold")
F_MONO  = ("Consolas", 13)
F_BTN   = ("Inter", 14, "bold")

class MeasurementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dual-Pair ArUco Measurement v16 (v14 Formula)")
        self.root.geometry("1600x960")
        self.root.configure(bg=C_BG)

        self.size_top         = tk.DoubleVar(value=DEFAULT_MARKER_SIZE)
        self.size_bot         = tk.DoubleVar(value=DEFAULT_MARKER_SIZE)
        self.fixed_side       = tk.StringVar(value="Left")
        self.rot_threshold    = tk.DoubleVar(value=12.0)
        self.use_angle_thresh = tk.BooleanVar(value=False)

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

        self.mv_state       = "idle"
        self.mv_init_buf    = {"top": [], "bottom": []}
        self.mv_final_buf   = {"top": [], "bottom": []}
        self.mv_dist_init   = {"top": None, "bottom": None}
        self.mv_dist_final  = {"top": None, "bottom": None}
        self._last_sc       = 0

        def _empty():
            return {
                "A": (0,0,0), "X": (0,0,0), "TR": (0,0,0), "BR": (0,0,0),
                "B": (0,0,0), "C": (0,0,0), "dist": 0.0, "k": 0.0,
                "L_A": (0.0, 0.0), "R_A": (0.0, 0.0), "rot_2d": 0.0,
                "L_z": 0.0, "R_z": 0.0, "p1_px": None, "p2_px": None,
                "L_det": False, "R_det": False,
            }

        self.last_data = {
            "top": _empty(),
            "bottom": _empty(),
            "session_count": 0,
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
        s.configure("TNotebook.Tab", background=C_PANEL, foreground=C_TEXT_MED, padding=[25, 12], font=F_HEAD)
        s.map("TNotebook.Tab", background=[("selected", C_ACCENT)], foreground=[("selected", C_BG)])
        s.configure("TFrame", background=C_BG)

    def setup_ui(self):
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill='both', expand=True, padx=8, pady=8)
        self._build_tab_live()
        self._build_tab_tele()
        self._build_tab_settings()
        self._build_tab_cam()

    def _card(self, parent, title, title_color, **pack_kw):
        f = tk.LabelFrame(parent, text=f"  {title.upper()}  ", font=F_HEAD, fg=title_color, bg=C_PANEL, bd=1, relief="solid", padx=20, pady=12)
        f.pack(**pack_kw)
        return f

    def _build_tab_live(self):
        tab = ttk.Frame(self.tabs); self.tabs.add(tab, text=" 📏  Movement "); tab.configure(style="TFrame")
        left = tk.Frame(tab, bg=C_BG); left.pack(side="left", padx=16, pady=16)
        self.canvas = tk.Canvas(left, width=960, height=540, bg="black", highlightthickness=1)
        self.canvas.pack()
        
        right = tk.Frame(tab, bg=C_BG); right.pack(side="right", fill="both", expand=True, padx=12, pady=10)
        for key, title, col in [("top", "UPPER SENSOR", C_TOP), ("bottom", "LOWER SENSOR", C_BOT)]:
            card = self._card(right, title, col, fill="x", pady=6, padx=12)
            dl = tk.Label(card, text="0.000 mm", font=F_DATA, fg=C_GREEN, bg=C_PANEL); dl.pack(pady=(4, 0))
            kl = tk.Label(card, text="k: 0.0000", font=F_SMALL, fg=C_TEXT_MED, bg=C_PANEL); kl.pack(pady=(0, 4))
            if key == "top": self.lbl_dist_top, self.lbl_k_top = dl, kl
            else: self.lbl_dist_bot, self.lbl_k_bot = dl, kl

        self.mv_status_lbl = tk.Label(right, text="Press START to capture initial gap", font=F_HEAD, fg=C_TEXT_MED, bg=C_BG); self.mv_status_lbl.pack(pady=10)
        self.mv_prog_bar = ttk.Progressbar(right, length=380, maximum=COLLECT_N, mode="determinate"); self.mv_prog_bar.pack(pady=5)
        
        bf = tk.Frame(right, bg=C_BG); bf.pack(pady=10, fill="x", padx=12)
        self.btn_reset = tk.Button(bf, text="↺", bg=C_CARD, fg="white", font=F_BTN, width=5, command=self._mv_reset); self.btn_reset.pack(side="right", padx=6)
        self.btn_start = tk.Button(bf, text="▶ START", bg=C_GREEN, fg=C_BG, font=F_BTN, command=self._mv_start); self.btn_start.pack(side="left", expand=True, fill="x", padx=6)
        self.btn_stop = tk.Button(bf, text="■ STOP", bg=C_RED, fg=C_BG, font=F_BTN, state="disabled", command=self._mv_stop); self.btn_stop.pack(side="left", expand=True, fill="x", padx=6)

        for key, title, col in [("top", "UPPER", C_TOP), ("bottom", "LOWER", C_BOT)]:
            row = tk.Frame(right, bg=C_PANEL, bd=1, relief="solid"); row.pack(fill="x", padx=12, pady=4)
            dl = tk.Label(row, text="—", font=F_DATA, fg=C_ACCENT, bg=C_PANEL); dl.pack(side="right", padx=15)
            info = tk.Frame(row, bg=C_PANEL); info.pack(side="left", expand=True)
            il = tk.Label(info, text="Init: —", font=F_BODY, fg=C_TEXT_MED, bg=C_PANEL); il.pack(fill="x")
            fl = tk.Label(info, text="Final: —", font=F_BODY, fg=C_TEXT_MED, bg=C_PANEL); fl.pack(fill="x")
            if key == "top": self.mv_init_lbl_top, self.mv_final_lbl_top, self.mv_delta_lbl_top = il, fl, dl
            else: self.mv_init_lbl_bot, self.mv_final_lbl_bot, self.mv_delta_lbl_bot = il, fl, dl

    def _build_tab_tele(self):
        tab = ttk.Frame(self.tabs); self.tabs.add(tab, text=" 🛸  Telemetry ")
        mtf = tk.Frame(tab, bg=C_BG); mtf.pack(fill="both", expand=True)
        self.tele_vars = {"top": {}, "bottom": {}}
        v_show = ["A", "X", "TR", "BR", "B", "C", "L_ROL", "L_ROT", "L_Z", "R_ROL", "R_ROT", "R_Z"]
        for key, title, col in [("top", "TOP", C_TOP), ("bottom", "BOT", C_BOT)]:
            cf = tk.LabelFrame(mtf, text=f" {title} DATA ", font=F_HEAD, fg=col, bg=C_BG); cf.pack(side="left", fill="both", expand=True, padx=10, pady=10)
            for v_name in v_show:
                row = tk.Frame(cf, bg=C_PANEL); row.pack(fill="x", pady=2)
                tk.Label(row, text=v_name, font=F_SMALL, bg=C_PANEL, fg=C_TEXT_MED).pack(side="left", padx=5)
                sv = tk.StringVar(value="—"); tk.Label(row, textvariable=sv, font=F_MONO, bg=C_PANEL, fg=C_TEXT_BRT).pack(side="right", padx=5)
                self.tele_vars[key][v_name] = sv

    def _build_tab_settings(self):
        tab = ttk.Frame(self.tabs); self.tabs.add(tab, text=" ⚙  Settings ")
        sc = tk.Frame(tab, bg=C_BG); sc.pack(fill="both", expand=True, padx=40, pady=20)
        rf = tk.LabelFrame(sc, text=" Reference (Fixed Side) ", bg=C_PANEL, fg=C_TEXT_BRT); rf.pack(fill="x", pady=10)
        for choice in ["Left", "Right"]: tk.Radiobutton(rf, text=choice, variable=self.fixed_side, value=choice, bg=C_PANEL, fg=C_TEXT_BRT).pack(side="left", padx=20)
        def sld(l, v, c, lo, hi):
            f = tk.LabelFrame(sc, text=f" {l} ", bg=C_PANEL, fg=c); f.pack(fill="x", pady=5)
            tk.Scale(f, from_=lo, to=hi, resolution=0.1, orient="horizontal", variable=v, bg=C_PANEL, length=600).pack(side="left", padx=10)
            tk.Entry(f, textvariable=v, width=10).pack(side="left")
        sld("Upper Marker (mm)", self.size_top, C_TOP, 10, 200); sld("Lower Marker (mm)", self.size_bot, C_BOT, 10, 200); sld("Rot Thresh °", self.rot_threshold, C_TEXT_BRT, 0, 45)
        tk.Checkbutton(sc, text="Use Angle Threshold", variable=self.use_angle_thresh, bg=C_BG, fg=C_TEXT_BRT, selectcolor=C_BG).pack(pady=10)

    def _build_tab_cam(self):
        tab = ttk.Frame(self.tabs); self.tabs.add(tab, text=" 📷  Camera ")
        cl = tk.Frame(tab, bg=C_BG); cl.pack(side="left", fill="both", expand=True)
        cr = tk.Frame(tab, bg=C_BG); cr.pack(side="right", fill="y", padx=10)
        self.cam_canvas = tk.Canvas(cl, width=800, height=480, bg="black"); self.cam_canvas.pack(pady=20)
        self.lbl_cam_status = tk.Label(cl, text="⏳ Wait...", font=F_HEAD, fg=C_AMBER, bg=C_BG); self.lbl_cam_status.pack()
        def crw(l, v, lo, hi, r):
            f = tk.LabelFrame(cr, text=f" {l} ", bg=C_PANEL, fg=C_TEXT_BRT); f.pack(fill="x", pady=2)
            tk.Scale(f, from_=lo, to=hi, resolution=r, orient="horizontal", variable=v, bg=C_PANEL, command=lambda _: self._apply_cam()).pack(fill="x")
        tk.Checkbutton(cr, text="Auto Exposure", variable=self.cam_ae, bg=C_BG, fg=C_TEXT_BRT, command=self._apply_cam).pack(fill="x")
        crw("Exposure (us)", self.cam_exposure, 100, 66000, 100); crw("Brightness", self.cam_brightness, -1.0, 1.0, 0.05); crw("Contrast", self.cam_contrast, 0.0, 8.0, 0.1)
        tk.Button(cr, text="Reset Camera", bg=C_RED, command=self._reset_cam).pack(pady=10, fill="x")

    def _mv_start(self):
        self.mv_state = "collecting_init"; self.mv_init_buf = {"top": [], "bottom": []}; self.btn_start.config(state="disabled"); self.mv_status_lbl.config(text="Collecting Initial...", fg=C_AMBER)
    def _mv_stop(self):
        self.mv_state = "collecting_final"; self.mv_final_buf = {"top": [], "bottom": []}; self.btn_stop.config(state="disabled"); self.mv_status_lbl.config(text="Collecting Final...", fg=C_AMBER)
    def _mv_reset(self):
        self.mv_state = "idle"; self.btn_start.config(state="normal"); self.btn_stop.config(state="disabled"); self.mv_status_lbl.config(text="Press START", fg=C_TEXT_MED)

    def _mv_tick(self):
        sc = self.last_data["session_count"]
        if sc == self._last_sc: return
        self._last_sc = sc
        if self.mv_state == "collecting_init":
            for k in ["top", "bottom"]:
                d = self.last_data[k]["dist"]
                if d > 0: self.mv_init_buf[k].append(d)
            n = min(len(self.mv_init_buf["top"]), len(self.mv_init_buf["bottom"]))
            if n >= COLLECT_N:
                self.mv_dist_init = {k: float(np.mean(self.mv_init_buf[k][:COLLECT_N])) for k in ["top", "bottom"]}
                self.mv_state = "ready"; self.btn_stop.config(state="normal"); self.mv_status_lbl.config(text="Initial Captured", fg=C_GREEN)
        elif self.mv_state == "collecting_final":
            for k in ["top", "bottom"]:
                d = self.last_data[k]["dist"]
                if d > 0: self.mv_final_buf[k].append(d)
            n = min(len(self.mv_final_buf["top"]), len(self.mv_final_buf["bottom"]))
            if n >= COLLECT_N:
                self.mv_dist_final = {k: float(np.mean(self.mv_final_buf[k][:COLLECT_N])) for k in ["top", "bottom"]}
                self.mv_state = "done"; self.btn_start.config(state="normal"); self.mv_status_lbl.config(text="Final Captured", fg=C_GREEN)
                for k, il, fl, dl in [("top", self.mv_init_lbl_top, self.mv_final_lbl_top, self.mv_delta_lbl_top), ("bottom", self.mv_init_lbl_bot, self.mv_final_lbl_bot, self.mv_delta_lbl_bot)]:
                    di, df = self.mv_dist_init[k], self.mv_dist_final[k]; delta = df - di
                    il.config(text=f"Init: {di:.3f}"); fl.config(text=f"Final: {df:.3f}"); dl.config(text=f"{delta:+.3f}")

    def load_calib(self):
        f = "camera_params_2.npz" if self.fixed_side.get() == "Right" else "camera_params.npz"
        if os.path.exists(f):
            d = np.load(f)
            # v14 Keys Compatibility
            k_m = 'camera_matrix' if 'camera_matrix' in d else 'mtx'
            k_d = 'dist_coeff' if 'dist_coeff' in d else 'dist'
            return d[k_m], d[k_d]
        return np.array([[1280,0,640],[0,1280,360],[0,0,1]], dtype=np.float32), np.zeros(5)

    def measurement_loop(self):
        try:
            from picamera2 import Picamera2
            pc = Picamera2(); pc.configure(pc.create_video_configuration(main={"size": RESOLUTION, "format": "RGB888"})); pc.start(); self.pc = pc
        except: return
        detector = cv.aruco.ArucoDetector(cv.aruco.getPredefinedDictionary(ARUCO_DICT), cv.aruco.DetectorParameters())
        log.init_log(); buffers = {"top": [], "bottom": []}; l_s = l_u = time.time() * 1000
        while self.is_running:
            K, dist_c = self.load_calib()
            try: frame = cv.cvtColor(pc.capture_array(), cv.COLOR_RGB2BGR)
            except: continue
            corners_raw, ids, _ = detector.detectMarkers(cv.cvtColor(frame, cv.COLOR_BGR2GRAY)); curr = time.time() * 1000
            if ids is not None and len(ids) >= 1:
                m_data = []
                for i in range(len(ids)):
                    raw = corners_raw[i][0]; idx_x = np.argsort(raw[:, 0])
                    lp = raw[idx_x[:2]]; rp = raw[idx_x[2:]]
                    tl = lp[np.argmin(lp[:, 1])]; bl = lp[np.argmax(lp[:, 1])]
                    tr = rp[np.argmin(rp[:, 1])]; br = rp[np.argmax(rp[:, 1])]
                    m_data.append({"c": np.array([tl, tr, br, bl], dtype=np.float32), "c_raw": raw, "y": (tl[1]+tr[1]+br[1]+bl[1])/4.0, "x": (tl[0]+tr[0]+br[0]+bl[0])/4.0})
                m_data.sort(key=lambda m: m["y"])
                top_m = m_data[:2]; bot_m = m_data[2:4] if len(m_data) >= 4 else []
                if len(m_data) == 2:
                    dy, dx = abs(m_data[0]["y"]-m_data[1]["y"]), abs(m_data[0]["x"]-m_data[1]["x"])
                    if dx > dy: top_m = m_data if (m_data[0]["y"]+m_data[1]["y"])/2 < RESOLUTION[1]/2 else []; bot_m = [] if not top_m else m_data
                    else: top_m = [m_data[0]]; bot_m = [m_data[1]]
                elif len(m_data) == 3: top_m = m_data[:2]; bot_m = m_data[2:]
                def proc(marker_list, key, size):
                    if len(marker_list) < 2: self.last_data[key].update({"dist": 0.0, "L_det": False, "R_det": False}); return
                    marker_list.sort(key=lambda m: m["x"]); is_rf = (self.fixed_side.get() == "Right")
                    S_m = marker_list[-1] if is_rf else marker_list[0]; T_m = marker_list[0] if is_rf else marker_list[-1]; h = size/2.0
                    obj = np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]], dtype=np.float32)
                    def get_p(c):
                        _, rv, tv = cv.solvePnP(obj, c, K, dist_c); R, _ = cv.Rodrigues(rv)
                        return np.array([R@pt+tv.ravel() for pt in obj]), math.degrees(math.atan2(R[1,0], R[0,0])), float(tv[2])
                    S_pts, S_roll, S_z = get_p(S_m["c_raw"]); T_pts, T_roll, T_z = get_p(T_m["c_raw"])
                    def i_o(pts, ir):
                        o = pts[:, 0].argsort(); i = pts[o[2:]] if ir else pts[o[:2]]; ot = pts[o[:2]] if ir else pts[o[2:]]
                        return i[i[:,1].argsort()][0], i[i[:,1].argsort()][1], ot[ot[:,1].argsort()][0]
                    S_it, S_ib, S_ot = i_o(S_pts, not is_rf); A = (S_it+S_ib)/2.0; T_it, T_ib, _ = i_o(T_pts, is_rf)
                    p1 = tuple(((S_m["c"][1]+S_m["c"][2])/2 if not is_rf else (S_m["c"][0]+S_m["c"][3])/2).astype(int))
                    p2 = tuple(((T_m["c"][0]+T_m["c"][3])/2 if not is_rf else (T_m["c"][1]+T_m["c"][2])/2).astype(int))
                    si = abs(math.degrees(math.atan2(S_m["c"][1,1]-S_m["c"][0,1], S_m["c"][1,0]-S_m["c"][0,0])))
                    ti = abs(math.degrees(math.atan2(T_m["c"][1,1]-T_m["c"][0,1], T_m["c"][1,0]-T_m["c"][0,0])))
                    buffers[key].append({"A": A, "X_alt": (T_it+T_ib)/2.0, "TR": S_it, "TL_ref": S_ot, "BR": S_ib, "B": T_it, "C": T_ib, "L_A": (S_roll, si), "R_A": (T_roll, ti), "rot": max(si, ti), "L_z": S_z, "R_z": T_z, "p1": p1, "p2": p2})
                    self.last_data[key].update({"L_det": True, "R_det": True})
                if curr-l_s >= 100: proc(top_m, "top", self.size_top.get()); proc(bot_m, "bottom", self.size_bot.get()); l_s = curr
                if curr-l_u >= 1000: # v14 STRICT
                    for k in ["top", "bottom"]:
                        if buffers[k]:
                            s = buffers[k]; aA = np.mean([x["A"] for x in s],0); aTR = np.mean([x["TR"] for x in s],0); aTL = np.mean([x["TL_ref"] for x in s],0)
                            aB = np.mean([x["B"] for x in s],0); aC = np.mean([x["C"] for x in s],0); aX_alt = np.mean([x["X_alt"] for x in s],0)
                            v_r = aTR-aTL; v = v_r/np.linalg.norm(v_r) if np.linalg.norm(v_r)>0 else np.zeros(3); w = aC-aB; u = aB-aA; den = np.dot(v,v)*np.dot(w,w)-np.dot(v,w)**2
                            def _prp():
                                if abs(den)>1e-6: kv = np.clip((np.dot(v,w)*np.dot(u,v)-np.dot(v,v)*np.dot(u,w))/den, 0, 1); ax = aB+kv*w; return ax, np.linalg.norm(ax-aA), kv
                                return aX_alt, np.linalg.norm(aX_alt-aA), 0.5
                            if self.use_angle_thresh.get() and np.mean([x["rot"] for x in s]) > self.rot_threshold.get(): aX, dv, kv = aX_alt, np.linalg.norm(aX_alt-aA), 0.5
                            else: aX, dv, kv = _prp()
                            self.last_data[k].update({"A": aA, "X": aX, "dist": dv, "k": kv, "rot_2d": np.mean([x["rot"] for x in s]), "L_z": np.mean([x["L_z"] for x in s]), "R_z": np.mean([x["R_z"] for x in s]), "L_A": np.mean([x["L_A"] for x in s],0), "R_A": np.mean([x["R_A"] for x in s],0), "TR": aTR, "BR": np.mean([x["BR"] for x in s],0), "B": aB, "C": aC, "p1_px": tuple(np.mean([x["p1"] for x in s],0).astype(int)), "p2_px": tuple(np.mean([x["p2"] for x in s],0).astype(int))})
                            try: log.record(dv, kv, aA, aX, aTR, np.mean([x["BR"] for x in s],0), aB, aC, aA, aTR-aTL, aC-aB, np.mean([x["L_A"] for x in s],0), np.mean([x["R_A"] for x in s],0))
                            except: pass
                            buffers[k].clear()
                    self.last_data["session_count"] += 1; l_u = curr
            else:
                for k in ["top", "bottom"]: self.last_data[k]["dist"] = 0.0
            for k, col in [("top", (0,165,255)), ("bottom", (255,0,255))]:
                d = self.last_data[k]
                if d["dist"]>0 and d["p1_px"]:
                    p1 = d["p1_px"]; p2 = tuple(cv.projectPoints(np.array([d["X"]]), np.zeros(3), np.zeros(3), K, dist_c)[0].ravel().astype(int)) if d["X"][2]>0 else d["p2_px"]
                    cv.line(frame, p1, p2, col, 3); cv.circle(frame, p2, 6, (0,255,0), -1); cv.putText(frame, f"{k.upper()}: {d['dist']:.2f}mm", (p1[0], p1[1]-12), 0, 0.6, col, 2)
            self.current_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB); self._cam_preview_frame = self.current_frame

    def _apply_cam(self):
        if self.pc: 
            try: self.pc.set_controls({"AeEnable": self.cam_ae.get(), "ExposureTime": int(self.cam_exposure.get()), "Brightness": self.cam_brightness.get(), "Contrast": self.cam_contrast.get()})
            except: pass
    def _reset_cam(self):
        self.cam_ae.set(True); self.cam_exposure.set(10000); self.cam_brightness.set(0.0); self.cam_contrast.set(1.0); self._apply_cam()

    def update_gui_loop(self):
        if self.current_frame is not None:
            img = ImageTk.PhotoImage(image=Image.fromarray(self.current_frame).resize((960, 540))); self.canvas.create_image(0, 0, anchor="nw", image=img); self.canvas.img = img
        self.lbl_dist_top.config(text=f"{self.last_data['top']['dist']:.3f} mm"); self.lbl_k_top.config(text=f"k: {self.last_data['top']['k']:.4f}")
        self.lbl_dist_bot.config(text=f"{self.last_data['bottom']['dist']:.3f} mm"); self.lbl_k_bot.config(text=f"k: {self.last_data['bottom']['k']:.4f}")
        self._mv_tick(); self.root.after(33, self.update_gui_loop)

if __name__ == "__main__":
    root = tk.Tk(); app = MeasurementApp(root); root.mainloop()
