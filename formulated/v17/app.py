import tkinter as tk
from tkinter import ttk, messagebox
import cv2 as cv
import numpy as np
import threading
import time
import os
from datetime import datetime

from constants import *
from utils import load_calibration
from measurement_engine import MeasurementEngine
import log

class MeasurementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dual-Pair ArUco Measurement v17")
        self.root.geometry("1600x960")
        self.root.configure(bg=C_BG)

        self.engine = MeasurementEngine()
        self.running = True
        self.paused = False
        
        # State
        self.mv_state = "ready"
        self._init_data = {"top": None, "bottom": None}
        self.last_data = {
            "top": self._empty(), "bottom": self._empty(),
            "lighting": {"status": "ok", "mean": 0, "std": 0},
            "session_count": 0
        }
        self._last_sc = 0

        # UI references for results
        self.ui_results = {}

        # Styles
        self._style()
        self.setup_ui()

        # Threading
        self.thread = threading.Thread(target=self.measurement_loop, daemon=True)
        self.thread.start()

        # Loops
        self._gui_update_loop()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _empty(self):
        return {
            "dist": 0.0, "k": 0.5, "rot_2d": 0.0,
            "L_det": False, "R_det": False,
            "L_pitch": 0.0, "L_yaw": 0.0, "R_pitch": 0.0, "R_yaw": 0.0,
            "L_z": 0, "R_z": 0, "L_A": 0, "R_A": 0,
            "A": None, "X": None, "p1_px": None, "p2_px": None
        }

    def _style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=C_BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=C_CARD, foreground=C_TEXT_MED,
                        font=F_HEAD, padding=[20, 10], borderwidth=0)
        style.map("TNotebook.Tab", background=[("selected", C_PANEL)], foreground=[("selected", C_ACCENT)])

    def setup_ui(self):
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True)

        self._build_tab_live()
        self._build_tab_tele()
        self._build_tab_settings()
        self._build_tab_cam()

    def _build_tab_live(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text=" 🔴  Live Measurement ")
        
        # Camera Feed
        self.cam_frame = tk.Label(tab, bg="black")
        self.cam_frame.pack(side="left", fill="both", expand=True, padx=20, pady=20)
        
        # Right Panel
        right = tk.Frame(tab, bg=C_BG)
        right.pack(side="right", fill="both", expand=True, padx=12, pady=10)

        # Movement Monitor Controls
        stf = tk.Frame(right, bg=C_BG); stf.pack(fill="x", pady=(15, 0))
        self.mv_status_lbl = tk.Label(stf, text="READY FOR INITIAL CAPTURE", font=F_HEAD, 
                                      fg=C_TEXT_MED, bg=C_BG, wraplength=400, justify="center")
        self.mv_status_lbl.pack(pady=(2, 2))
        
        prog_f = tk.Frame(right, bg=C_BG); prog_f.pack(fill="x", padx=12)
        self.mv_prog_lbl = tk.Label(prog_f, text="", font=F_HEAD, fg=C_GREEN, bg=C_BG); self.mv_prog_lbl.pack()
        self.mv_prog_bar = ttk.Progressbar(prog_f, length=380, maximum=COLLECT_N, mode="determinate"); self.mv_prog_bar.pack(pady=4)

        btn_f = tk.Frame(right, bg=C_BG); btn_f.pack(pady=10, fill="x", padx=12)
        btn_cfg = dict(font=F_BTN, relief="flat", padx=20, pady=12, bd=0, cursor="hand2")
        def add_hover(btn, normal_bg, hover_bg):
            btn.bind("<Enter>", lambda e: btn.config(bg=hover_bg)); btn.bind("<Leave>", lambda e: btn.config(bg=normal_bg))

        self.btn_reset = tk.Button(btn_f, text="↺", bg=C_CARD, fg="white", activebackground=C_TEXT_MED,
                                   font=F_BTN, relief="flat", width=5, pady=12, bd=0, cursor="hand2", 
                                   command=self._mv_reset); self.btn_reset.pack(side="right", padx=6); add_hover(self.btn_reset, C_CARD, C_TEXT_MED)
        
        self.btn_start = tk.Button(btn_f, text="▶ START SESSION", bg=C_GREEN, fg=C_BG, 
                                   activebackground="#86efac", command=self._mv_start, **btn_cfg)
        self.btn_start.pack(side="left", expand=True, fill="x", padx=6); add_hover(self.btn_start, C_GREEN, "#86efac")
        
        self.btn_stop = tk.Button(btn_f, text="■ STOP (SAVE)", bg=C_RED, fg=C_BG, 
                                  activebackground="#fca5a5", state="disabled", command=self._mv_stop, **btn_cfg)
        self.btn_stop.pack(side="left", expand=True, fill="x", padx=6); add_hover(self.btn_stop, C_RED, "#fca5a5")

        tk.Frame(right, bg=C_CARD, height=2).pack(fill="x", padx=12, pady=15)
        
        # Result Cards
        for key, title, col in [("top", "UPPER", C_TOP), ("bottom", "LOWER", C_BOT)]:
            row = tk.Frame(right, bg=C_PANEL, bd=1, relief="solid"); row.pack(fill="x", padx=12, pady=4)
            # DELTA packed first on right to guarantee space
            dl = tk.Label(row, text="—", font=F_DATA, fg=C_ACCENT, bg=C_PANEL, anchor="e"); dl.pack(side="right", padx=15, pady=8)
            tk.Label(row, text=title, font=F_HEAD, fg=col, bg=C_PANEL, width=8).pack(side="left", padx=12, pady=8)
            info = tk.Frame(row, bg=C_PANEL); info.pack(side="left", fill="x", expand=True)
            il = tk.Label(info, text="INIT: —", font=F_BODY, fg=C_TEXT_MED, bg=C_PANEL, anchor="w"); il.pack(fill="x", padx=8)
            fl = tk.Label(info, text="FINAL: —", font=F_BODY, fg=C_TEXT_MED, bg=C_PANEL, anchor="w"); fl.pack(fill="x", padx=8)
            self.ui_results[key] = (il, fl, dl)

        tk.Frame(right, bg=C_BG).pack(fill="both", expand=True)
        
        # Warnings Strip at bottom left
        self.warn_strip = tk.Text(tab, height=5, bg="#020617", fg=C_AMBER, font=F_SMALL, 
                                  relief="flat", padx=10, pady=5); self.warn_strip.pack(side="bottom", fill="x", padx=20, pady=(0, 20))
        self.lbl_lighting_cam = tk.Label(self.cam_frame, text="LIGHTING PROFILE: NOMINAL", font=F_SMALL, 
                                        fg=C_GREEN, bg="black"); self.lbl_lighting_cam.place(x=10, y=10)

    def _build_tab_tele(self):
        tab = ttk.Frame(self.tabs); self.tabs.add(tab, text=" 🛸  Dual Telemetry ")
        mtf = tk.Frame(tab, bg=C_BG); mtf.pack(fill="both", expand=True)
        self.tele_vars = {"top": {}, "bottom": {}}
        v_show = ["A", "X", "TR", "BR", "B", "C", "L_ROL", "L_ROT", "L_Z", "L_PITCH", "L_YAW", "R_ROL", "R_ROT", "R_Z", "R_PITCH", "R_YAW"]
        for key, title, col in [("top", "TOP", C_TOP), ("bottom", "BOTTOM", C_BOT)]:
            cf = tk.LabelFrame(mtf, text=f" {title} PAIR TELEMETRY ", font=F_TITLE, fg=col, bg=C_BG, padx=15, pady=15)
            cf.pack(side="left", fill="both", expand=True, padx=20, pady=20)
            for v_name in v_show:
                row = tk.Frame(cf, bg=C_PANEL, highlightbackground=C_CARD, highlightthickness=1); row.pack(fill="x", padx=10, pady=5)
                tk.Label(row, text=v_name, font=F_HEAD, fg=C_TEXT_MED, bg=C_PANEL).pack(side="left", padx=10, pady=5)
                sv = tk.StringVar(value="—"); tk.Label(row, textvariable=sv, font=F_MONO, fg=C_TEXT_BRT, bg=C_PANEL).pack(side="right", padx=10, pady=5)
                self.tele_vars[key][v_name] = sv

    def _build_tab_settings(self):
        tab = ttk.Frame(self.tabs); self.tabs.add(tab, text=" ⚙  Settings ")
        sf = tk.Frame(tab, bg=C_BG, padx=30, pady=30); sf.pack(fill="both", expand=True)
        
        self.size_top = tk.DoubleVar(value=MARKER_SIZE_TOP)
        self.size_bot = tk.DoubleVar(value=MARKER_SIZE_BOT)
        self.pitch_threshold = tk.DoubleVar(value=PITCH_THRESH)
        self.yaw_threshold = tk.DoubleVar(value=YAW_THRESH)
        self.rot_threshold = tk.DoubleVar(value=ROT_THRESH)
        self.use_angle_thresh = tk.BooleanVar(value=True)

        for title, var, min_v, max_v in [
            ("Upper Marker Size (mm)", self.size_top, 10, 50),
            ("Lower Marker Size (mm)", self.size_bot, 10, 50),
            ("Warning Tilt (Pitch)", self.pitch_threshold, 1, 30),
            ("Warning Tilt (Yaw)", self.yaw_threshold, 1, 30),
            ("Warning In-Plane Rotation", self.rot_threshold, 1, 20)
        ]:
            row = tk.Frame(sf, bg=C_BG); row.pack(fill="x", pady=10)
            tk.Label(row, text=title, font=F_HEAD, fg=C_TEXT_BRT, bg=C_BG, width=25, anchor="w").pack(side="left")
            tk.Scale(row, from_=min_v, to=max_v, resolution=0.1, variable=var, orient="horizontal", 
                     bg=C_BG, fg=C_TEXT_MED, highlightthickness=0, length=400).pack(side="left", padx=20)
        
        tk.Checkbutton(sf, text=" Use Rotation Threshold for Result Filtering", 
                       variable=self.use_angle_thresh, font=F_HEAD, fg=C_GREEN, bg=C_BG, 
                       selectcolor=C_PANEL, activebackground=C_BG).pack(fill="x", pady=20)

    def _build_tab_cam(self):
        tab = ttk.Frame(self.tabs); self.tabs.add(tab, text=" 📽  Camera Config ")
        cf = tk.Frame(tab, bg=C_BG, padx=30, pady=30); cf.pack(fill="both", expand=True)
        tk.Label(cf, text="Manual Camera Control Overrides", font=F_TITLE, fg=C_ACCENT, bg=C_BG).pack(anchor="w", pady=(0, 20))
        # Note: In Windows/Desktop, many Picamera2 controls won't apply.
        tk.Label(cf, text="Advanced camera parameters for Raspberry Pi PiCamera2 integration.", 
                 font=F_BODY, fg=C_TEXT_MED, bg=C_BG).pack(anchor="w")

    def _mv_start(self):
        self.mv_state = "collecting_init"; self.btn_start.config(state="disabled")
    def _mv_stop(self):
        self.mv_state = "collecting_final"; self.btn_stop.config(state="disabled")
    def _mv_reset(self):
        self.mv_state = "ready"; self._init_data = {"top": None, "bottom": None}
        self.btn_start.config(state="normal"); self.btn_stop.config(state="disabled")
        self.mv_status_lbl.config(text="READY FOR INITIAL CAPTURE", fg=C_TEXT_MED); self.mv_prog_bar["value"]=0; self.mv_prog_lbl.config(text="")
        for k in ["top", "bottom"]: il, fl, dl = self.ui_results[k]; il.config(text="INIT: —"); fl.config(text="FINAL: —"); dl.config(text="—", fg=C_ACCENT)

    def measurement_loop(self):
        # Load Calibrations
        K1, d1 = load_calibration(CALIB_FILE_1); K2, d2 = load_calibration(CALIB_FILE_2)
        cap = cv.VideoCapture(0); cap.set(cv.CAP_PROP_FRAME_WIDTH, RESOLUTION[0]); cap.set(cv.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
        
        bufs = {"top": [], "bottom": []}
        l_s = l_u = time.time()*1000

        while self.running:
            if self.paused: time.sleep(0.1); continue
            ret, frame = cap.read()
            if not ret: time.sleep(0.1); continue
            
            curr = time.time()*1000
            l_info, gray = self.engine.analyze_lighting(frame); self.last_data["lighting"] = l_info
            top_m, bot_m = self.engine.detect_markers(gray)
            
            # Reset detection flags
            for k in ["top", "bottom"]: self.last_data[k]["L_det"] = False; self.last_data[k]["R_det"] = False

            def proc(pair_list, pk, size, K, dc):
                if len(pair_list) < 2: return
                m1, m2 = sorted(pair_list, key=lambda x: x["x"])
                is_rf = (pk == "bottom")
                # L/R logic based on row
                left_m, right_m = (m1, m2)
                p_pts, roll, z, pitch, yaw = self.engine.get_pose(left_m["c"], K, dc, size)
                p2_pts, r2, z2, pitch2, yaw2 = self.engine.get_pose(right_m["c"], K, dc, size)
                
                # Formula inputs...
                Sit, Sib, Sot = self.engine.inner_outer_pts(p_pts, not is_rf)
                Tit, Tib, _   = self.engine.inner_outer_pts(p2_pts, is_rf)
                p1_px = tuple(((m1["c"][1 if not is_rf else 0] + m1["c"][2 if not is_rf else 3]) / 2).astype(int))
                p2_px = tuple(((m2["c"][0 if not is_rf else 1] + m2["c"][3 if not is_rf else 2]) / 2).astype(int))
                
                bufs[pk].append({
                    "A": (Sit+Sib)/2.0, "TR": Sit, "BR": Sib, "TL_ref": Sot, "B": Tit, "C": Tib, "rot": max(self.engine.inplane_rot(m1["c"]), self.engine.inplane_rot(m2["c"])),
                    "L_z": z, "R_z": z2, "L_p": pitch, "L_y": yaw, "R_p": pitch2, "R_y": yaw2, "p1": p1_px, "p2": p2_px
                })
                self.last_data[pk]["L_det"] = True; self.last_data[pk]["R_det"] = True

            # Sample every 100ms
            if (curr - l_s) >= 100:
                if top_m: proc(top_m, "top", self.size_top.get(), K1, d1)
                if bot_m: proc(bot_m, "bottom", self.size_bot.get(), K2, d2)
                l_s = curr

            # Live Drawing every frame
            for pair, k, col in [(top_m, "top", (0, 165, 255)), (bot_m, "bottom", (255, 0, 255))]:
                if len(pair) == 2:
                    m1, m2 = sorted(pair, key=lambda x: x["x"])
                    is_rf = (k == "bottom"); c1, c2 = m1["c"], m2["c"]
                    p1 = tuple(((c1[1 if not is_rf else 0] + c1[2 if not is_rf else 3]) / 2).astype(int))
                    p2 = tuple(((c2[0 if not is_rf else 1] + c2[3 if not is_rf else 2]) / 2).astype(int))
                    cv.line(frame, p1, p2, col, 2)

            # 200ms snappy update
            if (curr - l_u) >= 200:
                for k in ["top", "bottom"]:
                    if bufs[k]:
                        s = bufs[k]
                        aA  = np.mean([x["A"]  for x in s], axis=0)
                        aTR = np.mean([x["TR"] for x in s], axis=0)
                        aTL = np.mean([x["TL_ref"] for x in s], axis=0)
                        aB = np.mean([x["B"] for x in s], axis=0)
                        aC = np.mean([x["C"] for x in s], axis=0)
                        
                        v_raw = aTR - aTL; v = v_raw / np.linalg.norm(v_raw) if np.linalg.norm(v_raw)>0 else np.zeros(3)
                        w = aC - aB; u = aB - aA; den = np.dot(v,v)*np.dot(w,w) - np.dot(v,w)**2
                        if abs(den)>1e-6: kv = np.clip(((np.dot(v,w)*np.dot(u,v)) - (np.dot(v,v)*np.dot(u,w)))/den, 0, 1); aX = aB + kv*w
                        else: aX = (aB+aC)/2.0; kv = 0.5
                        dist = np.linalg.norm(aX-aA)
                        
                        last = s[-1]
                        self.last_data[k].update({
                            "A": aA, "X": aX, "dist": dist, "k": kv, "rot_2d": np.mean([x["rot"] for x in s]),
                            "L_pitch": np.mean([x["L_p"] for x in s]), "L_yaw": np.mean([x["L_y"] for x in s]),
                            "R_pitch": np.mean([x["R_p"] for x in s]), "R_yaw": np.mean([x["R_y"] for x in s]),
                            "L_z": np.mean([x["L_z"] for x in s]), "R_z": np.mean([x["R_z"] for x in s]),
                            "p1_px": last["p1"], "p2_px": last["p2"]
                        })
                        self.last_data["session_count"] += 1; bufs[k].clear()
                        # Final Drawing points
                        if dist > 0:
                            cv.circle(frame, last["p2"], 6, (0, 255, 0), -1)
                            cv.putText(frame, f"{k.upper()}: {dist:.2f}mm", (last["p1"][0], last["p1"][1]-12), 0, 0.6, (0, 220, 255), 2)
                l_u = curr

            self.current_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    def _gui_update_loop(self):
        if hasattr(self, "current_frame"):
            img = tk.PhotoImage(data=cv.imencode('.png', cv.cvtColor(self.current_frame, cv.COLOR_RGB2BGR))[1].tobytes())
            self.cam_frame.config(image=img); self.cam_frame.image = img
        self._update_movement_monitor(); self._update_telemetry(); self._update_warnings()
        self.root.after(30, self._gui_update_loop)

    def _update_movement_monitor(self):
        sc = self.last_data["session_count"]
        if sc == self._last_sc: return
        self._last_sc = sc

        p_top = len(self._init_data.get("top_buf", [])) if self.mv_state == "collecting_init" else 0
        p_bot = len(self._init_data.get("bot_buf", [])) if self.mv_state == "collecting_init" else 0
        
        if self.mv_state == "collecting_init":
            self.mv_status_lbl.config(text="CAPTURING INITIAL POSITION...", fg=C_ACCENT)
            if "top_buf" not in self._init_data: self._init_data["top_buf"] = []; self._init_data["bot_buf"] = []
            if self.last_data["top"]["L_det"]: self._init_data["top_buf"].append(self.last_data["top"]["dist"])
            if self.last_data["bottom"]["L_det"]: self._init_data["bot_buf"].append(self.last_data["bottom"]["dist"])
            self.mv_prog_bar["value"] = len(self._init_data["top_buf"])
            if len(self._init_data["top_buf"]) >= COLLECT_N:
                self._init_data["top"] = np.median(self._init_data["top_buf"])
                self._init_data["bottom"] = np.median(self._init_data["bot_buf"])
                self.mv_state = "ready_for_final"; self.btn_stop.config(state="normal")
                self.mv_status_lbl.config(text="INITIAL GAP CAPTURED - MEASURE NOW", fg=C_GREEN)
                il_t, _, _ = self.ui_results["top"]; il_t.config(text=f"INIT: {self._init_data['top']:.3f} mm")
                il_b, _, _ = self.ui_results["bottom"]; il_b.config(text=f"INIT: {self._init_data['bottom']:.3f} mm")

        elif self.mv_state == "ready_for_final":
            for k in ["top", "bottom"]:
                _, _, dl = self.ui_results[k]
                delta = self.last_data[k]["dist"] - (self._init_data[k] if self._init_data[k] else 0)
                dl.config(text=f"{'+' if delta>=0 else ''}{delta:.3f} mm", fg=C_GREEN if abs(delta)<0.5 else C_RED)

        elif self.mv_state == "collecting_final":
            if "final_buf_t" not in self._init_data: self._init_data["final_buf_t"] = []; self._init_data["final_buf_b"] = []
            if self.last_data["top"]["L_det"]: self._init_data["final_buf_t"].append(self.last_data["top"]["dist"])
            if self.last_data["bottom"]["L_det"]: self._init_data["final_buf_b"].append(self.last_data["bottom"]["dist"])
            self.mv_prog_bar["value"] = len(self._init_data["final_buf_t"])
            if len(self._init_data["final_buf_t"]) >= COLLECT_N:
                f_t = np.median(self._init_data["final_buf_t"]); f_b = np.median(self._init_data["final_buf_b"])
                d_t, d_b = f_t - self._init_data["top"], f_b - self._init_data["bottom"]
                _, fl_t, dl_t = self.ui_results["top"]; fl_t.config(text=f"FINAL: {f_t:.3f} mm"); dl_t.config(text=f"{'+' if d_t>=0 else ''}{d_t:.3f} mm")
                _, fl_b, dl_b = self.ui_results["bottom"]; fl_b.config(text=f"FINAL: {f_b:.3f} mm"); dl_b.config(text=f"{'+' if d_b>=0 else ''}{d_b:.3f} mm")
                self.mv_state = "finished"; self.mv_status_lbl.config(text="MEASUREMENT COMPLETE - PRESS RESET", fg=C_AMBER)

    def _update_telemetry(self):
        for k in ["top", "bottom"]:
            d = self.last_data[k]
            self.tele_vars[k]["L_ROT"].set(f"{d['rot_2d']:.1f}°")
            self.tele_vars[k]["L_Z"].set(f"{d['L_z']:.1f}")
            self.tele_vars[k]["L_PITCH"].set(f"{d['L_pitch']:.1f}°")
            self.tele_vars[k]["L_YAW"].set(f"{d['L_yaw']:.1f}°")
            self.tele_vars[k]["R_PITCH"].set(f"{d['R_pitch']:.1f}°")
            self.tele_vars[k]["R_YAW"].set(f"{d['R_yaw']:.1f}°")

    def _update_warnings(self):
        p_th, y_th, r_th = self.pitch_threshold.get(), self.yaw_threshold.get(), self.rot_threshold.get(); strip = []
        for k in ["top", "bottom"]:
            d = self.last_data[k]
            if not d["L_det"]: strip.append(f"[{k.upper()}] L-marker not detected")
            if not d["R_det"]: strip.append(f"[{k.upper()}] R-marker not detected")
            if d["rot_2d"] > r_th: strip.append(f"[{k.upper()}] Rotation {d['rot_2d']:.1f}°")
            lp, rp = d.get("L_pitch", 0), d.get("R_pitch", 0)
            if abs(lp)>p_th: strip.append(f"[{k.upper()}] L-Pitch {lp:+.1f}°")
            if abs(rp)>p_th: strip.append(f"[{k.upper()}] R-Pitch {rp:+.1f}°")
        
        self.warn_strip.config(state="normal"); self.warn_strip.delete("1.0", tk.END)
        if strip: self.warn_strip.insert(tk.END, "![!] ACTIVE ALERTS:\n" + "\n".join([f" • {x}" for x in strip]))
        else: self.warn_strip.insert(tk.END, "[OK] SYSTEM NOMINAL")
        self.warn_strip.config(state="disabled")

    def on_close(self):
        self.running = False; self.root.destroy()
