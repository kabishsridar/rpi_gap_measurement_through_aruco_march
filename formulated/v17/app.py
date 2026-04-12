import tkinter as tk
from tkinter import ttk
import cv2 as cv
import numpy as np
import threading
import os
from PIL import Image, ImageTk

from constants import *
from measurement_logic import measurement_loop
import log

class MeasurementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dual-Pair ArUco Measurement v17 (v16 GUI, v14 Math)")
        self.root.geometry("1600x960")
        self.root.configure(bg=C_BG)

        # ── Movement vars ──
        self.size_top         = tk.DoubleVar(value=DEFAULT_MARKER_SIZE)
        self.size_bot         = tk.DoubleVar(value=DEFAULT_MARKER_SIZE)
        self.fixed_side       = tk.StringVar(value="Left")
        self.rot_threshold    = tk.DoubleVar(value=12.0)
        self.use_angle_thresh = tk.BooleanVar(value=False)

        # ── Camera vars ──
        self.cam_ae         = tk.BooleanVar(value=True)
        self.cam_exposure   = tk.IntVar(value=10000)
        self.cam_brightness = tk.DoubleVar(value=0.0)
        self.cam_contrast   = tk.DoubleVar(value=1.0)
        self.pc             = None
        self._cam_preview_frame = None

        # ── Movement monitor state ──
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
        threading.Thread(target=measurement_loop, args=(self,), daemon=True).start()
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
        f = tk.LabelFrame(parent, text=f"  {title.upper()}  ", font=F_HEAD, fg=title_color, bg=C_PANEL, bd=2, relief="flat", highlightbackground=title_color, highlightthickness=1, padx=20, pady=12)
        f.pack(**pack_kw)
        return f

    def _build_tab_live(self):
        tab = ttk.Frame(self.tabs); self.tabs.add(tab, text=" 📏  Movement "); tab.configure(style="TFrame")
        left = tk.Frame(tab, bg=C_BG); left.pack(side="left", padx=16, pady=16)
        self.canvas = tk.Canvas(left, width=960, height=540, bg="black", highlightthickness=1, highlightbackground=C_PANEL); self.canvas.pack()
        
        warn_f = tk.Frame(left, bg=C_PANEL, bd=1, relief="solid"); warn_f.pack(fill="x", pady=(12, 0))
        self.warn_strip = tk.Text(warn_f, height=5, font=F_HEAD, fg=C_ACCENT, bg=C_PANEL, padx=15, pady=12, bd=0, highlightthickness=0, wrap="word", state="disabled")
        self.warn_strip.pack(side="left", fill="both", expand=True)
        ws = tk.Scrollbar(warn_f, orient="vertical", command=self.warn_strip.yview, width=12); ws.pack(side="right", fill="y"); self.warn_strip.config(yscrollcommand=ws.set)

        right = tk.Frame(tab, bg=C_BG); right.pack(side="right", fill="both", expand=True, padx=12, pady=10)
        for key, title, col in [("top", "UPPER SENSOR", C_TOP), ("bottom", "LOWER SENSOR", C_BOT)]:
            card = self._card(right, title, col, fill="x", pady=6, padx=12)
            dl = tk.Label(card, text="0.000 mm", font=F_DATA, fg=C_GREEN, bg=C_PANEL); dl.pack(pady=(8, 2))
            kl = tk.Label(card, text="INTERSECT RATIO: 0.0000", font=F_SMALL, fg=C_TEXT_MED, bg=C_PANEL); kl.pack(pady=(0, 8))
            if key == "top": self.lbl_dist_top, self.lbl_k_top = dl, kl
            else: self.lbl_dist_bot, self.lbl_k_bot = dl, kl

        sf = tk.Frame(right, bg=C_BG); sf.pack(fill="x", pady=(15, 0))
        self.mv_status_lbl = tk.Label(sf, text="READY FOR INITIAL CAPTURE", font=F_HEAD, fg=C_TEXT_MED, bg=C_BG); self.mv_status_lbl.pack(pady=2)
        self.mv_prog_bar = ttk.Progressbar(right, length=380, maximum=COLLECT_N, mode="determinate"); self.mv_prog_bar.pack(pady=5)
        
        bf = tk.Frame(right, bg=C_BG); bf.pack(pady=10, fill="x", padx=12)
        self.btn_reset = tk.Button(bf, text="↺", bg=C_CARD, fg="white", font=F_BTN, width=5, pady=10, command=self._mv_reset); self.btn_reset.pack(side="right", padx=6)
        self.btn_start = tk.Button(bf, text="▶ START SESSION", bg=C_GREEN, fg=C_BG, font=F_BTN, padx=20, pady=10, command=self._mv_start); self.btn_start.pack(side="left", expand=True, fill="x", padx=6)
        self.btn_stop = tk.Button(bf, text="■ STOP (SAVE)", bg=C_RED, fg=C_BG, font=F_BTN, state="disabled", padx=20, pady=10, command=self._mv_stop); self.btn_stop.pack(side="left", expand=True, fill="x", padx=6)

        for key, title, col in [("top", "UPPER", C_TOP), ("bottom", "LOWER", C_BOT)]:
            row = tk.Frame(right, bg=C_PANEL, bd=1, relief="solid"); row.pack(fill="x", padx=12, pady=4)
            dl = tk.Label(row, text="—", font=F_DATA, fg=C_ACCENT, bg=C_PANEL); dl.pack(side="right", padx=15, pady=8)
            tk.Label(row, text=title, font=F_HEAD, fg=col, bg=C_PANEL, width=8).pack(side="left", padx=12)
            info = tk.Frame(row, bg=C_PANEL); info.pack(side="left", expand=True)
            il = tk.Label(info, text="INIT: —", font=F_BODY, fg=C_TEXT_MED, bg=C_PANEL, anchor="w"); il.pack(fill="x")
            fl = tk.Label(info, text="FINAL: —", font=F_BODY, fg=C_TEXT_MED, bg=C_PANEL, anchor="w"); fl.pack(fill="x")
            if key == "top": self.mv_init_lbl_top, self.mv_final_lbl_top, self.mv_delta_lbl_top = il, fl, dl
            else: self.mv_init_lbl_bot, self.mv_final_lbl_bot, self.mv_delta_lbl_bot = il, fl, dl

    def _build_tab_tele(self):
        tab = ttk.Frame(self.tabs); self.tabs.add(tab, text=" 🛸  Telemetry ")
        mtf = tk.Frame(tab, bg=C_BG); mtf.pack(fill="both", expand=True)
        self.tele_vars = {"top": {}, "bottom": {}}
        v_show = ["A", "X", "TR", "BR", "B", "C", "L_ROL", "L_ROT", "L_Z", "R_ROL", "R_ROT", "R_Z"]
        for key, title, col in [("top", "TOP", C_TOP), ("bottom", "BOT", C_BOT)]:
            cf = tk.LabelFrame(mtf, text=f" {title} DATA ", font=F_HEAD, fg=col, bg=C_BG); cf.pack(side="left", fill="both", expand=True, padx=20, pady=20)
            for v_name in v_show:
                row = tk.Frame(cf, bg=C_PANEL); row.pack(fill="x", pady=4, padx=10)
                tk.Label(row, text=v_name, font=F_SMALL, bg=C_PANEL, fg=C_TEXT_MED).pack(side="left", padx=10)
                sv = tk.StringVar(value="—"); tk.Label(row, textvariable=sv, font=F_MONO, bg=C_PANEL, fg=C_TEXT_BRT).pack(side="right", padx=10)
                self.tele_vars[key][v_name] = sv

    def _build_tab_settings(self):
        tab = ttk.Frame(self.tabs); self.tabs.add(tab, text=" ⚙  Settings ")
        sc = tk.Frame(tab, bg=C_BG); sc.pack(fill="both", expand=True, padx=80, pady=40)
        rf = tk.LabelFrame(sc, text=" Reference (Fixed Side) ", bg=C_PANEL, fg=C_TEXT_BRT, font=F_HEAD); rf.pack(fill="x", pady=10)
        for choice in ["Left", "Right"]: tk.Radiobutton(rf, text=choice, variable=self.fixed_side, value=choice, bg=C_PANEL, fg=C_TEXT_BRT).pack(side="left", padx=30)
        def sld(l, v, c, lo, hi):
            f = tk.LabelFrame(sc, text=f" {l} ", bg=C_PANEL, fg=c, font=F_HEAD); f.pack(fill="x", pady=10)
            tk.Scale(f, from_=lo, to=hi, resolution=0.1, orient="horizontal", variable=v, bg=C_PANEL, length=600).pack(side="left", padx=20)
            tk.Entry(f, textvariable=v, width=10, font=F_MONO).pack(side="left")
        sld("Upper Marker (mm)", self.size_top, C_TOP, 10, 200); sld("Lower Marker (mm)", self.size_bot, C_BOT, 10, 200); sld("Rot Thresh °", self.rot_threshold, C_TEXT_BRT, 0, 45)
        tk.Checkbutton(sc, text="Use Angle Threshold", variable=self.use_angle_thresh, bg=C_BG, fg=C_TEXT_BRT, font=F_BODY).pack(pady=20)

    def _build_tab_cam(self):
        tab = ttk.Frame(self.tabs); self.tabs.add(tab, text=" 📷  Camera ")
        cl = tk.Frame(tab, bg=C_BG); cl.pack(side="left", fill="both", expand=True)
        cr = tk.Frame(tab, bg=C_BG); cr.pack(side="right", fill="y", padx=20)
        self.cam_canvas = tk.Canvas(cl, width=800, height=480, bg="black", highlightthickness=2, highlightbackground=C_PANEL); self.cam_canvas.pack(pady=30)
        self.lbl_cam_status = tk.Label(cl, text="⏳ Wait...", font=F_TITLE, fg=C_AMBER, bg=C_BG); self.lbl_cam_status.pack()
        def crw(l, v, lo, hi, r):
            f = tk.LabelFrame(cr, text=f" {l} ", bg=C_PANEL, fg=C_TEXT_BRT, font=F_HEAD); f.pack(fill="x", pady=5)
            tk.Scale(f, from_=lo, to=hi, resolution=r, orient="horizontal", variable=v, bg=C_PANEL, command=lambda _: self._apply_cam()).pack(fill="x", padx=10)
        tk.Checkbutton(cr, text="Auto Exposure", variable=self.cam_ae, bg=C_BG, fg=C_TEXT_BRT, font=F_BODY, command=self._apply_cam).pack(fill="x", pady=5)
        crw("Exposure", self.cam_exposure, 100, 66000, 100); crw("Brightness", self.cam_brightness, -1.0, 1.0, 0.05); crw("Contrast", self.cam_contrast, 0.0, 8.0, 0.1)
        tk.Button(cr, text="Reset Camera", bg=C_RED, fg="white", font=F_BTN, command=self._reset_cam).pack(pady=20, fill="x")

    def load_calib(self):
        f = "camera_params_2.npz" if self.fixed_side.get() == "Right" else "camera_params.npz"
        if os.path.exists(f):
            d = np.load(f)
            # v14 compatibility check (camera_matrix vs mtx)
            k_m = 'camera_matrix' if 'camera_matrix' in d else 'mtx'
            k_d = 'dist_coeff' if 'dist_coeff' in d else 'dist'
            return d[k_m], d[k_d]
        return np.array([[1280,0,640],[0,1280,360],[0,0,1]], dtype=np.float32), np.zeros(5)

    def _mv_start(self):
        self.mv_state = "collecting_init"; self.mv_init_buf = {"top": [], "bottom": []}; self.btn_start.config(state="disabled"); self.mv_status_lbl.config(text="COLLECTING INITIALS...", fg=C_AMBER)
    def _mv_stop(self):
        self.mv_state = "collecting_final"; self.mv_final_buf = {"top": [], "bottom": []}; self.btn_stop.config(state="disabled"); self.mv_status_lbl.config(text="COLLECTING FINALS...", fg=C_AMBER)
    def _mv_reset(self):
        self.mv_state = "idle"; self.btn_start.config(state="normal"); self.btn_stop.config(state="disabled"); self.mv_status_lbl.config(text="READY FOR INITIAL CAPTURE", fg=C_TEXT_MED)

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
                self.mv_state = "ready"; self.btn_stop.config(state="normal"); self.mv_status_lbl.config(text="INITIAL CAPTURED", fg=C_GREEN)
        elif self.mv_state == "collecting_final":
            for k in ["top", "bottom"]:
                d = self.last_data[k]["dist"]
                if d > 0: self.mv_final_buf[k].append(d)
            n = min(len(self.mv_final_buf["top"]), len(self.mv_final_buf["bottom"]))
            if n >= COLLECT_N:
                self.mv_dist_final = {k: float(np.mean(self.mv_final_buf[k][:COLLECT_N])) for k in ["top", "bottom"]}
                self.mv_state = "done"; self.btn_start.config(state="normal"); self.mv_status_lbl.config(text="FINAL CAPTURED", fg=C_GREEN)
                for k, il, fl, dl in [("top", self.mv_init_lbl_top, self.mv_final_lbl_top, self.mv_delta_lbl_top), ("bottom", self.mv_init_lbl_bot, self.mv_final_lbl_bot, self.mv_delta_lbl_bot)]:
                    di, df = self.mv_dist_init[k], self.mv_dist_final[k]; delta = df - di
                    il.config(text=f"INIT: {di:.3f} mm"); fl.config(text=f"FINAL: {df:.3f} mm"); dl.config(text=f"{delta:+.3f} mm")

    def _apply_cam(self):
        if self.pc: 
            try: self.pc.set_controls({"AeEnable": self.cam_ae.get(), "ExposureTime": int(self.cam_exposure.get()), "Brightness": self.cam_brightness.get(), "Contrast": self.cam_contrast.get()})
            except: pass
    def _reset_cam(self):
        self.cam_ae.set(True); self.cam_exposure.set(10000); self.cam_brightness.set(0.0); self.cam_contrast.set(1.0); self._apply_cam()

    def update_gui_loop(self):
        if self.current_frame is not None:
            img = ImageTk.PhotoImage(image=Image.fromarray(self.current_frame).resize((960, 540))); self.canvas.create_image(0, 0, anchor="nw", image=img); self.canvas.img = img
        self.lbl_dist_top.config(text=f"{self.last_data['top']['dist']:.3f} mm"); self.lbl_k_top.config(text=f"INTERSECT RATIO: {self.last_data['top']['k']:.4f}")
        self.lbl_dist_bot.config(text=f"{self.last_data['bottom']['dist']:.3f} mm"); self.lbl_k_bot.config(text=f"INTERSECT RATIO: {self.last_data['bottom']['k']:.4f}")
        
        # UI Alerts Logic
        strip = []; r_th = self.rot_threshold.get()
        for k in ["top", "bottom"]:
            d = self.last_data[k]
            if not d["L_det"]: strip.append(f"[{k.upper()}] Left Missing")
            if not d["R_det"]: strip.append(f"[{k.upper()}] Right Missing")
            if d["rot_2d"] > r_th: strip.append(f"[{k.upper()}] Rotation {d['rot_2d']:.1f}°")
        self.warn_strip.config(state="normal"); self.warn_strip.delete("1.0", tk.END)
        if strip: self.warn_strip.insert(tk.END, "![!] ACTIVE ALERTS:\n" + "\n".join([f" • {x}" for x in strip]))
        else: self.warn_strip.insert(tk.END, "[OK] SYSTEM NOMINAL")
        self.warn_strip.config(state="disabled")

        if self._cam_preview_frame is not None:
            cimg = ImageTk.PhotoImage(image=Image.fromarray(self._cam_preview_frame).resize((800, 480))); self.cam_canvas.create_image(0, 0, anchor="nw", image=cimg); self.cam_canvas.img = cimg
            self.lbl_cam_status.config(text="🟢 Live Feed Active" if self.last_data["top"]["dist"]>0 or self.last_data["bottom"]["dist"]>0 else "🟡 Waiting for markers...", fg=C_GREEN if self.last_data["top"]["dist"]>0 else C_AMBER)

        # Telemetry update
        for k in ["top", "bottom"]:
            d = self.last_data[k]; v = self.tele_vars[k]
            for kn in ["A", "X", "TR", "BR", "B", "C"]: v[kn].set(f"{d[kn][0]:.1f}, {d[kn][1]:.1f}, {d[kn][2]:.1f}")
            v["L_ROL"].set(f"{d['L_A'][0]:.1f}°"); v["L_ROT"].set(f"{d['L_A'][1]:.1f}°"); v["L_Z"].set(f"{d['L_z']:.1f}")
            v["R_ROL"].set(f"{d['R_A'][0]:.1f}°"); v["R_ROT"].set(f"{d['R_A'][1]:.1f}°"); v["R_Z"].set(f"{d['R_z']:.1f}")

        self._mv_tick(); self.root.after(33, self.update_gui_loop)
