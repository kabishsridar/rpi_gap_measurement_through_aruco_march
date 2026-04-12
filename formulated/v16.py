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

class SmartEntry(tk.Entry):
    def __init__(self, parent, variable, **kwargs):
        super().__init__(parent, textvariable=None, **kwargs)
        self.variable = variable
        self.insert(0, str(variable.get()))
        self.bind("<Return>", self._update_var)
        self.bind("<FocusOut>", self._update_var)
        self.variable.trace_add("write", self._update_entry)

    def _update_var(self, event=None):
        try:
            self.variable.set(float(self.get()))
            self.selection_clear()
            self.winfo_toplevel().focus_set()
            self.update_idletasks() # Force GUI to sync sliders
        except:
            self.delete(0, tk.END)
            self.insert(0, str(self.variable.get()))

    def _update_entry(self, *args):
        if self.focus_get() != self:
            self.delete(0, tk.END)
            self.insert(0, str(self.variable.get()))

class MeasurementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dual-Pair ArUco Measurement v17")
        self.root.geometry("1600x960")
        self.root.configure(bg=C_BG)

        # ── Machine Vars ──
        self.size_top         = tk.DoubleVar(value=DEFAULT_MARKER_SIZE)
        self.size_bot         = tk.DoubleVar(value=DEFAULT_MARKER_SIZE)
        self.fixed_side       = tk.StringVar(value="Left")
        self.rot_threshold    = tk.DoubleVar(value=5.0)
        self.pitch_threshold  = tk.DoubleVar(value=6.0)
        self.yaw_threshold    = tk.DoubleVar(value=4.0)
        self.use_v8_only      = tk.BooleanVar(value=False)

        # ── Camera Vars ──
        self.cam_ae           = tk.BooleanVar(value=True)
        self.cam_exposure   = tk.IntVar(value=10000)
        self.cam_gain       = tk.DoubleVar(value=2.0)
        self.cam_awb        = tk.BooleanVar(value=True)
        self.cam_awb_mode   = tk.StringVar(value="Auto")
        self.cam_brightness = tk.DoubleVar(value=0.0)
        self.cam_contrast   = tk.DoubleVar(value=1.0)
        self.cam_saturation = tk.DoubleVar(value=1.0)
        self.cam_sharpness  = tk.DoubleVar(value=1.0)
        
        self.pc = None
        self._cam_preview_frame = None

        # ── Movement State ──
        self.mv_state         = "idle"
        self.mv_init_buf      = {"top": [], "bottom": []}
        self.mv_final_buf     = {"top": [], "bottom": []}
        self.mv_dist_init     = {"top": None, "bottom": None}
        self.mv_dist_final    = {"top": None, "bottom": None}
        self._last_sc         = 0

        def _empty():
            return {
                "A": (0,0,0), "X": (0,0,0), "TR": (0,0,0), "BR": (0,0,0),
                "B": (0,0,0), "C": (0,0,0), "dist": 0.0, "k": 0.0,
                "L_rol": 0, "L_pit": 0, "L_yaw": 0, "L_z": 0,
                "R_rol": 0, "R_pit": 0, "R_yaw": 0, "R_z": 0,
                "rot_2d": 0.0, "L_det": False, "R_det": False,
                "p1_px": None, "p2_px": None, "mean_b": 0,
            }

        self.last_data = {"top": _empty(), "bottom": _empty(), "session_count": 0}
        self.is_running = True
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
        
        wf = tk.Frame(left, bg=C_PANEL, bd=1, relief="solid"); wf.pack(fill="x", pady=(12, 0))
        self.warn_strip = tk.Text(wf, height=6, font=F_HEAD, fg=C_AMBER, bg=C_PANEL, padx=15, pady=12, bd=0, highlightthickness=0, wrap="word", state="disabled")
        self.warn_strip.pack(side="left", fill="both", expand=True)
        ws = tk.Scrollbar(wf, orient="vertical", command=self.warn_strip.yview, width=12); ws.pack(side="right", fill="y"); self.warn_strip.config(yscrollcommand=ws.set)

        right = tk.Frame(tab, bg=C_BG); right.pack(side="right", fill="both", expand=True, padx=12, pady=10)
        for key, title, col in [("top", "UPPER SENSOR", C_TOP), ("bottom", "LOWER SENSOR", C_BOT)]:
            card = self._card(right, title, col, fill="x", pady=6, padx=12)
            dl = tk.Label(card, text="0.000 mm", font=F_DATA, fg=C_GREEN, bg=C_PANEL); dl.pack(pady=(8, 2))
            kl = tk.Label(card, text="INTERSECT RATIO: 0.0000", font=F_SMALL, fg=C_TEXT_MED, bg=C_PANEL); kl.pack(pady=(0, 8))
            if key == "top": self.lbl_dist_top, self.lbl_k_top = dl, kl
            else: self.lbl_dist_bot, self.lbl_k_bot = dl, kl

        tk.Label(right, text="READY FOR INITIAL CAPTURE", font=F_HEAD, fg=C_TEXT_MED, bg=C_BG).pack(pady=(25, 5))
        self.mv_prog_bar = ttk.Progressbar(right, length=450, maximum=COLLECT_N, mode="determinate"); self.mv_prog_bar.pack(pady=10)

        # Movement Buttons Row
        bf = tk.Frame(right, bg=C_BG); bf.pack(pady=10, fill="x")
        self.btn_start = tk.Button(bf, text="▶ START", bg=C_GREEN, fg=C_BG, font=F_BTN, width=12, pady=8, relief="flat", command=self._mv_start); self.btn_start.pack(side="left", padx=(15, 2))
        self.btn_stop = tk.Button(bf, text="■ STOP", bg=C_RED, fg=C_BG, font=F_BTN, width=12, pady=8, relief="flat", state="disabled", command=self._mv_stop); self.btn_stop.pack(side="left", padx=2)
        self.btn_reset = tk.Button(bf, text="↺ RESET", bg=C_CARD, fg="white", font=F_BTN, width=10, pady=8, relief="flat", command=self._mv_reset); self.btn_reset.pack(side="left", padx=2)

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
        tab = ttk.Frame(self.tabs); self.tabs.add(tab, text=" 🛸  Dual Telemetry ")
        mtf = tk.Frame(tab, bg=C_BG); mtf.pack(fill="both", expand=True)
        self.tele_vars = {"top": {}, "bottom": {}}
        v_show = ["A", "X", "TR", "BR", "B", "C", "L_ROL", "L_PIT", "L_YAW", "L_Z", "R_ROL", "R_PIT", "R_YAW", "R_Z"]
        for key, title, col in [("top", "TOP SENSOR", C_TOP), ("bottom", "BOTTOM SENSOR", C_BOT)]:
            cf = tk.LabelFrame(mtf, text=f" {title} TELEMETRY ", font=F_HEAD, fg=col, bg=C_BG, padx=20, pady=20); cf.pack(side="left", fill="both", expand=True, padx=25, pady=20)
            for v_name in v_show:
                row = tk.Frame(cf, bg=C_PANEL, highlightbackground=C_CARD, highlightthickness=1); row.pack(fill="x", pady=5, padx=10)
                tk.Label(row, text=v_name, font=F_BODY, bg=C_PANEL, fg=C_TEXT_MED).pack(side="left", padx=15, pady=10)
                sv = tk.StringVar(value="—"); tk.Label(row, textvariable=sv, font=F_MONO, bg=C_PANEL, fg=C_TEXT_BRT).pack(side="right", padx=15)
                self.tele_vars[key][v_name] = sv

    def _build_tab_settings(self):
        tab = ttk.Frame(self.tabs); self.tabs.add(tab, text=" ⚙  Machine Configuration ")
        sc = tk.Frame(tab, bg=C_BG); sc.pack(fill="both", expand=True, padx=60, pady=20)
        rf = tk.LabelFrame(sc, text=" Reference Side (Fixed ArUco) ", bg=C_PANEL, fg=C_TEXT_BRT, font=F_HEAD, padx=20, pady=10); rf.pack(fill="x", pady=10)
        self.rb_l = tk.Radiobutton(rf, text=" LEFT (FIXED) ", variable=self.fixed_side, value="Left", indicatoron=False, font=F_HEAD, width=20, pady=10, relief="flat", command=self._update_side_ui); self.rb_l.pack(side="left", padx=(40, 10))
        self.rb_r = tk.Radiobutton(rf, text=" RIGHT (FIXED) ", variable=self.fixed_side, value="Right", indicatoron=False, font=F_HEAD, width=20, pady=10, relief="flat", command=self._update_side_ui); self.rb_r.pack(side="left", padx=10)
        self._update_side_ui()
        def sld(l, v, c, lo, hi):
            f = tk.LabelFrame(sc, text=f" {l} ", bg=C_PANEL, fg=c, font=F_HEAD, padx=15, pady=5); f.pack(fill="x", pady=5)
            tk.Scale(f, from_=lo, to=hi, resolution=0.1, orient="horizontal", variable=v, bg=C_PANEL, troughcolor=C_BG, length=1000, font=F_SMALL, highlightthickness=0).pack(side="left", padx=20)
            SmartEntry(f, v, width=12, font=F_MONO, bg=C_BG, fg=C_ACCENT).pack(side="left", padx=10)
        sld("Upper Pair Marker Size (mm)", self.size_top, C_TOP, 10, 200)
        sld("Bottom Pair Marker Size (mm)", self.size_bot, C_BOT, 10, 200)
        sld("Rotation Threshold ° (in-plane / formula switch)", self.rot_threshold, C_TEXT_BRT, 0, 45)
        sld("Pitch Threshold ° (forward / backward tilt warning)", self.pitch_threshold, C_BOT, 0, 45)
        sld("Yaw Threshold ° (left / right tilt — 'one side in' warning)", self.yaw_threshold, C_GREEN, 0, 45)
        lf = tk.LabelFrame(sc, text=" Measurement Logic Options ", bg=C_PANEL, fg=C_TEXT_BRT, font=F_HEAD, padx=20, pady=10); lf.pack(fill="x", pady=10)
        tf = tk.Frame(lf, bg=C_PANEL); tf.pack(fill="x", expand=True, pady=10)
        self.logic_canvas = tk.Canvas(tf, width=60, height=30, bg=C_PANEL, highlightthickness=0); self.logic_canvas.pack(side="left", padx=20)
        self.logic_lbl = tk.Label(tf, text="Measurement Logic Status", font=F_BODY, bg=C_PANEL); self.logic_lbl.pack(side="left")
        self.logic_canvas.bind("<Button-1>", lambda e: self._toggle_logic())
        self._toggle_logic(init=True)

    def _toggle_logic(self, init=False):
        if not init: self.use_v8_only.set(not self.use_v8_only.get())
        active = self.use_v8_only.get()
        self.logic_lbl.config(text="ON — Using Pure Perpendicular Logic Only" if active else "OFF — Always use Perpendicular Formula (v8 High-Precision)", fg=C_ACCENT if active else C_RED)
        self.logic_canvas.delete("all")
        bg_col = C_ACCENT if active else "#64748b"
        self.logic_canvas.create_oval(2, 2, 58, 28, fill=bg_col, outline="")
        if active: self.logic_canvas.create_oval(32, 4, 56, 26, fill="white", outline="")
        else: self.logic_canvas.create_oval(4, 4, 28, 26, fill="white", outline="")

    def _build_tab_cam(self):
        tab = ttk.Frame(self.tabs); self.tabs.add(tab, text=" 📷  Camera Controls ")
        cl, cr = tk.Frame(tab, bg=C_BG), tk.Frame(tab, bg=C_BG); cl.pack(side="left", fill="both", expand=True); cr.pack(side="right", fill="y", padx=20, pady=20)
        tk.Label(cl, text="Live Camera Preview", font=F_TITLE, fg=C_TEXT_BRT, bg=C_BG).pack(pady=10)
        self.cam_canvas = tk.Canvas(cl, width=800, height=480, bg="black", highlightthickness=2, highlightbackground=C_PANEL); self.cam_canvas.pack()
        self.lbl_det_status = tk.Label(cl, text="STATUS: Searching...", font=F_TITLE, fg=C_AMBER, bg=C_BG); self.lbl_det_status.pack(pady=10)
        self.lbl_light_status = tk.Label(cl, text="LIGHTING PROFILE: NOMINAL", font=F_HEAD, fg=C_GREEN, bg=C_BG); self.lbl_light_status.pack()
        tk.Label(cr, text="Sensor Configuration", font=F_TITLE, fg=C_TEXT_BRT, bg=C_BG).pack(pady=(0, 10))
        def crw(l, v, lo, hi, r, unit=""):
            f = tk.LabelFrame(cr, text=f" {l} ", bg=C_PANEL, fg=C_TEXT_BRT, font=F_SMALL, padx=10, pady=5); f.pack(fill="x", pady=2)
            tk.Scale(f, from_=lo, to=hi, resolution=r, orient="horizontal", variable=v, bg=C_PANEL, troughcolor=C_BG, showvalue=False, length=450, highlightthickness=0, font=F_SMALL, command=lambda _: self._apply_cam()).pack(side="left", padx=5)
            SmartEntry(f, v, width=8, font=F_MONO, bg=C_BG, fg=C_ACCENT).pack(side="left")
            if unit: tk.Label(f, text=unit, font=F_SMALL, bg=C_PANEL, fg=C_TEXT_MED).pack(side="left")
        ef = tk.LabelFrame(cr, text=" Exposure Control ", bg=C_PANEL, fg=C_TEXT_BRT, font=F_SMALL); ef.pack(fill="x", pady=2)
        tk.Checkbutton(ef, text="Enable Software Auto-Exposure", variable=self.cam_ae, bg=C_PANEL, fg=C_TEXT_MED, selectcolor=C_BG, command=self._apply_cam).pack(padx=10, pady=5)
        crw("Exposure Time", self.cam_exposure, 100, 66000, 100, "us"); crw("ISO / Gain", self.cam_gain, 1.0, 16.0, 0.1, "x")
        af = tk.LabelFrame(cr, text=" Color Profile / AWB ", bg=C_PANEL, fg=C_TEXT_BRT, font=F_SMALL); af.pack(fill="x", pady=2)
        tk.Checkbutton(af, text="Hardware Auto White Balance", variable=self.cam_awb, bg=C_PANEL, fg=C_TEXT_MED, selectcolor=C_BG, command=self._apply_cam).pack(padx=10, pady=5)
        mf = tk.Frame(af, bg=C_PANEL); mf.pack(fill="x", pady=2)
        tk.Label(mf, text="Mode:", font=F_SMALL, bg=C_PANEL, fg=C_TEXT_MED).pack(side="left", padx=5)
        self.AWB_MODES = {"Auto": 0, "Tungsten": 1, "Fluorescent": 2, "Indoor": 3, "Daylight": 4, "Cloudy": 5}
        for m in self.AWB_MODES: tk.Radiobutton(mf, text=m, variable=self.cam_awb_mode, value=m, bg=C_PANEL, fg=C_TEXT_MED, font=("Inter", 8), command=self._apply_cam).pack(side="left")
        crw("Brightness", self.cam_brightness, -1.0, 1.0, 0.05); crw("Contrast", self.cam_contrast, 0.0, 8.0, 0.1); crw("Saturation", self.cam_saturation, 0.0, 8.0, 0.1); crw("Sharpness", self.cam_sharpness, 0.0, 8.0, 0.1)
        tk.Button(cr, text="↺ Restore Factory Defaults", bg=C_RED, fg=C_BG, font=F_BTN, command=self._reset_cam).pack(fill="x", pady=10)

    def load_calib(self):
        f = "camera_params_2.npz" if self.fixed_side.get() == "Right" else "camera_params.npz"
        if os.path.exists(f): 
            d = np.load(f); k_m = 'camera_matrix' if 'camera_matrix' in d else 'mtx'; k_d = 'dist_coeff' if 'dist_coeff' in d else 'dist'; return d[k_m], d[k_d]
        return np.array([[1280,0,640],[0,1280,360],[0,0,1]], dtype=np.float32), np.zeros(5)

    def _mv_start(self): self.mv_state = "collecting_init"; self.mv_init_buf = {"top": [], "bottom": []}; self.btn_start.config(state="disabled"); self.btn_stop.config(state="disabled")
    def _mv_stop(self): self.mv_state = "collecting_final"; self.mv_final_buf = {"top": [], "bottom": []}; self.btn_stop.config(state="disabled")
    def _mv_reset(self):
        self.mv_state = "idle"
        self.mv_init_buf = {"top": [], "bottom": []}
        self.mv_final_buf = {"top": [], "bottom": []}
        self.mv_dist_init = {"top": None, "bottom": None}
        self.mv_dist_final = {"top": None, "bottom": None}
        
        # Reset GUI Labels
        for lbl_init, lbl_final, lbl_delta in [
            (self.mv_init_lbl_top, self.mv_final_lbl_top, self.mv_delta_lbl_top),
            (self.mv_init_lbl_bot, self.mv_final_lbl_bot, self.mv_delta_lbl_bot)
        ]:
            lbl_init.config(text="INIT: —")
            lbl_final.config(text="FINAL: —")
            lbl_delta.config(text="—")
        
        self.mv_prog_bar["value"] = 0
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")

    def _mv_tick(self):
        sc = self.last_data["session_count"]
        if sc == self._last_sc: return
        self._last_sc = sc
        if self.mv_state == "collecting_init":
            for k in ["top", "bottom"]:
                d = self.last_data[k]["dist"]
                if d > 0: self.mv_init_buf[k].append(d)
            if min(len(self.mv_init_buf["top"]), len(self.mv_init_buf["bottom"])) >= COLLECT_N:
                self.mv_dist_init = {k: float(np.mean(self.mv_init_buf[k][:COLLECT_N])) for k in ["top", "bottom"]}
                self.mv_state = "ready"; self.btn_stop.config(state="normal")
                self.mv_init_lbl_top.config(text=f"INIT: {self.mv_dist_init['top']:.3f} mm"); self.mv_init_lbl_bot.config(text=f"INIT: {self.mv_dist_init['bottom']:.3f} mm")
        elif self.mv_state == "collecting_final":
            for k in ["top", "bottom"]:
                d = self.last_data[k]["dist"]
                if d > 0: self.mv_final_buf[k].append(d)
            if min(len(self.mv_final_buf["top"]), len(self.mv_final_buf["bottom"])) >= COLLECT_N:
                self.mv_dist_final = {k: float(np.mean(self.mv_final_buf[k][:COLLECT_N])) for k in ["top", "bottom"]}
                self.mv_state = "done"; self.btn_start.config(state="normal")
                for k, il, fl, dl in [("top", self.mv_init_lbl_top, self.mv_final_lbl_top, self.mv_delta_lbl_top), ("bottom", self.mv_init_lbl_bot, self.mv_final_lbl_bot, self.mv_delta_lbl_bot)]:
                    di, df = self.mv_dist_init[k], self.mv_dist_final[k]; il.config(text=f"INIT: {di:.3f} mm"); fl.config(text=f"FINAL: {df:.3f} mm"); dl.config(text=f"{df-di:+.3f} mm")

    def _apply_cam(self, *_):
        if self.pc is None: return
        ctrls = {}
        ae_on = self.cam_ae.get()
        ctrls["AeEnable"] = ae_on
        if not ae_on:
            ctrls["ExposureTime"] = int(self.cam_exposure.get())
            ctrls["AnalogueGain"] = float(self.cam_gain.get())
        awb_on = self.cam_awb.get()
        ctrls["AwbEnable"] = awb_on
        if not awb_on:
            ctrls["AwbMode"] = self.AWB_MODES.get(self.cam_awb_mode.get(), 0)
        ctrls["Brightness"] = float(self.cam_brightness.get())
        ctrls["Contrast"] = float(self.cam_contrast.get())
        ctrls["Saturation"] = float(self.cam_saturation.get())
        ctrls["Sharpness"] = float(self.cam_sharpness.get())
        try:
            self.pc.set_controls(ctrls)
        except Exception as e:
            print(f"[CAM ERROR] {e}")

    def _reset_cam(self): self.cam_ae.set(True); self.cam_exposure.set(10000); self.cam_gain.set(2.0); self.cam_awb.set(True); self.cam_awb_mode.set("Auto"); self.cam_brightness.set(0.0); self.cam_contrast.set(1.0); self.cam_saturation.set(1.0); self.cam_sharpness.set(1.0); self._apply_cam()

    def _update_side_ui(self):
        if self.fixed_side.get() == "Left":
            self.rb_l.config(bg=C_GREEN, fg=C_BG, activebackground="#86efac")
            self.rb_r.config(bg=C_PANEL, fg="white", activebackground=C_BG)
        else:
            self.rb_r.config(bg=C_GREEN, fg=C_BG, activebackground="#86efac")
            self.rb_l.config(bg=C_PANEL, fg="white", activebackground=C_BG)

    def update_gui_loop(self):
        if self.current_frame is not None:
            img = ImageTk.PhotoImage(image=Image.fromarray(self.current_frame).resize((960, 540))); self.canvas.create_image(0, 0, anchor="nw", image=img); self.canvas.img = img
        self.lbl_dist_top.config(text=f"{self.last_data['top']['dist']:.3f} mm"); self.lbl_k_top.config(text=f"INTERSECT RATIO: {self.last_data['top']['k']:.4f}")
        self.lbl_dist_bot.config(text=f"{self.last_data['bottom']['dist']:.3f} mm"); self.lbl_k_bot.config(text=f"INTERSECT RATIO: {self.last_data['bottom']['k']:.4f}")
        strip = []; r_th, p_th, y_th = self.rot_threshold.get(), self.pitch_threshold.get(), self.yaw_threshold.get()
        for k in ["top", "bottom"]:
            d = self.last_data[k]
            if not d["L_det"]: strip.append(f"• [{k.upper()}] Left Marker Missing")
            if not d["R_det"]: strip.append(f"• [{k.upper()}] Right Marker Missing")
            if d["L_det"]:
                if abs(d["L_pit"]) > p_th: strip.append(f"• [{k.upper()}] L-pitch {d['L_pit']:+.1f}° ({'FWD' if d['L_pit']>0 else 'BWD'})")
                if abs(d["L_yaw"]) > y_th: strip.append(f"• [{k.upper()}] L-yaw {d['L_yaw']:+.1f}° ({'RIGHT' if d['L_yaw']>0 else 'LEFT'})")
            if d["R_det"]:
                if abs(d["R_pit"]) > p_th: strip.append(f"• [{k.upper()}] R-pitch {d['R_pit']:+.1f}° ({'FWD' if d['R_pit']>0 else 'BWD'})")
                if abs(d["R_yaw"]) > y_th: strip.append(f"• [{k.upper()}] R-yaw {d['R_yaw']:+.1f}° ({'RIGHT' if d['R_yaw']>0 else 'LEFT'})")
        self.warn_strip.config(state="normal"); self.warn_strip.delete("1.0", tk.END)
        if strip: self.warn_strip.insert(tk.END, "![!] ACTIVE ALERTS:\n" + "\n".join(strip))
        else: self.warn_strip.insert(tk.END, "[OK] SYSTEM NOMINAL")
        self.warn_strip.config(state="disabled")
        if self._cam_preview_frame is not None:
            cimg = ImageTk.PhotoImage(image=Image.fromarray(self._cam_preview_frame).resize((800, 480))); self.cam_canvas.create_image(0, 0, anchor="nw", image=cimg); self.cam_canvas.img = cimg
            any_det = self.last_data["top"]["L_det"] or self.last_data["bottom"]["L_det"]
            self.lbl_det_status.config(text="STATUS: Markers detected" if any_det else "STATUS: Searching...", fg=C_GREEN if any_det else C_AMBER)
            mb = self.last_data["top"]["mean_b"]
            if mb < 40: self.lbl_light_status.config(text="LIGHTING PROFILE: TOO DARK", fg=C_RED)
            elif mb > 220: self.lbl_light_status.config(text="LIGHTING PROFILE: TOO BRIGHT", fg=C_RED)
            else: self.lbl_light_status.config(text="LIGHTING PROFILE: NOMINAL", fg=C_GREEN)
        for k in ["top", "bottom"]:
            d = self.last_data[k]; v = self.tele_vars[k]
            for kn in ["A", "X", "TR", "BR", "B", "C"]: v[kn].set(f"{d[kn][0]:.1f}, {d[kn][1]:.1f}, {d[kn][2]:.1f}")
            v["L_ROL"].set(f"{d['L_rol']:.1f}°"); v["L_PIT"].set(f"{d['L_pit']:+.1f}°"); v["L_YAW"].set(f"{d['L_yaw']:+.1f}°"); v["L_Z"].set(f"{d['L_z']:.1f}")
            v["R_ROL"].set(f"{d['R_rol']:.1f}°"); v["R_PIT"].set(f"{d['R_pit']:+.1f}°"); v["R_YAW"].set(f"{d['R_yaw']:+.1f}°"); v["R_Z"].set(f"{d['R_z']:.1f}")
        self._mv_tick(); self.root.after(33, self.update_gui_loop)

if __name__ == "__main__":
    root = tk.Tk(); app = MeasurementApp(root); root.mainloop()
