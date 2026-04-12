import tkinter as tk
from tkinter import ttk
import cv2 as cv
import numpy as np
import threading, math, os, time
from PIL import Image, ImageTk
from datetime import datetime
import log

# ── Constants & Aesthetic ──────────────────────────────────────────────────
SZ, DICT, RES, N = 53.8, cv.aruco.DICT_4X4_50, (1280, 720), 5
C = {'bg':"#0f172a", 'p':"#1e293b", 'c':"#334155", 'a':"#38bdf8", 't':"#fb923c", 
     'b':"#c084fc", 'g':"#4ade80", 'r':"#f87171", 'tx':"#f8fafc", 'm':"#94a3b8", 'w':"#fbbf24"}
F = {'t':("Inter",20,"bold"), 'h':("Inter",18,"bold"), 'b':("Inter",16), 
     's':("Inter",14), 'd':("Inter",32,"bold"), 'm':("Consolas",18), 'bt':("Inter",14,"bold")}

def rot_to_euler(R):
    """
    Decompose an OpenCV 3×3 rotation matrix into (pitch, yaw, roll) degrees.
    """
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        return math.degrees(math.atan2(R[2,1], R[2,2])), math.degrees(math.atan2(-R[2,0], sy)), math.degrees(math.atan2(R[1,0], R[0,0]))
    return math.degrees(math.atan2(-R[1,2], R[1,1])), math.degrees(math.atan2(-R[2,0], sy)), 0.0

class MeasurementApp:
    """
    Dual-Pair ArUco Measurement v17 (Compact)
    """
    def __init__(self, root):
        """
        Initialize the application.
        """
        self.root = root # Root window
        root.title("Dual-Pair ArUco v17"); root.geometry("1600x960"); root.configure(bg=C['bg'])
        
        # Vars
        V = lambda v, t=tk.DoubleVar: t(value=v)
        self.size_top, self.size_bot = V(SZ), V(SZ)
        self.fixed_side, self.rot_threshold = tk.StringVar(value="Left"), V(12.0)
        self.p_th, self.y_th, self.u_ang = V(10.0), V(10.0), tk.BooleanVar(value=False)
        self.cam_ae, self.cam_awb = tk.BooleanVar(value=True), tk.BooleanVar(value=True)
        self.cam_exp, self.cam_gain = tk.IntVar(value=10000), V(2.0)
        self.cam_awbm, self.cam_br, self.cam_ct, self.cam_sa, self.cam_sh = tk.StringVar(value="Auto"), V(0.0), V(1.0), V(1.0), V(1.0)
        
        # State
        self.mv_state, self._last_sc, self.pc = "idle", 0, None
        self.bufs = {"init":{"top":[],"bottom":[]}, "final":{"top":[],"bottom":[]}}
        self.mv_d = {"init":{"top":None,"bottom":None}, "final":{"top":None,"bottom":None}}
        _e = lambda: {k: (0,0,0) if k in "AXTRBRBC" else 0.0 for k in ["A","X","TR","BR","B","C","dist","k","L_z","R_z","L_pitch","L_yaw","R_pitch","R_yaw"]}
        self.last_data = {"top":_e(), "bottom":_e(), "sc":0, "light":{"s":"no_frame","m":0,"v":0}}
        for k in ["top","bottom"]: self.last_data[k].update({"L_A":(0,0),"R_A":(0,0),"p1":None,"p2":None,"L_d":0,"R_d":0})

        self._style(); self.setup_ui()
        threading.Thread(target=self.m_loop, daemon=True).start()
        self.gui_loop()

    def _style(self):
        s = ttk.Style(); s.theme_use('clam')
        s.configure("TNotebook", background=C['bg'], borderwidth=0)
        s.configure("TNotebook.Tab", background=C['p'], foreground=C['m'], padding=[25,12], font=F['h'])
        s.map("TNotebook.Tab", background=[("selected",C['a'])], foreground=[("selected",C['bg'])])
        s.configure("TProgressbar", thickness=24, background=C['a'], troughcolor=C['p'])

    def _card(self, p, t, col, **kw):
        f = tk.LabelFrame(p, text=f" {t.upper()} ", font=F['h'], fg=col, bg=C['p'], bd=2, relief="flat", highlightbackground=col, highlightthickness=1, padx=20, pady=20)
        f.pack(**kw); return f

    def setup_ui(self):
        self.tabs = ttk.Notebook(self.root); self.tabs.pack(fill='both', expand=True, padx=8, pady=8)
        self._build_live(); self._build_tele(); self._build_set(); self._build_cam()

    def _build_live(self):
        """
        Build the live tab.
        """
        t = ttk.Frame(self.tabs); self.tabs.add(t, text=" 📏 Movement "); t.configure(style="TFrame")
        l = tk.Frame(t, bg=C['bg']); l.pack(side="left", padx=16, pady=16)
        self.can = tk.Canvas(l, width=960, height=540, bg="black", highlightthickness=1, highlightbackground=C['p']); self.can.pack()
        wf = tk.Frame(l, bg=C['p'], bd=1, relief="solid"); wf.pack(fill="x", pady=(12,0))
        self.wst = tk.Text(wf, height=6, font=F['h'], fg=C['a'], bg=C['p'], padx=15, pady=12, bd=0, wrap="word", state="disabled"); self.wst.pack(side="left", fill="both", expand=True)
        r = tk.Frame(t, bg=C['bg']); r.pack(side="right", fill="both", expand=True, padx=12, pady=10)
        
        self.l_dist, self.l_k = {}, {}
        for k, n, col in [("top","UPPER SENSOR",C['t']), ("bottom","LOWER SENSOR",C['b'])]:
            cd = self._card(r, n, col, fill="x", pady=10, padx=12)
            self.l_dist[k] = tk.Label(cd, text="0.000 mm", font=F['d'], fg=C['g'], bg=C['p']); self.l_dist[k].pack(pady=(12,4))
            self.l_k[k] = tk.Label(cd, text="RATIO: 0.0000", font=F['s'], fg=C['m'], bg=C['p']); self.l_k[k].pack(pady=(0,12))

        self.mv_st_l = tk.Label(r, text="READY", font=F['h'], fg=C['m'], bg=C['bg']); self.mv_st_l.pack(pady=10)
        self.p_bar = ttk.Progressbar(r, length=380, maximum=N); self.p_bar.pack(pady=8)
        
        bf = tk.Frame(r, bg=C['bg']); bf.pack(pady=20, fill="x", padx=12)
        h = lambda b, n, h: (b.bind("<Enter>", lambda e: b.config(bg=h)), b.bind("<Leave>", lambda e: b.config(bg=n)))
        self.b_st = tk.Button(bf, text="▶ START SESSION", bg=C['g'], fg=C['bg'], font=F['bt'], relief="flat", padx=10, pady=10, command=self._mv_st); self.b_st.pack(side="left", expand=True, fill="x", padx=6); h(self.b_st, C['g'], "#86efac")
        self.b_sp = tk.Button(bf, text="■ STOP (SAVE)", bg=C['r'], fg=C['bg'], font=F['bt'], relief="flat", state="disabled", padx=10, pady=10, command=self._mv_sp); self.b_sp.pack(side="left", expand=True, fill="x", padx=6); h(self.b_sp, C['r'], "#fca5a5")
        tk.Button(bf, text="↺", bg=C['c'], fg="white", font=F['bt'], relief="flat", width=5, pady=10, command=self._mv_rs).pack(side="left", padx=6)

        self.m_res = {}
        for k, n, col in [("top","UPPER",C['t']), ("bottom","LOWER",C['b'])]:
            row = tk.Frame(r, bg=C['p'], bd=1, relief="solid"); row.pack(fill="x", padx=12, pady=8)
            tk.Label(row, text=n, font=F['h'], fg=col, bg=C['p'], width=8).pack(side="left", padx=12)
            inf = tk.Frame(row, bg=C['p']); inf.pack(side="left", expand=True)
            i, f = tk.Label(inf, text="INIT: —", font=F['b'], fg=C['m'], bg=C['p'], anchor="w"), tk.Label(inf, text="FINAL: —", font=F['b'], fg=C['m'], bg=C['p'], anchor="w")
            i.pack(fill="x"); f.pack(fill="x")
            d = tk.Label(row, text="—", font=F['d'], fg=C['a'], bg=C['p']); d.pack(side="right", padx=20)
            self.m_res[k] = (i, f, d)

        df = tk.LabelFrame(r, text=" DIAGNOSTICS ", font=F['h'], fg=C['r'], bg=C['p'], bd=0, padx=15, pady=15); df.pack(fill="x", padx=12, pady=20)
        self.dots = {}
        for i, (k, n, col) in enumerate([("top","UPPER",C['t']), ("bottom","LOWER",C['b'])]):
            tk.Label(df, text=n, font=F['h'], fg=col, bg=C['p'], width=8).grid(row=i+1, column=0, pady=5)
            self.dots[k] = {f: tk.Label(df, text="■", font=("Inter",20), fg=C['c'], bg=C['p'], width=4) for f in ["L_d","R_d","rot","p","y"]}
            for j, f in enumerate(["L_d","R_d","rot","p","y"]): self.dots[k][f].grid(row=i+1, column=j+1)
        
        lf = tk.Frame(df, bg=C['p']); lf.grid(row=3, column=0, columnspan=6, sticky="w", pady=15)
        self.lc = {k: tk.Label(lf, text="—", font=F['h'], fg=C['tx'], bg=C['c']) for k in ["b","c"]}
        tk.Label(lf, text="LIGHT:", font=F['h'], fg=C['tx'], bg=C['p']).pack(side="left")
        for k in ["b","c"]: f=tk.Frame(lf, bg=C['c'], padx=8); f.pack(side="left", padx=5); tk.Label(f, text=k.upper(), font=F['s'], bg=C['c'], fg=C['m']).pack(side="left"); self.lc[k].pack(side="left"); self.lc[k].master=f
        self.l_st = tk.Label(lf, text="SCAN", font=F['h'], fg=C['a'], bg=C['p']); self.l_st.pack(side="right", padx=20)

    def _build_tele(self):
        t = ttk.Frame(self.tabs); self.tabs.add(t, text=" 🛸 Telemetry ")
        self.tv = {"top":{}, "bottom":{}}
        for k, n, col in [("top","TOP",C['t']), ("bottom","BOTTOM",C['b'])]:
            f = tk.LabelFrame(t, text=f" {n} ", font=F['t'], fg=col, bg=C['bg'], padx=15, pady=15); f.pack(side="left", fill="both", expand=True, padx=20, pady=20)
            for v in ["A","X","TR","BR","B","C","L_ROL","L_ROT","L_Z","L_PITCH","L_YAW","R_ROL","R_ROT","R_Z","R_PITCH","R_YAW"]:
                r = tk.Frame(f, bg=C['p'], highlightbackground=C['c'], highlightthickness=1); r.pack(fill="x", pady=4)
                tk.Label(r, text=v, font=F['b'], fg=C['m'], bg=C['p']).pack(side="left", padx=10)
                sv = tk.StringVar(value="—"); tk.Label(r, textvariable=sv, font=F['m'], fg=C['tx'], bg=C['p']).pack(side="right", padx=10); self.tv[k][v] = sv

    def _build_set(self):
        t = ttk.Frame(self.tabs); self.tabs.add(t, text=" ⚙ Machine "); sc = tk.Frame(t, bg=C['bg']); sc.pack(fill="both", expand=True, padx=80, pady=40)
        rf = tk.LabelFrame(sc, text=" Ref Side ", bg=C['p'], fg=C['tx'], font=F['h'], padx=20, pady=10); rf.pack(fill="x", pady=10)
        for c in ["Left","Right"]: tk.Radiobutton(rf, text=c, variable=self.fixed_side, value=c, bg=C['p'], fg=C['tx'], selectcolor=C['bg'], font=F['b']).pack(side="left", padx=20)
        def sld(l, v, col, lo, hi):
            f = tk.LabelFrame(sc, text=f" {l} ", bg=C['p'], font=F['h'], fg=col, padx=20, pady=5); f.pack(fill="x", pady=5)
            s = tk.Scale(f, from_=lo, to=hi, resolution=0.1, orient="horizontal", variable=v, bg=C['p'], fg=C['tx'], troughcolor=C['bg'], length=800, highlightthickness=0, font=F['s']); s.pack(side="left", padx=20)
            e = tk.Entry(f, width=8, bg=C['bg'], fg=C['m'], bd=0, font=F['m'])
            e.insert(0, str(v.get())); e.pack(side="left", padx=10)
            def _sync(*_):
                if self.root.focus_get() != e:
                    e.delete(0, tk.END); e.insert(0, f"{v.get():.1f}"); e.config(fg=C['m'])
            v.trace_add("write", _sync)
            def _cmt(*_):
                try: 
                    v.set(float(e.get()))
                    self.root.update_idletasks() # Force sync
                except: _sync()
                self.root.focus()
            e.bind("<FocusIn>", lambda _: (e.config(fg=C['tx']), e.delete(0, tk.END)))
            e.bind("<Return>", _cmt); e.bind("<FocusOut>", _cmt); return s
        sld("Upper Pair Marker Size (mm)", self.size_top, C['t'], 10, 200)
        sld("Bottom Pair Marker Size (mm)", self.size_bot, C['b'], 10, 200)
        self.rs = sld("Rotation Threshold ° (in-plane / formula switch)", self.rot_threshold, C['tx'], 0, 45)
        sld("Pitch Threshold ° (forward / backward tilt warning)", self.p_th, C['b'], 0, 45)
        sld("Yaw Threshold ° (left / right tilt — 'one side in' warning)", self.y_th, C['g'], 0, 45)
        tf = tk.LabelFrame(sc, text=" Measurement Logic Options ", bg=C['p'], fg=C['tx'], font=F['h'], padx=20, pady=10); tf.pack(fill="x", pady=10)
        self.tc = tk.Canvas(tf, width=60, height=30, bg=C['p'], highlightthickness=0); self.tc.pack(side="left", padx=20)
        self.tl = tk.Label(tf, text="", font=F['b'], bg=C['p']); self.tl.pack(side="left")
        def _ut(*_):
            on = self.u_ang.get()
            self.tl.config(text="ON — Using Pure Perpendicular Logic Only" if on else "OFF — Always use Perpendicular Formula (v8 High-Precision)", fg=C['a'] if on else C['r'])
            self.tc.delete("all")
            col = C['a'] if on else "#64748b"
            self.tc.create_oval(2, 2, 58, 28, fill=col, outline="")
            if on: self.tc.create_oval(32, 4, 56, 26, fill="white", outline="")
            else: self.tc.create_oval(4, 4, 28, 26, fill="white", outline="")
        self.tc.bind("<Button-1>", lambda e: (self.u_ang.set(not self.u_ang.get()), _ut())); _ut()

    def _build_cam(self):
        t = ttk.Frame(self.tabs); self.tabs.add(t, text=" 📷 Camera "); cl, cr = tk.Frame(t, bg=C['bg']), tk.Frame(t, bg=C['bg'])
        cl.pack(side="left", fill="both", expand=True); cr.pack(side="right", fill="y", padx=10)
        self.c_can = tk.Canvas(cl, width=800, height=480, bg="black"); self.c_can.pack(pady=20)
        self.c_st = tk.Label(cl, text="WAIT", font=F['t'], fg=C['w'], bg=C['bg']); self.c_st.pack()
        self.l_cam = tk.Label(cl, text="", font=F['h'], fg=C['m'], bg=C['bg']); self.l_cam.pack()
        def crow(l, v, lo, hi, r, u=""):
            f = tk.LabelFrame(cr, text=l, bg=C['p'], font=F['h'], fg=C['tx'], padx=10, pady=5); f.pack(fill="x", pady=2)
            e = tk.Entry(f, width=6, bg=C['bg'], fg=C['m'], bd=0, font=F['m'])
            e.insert(0, str(v.get())); e.pack(side="right", padx=5)
            def _up(_=None):
                if l=="EXP": self.cam_ae.set(0)
                self._app()
            def _sync(*_):
                if self.root.focus_get() != e:
                    e.delete(0, tk.END); e.insert(0, str(v.get())); e.config(fg=C['m'])
            v.trace_add("write", _sync)
            def _cmt(*_):
                try: v.set(type(v.get())(e.get())); _up()
                except: _sync()
                self.root.focus()
            tk.Scale(f, from_=lo, to=hi, resolution=r, orient="horizontal", variable=v, bg=C['p'], fg=C['tx'], troughcolor=C['bg'], length=300, command=_up).pack(side="left")
            def _on_foc(_):
                e.config(fg=C['tx']); e.delete(0, tk.END)
            e.bind("<FocusIn>", _on_foc)
            e.bind("<Return>", _cmt); e.bind("<FocusOut>", _cmt)
        tk.Checkbutton(cr, text="AE", variable=self.cam_ae, bg=C['p'], fg=C['tx'], selectcolor=C['bg'], command=self._app).pack(fill="x")
        crow("EXP", self.cam_exp, 100, 66000, 100, "us")
        # GAIN row removed as requested
        tk.Checkbutton(cr, text="AWB", variable=self.cam_awb, bg=C['p'], fg=C['tx'], selectcolor=C['bg'], command=self._app).pack(fill="x")
        wf = tk.Frame(cr, bg=C['p']); wf.pack(fill="x")
        self.AWB_MAP = {"Auto": 0, "Tungsten": 1, "Fluorescent": 2, "Indoor": 3, "Daylight": 4, "Cloudy": 5}
        for m in self.AWB_MAP.keys(): tk.Radiobutton(wf, text=m, variable=self.cam_awbm, value=m, bg=C['p'], fg=C['tx'], selectcolor=C['bg'], font=F['s'], command=self._app).pack(side="left")
        for l,v,lo,hi in [("BR",self.cam_br,-1,1),("CT",self.cam_ct,0,8),("SA",self.cam_sa,0,8),("SH",self.cam_sh,0,8)]: crow(l,v,lo,hi,0.1)
        tk.Button(cr, text="RESET", bg=C['r'], fg="white", font=F['bt'], relief="flat", command=self._r_cam).pack(pady=10, fill="x")

    def _mv_st(self): self.mv_state="collecting_init"; self.bufs["init"]={"top":[],"bottom":[]}; self.b_st.config(state="disabled"); self.b_sp.config(state="disabled")
    def _mv_sp(self): self.mv_state="collecting_final"; self.bufs["final"]={"top":[],"bottom":[]}; self.b_sp.config(state="disabled")
    def _mv_rs(self):
        self.mv_state = "idle"
        self.mv_init_buf = {"top":[], "bottom":[]}
        self.mv_final_buf = {"top":[], "bottom":[]}
        self.mv_d = {"init":{"top":None,"bottom":None}, "final":{"top":None,"bottom":None}}
        
        # Reset Labels
        for k in ["top", "bottom"]:
            il, fl, dl = self.m_res[k]
            il.config(text="INIT: —")
            fl.config(text="FINAL: —")
            dl.config(text="—")
            
        self.b_st.config(state="normal")
        self.b_sp.config(state="disabled")
        self.p_bar["value"] = 0
        self.wst.config(state="normal")
        self.wst.delete("1.0", tk.END)
        self.wst.config(state="disabled")
    def _app(self):
        if not self.pc: return
        try:
            p = {
                "AeEnable": bool(self.cam_ae.get()), 
                "AwbEnable": bool(self.cam_awb.get()), 
                "Brightness": float(self.cam_br.get()), 
                "Contrast": float(self.cam_ct.get()), 
                "Saturation": float(self.cam_sa.get()), 
                "Sharpness": float(self.cam_sh.get())
            }
            if not p["AeEnable"]: 
                p["ExposureTime"] = int(self.cam_exp.get())
                p["AnalogueGain"] = float(self.cam_gain.get())
            if not p["AwbEnable"]: 
                p["AwbMode"] = self.AWB_MAP.get(self.cam_awbm.get(), 0)
            self.pc.set_controls(p)
        except Exception as e:
            print(f"[CAM ERROR] {e}")
    def _r_cam(self): self.cam_ae.set(1); self.cam_awb.set(1); self.cam_br.set(0); self.cam_ct.set(1); self.cam_sa.set(1); self.cam_sh.set(1); self._app()

    def m_loop(self):
        try:
            from picamera2 import Picamera2
            pc = Picamera2(); pc.configure(pc.create_video_configuration(main={"size":RES, "format":"RGB888"})); pc.start(); self.pc = pc
        except: return
        det = cv.aruco.ArucoDetector(cv.aruco.getPredefinedDictionary(DICT), cv.aruco.DetectorParameters()); log.init_log()
        bufs = {"top":[], "bottom":[]}; l_s = l_u = time.time()*1000
        while self.is_running:
            try: frame = cv.cvtColor(pc.capture_array(), cv.COLOR_RGB2BGR)
            except: continue
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY); mb, sb = np.mean(gray), np.std(gray)
            ls = "dark" if mb<40 else "bright" if mb>215 else "flat" if sb<18 else "ok"
            self.last_data["light"] = {"s":ls, "m":mb, "v":sb}
            corners, ids, _ = det.detectMarkers(gray); curr = time.time()*1000
            if ids is not None:
                md = []
                for i in range(len(ids)):
                    r = corners[i][0]; ix = np.argsort(r[:,0]); lp, rp = r[ix[:2]], r[ix[2:]]
                    tl, bl, tr, br = lp[np.argmin(lp[:,1])], lp[np.argmax(lp[:,1])], rp[np.argmin(rp[:,1])], rp[np.argmax(rp[:,1])]
                    md.append({"c":np.array([tl,tr,br,bl], np.float32), "r":r, "y":(tl[1]+tr[1]+br[1]+bl[1])/4, "x":(tl[0]+tr[0]+br[0]+bl[0])/4})
                md.sort(key=lambda m: m["y"]); top, bot = md[:2], md[2:4] if len(md)>=4 else []
                if len(md)==2: (top, botanical) = (md, []) if (md[0]["y"]+md[1]["y"])/2 < RES[1]/2 else ([], md)
                elif len(md)==3: top, bot = md[:2], md[2:]
                
                K, dist_c = self.load_calib()
                def proc(ml, pk, sz):
                    if len(ml)<2: 
                        for f in ["dist","p1","L_d","R_d"]: self.last_data[pk][f] = 0 if f!="p1" else None
                        return
                    ml.sort(key=lambda m: m["x"]); is_rf = self.fixed_side.get()=="Right"
                    Sm, Tm = (ml[1] if is_rf else ml[0]), (ml[0] if is_rf else ml[1])
                    obj = np.array([[-sz/2,sz/2,0],[sz/2,sz/2,0],[sz/2,-sz/2,0],[-sz/2,-sz/2,0]], np.float32)
                    def gp(cr):
                        _, rv, tv = cv.solvePnP(obj, cr, K, dist_c); R, _ = cv.Rodrigues(rv); p3 = np.array([R@pt+tv.ravel() for pt in obj])
                        pit, yaw, _ = rot_to_euler(R); return p3, math.degrees(math.atan2(R[1,0], R[0,0])), tv[2,0], pit, yaw
                    S_pts, S_r, S_z, S_p, S_y = gp(Sm["r"]); T_pts, T_r, T_z, T_p, T_y = gp(Tm["r"])
                    Si, Ti = abs(math.degrees(math.atan2(Sm["c"][1,1]-Sm["c"][0,1], Sm["c"][1,0]-Sm["c"][0,0]))), abs(math.degrees(math.atan2(Tm["c"][1,1]-Tm["c"][0,1], Tm["c"][1,0]-Tm["c"][0,0])))
                    def io(p, ir): ox = p[:,0].argsort(); i, o = (p[ox[2:]] if ir else p[ox[:2]]), (p[ox[:2]] if ir else p[ox[2:]]); return i[i[:,1].argsort()[0]], i[i[:,1].argsort()[1]], o[o[:,1].argsort()[0]]
                    Sit, Sib, Sot = io(S_pts, not is_rf); Tit, Tib, _ = io(T_pts, is_rf)
                    p1 = tuple(((Sm["c"][1 if not is_rf else 0]+Sm["c"][2 if not is_rf else 3])/2).astype(int))
                    p2 = tuple(((Tm["c"][0 if not is_rf else 1]+Tm["c"][3 if not is_rf else 2])/2).astype(int))
                    bufs[pk].append({"A":(Sit+Sib)/2, "X_alt":(Tit+Tib)/2, "TR":Sit, "BR":Sib, "TL_ref":Sot, "B":Tit, "C":Tib, "LA":(S_r,Si), "RA":(T_r,Ti), "rot":max(Si,Ti), "Lz":S_z, "Rz":T_z, "Lp":S_p, "Ly":S_y, "Rp":T_p, "Ry":T_y, "p1":p1, "p2":p2})
                    self.last_data[pk]["L_d"] = self.last_data[pk]["R_d"] = 1
                if (curr-l_s)>=100: proc(top, "top", self.size_top.get()); proc(bot, "bottom", self.size_bot.get()); l_s=curr
                if (curr-l_u)>=1000:
                    for k in ["top","bottom"]:
                        if bufs[k]:
                            s = bufs[k]; m = lambda f: np.mean([x[f] for x in s], axis=0)
                            aA, aTR, aTL, aBR, aB, aC, aXa = m("A"), m("TR"), m("TL_ref"), m("BR"), m("B"), m("C"), m("X_alt")
                            v, w, u = (aTR-aTL)/np.linalg.norm(aTR-aTL), aC-aB, aB-aA; den = np.dot(v,v)*np.dot(w,w)-np.dot(v,w)**2
                            kv = np.clip((np.dot(v,w)*np.dot(u,v)-np.dot(v,v)*np.dot(u,w))/den, 0, 1) if abs(den)>1e-6 else 0.5
                            aX = aB + kv*w; dv = np.linalg.norm(aX-aA)
                            if self.u_ang.get() and float(np.mean([x["rot"] for x in s])) > self.rot_threshold.get(): aX, dv, kv = aXa, np.linalg.norm(aXa-aA), 0.5
                            self.last_data[k].update({"A":aA,"X":aX,"TR":aTR,"BR":aBR,"B":aB,"C":aC,"dist":dv,"k":kv,"rot_2d":float(np.mean([x["rot"] for x in s])),"L_A":m("LA"),"R_A":m("RA"),"L_z":float(np.mean([x["Lz"] for x in s])),"R_z":float(np.mean([x["Rz"] for x in s])),"L_pitch":float(np.mean([x["Lp"] for x in s])),"L_yaw":float(np.mean([x["Ly"] for x in s])),"R_pitch":float(np.mean([x["Rp"] for x in s])),"R_yaw":float(np.mean([x["Ry"] for x in s])),"p1":tuple(m("p1").astype(int)),"p2":tuple(m("p2").astype(int))})
                            bufs[k].clear(); log.record(dv, kv, aA, aX, aTR, aBR, aB, aC, aA, aTR-aTL, aC-aB, m("LA"), m("RA"))
                            if self.mv_state in ["collecting_init","ready","collecting_final"]:
                                idir = "captured_images"; os.makedirs(idir, exist_ok=True); cv.imwrite(f"{idir}/mv_{datetime.now().strftime('%Y%m%d_%H%M%S%f')[:-3]}.jpg", frame)
                    l_u=curr
            else:
                for k in ["top","bottom"]: self.last_data[k].update({"dist":0, "L_d":0, "R_d":0})
            for k, col in [("top",(0,165,255)),("bottom",(255,0,255))]:
                d = self.last_data[k]
                issues = 0
                if not d["L_d"]: issues += 1
                if not d["R_d"]: issues += 1
                if d["L_d"]:
                    if abs(d["L_pitch"]) > self.p_th.get(): issues += 1
                    if abs(d["L_yaw"]) > self.y_th.get(): issues += 1
                if d["R_d"]:
                    if abs(d["R_pitch"]) > self.p_th.get(): issues += 1
                    if abs(d["R_yaw"]) > self.y_th.get(): issues += 1

                if d["dist"]>0 and d["p1"]:
                    p1, X3 = d["p1"], np.array([d["X"]], np.float64).reshape(1,1,3)
                    p2 = tuple(cv.projectPoints(X3, np.zeros(3), np.zeros(3), K, dist_c)[0][0].ravel().astype(int)) if d["X"][2]>0 else d["p2"]
                    cv.line(frame, p1, p2, col, 3); cv.circle(frame, p2, 6, (0,255,0), -1); 
                    
                    # Draw Measurement Text
                    cv.putText(frame, f"{d['dist']:.2f}mm", (p2[0]+10, p2[1]), 0, 0.6, (255, 0, 255), 2)

                    # Draw Issues Box (if any)
                    if issues > 0:
                        mid_x, mid_y = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
                        txt = f"ISSUES: {issues}"
                        (tw, th), _ = cv.getTextSize(txt, 0, 0.5, 2)
                        bx, by, bw, bh = mid_x - tw//2 - 5, mid_y - 30 - th - 5, tw + 10, th + 10
                        cv.rectangle(frame, (bx, by), (bx+bw, by+bh), (0,0,0), -1)
                        cv.rectangle(frame, (bx, by), (bx+bw, by+bh), (0,165,255), 2)
                        cv.putText(frame, txt, (bx+5, by+bh-7), 0, 0.5, (0,255,255), 2)
            self.current_frame = self._cam_preview_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    def load_calib(self):
        f = "camera_params_2.npz" if self.fixed_side.get()=="Right" else "camera_params.npz"
        if os.path.exists(f): d = np.load(f); return d['camera_matrix'], d['dist_coeff']
        return np.array([[1280,0,640],[0,1280,360],[0,0,1]], np.float32), np.zeros(5)

    def gui_loop(self):
        if self.current_frame is not None:
            im = ImageTk.PhotoImage(Image.fromarray(self.current_frame).resize((960, 540))); self.can.create_image(0,0, anchor="nw", image=im); self.can.img=im
            ci = ImageTk.PhotoImage(Image.fromarray(self.current_frame).resize((800, 480))); self.c_can.create_image(0,0, anchor="nw", image=ci); self.c_can.img=ci
        for k in ["top","bottom"]: d=self.last_data[k]; self.l_dist[k].config(text=f"{d['dist']:.3f} mm"); self.l_k[k].config(text=f"RATIO: {d['k']:.4f}")
        for k in ["top","bottom"]:
            d, v = self.last_data[k], self.tv[k]; v["A"].set(f"{d['A'][2]:.2f}"); v["X"].set(f"{d['X'][2]:.2f}")
            for n in ["TR","BR","B","C"]: v[n].set(f"{d[n][0]:.1f},{d[n][1]:.1f},{d[n][2]:.1f}")
            lr, rr = d["L_A"], d["R_A"]; v["L_ROL"].set(f"{lr[0]:.2f}"); v["L_ROT"].set(f"{lr[1]:.2f}"); v["L_Z"].set(f"{d['L_z']:.1f}"); v["L_PITCH"].set(f"{d['L_pitch']:+.2f}"); v["L_YAW"].set(f"{d['L_yaw']:+.2f}")
            v["R_ROL"].set(f"{rr[0]:.2f}"); v["R_ROT"].set(f"{rr[1]:.2f}"); v["R_Z"].set(f"{d['R_z']:.1f}"); v["R_PITCH"].set(f"{d['R_pitch']:+.2f}"); v["R_YAW"].set(f"{d['R_yaw']:+.2f}")
        
        ok1, ok2 = self.last_data["top"]["dist"]>0, self.last_data["bottom"]["dist"]>0
        self.c_st.config(text="OK" if ok1 and ok2 else "PARTIAL" if ok1 or ok2 else "NONE", fg=C['g'] if ok1 and ok2 else C['w'] if ok1 or ok2 else C['r'])
        
        # Warnings & State
        sp = []
        for k in ["top","bottom"]:
            d = self.last_data[k]; dt = self.dots[k]; r, p, y = d["rot_2d"], max(abs(d["L_pitch"]),abs(d["R_pitch"])), max(abs(d["L_yaw"]),abs(d["R_yaw"]))
            dt["L_d"].config(fg=C['g'] if d["L_d"] else C['r']); dt["R_d"].config(fg=C['g'] if d["R_d"] else C['r'])
            dt["rot"].config(fg=C['g'] if r<=self.rot_threshold.get() else C['w']); dt["p"].config(fg=C['g'] if p<=self.p_th.get() else C['r']); dt["y"].config(fg=C['g'] if y<=self.y_th.get() else C['r'])
            if not d["L_d"]: sp.append(f"{k} L-MISS")
            if not d["R_d"]: sp.append(f"{k} R-MISS")
            if r>self.rot_threshold.get(): sp.append(f"{k} ROT {r:.1f}")
            if p>self.p_th.get(): sp.append(f"{k} PITCH {p:.1f}")
            if y>self.y_th.get(): sp.append(f"{k} YAW {y:.1f}")
        
        li = self.last_data["light"]; self.lc["b"].config(text=f"{li['m']:.0f}"); self.lc["c"].config(text=f"{li['v']:.0f}")
        self.lc["b"].master.config(bg=C['g'] if 40<li['m']<215 else C['r']); self.lc["c"].master.config(bg=C['g'] if li['v']>=18 else C['w'])
        st = "NOMINAL" if li['s']=="ok" else li['s'].upper(); self.l_st.config(text=st, fg=C['g'] if li['s']=="ok" else C['w'])
        if li['s']!="ok": sp.append(f"LIGHT {st}")
        
        self.wst.config(state="normal"); self.wst.delete("1.0",tk.END); self.wst.insert(tk.END, ("\n".join(sp) if sp else "ALL SYSTEMS OK")); self.wst.config(fg=C['r'] if sp else C['g'], state="disabled")
        
        # Mv tick
        if self.last_data["sc"] != self._last_sc:
            self._last_sc = self.last_data["sc"]
            if self.mv_state == "collecting_init":
                for k in ["top","bottom"]: 
                    if self.last_data[k]["dist"]>0: self.bufs["init"][k].append(self.last_data[k]["dist"])
                n = min(len(self.bufs["init"]["top"]), len(self.bufs["init"]["bottom"])); self.p_bar["value"]=n; self.mv_st_l.config(text=f"INIT {n}/{N}")
                if n>=N: self.mv_d["init"]={k:np.mean(self.bufs["init"][k][:N]) for k in ["top","bottom"]}; self.mv_state="ready"; self.b_sp.config(state="normal"); self.mv_st_l.config(text="READY - MOVE PANEL")
                for k in ["top","bottom"]: self.m_res[k][0].config(text=f"INIT: {self.mv_d['init'][k] or 0:.3f}")
            elif self.mv_state == "collecting_final":
                for k in ["top","bottom"]: 
                    if self.last_data[k]["dist"]>0: self.bufs["final"][k].append(self.last_data[k]["dist"])
                n = min(len(self.bufs["final"]["top"]), len(self.bufs["final"]["bottom"])); self.p_bar["value"]=n; self.mv_st_l.config(text=f"FINAL {n}/{N}")
                if n>=N: 
                    self.mv_d["final"]={k:np.mean(self.bufs["final"][k][:N]) for k in ["top","bottom"]}; self.mv_state="done"; self.b_st.config(state="normal"); self.mv_st_l.config(text="DONE")
                    for k in ["top","bottom"]:
                        self.m_res[k][1].config(text=f"FINAL: {self.mv_d['final'][k]:.3f}"); d = self.mv_d['final'][k]-self.mv_d['init'][k]
                        self.m_res[k][2].config(text=f"{'+' if d>=0 else ''}{d:.3f}", fg=C['r'] if d>0.5 else C['g'] if d<-0.5 else C['a'])

        self.root.after(50, self.gui_loop)

if __name__ == "__main__":
    r = tk.Tk(); app = MeasurementApp(r); r.mainloop()
