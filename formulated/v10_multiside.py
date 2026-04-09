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

class MeasurementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Agnostic ArUco Measurement Dashboard v11")
        self.root.geometry("1500x950")
        self.root.configure(bg="#2c3e50")

        # UI Variables
        self.size_top = tk.DoubleVar(value=DEFAULT_MARKER_SIZE)
        self.size_bot = tk.DoubleVar(value=DEFAULT_MARKER_SIZE)
        self.fixed_side = tk.StringVar(value="Left")
        self.rot_threshold = tk.DoubleVar(value=10.0) # Default 10 degrees

        temp = {"A":(0,0,0), "X":(0,0,0), "TR":(0,0,0), "BR":(0,0,0), "B":(0,0,0), "C":(0,0,0), 
                "dist":0.0, "k":0.0, "L_Ang":(0,0), "R_Ang":(0,0), "rot_2d": 0.0}
        self.last_data = {"top": temp.copy(), "bottom": temp.copy(), "session_count": 0}
        self.is_running = True
        self.current_frame = None

        style = ttk.Style(); style.theme_use('clam')
        self.setup_ui()
        threading.Thread(target=self.measurement_loop, daemon=True).start()
        self.update_gui_loop()

    def setup_ui(self):
        self.tabs = ttk.Notebook(self.root); self.tabs.pack(fill='both', expand=True, padx=10, pady=10)
        
        # TAB 1: LIVE FEED
        self.tab_live = ttk.Frame(self.tabs); self.tabs.add(self.tab_live, text=" 📽 Live Monitor ")
        self.canvas = tk.Canvas(self.tab_live, width=960, height=540, bg="black"); self.canvas.pack(side="left", padx=20, pady=20)
        st = tk.Frame(self.tab_live, bg="#ecf0f1"); st.pack(side="right", fill="both", expand=True, padx=10, pady=20)

        for key, title, color in [("top", "TOP PAIR", "#e67e22"), ("bottom", "BOTTOM PAIR", "#9b59b6")]:
            f = tk.LabelFrame(st, text=f" {title} ", font=("Helvetica", 12, "bold"), bg="#ecf0f1", fg=color); f.pack(fill="x", pady=10, padx=10)
            dl = tk.Label(f, text="0.000 mm", font=("Helvetica", 24, "bold"), fg="#27ae60", bg="#ecf0f1"); dl.pack(pady=10)
            kl = tk.Label(f, text="k: 0.000", font=("Helvetica", 11), bg="#ecf0f1"); kl.pack()
            if key == "top": self.lbl_dist_top, self.lbl_k_top = dl, kl
            else: self.lbl_dist_bot, self.lbl_k_bot = dl, kl
        
        # TAB 2: SETTINGS
        self.tab_settings = ttk.Frame(self.tabs); self.tabs.add(self.tab_settings, text=" ⚙ Machine Configuration ")
        sc = tk.Frame(self.tab_settings, bg="#ecf0f1"); sc.pack(fill="both", expand=True, padx=100, pady=50)
        
        rf = tk.LabelFrame(sc, text=" Reference Setup ", bg="white", font=("Helvetica", 11, "bold"), padx=20, pady=10); rf.pack(fill="x")
        for choice in ["Left", "Right"]: tk.Radiobutton(rf, text=choice, variable=self.fixed_side, value=choice, bg="white").pack(side="left", padx=20)

        def create_sc(parent, label, var, color, min_v, max_v):
            f = tk.LabelFrame(parent, text=f" {label} ", bg="white", font=("Helvetica", 10), fg=color, padx=20, pady=5)
            f.pack(fill="x", pady=5)
            tk.Scale(f, from_=min_v, to=max_v, resolution=0.1, orient="horizontal", variable=var, bg="white", length=400).pack(side="left", padx=20)
            tk.Entry(f, textvariable=var, width=8).pack(side="left")

        create_sc(sc, "Upper Pair Marker Size (mm)", self.size_top, "#e67e22", 10, 200)
        create_sc(sc, "Down Pair Marker Size (mm)", self.size_bot, "#9b59b6", 10, 200)
        create_sc(sc, "Rotation Threshold Deg (Fallback Trigger)", self.rot_threshold, "#34495e", 0, 90)

    def load_calibrated_params(self):
        target = "camera_params_2.npz" if self.fixed_side.get() == "Right" else "camera_params.npz"
        if os.path.exists(target):
            d = np.load(target); return d['camera_matrix'], d['dist_coeff']
        return np.array([[1280, 0, 640], [0, 1280, 360], [0, 0, 1]], dtype=np.float32), np.zeros(5)

    def measurement_loop(self):
        try:
            from picamera2 import Picamera2
            pc = Picamera2(); pc.configure(pc.create_video_configuration(main={"size": RESOLUTION, "format": "RGB888"})); pc.start()
        except: return
        
        detector = cv.aruco.ArucoDetector(cv.aruco.getPredefinedDictionary(ARUCO_DICT), cv.aruco.DetectorParameters())
        log.init_log(); buffers = {"top": [], "bottom": []}
        l_s = l_u = time.time() * 1000

        while self.is_running:
            K, dist = self.load_calibrated_params()
            frame = cv.cvtColor(pc.capture_array(), cv.COLOR_RGB2BGR)
            corners, ids, _ = detector.detectMarkers(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
            curr = time.time() * 1000

            if ids is not None and len(ids) >= 2:
                m_data = [{"c": corners[i][0], "y": np.mean(corners[i][0], axis=0)[1]} for i in range(len(ids))]
                top_m, bot_m = [m for m in m_data if m["y"] < 360], [m for m in m_data if m["y"] >= 360]

                def proc(marker_list, key, size):
                    if len(marker_list) < 2: self.last_data[key]["dist"] = 0.0; return
                    marker_list.sort(key=lambda m: np.mean(m["c"], axis=0)[0])
                    is_rf = (self.fixed_side.get() == "Right")

                    def get_full_data(c2d):
                        h = size / 2.0; obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)
                        _, rv, tv = cv.solvePnP(obj, c2d, K, dist); R, _ = cv.Rodrigues(rv)
                        pts3d = np.array([np.dot(R, pt) + tv.ravel() for pt in obj])
                        idx = np.argsort(c2d[:, 0]); dy, dx = c2d[idx[2], 1]-c2d[idx[0], 1], c2d[idx[2], 0]-c2d[idx[0], 0]
                        rot2d = math.degrees(math.atan2(dy, dx))
                        return pts3d, rot2d, math.degrees(math.atan2(R[1,0], R[0,0])), math.degrees(math.acos(np.clip(R[2,2], -1,1)))

                    def extract_inner(pts3d, p2d, find_right_side=True):
                        idx = np.argsort(p2d[:, 0])
                        inner_sub = idx[-2:] if find_right_side else idx[:2]
                        inner_sub = inner_sub[np.argsort(p2d[inner_sub, 1])] 
                        return pts3d[inner_sub[0]], pts3d[inner_sub[1]] 

                    S_pts, S_rot, S_rl, S_tl = get_full_data(marker_list[1 if is_rf else 0]["c"])
                    T_pts, T_rot, T_rl, T_tl = get_full_data(marker_list[0 if is_rf else 1]["c"])

                    trl, brl = extract_inner(S_pts, marker_list[1 if is_rf else 0]["c"], find_right_side=not is_rf)
                    trr, brr = extract_inner(T_pts, marker_list[0 if is_rf else 1]["c"], find_right_side=is_rf)
                    
                    buffers[key].append({"A":(trl+brl)/2, "X_alt":(trr+brr)/2, "TR":trl, "BR":brl, "B":trr, "C":brr, 
                                         "L_A":(S_rl, S_tl), "R_A":(T_rl, T_tl), "rot": max(abs(S_rot), abs(T_rot))})

                if (curr - l_s) >= 100: proc(top_m, "top", self.size_top.get()); proc(bot_m, "bottom", self.size_bot.get()); l_s = curr

                if (curr - l_u) >= 500:
                    for key in ["top", "bottom"]:
                        if buffers[key]:
                            s = buffers[key]; aA, aTR, aBR = [np.mean([x[k] for x in s], axis=0) for k in ["A", "TR", "BR"]]
                            aXalt, aB, aC = [np.mean([x[k] for x in s], axis=0) for k in ["X_alt", "B", "C"]]
                            avg_rot = np.mean([x["rot"] for x in s])
                            
                            # Rotation Trigger using slider value
                            if avg_rot > self.rot_threshold.get():
                                aX = aXalt; dist_v = np.linalg.norm(aX - aA); kv = 0.5
                            else:
                                v_vec = (aTR-aA)/np.linalg.norm(aTR-aA); w_vec, u_vec = aC-aB, aB-aA
                                den = (np.dot(v_vec,v_vec)*np.dot(w_vec,w_vec))-(np.dot(v_vec,w_vec)**2)
                                if abs(den)>1e-6:
                                    kv = ((np.dot(v_vec,w_vec)*np.dot(u_vec,v_vec))-(np.dot(v_vec,v_vec)*np.dot(u_vec,w_vec)))/den
                                    aX = aB + kv * w_vec; dist_v = np.linalg.norm(aX-aA)
                                else: aX = aB; dist_v = 0.0; kv = 0.0
                            
                            self.last_data[key].update({"A":aA, "X":aX, "dist":dist_v, "k":kv, "rot_2d":avg_rot})
                            buffers[key].clear()
                    l_u = curr
            else: self.last_data["top"]["dist"] = self.last_data["bottom"]["dist"] = 0.0

            for key, color in [("top", (0, 165, 255)), ("bottom", (255, 0, 255))]:
                d = self.last_data[key]
                if d["dist"] > 0:
                    p_disp, _ = cv.projectPoints(np.array([d["A"], d["X"]]), np.zeros(3), np.zeros(3), K, dist)
                    p1, p2 = tuple(p_disp[0].ravel().astype(int)), tuple(p_disp[1].ravel().astype(int))
                    cv.line(frame, p1, p2, color, 3); cv.circle(frame, p2, 6, (0, 255, 0), -1)
                    if not (160.0 <= d["L_A"][1] <= 200.0): 
                        cv.rectangle(frame, (p1[0]-80, p1[1]-85), (p1[0]+50, p1[1]-50), (0,0,255), -1)
                        cv.putText(frame, f"TILT: {d['L_A'][1]:.1f}", (p1[0]-75, p1[1]-65), 0, 0.5, (0,255,255), 2)
            self.current_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    def update_gui_loop(self):
        if self.current_frame is not None:
            img = ImageTk.PhotoImage(image=Image.fromarray(self.current_frame).resize((960, 540)))
            self.canvas.create_image(0, 0, anchor="nw", image=img); self.canvas.img = img
        self.lbl_dist_top.config(text=f"{self.last_data['top']['dist']:.3f} mm")
        self.lbl_dist_bot.config(text=f"{self.last_data['bottom']['dist']:.3f} mm")
        self.root.after(50, self.update_gui_loop)

if __name__ == "__main__":
    tk.Tk().mainloop() if not MeasurementApp(tk.Tk()) else None
