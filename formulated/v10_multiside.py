import tkinter as tk
from tkinter import ttk
import cv2 as cv
import numpy as np
import threading
import math
import os
import time
from PIL import Image, ImageTk
import log  # Uses your existing log.py module

# --- DEFAULT SETTINGS ---
DEFAULT_MARKER_SIZE = 53.8             
ARUCO_DICT = cv.aruco.DICT_4X4_50  
RESOLUTION = (1280, 720)          
CALIB_NORMAL = "camera_params.npz"
CALIB_ALT = "camera_params_2.npz"

class MeasurementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dual-Pair Multi-Side Dashboard v10")
        self.root.geometry("1500x950")
        self.root.configure(bg="#2c3e50")

        # --- DYNAMIC SETTINGS ---
        self.size_top = tk.DoubleVar(value=DEFAULT_MARKER_SIZE)
        self.size_bot = tk.DoubleVar(value=DEFAULT_MARKER_SIZE)
        self.fixed_side = tk.StringVar(value="Left") # "Left" or "Right"

        # --- DATA STORAGE ---
        template = {"A":(0,0,0), "X":(0,0,0), "TR":(0,0,0), "BR":(0,0,0), "B":(0,0,0), "C":(0,0,0), 
                    "dist":0.0, "k":0.0, "U":(0,0,0), "V":(0,0,0), "W":(0,0,0), "L_Ang":(0,0), "R_Ang":(0,0)}
        
        self.last_data = {"top": template.copy(), "bottom": template.copy(), "session_count": 0}
        self.is_running = True
        self.current_frame = None

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TNotebook", background="#2c3e50")
        style.configure("TFrame", background="#ecf0f1")
        
        self.setup_ui()
        
        self.thread = threading.Thread(target=self.measurement_loop, daemon=True)
        self.thread.start()
        self.update_gui_loop()

    def setup_ui(self):
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill='both', expand=True, padx=10, pady=10)

        # TAB 1: LIVE FEED
        self.tab_live = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_live, text=" 📽 Live Monitor ")
        self.canvas = tk.Canvas(self.tab_live, width=960, height=540, bg="black")
        self.canvas.pack(side="left", padx=20, pady=20)

        stats_container = tk.Frame(self.tab_live, bg="#ecf0f1")
        stats_container.pack(side="right", fill="both", expand=True, padx=10, pady=20)

        for key, title, color in [("top", "TOP PAIR", "#e67e22"), ("bottom", "BOTTOM PAIR", "#9b59b6")]:
            f = tk.LabelFrame(stats_container, text=f" {title} ", font=("Helvetica", 12, "bold"), bg="#ecf0f1", fg=color)
            f.pack(fill="x", pady=10, padx=10)
            dist_lbl = tk.Label(f, text="0.000 mm", font=("Helvetica", 24, "bold"), fg="#27ae60", bg="#ecf0f1")
            dist_lbl.pack(pady=10)
            k_lbl = tk.Label(f, text="k: 0.000", font=("Helvetica", 11), bg="#ecf0f1")
            k_lbl.pack()
            if key == "top": self.lbl_dist_top, self.lbl_k_top = dist_lbl, k_lbl
            else: self.lbl_dist_bot, self.lbl_k_bot = dist_lbl, k_lbl

        self.lbl_session = tk.Label(stats_container, text="Logged Points: 0", font=("Helvetica", 11), fg="#7f8c8d", bg="#ecf0f1")
        self.lbl_session.pack(side="bottom", pady=20)

        # TAB 2: TELEMETRY
        self.tab_tele = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_tele, text=" 🛸 Dual Telemetry ")
        main_tele_frame = tk.Frame(self.tab_tele, bg="#ecf0f1")
        main_tele_frame.pack(fill="both", expand=True)
        self.tele_vars = {"top": {}, "bottom": {}}
        vars_to_show = ["A", "X", "TR", "BR", "B", "C", "L_Roll", "L_Tilt", "R_Roll", "R_Tilt"]

        for key, title, color in [("top", "TOP", "#e67e22"), ("bottom", "BOTTOM", "#9b59b6")]:
            col_frame = tk.LabelFrame(main_tele_frame, text=f" {title} DATA ", bg="#ecf0f1", fg=color, font=("Helvetica", 12, "bold"))
            col_frame.pack(side="left", fill="both", expand=True, padx=20, pady=20)
            for var in vars_to_show:
                f = tk.Frame(col_frame, bg="white", highlightbackground="#bdc3c7", highlightthickness=1)
                f.pack(fill="x", padx=10, pady=4)
                tk.Label(f, text=var, font=("Helvetica", 9, "bold"), bg="white", fg="#7f8c8d").pack(side="left", padx=5)
                v = tk.StringVar(value="0.000, 0.000, 0.000")
                tk.Label(f, textvariable=v, font=("Courier", 11), bg="white").pack(side="right", padx=5)
                self.tele_vars[key][var] = v

        # TAB 3: SETTINGS
        self.tab_settings = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_settings, text=" ⚙ Machine Configuration ")
        settings_canvas = tk.Frame(self.tab_settings, bg="#ecf0f1")
        settings_canvas.pack(fill="both", expand=True, padx=100, pady=50)

        # Reference Side Toggle
        ref_frame = tk.LabelFrame(settings_canvas, text=" Reference Setup ", bg="white", font=("Helvetica", 12, "bold"), padx=20, pady=10)
        ref_frame.pack(fill="x", pady=10)
        tk.Label(ref_frame, text="Which ArUco is FIXED to the machine?", bg="white").pack(side="left")
        for choice in ["Left", "Right"]:
            tk.Radiobutton(ref_frame, text=f" {choice} marker is Fixed ", variable=self.fixed_side, value=choice, bg="white").pack(side="left", padx=20)

        def create_size_control(parent, label, var, color):
            frame = tk.LabelFrame(parent, text=f" {label} ", bg="white", font=("Helvetica", 12), fg=color, padx=20, pady=10)
            frame.pack(fill="x", pady=10)
            tk.Label(frame, text="Marker Size (mm):", bg="white").pack(side="left")
            tk.Scale(frame, from_=10.0, to=200.0, resolution=0.1, orient="horizontal", variable=var, bg="white", length=400).pack(side="left", padx=20)
            tk.Entry(frame, textvariable=var, width=8).pack(side="left")

        create_size_control(settings_canvas, "Upper Pair Size", self.size_top, "#e67e22")
        create_size_control(settings_canvas, "Down Pair Size", self.size_bot, "#9b59b6")

    def load_calib(self):
        # Determine file based on settings
        target_file = CALIB_ALT if self.fixed_side.get() == "Right" else CALIB_NORMAL
        if os.path.exists(target_file):
            d = np.load(target_file)
            return d['camera_matrix'], d['dist_coeff']
        # Fallback
        return np.array([[RESOLUTION[0], 0, RESOLUTION[0]/2], [0, RESOLUTION[0], RESOLUTION[1]/2], [0, 0, 1]], dtype=np.float32), np.zeros(5)

    def measurement_loop(self):
        try:
            from picamera2 import Picamera2
            picam2 = Picamera2(); picam2.configure(picam2.create_video_configuration(main={"size": RESOLUTION, "format": "RGB888"})); picam2.start()
        except: return

        detector = cv.aruco.ArucoDetector(cv.aruco.getPredefinedDictionary(ARUCO_DICT), cv.aruco.DetectorParameters())
        log.init_log()
        buffers = {"top": [], "bottom": []}
        l_sample = l_update = time.time() * 1000

        while self.is_running:
            # RELOAD K/dist in case settings changed
            K, dist = self.load_calib()
            
            frame_rgb = picam2.capture_array()
            frame = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
            corners, ids, _ = detector.detectMarkers(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
            curr = time.time() * 1000

            if ids is not None and len(ids) >= 2:
                marker_data = [{"corners": corners[i][0], "y": np.mean(corners[i][0], axis=0)[1]} for i in range(len(ids))]
                top_m = [m for m in marker_data if m["y"] < RESOLUTION[1]/2]
                bot_m = [m for m in marker_data if m["y"] >= RESOLUTION[1]/2]

                def process_pair(marker_list, key, size_mm):
                    if len(marker_list) < 2: 
                        self.last_data[key]["dist"] = 0.0
                        return
                    
                    marker_list.sort(key=lambda m: np.mean(m["corners"], axis=0)[0])
                    is_right_fixed = (self.fixed_side.get() == "Right")

                    def solve_p(c_pts):
                        h = size_mm / 2.0
                        obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)
                        _, rv, tv = cv.solvePnP(obj, c_pts, K, dist)
                        R, _ = cv.Rodrigues(rv)
                        pts = np.array([np.dot(R, pt) + tv.ravel() for pt in obj])
                        return {"pts": pts, "r": math.degrees(math.atan2(R[1,0], R[0,0])), "t": math.degrees(math.acos(np.clip(R[2,2], -1,1)))}

                    S = solve_p(marker_list[1 if is_right_fixed else 0]["corners"]) # Source (Fixed)
                    T = solve_p(marker_list[0 if is_right_fixed else 1]["corners"]) # Target (Moving)
                    
                    # Logic for Side A (Source)
                    if not is_right_fixed: # Left is Fix (Source)
                        trl, brl = S["pts"][S["pts"][:, 0].argsort()[::-1]][:2][S["pts"][S["pts"][:, 0].argsort()[::-1]][:2][:, 1].argsort()]
                        tll = S["pts"][S["pts"][:, 0].argsort()[::-1]][2:][S["pts"][S["pts"][:, 0].argsort()[::-1]][2:][:, 1].argsort()][0]
                        b, c = T["pts"][T["pts"][:, 0].argsort()][:2][T["pts"][T["pts"][:, 0].argsort()][:2][:, 1].argsort()]
                        trr, brr = T["pts"][T["pts"][:, 0].argsort()[::-1]][:2][T["pts"][T["pts"][:, 0].argsort()[::-1]][:2][:, 1].argsort()]
                    else: # Right is Fix (Source)
                        trl, brl = S["pts"][S["pts"][:, 0].argsort()][:2][S["pts"][S["pts"][:, 0].argsort()][:2][:, 1].argsort()] # Left edge of Right marker
                        tll = S["pts"][S["pts"][:, 0].argsort()][2:][S["pts"][S["pts"][:, 0].argsort()][2:][:, 1].argsort()][0]
                        b, c = T["pts"][T["pts"][:, 0].argsort()[::-1]][:2][T["pts"][T["pts"][:, 0].argsort()[::-1]][:2][:, 1].argsort()] # Right edge of Left marker
                        trr, brr = T["pts"][T["pts"][:, 0].argsort()][:2][T["pts"][T["pts"][:, 0].argsort()][:2][:, 1].argsort()]

                    buffers[key].append({"A":(trl+brl)/2, "B":b, "C":c, "TR_R":trr, "BR_R":brr, "TR":trl, "BR":brl, "TL":tll, "L_Ang":(S["r"], S["t"]), "R_Ang":(T["r"], T["t"])})

                if (curr - l_sample) >= 100:
                    process_pair(top_m, "top", self.size_top.get())
                    process_pair(bot_m, "bottom", self.size_bot.get())
                    l_sample = curr

                if (curr - l_update) >= 500:
                    for key in ["top", "bottom"]:
                        if len(buffers[key]) > 0:
                            s = buffers[key]
                            aA, aB, aC, aTR, aBR, aTL = [np.mean([x[k] for x in s], axis=0) for k in ["A", "B", "C", "TR", "BR", "TL"]]
                            aTRR, aBRR = [np.mean([x[k] for x in s], axis=0) for k in ["TR_R", "BR_R"]]
                            aL, aR = np.mean([x["L_Ang"] for x in s], axis=0), np.mean([x["R_Ang"] for x in s], axis=0)
                            is_rot = any(abs(ang) > 20 for ang in [aL[0], aL[1], aR[0], aR[1]])
                            
                            if is_rot:
                                aX = (aTRR + aBRR) / 2; dist_v = np.linalg.norm(aX - aA)
                                k_v, v_vec, w_vec, u_vec = 0.5, np.zeros(3), np.zeros(3), np.zeros(3)
                            else:
                                v_vec = (aTR-aTL)/np.linalg.norm(aTR-aTL); w_vec, u_vec = aC-aB, aB-aA
                                den = (np.dot(v_vec,v_vec)*np.dot(w_vec,w_vec))-(np.dot(v_vec,w_vec)**2)
                                if abs(den)>1e-6:
                                    k_v = ((np.dot(v_vec,w_vec)*np.dot(u_vec,v_vec))-(np.dot(v_vec,v_vec)*np.dot(u_vec,w_vec)))/den
                                    aX = aB + k_v * w_vec; dist_v = np.linalg.norm(aX-aA)
                                else: aX = aB; dist_v = 0.0; k_v = 0.0
                            
                            self.last_data[key].update({"A":aA, "X":aX, "TR":aTR, "BR":aBR, "B":aB, "C":aC, "dist":dist_v, "k":k_v, "L_Ang":aL, "R_Ang":aR})
                            self.last_data["session_count"] += 1
                            log.record(dist_v, k_v, aA, aX, aTR, aBR, aB, aC, u_vec, v_vec, w_vec, aL, aR)
                            buffers[key].clear()
                    l_update = curr
            else: self.last_data["top"]["dist"] = self.last_data["bottom"]["dist"] = 0.0

            for key, color in [("top", (0, 165, 255)), ("bottom", (255, 0, 255))]:
                d = self.last_data[key]
                if d["dist"] > 0:
                    p_disp, _ = cv.projectPoints(np.array([d["A"], d["X"]]), np.zeros(3), np.zeros(3), K, dist)
                    p1, p2 = tuple(p_disp[0].ravel().astype(int)), tuple(p_disp[1].ravel().astype(int))
                    cv.line(frame, p1, p2, color, 3); cv.circle(frame, p2, 6, (0, 255, 0), -1)
                    cv.putText(frame, f"{key.upper()}: {d['dist']:.2f}mm", (p1[0], p1[1]-10), 0, 0.6, color, 2)
                    if abs(d["L_Ang"][1]) > 5.0:
                        bx, by = p1[0]-80, p1[1]-60
                        cv.rectangle(frame, (bx, by-25), (bx+130, by+10), (0,0,255), -1)
                        cv.putText(frame, f"REF TILT: {d['L_Ang'][1]:.2f}", (bx+5, by), 0, 0.6, (0,255,255), 2)
            
            log.save_image(frame)
            self.current_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    def update_gui_loop(self):
        if self.current_frame is not None:
            img = Image.fromarray(self.current_frame).resize((960, 540))
            self.tk_img = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

        self.lbl_dist_top.config(text=f"{self.last_data['top']['dist']:.3f} mm")
        self.lbl_k_top.config(text=f"k: {self.last_data['top']['k']:.4f}")
        self.lbl_dist_bot.config(text=f"{self.last_data['bottom']['dist']:.3f} mm")
        self.lbl_k_bot.config(text=f"k: {self.last_data['bottom']['k']:.4f}")
        self.lbl_session.config(text=f"Total Measurements Recorded: {self.last_data['session_count']}")

        for key in ["top", "bottom"]:
            d = self.last_data[key]
            for var in ["A", "X", "TR", "BR", "B", "C"]:
                v = d[var]; self.tele_vars[key][var].set(f"{v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}")
            self.tele_vars[key]["L_Roll"].set(f"{d['L_Ang'][0]:.2f}°"); self.tele_vars[key]["L_Tilt"].set(f"{d['L_Ang'][1]:.2f}°")
            self.tele_vars[key]["R_Roll"].set(f"{d['R_Ang'][0]:.2f}°"); self.tele_vars[key]["R_Tilt"].set(f"{d['R_Ang'][1]:.2f}°")
        self.root.after(50, self.update_gui_loop)

if __name__ == "__main__":
    root = tk.Tk(); app = MeasurementApp(root); root.mainloop()
