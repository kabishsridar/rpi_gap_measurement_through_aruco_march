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

# --- SETTINGS ---
MARKER_SIZE_MM = 53.8             
ARUCO_DICT = cv.aruco.DICT_4X4_50  
RESOLUTION = (1280, 720)          
CALIB_FILE = "camera_params.npz"

class MeasurementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Precision ArUco Measurement Dashboard v6")
        self.root.geometry("1400x900")
        self.root.configure(bg="#2c3e50")

        # --- DATA STORAGE ---
        self.last_data = {"A":(0,0,0), "X":(0,0,0), "TR":(0,0,0), "BR":(0,0,0), "B":(0,0,0), "C":(0,0,0), "dist":0.0, "k":0.0,
                          "U":(0,0,0), "V":(0,0,0), "W":(0,0,0), "L_Ang":(0,0), "R_Ang":(0,0), "session_count":0}
        self.is_running = True
        self.current_frame = None

        # --- UI STYLING ---
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TNotebook", background="#2c3e50")
        style.configure("TFrame", background="#ecf0f1")
        style.configure("Header.TLabel", font=("Helvetica", 14, "bold"), foreground="#2980b9")

        self.setup_ui()
        
        # --- START BACKGROUND THREAD ---
        self.thread = threading.Thread(target=self.measurement_loop, daemon=True)
        self.thread.start()

        # Update UI timer
        self.update_gui_loop()

    def setup_ui(self):
        # Navigation Tabs
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill='both', expand=True, padx=10, pady=10)

        # TAB 1: LIVE FEED
        self.tab_live = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_live, text=" ?? Live Monitor ")
        
        # Left Side: Video
        self.canvas = tk.Canvas(self.tab_live, width=960, height=540, bg="black")
        self.canvas.pack(side="left", padx=20, pady=20)

        # Right Side: Quick Stats
        stats_frame = tk.LabelFrame(self.tab_live, text=" REAL-TIME STATISTICS ", font=("Helvetica", 12, "bold"), bg="#ecf0f1")
        stats_frame.pack(side="right", fill="both", expand=True, padx=10, pady=20)

        self.lbl_dist = tk.Label(stats_frame, text="0.000 mm", font=("Helvetica", 24, "bold"), fg="#27ae60", bg="#ecf0f1")
        self.lbl_dist.pack(pady=30)
        
        self.lbl_k = tk.Label(stats_frame, text="k: 0.000", font=("Helvetica", 12), bg="#ecf0f1")
        self.lbl_k.pack(pady=5)
        
        self.lbl_session = tk.Label(stats_frame, text="Logged Points: 0", font=("Helvetica", 10), fg="#7f8c8d", bg="#ecf0f1")
        self.lbl_session.pack(side="bottom", pady=20)

        # TAB 2: TELEMETRY (GRID + GLOSSARY)
        self.tab_tele = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_tele, text=" ?? All Telemetry ")
        tele_grid = tk.Frame(self.tab_tele, bg="#ecf0f1")
        tele_grid.pack(fill="both", expand=True, padx=50, pady=30)
        self.tele_vars = {}
        vars_to_show = ["A", "X", "TR", "BR", "B", "C", "U", "V", "W", "L_Roll", "L_Tilt", "R_Roll", "R_Tilt"]
        for i, var in enumerate(vars_to_show):
            row, col = divmod(i, 3)
            f = tk.LabelFrame(tele_grid, text=f" {var} ", bg="white", padx=10, pady=10)
            f.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            v = tk.StringVar(value="0.000, 0.000, 0.000")
            tk.Label(f, textvariable=v, font=("Courier", 11), bg="white").pack()
            self.tele_vars[var] = v
        # NEW: Variable Glossary at the bottom of Tab 2
        legend_frame = tk.LabelFrame(self.tab_tele, text=" ?? Telemetry Variable Key (Definitions) ", font=("Helvetica", 10, "bold"), bg="#ecf0f1", pady=15)
        legend_frame.pack(fill="x", padx=50, pady=(0, 30))
        glossary = [
            ("A / X", "Measurement Start Point (A) and Calculated End Point (X) on Target Edge."),
            ("TR / BR", "Top-Right and Bottom-Right corners of Left Marker (Source Edge)."),
            ("B / C", "Top-Left and Bottom-Left corners of Right Marker (Target Edge boundary)."),
            ("U / V / W", "U: A-to-B Vector. V: Ray Direction (Unit Vector). W: B-to-C Target Vector."),
            ("k-value", "The linear intersection factor (0 = Start of Edge B, 1 = End of Edge C)."),
            ("Roll / Tilt", "Roll: Flat clock-rotation. Tilt: Inclination angle from camera plane.")
        ]
        for i, (term, desc) in enumerate(glossary):
            r, c = divmod(i, 2)
            tk.Label(legend_frame, text=f"� {term}: {desc}", font=("Helvetica", 9), bg="#ecf0f1", anchor="w").grid(row=r, column=c, sticky="w", padx=30, pady=3)

        # TAB 3: FORMULA GUIDE
        self.tab_math = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_math, text=" ?? Formula Guide ")
        
        math_txt = tk.Text(self.tab_math, wrap="word", font=("Helvetica", 12), bg="#ecf0f1", relief="flat")
        math_txt.insert("1.0", "--- 3D GEOMETRIC INTERSECTION (LASER-HIT METHOD) ---\n\n")
        math_txt.insert("end", "1. RAY FORMULA (Left Marker):\n   P(t) = A + t * v\n   A = Midpoint of inner edge. v = Unit direction across marker.\n\n")
        math_txt.insert("end", "2. SEGMENT FORMULA (Right Marker):\n   S(k) = B + k * w\n   B = Top-Left corner. w = Edge vector towards Bottom-Left.\n\n")
        math_txt.insert("end", "3. SOLVING FOR k (Intersection Point X):\n   k = [(v�w)(u�v) - (v�v)(u�w)] / [(v�v)(w�w) - (v�w)�]\n   Where u = B - A.\n\n")
        math_txt.insert("end", "This math calculates exactly where the 'laser' from the left hits the edge of the right marker, even if they are tilted or at different depths.")
        math_txt.config(state="disabled")
        math_txt.pack(fill="both", expand=True, padx=40, pady=40)

    def load_calib(self):
        if os.path.exists(CALIB_FILE):
            d = np.load(CALIB_FILE)
            return d['camera_matrix'], d['dist_coeff']
        return np.array([[RESOLUTION[0], 0, RESOLUTION[0]/2], [0, RESOLUTION[0], RESOLUTION[1]/2], [0, 0, 1]], dtype=np.float32), np.zeros(5)

    def measurement_loop(self):
        try:
            from picamera2 import Picamera2
            picam2 = Picamera2()
            picam2.configure(picam2.create_video_configuration(main={"size": RESOLUTION, "format": "RGB888"}))
            picam2.start()
        except:
            print("Video capture error.")
            return

        K, dist = self.load_calib()
        params = cv.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
        detector = cv.aruco.ArucoDetector(cv.aruco.getPredefinedDictionary(ARUCO_DICT), params)
        log.init_log()

        buffer = []
        l_sample = l_update = time.time() * 1000

        while self.is_running:
            frame_rgb = picam2.capture_array()
            frame = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            curr = time.time() * 1000

            if ids is not None and len(ids) >= 2:
                if (curr - l_sample) >= 100:
                    order = np.argsort([np.mean(c[0], axis=0)[0] for c in corners])
                    def get_p(idx):
                        half = MARKER_SIZE_MM / 2.0
                        obj = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
                        _, rv, tv = cv.solvePnP(obj, corners[order[idx]][0], K, dist)
                        R, _ = cv.Rodrigues(rv)
                        pts = np.array([np.dot(R, pt) + tv.ravel() for pt in obj])
                        l_sorted = pts[pts[:, 0].argsort()[::-1]]
                        r_sorted = pts[pts[:, 0].argsort()]
                        roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
                        tilt = math.degrees(math.acos(np.clip(R[2, 2], -1, 1)))
                        return {"pts":pts, "rv":rv, "tv":tv, "r":roll, "t":tilt}
                    
                    L = get_p(0); R = get_p(1)
                    # Inner Edges
                    trl, brl = L["pts"][L["pts"][:, 0].argsort()[::-1]][:2][L["pts"][L["pts"][:, 0].argsort()[::-1]][:2][:, 1].argsort()]
                    tll = L["pts"][L["pts"][:, 0].argsort()[::-1]][2:][L["pts"][L["pts"][:, 0].argsort()[::-1]][2:][:, 1].argsort()][0]
                    b, c = R["pts"][R["pts"][:, 0].argsort()][:2][R["pts"][R["pts"][:, 0].argsort()][:2][:, 1].argsort()]
                    
                    buffer.append({"A":(trl+brl)/2, "B":b, "C":c, "TR":trl, "BR":brl, "TL":tll, "L_Ang":(L["r"], L["t"]), "R_Ang":(R["r"], R["t"])})
                    l_sample = curr

                if (curr - l_update) >= 500 and len(buffer) > 0:
                    aA, aB, aC, aTR, aBR, aTL = [np.mean([s[k] for s in buffer], axis=0) for k in ["A", "B", "C", "TR", "BR", "TL"]]
                    aL, aR = np.mean([s["L_Ang"] for s in buffer], axis=0), np.mean([s["R_Ang"] for s in buffer], axis=0)
                    v_vec = (aTR-aTL)/np.linalg.norm(aTR-aTL); w_vec, u_vec = aC-aB, aB-aA
                    den = (np.dot(v_vec,v_vec)*np.dot(w_vec,w_vec))-(np.dot(v_vec,w_vec)**2)
                    if abs(den)>1e-6:

                        k = ((np.dot(v_vec,w_vec)*np.dot(u_vec,v_vec))-(np.dot(v_vec,v_vec)*np.dot(u_vec,w_vec)))/den
                        aX = aB + k * w_vec
                        dist_val = np.linalg.norm(aX-aA)
                        self.last_data.update({"A":aA, "X":aX, "TR":aTR, "BR":aBR, "B":aB, "C":aC, "dist":dist_val, "k":k, "U":u_vec, "V":v_vec, "W":w_vec, "L_Ang":aL, "R_Ang":aR})
                        self.last_data["session_count"] += 1
                        log.record(dist_val, k, aA, aX, aTR, aBR, aB, aC, u_vec, v_vec, w_vec, aL, aR)
                    buffer.clear(); l_update = curr
            else:
                self.last_data["dist"] = 0.0 # Reset if lost

            # Draw View
            if self.last_data["dist"] > 0:
                y = 35
                for l in ["A", "X", "TR", "BR", "B", "C"]:
                    v = self.last_data[l]; cv.putText(frame, f"{l}:({v[0]:.2f},{v[1]:.2f})", (15, y), 0, 0.5, (0, 255, 255), 1)
                    y += 20
                p_disp, _ = cv.projectPoints(np.array([self.last_data["A"], self.last_data["X"]]), np.zeros(3), np.zeros(3), K, dist)
                p1, p2 = tuple(p_disp[0].ravel().astype(int)), tuple(p_disp[1].ravel().astype(int))
                cv.line(frame, p1, p2, (0, 165, 255), 3); cv.circle(frame, p2, 8, (255, 0, 255), -1)
                cv.putText(frame, f"AVG: {self.last_data['dist']:.3f} mm", ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2-15), 0, 0.8, (0, 255, 0), 2)
            
            log.save_image(frame)
            self.current_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    def update_gui_loop(self):
        # Update Video
        if self.current_frame is not None:
            img = Image.fromarray(self.current_frame).resize((960, 540))
            self.tk_img = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

        # Update Statistics Tab 1
        d = self.last_data
        self.lbl_dist.config(text=f"{d['dist']:.3f} mm")
        self.lbl_k.config(text=f"Laser intersection (k): {d['k']:.4f}")
        self.lbl_session.config(text=f"Total Measurements Recorded: {d['session_count']}")

        # Update Telemetry Tab 2
        for key in ["A", "X", "TR", "BR", "B", "C", "U", "V", "W"]:
            val = d[key]
            self.tele_vars[key].set(f"{val[0]:.3f}, {val[1]:.3f}, {val[2]:.3f}")
        
        self.tele_vars["L_Roll"].set(f"{d['L_Ang'][0]:.3f}�")
        self.tele_vars["L_Tilt"].set(f"{d['L_Ang'][1]:.3f}�")
        self.tele_vars["R_Roll"].set(f"{d['R_Ang'][0]:.3f}�")
        self.tele_vars["R_Tilt"].set(f"{d['R_Ang'][1]:.3f}�")

        self.root.after(50, self.update_gui_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = MeasurementApp(root)
    root.mainloop()

