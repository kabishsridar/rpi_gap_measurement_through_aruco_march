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
        self.root.title("Dual-Pair Precision Telemetry v11.2")
        self.root.geometry("1500x950")
        self.root.configure(bg="#2c3e50")

        self.size_top = tk.DoubleVar(value=DEFAULT_MARKER_SIZE)
        self.size_bot = tk.DoubleVar(value=DEFAULT_MARKER_SIZE)
        self.fixed_side = tk.StringVar(value="Left")
        self.rot_threshold = tk.DoubleVar(value=15.0)

        temp = {"A":(0,0,0), "X":(0,0,0), "TR":(0,0,0), "BR":(0,0,0), "B":(0,0,0), "C":(0,0,0), 
                "dist":0.0, "k":0.0, "L_A":(0,0), "R_A":(0,0), "rot_2d": 0.0,
                "L_z": 0.0, "R_z": 0.0}
        self.last_data = {"top": temp.copy(), "bottom": temp.copy(), "session_count": 0}
        self.is_running = True
        self.current_frame = None

        style = ttk.Style(); style.theme_use('clam')
        self.setup_ui()
        threading.Thread(target=self.measurement_loop, daemon=True).start()
        self.update_gui_loop()

    def setup_ui(self):
        self.tabs = ttk.Notebook(self.root); self.tabs.pack(fill='both', expand=True, padx=10, pady=10)
        self.tab_live = ttk.Frame(self.tabs); self.tabs.add(self.tab_live, text=" 📽 Live Monitor ")
        self.canvas = tk.Canvas(self.tab_live, width=960, height=540, bg="black"); self.canvas.pack(side="left", padx=20, pady=20)
        st = tk.Frame(self.tab_live, bg="#ecf0f1"); st.pack(side="right", fill="both", expand=True, padx=10, pady=20)

        for key, title, color in [("top", "TOP PAIR", "#e67e22"), ("bottom", "BOTTOM PAIR", "#9b59b6")]:
            f = tk.LabelFrame(st, text=f" {title} ", font=("Helvetica", 12, "bold"), bg="#ecf0f1", fg=color); f.pack(fill="x", pady=10, padx=10)
            dl = tk.Label(f, text="0.000 mm", font=("Helvetica", 24, "bold"), fg="#27ae60", bg="#ecf0f1"); dl.pack(pady=10)
            kl = tk.Label(f, text="k: 0.000", font=("Helvetica", 11), bg="#ecf0f1"); kl.pack()
            if key == "top": self.lbl_dist_top, self.lbl_k_top = dl, kl
            else: self.lbl_dist_bot, self.lbl_k_bot = dl, kl

        self.tab_tele = ttk.Frame(self.tabs); self.tabs.add(self.tab_tele, text=" 🛸 Dual Telemetry ")
        mtf = tk.Frame(self.tab_tele, bg="#ecf0f1"); mtf.pack(fill="both", expand=True)
        self.tele_vars = {"top": {}, "bottom": {}}
        v_show = ["A", "X", "TR", "BR", "B", "C", "L_ROL", "L_ROT", "L_Z", "R_ROL", "R_ROT", "R_Z"]
        for key, title, color in [("top", "TOP", "#e67e22"), ("bottom", "BOTTOM", "#9b59b6")]:
            cf = tk.LabelFrame(mtf, text=f" {title} DATA ", font=("Helvetica", 12, "bold"), fg=color, bg="#ecf0f1"); cf.pack(side="left", fill="both", expand=True, padx=10, pady=10)
            for v_name in v_show:
                f = tk.Frame(cf, bg="white", highlightbackground="#bdc3c7", highlightthickness=1); f.pack(fill="x", padx=10, pady=3)
                tk.Label(f, text=v_name, font=("Helvetica", 9), bg="white").pack(side="left", padx=5)
                sv = tk.StringVar(value="0"); tk.Label(f, textvariable=sv, font=("Courier", 10), bg="white").pack(side="right", padx=5)
                self.tele_vars[key][v_name] = sv
        
        self.tab_settings = ttk.Frame(self.tabs); self.tabs.add(self.tab_settings, text=" ⚙ Machine Configuration ")
        sc = tk.Frame(self.tab_settings, bg="#ecf0f1"); sc.pack(fill="both", expand=True, padx=100, pady=50)
        rf = tk.LabelFrame(sc, text=" Reference Setup ", bg="white", font=("Helvetica", 11, "bold"), padx=20, pady=10); rf.pack(fill="x")
        for choice in ["Left", "Right"]: tk.Radiobutton(rf, text=choice, variable=self.fixed_side, value=choice, bg="white").pack(side="left", padx=20)
        def create_sc(parent, label, var, c, min_v, max_v):
            f = tk.LabelFrame(parent, text=f" {label} ", bg="white", font=("Helvetica", 10), fg=c, padx=20, pady=5); f.pack(fill="x", pady=5)
            tk.Scale(f, from_=min_v, to=max_v, resolution=0.1, orient="horizontal", variable=var, bg="white", length=400).pack(side="left", padx=20)
            tk.Entry(f, textvariable=var, width=8).pack(side="left")
        create_sc(sc, "Upper Pair Size", self.size_top, "#e67e22", 10, 200)
        create_sc(sc, "Bottom Pair Size", self.size_bot, "#9b59b6", 10, 200)
        create_sc(sc, "Agnostic Rotation Threshold", self.rot_threshold, "#34495e", 0, 90)

    def load_calib(self):
        target = "camera_params_2.npz" if self.fixed_side.get() == "Right" else "camera_params.npz"
        if os.path.exists(target): d = np.load(target); return d['camera_matrix'], d['dist_coeff']
        return np.array([[1280, 0, 640], [0, 1280, 360], [0, 0, 1]], dtype=np.float32), np.zeros(5)

    def measurement_loop(self):
        try:
            from picamera2 import Picamera2
            pc = Picamera2(); pc.configure(pc.create_video_configuration(main={"size": RESOLUTION, "format": "RGB888"})); pc.start()
        except: return
        detector = cv.aruco.ArucoDetector(cv.aruco.getPredefinedDictionary(ARUCO_DICT), cv.aruco.DetectorParameters())
        log.init_log(); buffers = {"top": [], "bottom": []}; l_s = l_u = time.time() * 1000

        while self.is_running:
            K, dist = self.load_calib()
            try: frame = cv.cvtColor(pc.capture_array(), cv.COLOR_RGB2BGR)
            except: continue
            corners, ids, _ = detector.detectMarkers(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
            curr = time.time() * 1000

            if ids is not None and len(ids) >= 1:
                # Reorder each marker's corners to Camera Frame [TL, TR, BR, BL]
                # This is fully rotation-agnostic - we ignore ArUco's default corner order
                m_data = []
                for i in range(len(ids)):
                    c = corners[i][0]
                    idx_x = np.argsort(c[:, 0]); lp = c[idx_x[:2]]; rp = c[idx_x[2:]]
                    tl = lp[np.argmin(lp[:, 1])]; bl = lp[np.argmax(lp[:, 1])]
                    tr = rp[np.argmin(rp[:, 1])]; br = rp[np.argmax(rp[:, 1])]
                    cy = (tl[1]+tr[1]+br[1]+bl[1])/4.0
                    cx = (tl[0]+tr[0]+br[0]+bl[0])/4.0
                    m_data.append({"c": np.array([tl, tr, br, bl], dtype=np.float32), "y": cy, "x": cx})

                # --- Y-RANK PAIRING ---
                # Sort ALL markers by Y ascending (smaller Y = higher on screen = top)
                # Top pair  = 2 markers with the SMALLEST Y values
                # Bottom pair = 2 markers with the LARGEST Y values
                # Each pair is evaluated independently — missing one => only that pair clears
                m_data.sort(key=lambda m: m["y"])
                top_m = m_data[:2]                              # 2 highest markers (smallest Y)
                bot_m = m_data[2:4] if len(m_data) >= 4 else [] # 2 lowest markers (largest Y)
                # Edge case: only 2 markers total
                # Determine if they are side-by-side (same pair) or vertically separated (one each)
                if len(m_data) == 2:
                    dy = abs(m_data[0]["y"] - m_data[1]["y"])
                    dx = abs(m_data[0]["x"] - m_data[1]["x"])
                    if dx > dy:
                        # Side-by-side → same pair; assign by average Y
                        avg_y = (m_data[0]["y"] + m_data[1]["y"]) / 2
                        top_m = m_data if avg_y < RESOLUTION[1]/2 else []
                        bot_m = [] if avg_y < RESOLUTION[1]/2 else m_data
                    else:
                        # Vertically separated → one from each pair (both incomplete)
                        top_m = [m_data[0]]   # upper one → incomplete top pair
                        bot_m = [m_data[1]]   # lower one → incomplete bottom pair
                elif len(m_data) == 3:
                    # 3 markers: top 2 (by Y) form top pair, bottom 1 is incomplete
                    top_m = m_data[:2]
                    bot_m = m_data[2:]

                def proc(marker_list, key, size):
                    # Need exactly 2 markers per pair - clear if broken
                    if len(marker_list) < 2:
                        self.last_data[key]["dist"] = 0.0
                        self.last_data[key]["p1_px"] = None  # clear pixel draw points
                        return
                    # Pick the two horizontally-closest opposite markers (leftmost and rightmost)
                    marker_list.sort(key=lambda m: m["x"])
                    left_m, right_m = marker_list[0], marker_list[-1]  # Always use outermost pair
                    is_rf = (self.fixed_side.get() == "Right")
                    S_m = right_m if is_rf else left_m   # Fixed = Source
                    T_m = left_m  if is_rf else right_m  # Moving = Target

                    def get_full(c2d):
                        h = size / 2.0; obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)
                        _, rv, tv = cv.solvePnP(obj, c2d, K, dist); R, _ = cv.Rodrigues(rv)
                        pts3d = np.array([np.dot(R, pt) + tv.ravel() for pt in obj])
                        # TL→TR pixel edge vector (always left-to-right after our corner re-sort)
                        dy, dx = c2d[1, 1]-c2d[0, 1], c2d[1, 0]-c2d[0, 0]
                        rot2d = math.degrees(math.atan2(dy, dx))   # Signed slope of top edge
                        inplane_rot = abs(rot2d)                    # 0-45° in-plane rotation
                        roll = math.degrees(math.atan2(R[1,0], R[0,0]))  # 3D roll (reference)
                        z_depth = float(tv[2])                     # Z distance camera→marker (mm)
                        return pts3d, rot2d, inplane_rot, roll, z_depth

                    # Returns: pts3d, rot2d (signed), inplane_rot (0-45°), roll (3D), z_depth (mm)
                    S_pts, S_rot2d, S_irot, S_roll, S_z = get_full(S_m["c"])
                    T_pts, T_rot2d, T_irot, T_roll, T_z = get_full(T_m["c"])

                    # Inner edge of Source (facing the gap)
                    # Left-fixed: source=left marker, inner edge = its RIGHT side (c[1]=TR, c[2]=BR)
                    # Right-fixed: source=right marker, inner edge = its LEFT side (c[0]=TL, c[3]=BL)
                    trl, brl = (S_pts[1], S_pts[2]) if not is_rf else (S_pts[0], S_pts[3])
                    # Inner edge of Target
                    # Left-fixed: target=right marker, left side (c[0]=TL, c[3]=BL)
                    # Right-fixed: target=left marker, right side (c[1]=TR, c[2]=BR)
                    trr, brr = (T_pts[0], T_pts[3]) if not is_rf else (T_pts[1], T_pts[2])
                    # 2D pixel inner-edge midpoints for screen line (always exact to marker edge)
                    p_src_2d = tuple(((S_m["c"][1]+S_m["c"][2])/2).astype(int)) if not is_rf else tuple(((S_m["c"][0]+S_m["c"][3])/2).astype(int))
                    p_tgt_2d = tuple(((T_m["c"][0]+T_m["c"][3])/2).astype(int)) if not is_rf else tuple(((T_m["c"][1]+T_m["c"][2])/2).astype(int))
                    buffers[key].append({"A":(trl+brl)/2, "X_alt":(trr+brr)/2, "TR":trl, "BR":brl, "B":trr, "C":brr,
                                         "L_A":(S_roll, S_irot), "R_A":(T_roll, T_irot),
                                         "rot": max(abs(S_rot2d), abs(T_rot2d)),
                                         "L_z": S_z, "R_z": T_z,
                                         "p1_px": p_src_2d, "p2_px": p_tgt_2d})

                if (curr - l_s) >= 100: proc(top_m, "top", self.size_top.get()); proc(bot_m, "bottom", self.size_bot.get()); l_s = curr

                if (curr - l_u) >= 500:
                    for k in ["top", "bottom"]:
                        if buffers[k]:
                            s = buffers[k]; aA, aTR, aBR, aX_alt, aB, aC = [np.mean([x[j] for x in s], axis=0) for j in ["A", "TR", "BR", "X_alt", "B", "C"]]
                            aL, aR = [np.mean([x[j] for x in s], axis=0) for j in ["L_A", "R_A"]]; avg_rot = np.mean([x["rot"] for x in s])
                            # Pixel draw points: average of last buffer samples
                            p1_px = tuple(np.mean([x["p1_px"] for x in s], axis=0).astype(int))
                            p2_px = tuple(np.mean([x["p2_px"] for x in s], axis=0).astype(int))
                            if avg_rot > self.rot_threshold.get():
                                # HIGH ROTATION FALLBACK (> threshold):
                                # Marker too tilted for reliable ray-intersection.
                                # Use midpoint of target inner edge directly as X.
                                aX = aX_alt
                                dv = np.linalg.norm(aX - aA)
                                kv = 0.5
                            else:
                                # PRIMARY METHOD — v8 perpendicular ray-intersection:
                                # v_vec: unit direction down the source marker's inner edge (3D TR→BR)
                                inner_edge_vec = aBR - aTR
                                inner_len = np.linalg.norm(inner_edge_vec)
                                v = inner_edge_vec / inner_len if inner_len > 0 else np.zeros(3)
                                # w: target inner edge direction (B→C)
                                # u: vector from A to B
                                w, u = aC - aB, aB - aA
                                den = (np.dot(v,v)*np.dot(w,w)) - (np.dot(v,w)**2)
                                if abs(den) > 1e-6:
                                    kv = ((np.dot(v,w)*np.dot(u,v)) - (np.dot(v,v)*np.dot(u,w))) / den
                                    # --- CLAMP k to [0, 1] ---
                                    # k outside this range means X is extrapolated beyond the
                                    # physical marker edge, which causes wildly wrong distances.
                                    # Clamping keeps X on the actual B-C segment at all times.
                                    kv = float(np.clip(kv, 0.0, 1.0))
                                    aX = aB + kv * w
                                    dv = np.linalg.norm(aX - aA)
                                    # Sanity check: if distance is impossibly large
                                    # (> 3x the target edge length), geometry is degenerate.
                                    # Fall back to midpoint to prevent wild readings.
                                    edge_len = np.linalg.norm(w)
                                    if dv > max(edge_len * 3.0, 10.0):
                                        aX = aX_alt; dv = np.linalg.norm(aX - aA); kv = 0.5
                                else:
                                    # Parallel edges — use midpoint
                                    aX = aX_alt; dv = np.linalg.norm(aX - aA); kv = 0.5
                            avg_Lz = float(np.mean([x["L_z"] for x in s]))
                            avg_Rz = float(np.mean([x["R_z"] for x in s]))
                            self.last_data[k].update({"A":aA, "X":aX, "TR":aTR, "BR":aBR, "B":aB, "C":aC, "dist":dv, "k":kv, "rot_2d":avg_rot, "L_A":aL, "R_A":aR, "L_z":avg_Lz, "R_z":avg_Rz, "p1_px":p1_px, "p2_px":p2_px})
                            self.last_data["session_count"] += 1; buffers[k].clear()
                    l_u = curr
            else:
                self.last_data["top"]["dist"] = 0.0
                self.last_data["bottom"]["dist"] = 0.0

            for k, color in [("top", (0, 165, 255)), ("bottom", (255, 0, 255))]:
                d = self.last_data[k]
                # Only draw if valid pair was detected and pixel coords exist
                if d["dist"] > 0 and d.get("p1_px") is not None and d.get("p2_px") is not None:
                    p1, p2 = d["p1_px"], d["p2_px"]
                    cv.line(frame, p1, p2, color, 3)
                    cv.circle(frame, p2, 6, (0, 255, 0), -1)
                    cv.putText(frame, f"{k.upper()}: {d['dist']:.2f}mm", (p1[0], p1[1]-12), 0, 0.6, color, 2)
                    # L_A[1] = in-plane rotation of fixed (source) marker (0-45°)
                    # R_A[1] = in-plane rotation of moving (target) marker (0-45°)
                    # Warn if EITHER marker is rotated beyond the threshold
                    max_rot = max(d["L_A"][1], d["R_A"][1])
                    if max_rot > self.rot_threshold.get():
                        cv.rectangle(frame, (p1[0]-90, p1[1]-85), (p1[0]+60, p1[1]-50), (0,0,255), -1)
                        cv.putText(frame, f"ROT: {max_rot:.1f}deg", (p1[0]-85, p1[1]-60), 0, 0.55, (0,255,255), 2)
            self.current_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    def update_gui_loop(self):
        if self.current_frame is not None:
            img = ImageTk.PhotoImage(image=Image.fromarray(self.current_frame).resize((960, 540)))
            self.canvas.create_image(0, 0, anchor="nw", image=img); self.canvas.img = img
        self.lbl_dist_top.config(text=f"{self.last_data['top']['dist']:.3f} mm")
        self.lbl_dist_bot.config(text=f"{self.last_data['bottom']['dist']:.3f} mm")
        for key in ["top", "bottom"]:
            d = self.last_data[key]
            for var in ["A", "X", "TR", "BR", "B", "C"]: self.tele_vars[key][var].set(f"{d[var][0]:.1f}, {d[var][1]:.1f}, {d[var][2]:.1f}")
            self.tele_vars[key]["L_ROL"].set(f"{d['L_A'][0]:.2f}° (3D roll)")
            self.tele_vars[key]["L_ROT"].set(f"{d['L_A'][1]:.2f}° (in-plane)")
            self.tele_vars[key]["L_Z"].set(f"{d['L_z']:.1f} mm")
            self.tele_vars[key]["R_ROL"].set(f"{d['R_A'][0]:.2f}° (3D roll)")
            self.tele_vars[key]["R_ROT"].set(f"{d['R_A'][1]:.2f}° (in-plane)")
            self.tele_vars[key]["R_Z"].set(f"{d['R_z']:.1f} mm")
        self.root.after(50, self.update_gui_loop)

if __name__ == "__main__":
    root = tk.Tk(); app = MeasurementApp(root); root.mainloop()
