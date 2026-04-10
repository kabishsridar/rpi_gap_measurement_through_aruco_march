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
        self.root.title("Dual-Pair ArUco Measurement v11")
        self.root.geometry("1500x950")
        self.root.configure(bg="#2c3e50")

        self.size_top = tk.DoubleVar(value=DEFAULT_MARKER_SIZE)
        self.size_bot = tk.DoubleVar(value=DEFAULT_MARKER_SIZE)
        self.fixed_side = tk.StringVar(value="Left")
        self.rot_threshold = tk.DoubleVar(value=7.0)

        temp = {"A": (0,0,0), "X": (0,0,0), "TR": (0,0,0), "BR": (0,0,0),
                "B": (0,0,0), "C": (0,0,0), "dist": 0.0, "k": 0.0,
                "L_A": (0.0, 0.0), "R_A": (0.0, 0.0), "rot_2d": 0.0,
                "L_z": 0.0, "R_z": 0.0, "p1_px": None, "p2_px": None}
        self.last_data = {"top": temp.copy(), "bottom": temp.copy(), "session_count": 0}
        self.is_running = True
        self.current_frame = None

        ttk.Style().theme_use('clam')
        self.setup_ui()
        threading.Thread(target=self.measurement_loop, daemon=True).start()
        self.update_gui_loop()

    # ─────────────────────────────── UI ────────────────────────────────
    def setup_ui(self):
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill='both', expand=True, padx=10, pady=10)

        # ── TAB 1: LIVE MONITOR ──
        self.tab_live = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_live, text=" 📽 Live Monitor ")
        self.canvas = tk.Canvas(self.tab_live, width=960, height=540, bg="black")
        self.canvas.pack(side="left", padx=20, pady=20)
        st = tk.Frame(self.tab_live, bg="#ecf0f1")
        st.pack(side="right", fill="both", expand=True, padx=10, pady=20)

        for key, title, color in [("top", "TOP PAIR", "#e67e22"),
                                   ("bottom", "BOTTOM PAIR", "#9b59b6")]:
            f = tk.LabelFrame(st, text=f" {title} ", font=("Helvetica", 12, "bold"),
                              bg="#ecf0f1", fg=color)
            f.pack(fill="x", pady=10, padx=10)
            dl = tk.Label(f, text="0.000 mm", font=("Helvetica", 24, "bold"),
                          fg="#27ae60", bg="#ecf0f1")
            dl.pack(pady=10)
            kl = tk.Label(f, text="k: 0.0000", font=("Helvetica", 11), bg="#ecf0f1")
            kl.pack()
            if key == "top":
                self.lbl_dist_top, self.lbl_k_top = dl, kl
            else:
                self.lbl_dist_bot, self.lbl_k_bot = dl, kl

        # ── TAB 2: DUAL TELEMETRY ──
        self.tab_tele = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_tele, text=" 🛸 Dual Telemetry ")
        mtf = tk.Frame(self.tab_tele, bg="#ecf0f1")
        mtf.pack(fill="both", expand=True)
        self.tele_vars = {"top": {}, "bottom": {}}
        v_show = ["A", "X", "TR", "BR", "B", "C",
                  "L_ROL", "L_ROT", "L_Z", "R_ROL", "R_ROT", "R_Z"]
        for key, title, color in [("top", "TOP", "#e67e22"), ("bottom", "BOTTOM", "#9b59b6")]:
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

        # ── TAB 3: MACHINE CONFIGURATION ──
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

        create_slider(sc, "Upper Pair Marker Size (mm)", self.size_top, "#e67e22", 10, 200)
        create_slider(sc, "Bottom Pair Marker Size (mm)", self.size_bot, "#9b59b6", 10, 200)
        create_slider(sc, "Rotation Threshold °  (fallback trigger)",
                      self.rot_threshold, "#34495e", 0, 45)

    # ──────────────────────────── CALIBRATION ──────────────────────────
    def load_calib(self):
        f = "camera_params_2.npz" if self.fixed_side.get() == "Right" else "camera_params.npz"
        if os.path.exists(f):
            d = np.load(f)
            return d['camera_matrix'], d['dist_coeff']
        return (np.array([[1280,0,640],[0,1280,360],[0,0,1]], dtype=np.float32),
                np.zeros(5))

    # ─────────────────────────── MEASUREMENT ───────────────────────────
    def measurement_loop(self):
        try:
            from picamera2 import Picamera2
            pc = Picamera2()
            pc.configure(pc.create_video_configuration(
                main={"size": RESOLUTION, "format": "RGB888"}))
            pc.start()
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

            # ── Preprocessing for harsh lighting (overexposed backgrounds) ──
            # CLAHE (Contrast Limited Adaptive Histogram Equalization):
            # Divides image into tiles and equalizes each locally.
            # This recovers the black/white marker borders even when the
            # global exposure is blown out by bright windows/skylights.
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_eq = clahe.apply(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))

            corners_raw, ids, _ = detector.detectMarkers(gray_eq)
            curr = time.time() * 1000

            if ids is not None and len(ids) >= 1:

                # ── Build marker list ──────────────────────────────────────────
                # For EACH marker we store:
                #   c     : pixel-sorted corners [TL,TR,BR,BL] – for display & rot2d
                #   c_raw : raw ArUco corners – for solvePnP (preserves physical identity)
                #   y, x  : pixel centre – for pairing
                m_data = []
                for i in range(len(ids)):
                    raw = corners_raw[i][0]          # default ArUco corner order
                    idx_x = np.argsort(raw[:, 0])
                    lp = raw[idx_x[:2]];  rp = raw[idx_x[2:]]
                    tl = lp[np.argmin(lp[:, 1])];  bl = lp[np.argmax(lp[:, 1])]
                    tr = rp[np.argmin(rp[:, 1])];  br = rp[np.argmax(rp[:, 1])]
                    cy = (tl[1]+tr[1]+br[1]+bl[1]) / 4.0
                    cx = (tl[0]+tr[0]+br[0]+bl[0]) / 4.0
                    m_data.append({
                        "c":     np.array([tl, tr, br, bl], dtype=np.float32),
                        "c_raw": raw,          # ← raw corners for correct solvePnP
                        "y": cy, "x": cx})

                # ── Y-rank pairing ─────────────────────────────────────────────
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

                # ── Process one pair ───────────────────────────────────────────
                def proc(marker_list, key, size):
                    if len(marker_list) < 2:
                        self.last_data[key]["dist"] = 0.0
                        self.last_data[key]["p1_px"] = None
                        return

                    marker_list.sort(key=lambda m: m["x"])
                    left_m  = marker_list[0]
                    right_m = marker_list[-1]
                    is_rf   = (self.fixed_side.get() == "Right")
                    S_m = right_m if is_rf else left_m   # Source = Fixed
                    T_m = left_m  if is_rf else right_m  # Target = Moving

                    h = size / 2.0
                    # Object points: physical marker corners in marker-local space.
                    # Convention (x-right, y-up, z=0):
                    #   obj[0]=(-h,+h,0)=TL  obj[1]=(+h,+h,0)=TR
                    #   obj[2]=(+h,-h,0)=BR  obj[3]=(-h,-h,0)=BL
                    obj = np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]],
                                   dtype=np.float32)

                    def get_pose(c_raw):
                        """Use RAW ArUco corners for correct physical pose (like v8)."""
                        _, rv, tv = cv.solvePnP(obj, c_raw, K, dist_c)
                        R, _ = cv.Rodrigues(rv)
                        pts3d = np.array([R @ pt + tv.ravel() for pt in obj])
                        roll    = math.degrees(math.atan2(R[1,0], R[0,0]))
                        z_depth = float(tv[2])
                        return pts3d, roll, z_depth

                    def inplane_rot(c_sorted):
                        """In-plane rotation from pixel-sorted TL→TR edge angle."""
                        dy_e = c_sorted[1,1] - c_sorted[0,1]
                        dx_e = c_sorted[1,0] - c_sorted[0,0]
                        return math.degrees(math.atan2(dy_e, dx_e))   # signed

                    S_pts, S_roll, S_z = get_pose(S_m["c_raw"])
                    T_pts, T_roll, T_z = get_pose(T_m["c_raw"])
                    S_rot2d = inplane_rot(S_m["c"])
                    T_rot2d = inplane_rot(T_m["c"])
                    S_irot  = abs(S_rot2d)   # 0–45° (square 4-fold symmetry)
                    T_irot  = abs(T_rot2d)

                    # ── v8-style 3D inner/outer edge selection ─────────────────
                    # Sort 3D points by their camera-space X-coordinate to find
                    # which corners are on the gap-facing (inner) side vs outer side.
                    # This is rotation-agnostic: no assumption about which obj index
                    # corresponds to which physical corner.
                    #
                    # Left marker  → inner edge = 2 rightmost X (right side faces gap)
                    # Right marker → inner edge = 2 leftmost  X (left side faces gap)

                    def inner_outer(pts, inner_is_right):
                        """
                        Returns (inner_top, inner_bot, outer_top) in camera 3D space.
                        inner_is_right=True  → pick 2 pts with largest X as inner edge
                        inner_is_right=False → pick 2 pts with smallest X as inner edge
                        Camera Y: smaller Y = higher in image = 'top'
                        """
                        order = pts[:, 0].argsort()       # X ascending
                        if inner_is_right:
                            inner = pts[order[2:]]        # 2 rightmost
                            outer = pts[order[:2]]        # 2 leftmost
                        else:
                            inner = pts[order[:2]]        # 2 leftmost
                            outer = pts[order[2:]]        # 2 rightmost
                        inner = inner[inner[:, 1].argsort()]  # sort by Y: [top, bot]
                        outer = outer[outer[:, 1].argsort()]
                        return inner[0], inner[1], outer[0]  # inner_top, inner_bot, outer_top

                    # Source: left marker faces right (inner=right); right marker faces left (inner=left)
                    S_inner_top, S_inner_bot, S_outer_top = inner_outer(S_pts, not is_rf)
                    A = (S_inner_top + S_inner_bot) / 2.0   # midpoint of source inner edge

                    # Target: opposite alignment from source
                    T_inner_top, T_inner_bot, _ = inner_outer(T_pts, is_rf)
                    B = T_inner_top    # top of target inner edge  (v8's b)
                    C = T_inner_bot    # bottom of target inner edge (v8's c)
                    X_alt = (B + C) / 2.0   # midpoint – high-rotation fallback

                    # ── 2D pixel inner-edge midpoints for on-screen line ───────
                    if not is_rf:
                        p_src_2d = tuple(((S_m["c"][1]+S_m["c"][2])/2).astype(int))
                        p_tgt_2d = tuple(((T_m["c"][0]+T_m["c"][3])/2).astype(int))
                    else:
                        p_src_2d = tuple(((S_m["c"][0]+S_m["c"][3])/2).astype(int))
                        p_tgt_2d = tuple(((T_m["c"][1]+T_m["c"][2])/2).astype(int))

                    buffers[key].append({
                        "A":      A,
                        "X_alt":  X_alt,
                        "TR":     S_inner_top,   # inner-top of source (for v_vec)
                        "BR":     S_inner_bot,   # inner-bot of source
                        "TL_ref": S_outer_top,   # outer-top of source (for v_vec)
                        "B":  B,  "C":  C,
                        "L_A":  (S_roll, S_irot),
                        "R_A":  (T_roll, T_irot),
                        "rot":  max(S_irot, T_irot),   # max in-plane rot (for threshold)
                        "L_z":  S_z,   "R_z":  T_z,
                        "p1_px": p_src_2d,
                        "p2_px": p_tgt_2d,
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
                                # ── FALLBACK: marker too rotated ──────────────
                                # Use midpoint of target inner edge directly as X
                                aX = aX_alt
                                dv = np.linalg.norm(aX - aA)
                                kv = 0.5
                            else:
                                # ── PRIMARY: v8 exact perpendicular formula ───
                                #
                                # v_vec = direction across source from outer-top → inner-top
                                #       = (aTR - aTL_outer) / |...|
                                # This points INTO the gap (perpendicular to gap plane).
                                #
                                # w_vec = target inner edge direction B→C (top→bottom)
                                # u_vec = A→B
                                #
                                # Solve: find k s.t. X = B + k*w is closest to ray A + t*v
                                #
                                v_raw = aTR - aTL
                                v_len = np.linalg.norm(v_raw)
                                v = v_raw / v_len if v_len > 0 else np.zeros(3)
                                w = aC - aB
                                u = aB - aA
                                den = (np.dot(v,v)*np.dot(w,w)) - (np.dot(v,w)**2)

                                if abs(den) > 1e-6:
                                    kv = ((np.dot(v,w)*np.dot(u,v)) -
                                          (np.dot(v,v)*np.dot(u,w))) / den
                                    # Clamp k → X stays on physical target edge [0,1]
                                    kv = float(np.clip(kv, 0.0, 1.0))
                                    aX = aB + kv * w
                                    dv = np.linalg.norm(aX - aA)
                                else:
                                    # Parallel edges – midpoint fallback
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

            # ── Draw lines on frame ────────────────────────────────────────────
            for key, color in [("top", (0, 165, 255)), ("bottom", (255, 0, 255))]:
                d = self.last_data[key]
                if (d["dist"] > 0
                        and d.get("p1_px") is not None
                        and d.get("p2_px") is not None):
                    p1 = d["p1_px"]   # 2D: midpoint of source inner edge

                    # Project aX (3D camera-space perpendicular intersection)
                    # to 2D screen. aX is already in camera space, so rvec/tvec = 0.
                    X_3d = np.array([d["X"]], dtype=np.float64).reshape(1, 1, 3)
                    if d["X"][2] > 0:
                        p_proj, _ = cv.projectPoints(X_3d, np.zeros(3), np.zeros(3),
                                                     K, dist_c)
                        p2 = tuple(p_proj[0].ravel().astype(int))
                    else:
                        p2 = d["p2_px"]   # fallback if X behind camera

                    cv.line(frame, p1, p2, color, 3)
                    cv.circle(frame, p2, 6, (0, 255, 0), -1)
                    cv.putText(frame,
                               f"{key.upper()}: {d['dist']:.2f}mm",
                               (p1[0], p1[1]-12), 0, 0.6, color, 2)
                    max_rot = max(d["L_A"][1], d["R_A"][1])
                    if max_rot > self.rot_threshold.get():
                        cv.rectangle(frame,
                                     (p1[0]-90, p1[1]-85), (p1[0]+60, p1[1]-50),
                                     (0, 0, 255), -1)
                        cv.putText(frame, f"ROT: {max_rot:.1f}deg",
                                   (p1[0]-85, p1[1]-60), 0, 0.55, (0,255,255), 2)

            self.current_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # ─────────────────────────── GUI REFRESH ───────────────────────────
    def update_gui_loop(self):
        if self.current_frame is not None:
            img = ImageTk.PhotoImage(
                image=Image.fromarray(self.current_frame).resize((960, 540)))
            self.canvas.create_image(0, 0, anchor="nw", image=img)
            self.canvas.img = img

        self.lbl_dist_top.config(text=f"{self.last_data['top']['dist']:.3f} mm")
        self.lbl_k_top.config(text=f"k: {self.last_data['top']['k']:.4f}")
        self.lbl_dist_bot.config(text=f"{self.last_data['bottom']['dist']:.3f} mm")
        self.lbl_k_bot.config(text=f"k: {self.last_data['bottom']['k']:.4f}")

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

        self.root.after(50, self.update_gui_loop)


if __name__ == "__main__":
    root = tk.Tk()
    app = MeasurementApp(root)
    root.mainloop()
