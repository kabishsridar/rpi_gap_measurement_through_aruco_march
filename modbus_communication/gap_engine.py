import cv2 as cv
import numpy as np
import time
import math
import os
from datetime import datetime

"""
ARUCO MARKER NAMING CONVENTION (Updated):

Position-based naming for 8 markers:
- LT1, LT2: Left Top markers (Left side, Top position)
- LB1, LB2: Left Bottom markers (Left side, Bottom position)
- RT1, RT2: Right Top markers (Right side, Top position)
- RB1, RB2: Right Bottom markers (Right side, Bottom position)

This enables clear references to:
- "L1-L2 pair": Left side measurement (LT + LB markers combined)
- "R1-R2 pair": Right side measurement (RT + RB markers combined)

Legacy variables in code still use "top"/"bottom" for compatibility,
but semantically now represent Left/Right side measurements.
"""

# Import constants (Assuming constants.py exists in formulated)
# We will define missing ones here for the simulation/project to be standalone
RESOLUTION = (1280, 720)
ARUCO_DICT = cv.aruco.DICT_4X4_50

def run_gap_engine(shared_data, config):
    """
    STRICT Port of legacy math engine (v16) with updated position-based naming.

    Uses new ArUco marker naming convention:
    - LT1/LT2, LB1/LB2 (Left side markers)
    - RT1/RT2, RB1/RB2 (Right side markers)

    Groups into "L1-L2 pair" (Left side) and "R1-R2 pair" (Right side) for clearer
    semantic meaning than the legacy "Top pair"/"Bottom pair" terminology.

    Headless version: No GUI, updates Shared Dictionary instead.
    """
    print("[GAP ENGINE] Initializing Camera...")
    
    try:
        from picamera2 import Picamera2
        pc = Picamera2()
        pc.configure(pc.create_video_configuration(main={"size": RESOLUTION, "format": "RGB888"}))
        pc.start()
    except Exception as e:
        print(f"[GAP ENGINE] FAILED to start Camera: {e}")
        shared_data["error_code"] = 99 # Camera Error
        return

    detector = cv.aruco.ArucoDetector(
        cv.aruco.getPredefinedDictionary(ARUCO_DICT),
        cv.aruco.DetectorParameters())
    
    # Buffers for averaging (v16 logic)
    b = {"top": [], "bottom": []}
    l_s = l_u = time.time() * 1000

    def load_calib(fixed_side):
        f = os.path.join("..", "calibration", "camera_params_2.npz" if fixed_side == "Right" else "camera_params.npz")
        if os.path.exists(f): 
            d = np.load(f)
            k_m = 'camera_matrix' if 'camera_matrix' in d else 'mtx'
            k_d = 'dist_coeff' if 'dist_coeff' in d else 'dist'
            return d[k_m], d[k_d]
        return np.array([[1280,0,640],[0,1280,360],[0,0,1]], dtype=np.float32), np.zeros(5)

    print("[GAP ENGINE] Engine Loop Started.")
    
    while True:
        # Load calibration based on current config
        K, dist_c = load_calib(config.get("fixed_side", "Left"))
        
        try:
            # Capture frame
            frame_raw = pc.capture_array()
            frame = cv.cvtColor(frame_raw, cv.COLOR_RGB2BGR)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            mean_b = np.mean(gray)
            shared_data["brightness"] = int(mean_b)
        except Exception:
            continue

        corners_raw, ids, _ = detector.detectMarkers(gray)
        curr = time.time() * 1000

        if ids is not None and len(ids) >= 1:
            m_data = []
            for i in range(len(ids)):
                raw = corners_raw[i][0]
                idx_x = np.argsort(raw[:, 0])
                lp, rp = raw[idx_x[:2]], raw[idx_x[2:]]
                tl, bl = lp[np.argmin(lp[:, 1])], lp[np.argmax(lp[:, 1])]
                tr, br = rp[np.argmin(rp[:, 1])], rp[np.argmax(rp[:, 1])]
                m_data.append({"c": np.array([tl, tr, br, bl], dtype=np.float32), "c_raw": raw, "y": np.mean(raw[:, 1]), "x": np.mean(raw[:, 0])})

            m_data.sort(key=lambda m: m["y"])
            top_m = m_data[:2]
            bot_m = m_data[2:4] if len(m_data) >= 4 else []
            
            # v16 Sorting logic
            if len(m_data) == 2:
                dy, dx = abs(m_data[0]["y"]-m_data[1]["y"]), abs(m_data[0]["x"]-m_data[1]["x"])
                if dx > dy: 
                    top_m = m_data if np.mean([m["y"] for m in m_data]) < RESOLUTION[1]/2 else []
                    bot_m = m_data if not top_m else []
                else: 
                    top_m = [m_data[0]]; bot_m = [m_data[1]]
            elif len(m_data) == 3:
                top_m = m_data[:2]; bot_m = m_data[2:]

            def process_pair(marker_list, key, size):
                if len(marker_list) < 2:
                    return {"detected": False}
                
                marker_list.sort(key=lambda m: m["x"])
                is_rf = (config.get("fixed_side", "Left") == "Right")
                S_m = marker_list[-1] if is_rf else marker_list[0]
                T_m = marker_list[0] if is_rf else marker_list[-1]
                h = size / 2.0
                obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)

                def get_pose(c):
                    _, rv, tv = cv.solvePnP(obj, c, K, dist_c)
                    R_m, _ = cv.Rodrigues(rv)
                    sy = math.sqrt(R_m[0,0]**2 + R_m[1,0]**2)
                    if sy >= 1e-6:
                        pit = math.degrees(math.atan2(-R_m[2,0], sy))
                        yaw = math.degrees(math.atan2(R_m[2,1], R_m[2,2]))
                        rol = math.degrees(math.atan2(R_m[1,0], R_m[0,0]))
                    else:
                        pit = math.degrees(math.atan2(-R_m[2,0], sy))
                        yaw = 0
                        rol = math.degrees(math.atan2(-R_m[0,1], R_m[1,1]))
                    return np.array([R_m @ pt + tv.ravel() for pt in obj]), rol, pit, yaw, float(tv[2])

                S_pts, S_r, S_p, S_y, S_z = get_pose(S_m["c_raw"])
                T_pts, T_r, T_p, T_y, T_z = get_pose(T_m["c_raw"])
                
                def identify_inner_outer(pts, is_right_fixed):
                    o = pts[:, 0].argsort()
                    inner = pts[o[2:]] if not is_right_fixed else pts[o[:2]]
                    outer = pts[o[:2]] if not is_right_fixed else pts[o[2:]]
                    return inner[inner[:,1].argsort()][0], inner[inner[:,1].argsort()][1], outer[outer[:,1].argsort()][0]

                S_it, S_ib, S_ot = identify_inner_outer(S_pts, is_rf)
                T_it, T_ib, _ = identify_inner_outer(T_pts, not is_rf)
                
                si = abs(math.degrees(math.atan2(S_m["c"][1,1]-S_m["c"][0,1], S_m["c"][1,0]-S_m["c"][0,0])))
                ti = abs(math.degrees(math.atan2(T_m["c"][1,1]-T_m["c"][0,1], T_m["c"][1,0]-T_m["c"][0,0])))
                
                return {
                    "detected": True,
                    "A": (S_it+S_ib)/2.0, "XB": (T_it+T_ib)/2.0, "TR": S_it, "TL": S_ot, "BR": S_ib, "B": T_it, "C": T_ib,
                    "L_euler": (S_r, S_p, S_y, S_z), "R_euler": (T_r, T_p, T_y, T_z), "rot_2d": max(si, ti)
                }

            # 100ms processing interval for accumulation
            if (curr - l_s) >= 100:
                res_t = process_pair(top_m, "top", config.get("marker_size_top", 100.0))
                res_b = process_pair(bot_m, "bottom", config.get("marker_size_bot", 100.0))
                if res_t["detected"]: b["top"].append(res_t)
                if res_b["detected"]: b["bottom"].append(res_b)
                l_s = curr

            # 500ms update interval for final result push
            if (curr - l_u) >= 500:
                error_flags = 0
                for k in ["top", "bottom"]:
                    if b[k]:
                        s = b[k]
                        aA = np.mean([x["A"] for x in s], 0)
                        aTR = np.mean([x["TR"] for x in s], 0)
                        aTL = np.mean([x["TL"] for x in s], 0)
                        aB = np.mean([x["B"] for x in s], 0)
                        aC = np.mean([x["C"] for x in s], 0)
                        aXB = np.mean([x["XB"] for x in s], 0)
                        
                        vr = aTR-aTL
                        v = vr/np.linalg.norm(vr) if np.linalg.norm(vr)>0 else np.zeros(3)
                        w = aC-aB
                        u = aB-aA
                        den = np.dot(v,v)*np.dot(w,w)-np.dot(v,w)**2
                        
                        def _perp():
                            if abs(den)>1e-6:
                                kv = np.clip((np.dot(v,w)*np.dot(u,v)-np.dot(v,v)*np.dot(u,w))/den, 0, 1)
                                return np.linalg.norm((aB+kv*w)-aA), kv
                            return np.linalg.norm(aXB-aA), 0.5
                        
                        if np.mean([x["rot_2d"] for x in s]) > config.get("rot_threshold", 5.0):
                            dv, kv = np.linalg.norm(aXB-aA), 0.5
                        else:
                            dv, kv = _perp()
                        
                        # Update Shared Dictionary
                        shared_data[f"{k}_dist"] = float(dv)
                        shared_data[f"{k}_k"] = float(kv)
                        
                        # Warning logic
                        L_avg = np.mean([x["L_euler"] for x in s], 0)
                        R_avg = np.mean([x["R_euler"] for x in s], 0)
                        
                        # If excessive tilt, flag error
                        if abs(L_avg[1]) > config.get("pitch_threshold", 6.0) or abs(R_avg[1]) > config.get("pitch_threshold", 6.0):
                            error_flags |= 2 # Pitch Error
                        
                        b[k].clear()
                    else:
                        shared_data[f"{k}_dist"] = 0.0
                        error_flags |= 1 # Marker Missing Error
                
                shared_data["error_code"] = error_flags
                l_u = curr
        else:
            # Map to new semantic naming: top_dist = Left side (L1-L2 pair), bottom_dist = Right side (R1-R2 pair)
            shared_data["top_dist"] = 0.0    # Left side (L1-L2 pair) distance
            shared_data["bottom_dist"] = 0.0 # Right side (R1-R2 pair) distance
            shared_data["error_code"] = 1 # No Markers Found
        
        # CPU Throttling
        time.sleep(0.01)
