import cv2 as cv
import numpy as np
import time
import math
import os
from datetime import datetime
from constants import *
import log

def measurement_loop(app):
    """
    STRICT Port of v14.py's measurement logic.
    Ensures 1:1 mathematical parity with the version the user identified as 'accurate'.
    """
    try:
        from picamera2 import Picamera2
        pc = Picamera2()
        pc.configure(pc.create_video_configuration(main={"size": RESOLUTION, "format": "RGB888"}))
        pc.start()
        app.pc = pc
    except Exception:
        return

    detector = cv.aruco.ArucoDetector(
        cv.aruco.getPredefinedDictionary(ARUCO_DICT),
        cv.aruco.DetectorParameters())
    
    log.init_log()
    buffers = {"top": [], "bottom": []}
    l_s = l_u = time.time() * 1000

    while app.is_running:
        # load_calib is called every loop as in v14
        K, dist_c = app.load_calib()
        try:
            frame = cv.cvtColor(pc.capture_array(), cv.COLOR_RGB2BGR)
        except Exception:
            continue

        corners_raw, ids, _ = detector.detectMarkers(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
        curr = time.time() * 1000

        if ids is not None and len(ids) >= 1:
            m_data = []
            for i in range(len(ids)):
                raw = corners_raw[i][0]
                idx_x = np.argsort(raw[:, 0])
                lp = raw[idx_x[:2]]
                rp = raw[idx_x[2:]]
                tl = lp[np.argmin(lp[:, 1])]
                bl = lp[np.argmax(lp[:, 1])]
                tr = rp[np.argmin(rp[:, 1])]
                br = rp[np.argmax(rp[:, 1])]
                cy = (tl[1]+tr[1]+br[1]+bl[1]) / 4.0
                cx = (tl[0]+tr[0]+br[0]+bl[0]) / 4.0
                m_data.append({
                    "c": np.array([tl, tr, br, bl], dtype=np.float32),
                    "c_raw": raw, "y": cy, "x": cx
                })

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
                    top_m = [m_data[0]]
                    bot_m = [m_data[1]]
            elif len(m_data) == 3:
                top_m = m_data[:2]
                bot_m = m_data[2:]

            def proc(marker_list, key, size):
                if len(marker_list) < 2:
                    app.last_data[key]["dist"] = 0.0
                    app.last_data[key]["p1_px"] = None
                    app.last_data[key]["L_det"] = False
                    app.last_data[key]["R_det"] = False
                    return
                marker_list.sort(key=lambda m: m["x"])
                left_m = marker_list[0]
                right_m = marker_list[-1]
                is_rf = (app.fixed_side.get() == "Right")
                S_m = right_m if is_rf else left_m
                T_m = left_m if is_rf else right_m
                h = size / 2.0
                obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)

                def get_pose(c_raw):
                    _, rv, tv = cv.solvePnP(obj, c_raw, K, dist_c)
                    R, _ = cv.Rodrigues(rv)
                    pts3d = np.array([R @ pt + tv.ravel() for pt in obj])
                    return pts3d, math.degrees(math.atan2(R[1, 0], R[0, 0])), float(tv[2])

                def inplane_rot(c_sorted):
                    return math.degrees(math.atan2(
                        c_sorted[1, 1] - c_sorted[0, 1],
                        c_sorted[1, 0] - c_sorted[0, 0]))

                S_pts, S_roll, S_z = get_pose(S_m["c_raw"])
                T_pts, T_roll, T_z = get_pose(T_m["c_raw"])
                S_irot = abs(inplane_rot(S_m["c"]))
                T_irot = abs(inplane_rot(T_m["c"]))

                def inner_outer(pts, inner_is_right):
                    order = pts[:, 0].argsort()
                    inner = pts[order[2:]] if inner_is_right else pts[order[:2]]
                    outer = pts[order[:2]] if inner_is_right else pts[order[2:]]
                    inner = inner[inner[:, 1].argsort()]
                    outer = outer[outer[:, 1].argsort()]
                    return inner[0], inner[1], outer[0]

                S_inner_top, S_inner_bot, S_outer_top = inner_outer(S_pts, not is_rf)
                A = (S_inner_top + S_inner_bot) / 2.0
                T_inner_top, T_inner_bot, _ = inner_outer(T_pts, is_rf)
                B = T_inner_top
                C = T_inner_bot
                X_alt = (B + C) / 2.0

                if not is_rf:
                    p_src_2d = tuple(((S_m["c"][1]+S_m["c"][2])/2).astype(int))
                    p_tgt_2d = tuple(((T_m["c"][0]+T_m["c"][3])/2).astype(int))
                else:
                    p_src_2d = tuple(((S_m["c"][0]+S_m["c"][3])/2).astype(int))
                    p_tgt_2d = tuple(((T_m["c"][1]+T_m["c"][2])/2).astype(int))

                buffers[key].append({
                    "A": A, "X_alt": X_alt,
                    "TR": S_inner_top, "BR": S_inner_bot, "TL_ref": S_outer_top,
                    "B": B, "C": C,
                    "L_A": (S_roll, S_irot), "R_A": (T_roll, T_irot),
                    "rot": max(S_irot, T_irot),
                    "L_z": S_z, "R_z": T_z,
                    "p1_px": p_src_2d, "p2_px": p_tgt_2d,
                })
                app.last_data[key]["L_det"] = True
                app.last_data[key]["R_det"] = True

            if (curr - l_s) >= 100:
                proc(top_m, "top", app.size_top.get())
                proc(bot_m, "bottom", app.size_bot.get())
                l_s = curr

            if (curr - l_u) >= 1000:  # v14 strict 1-second window
                for key in ["top", "bottom"]:
                    if buffers[key]:
                        s = buffers[key]
                        aA      = np.mean([x["A"]      for x in s], axis=0)
                        aTR     = np.mean([x["TR"]     for x in s], axis=0)
                        aTL     = np.mean([x["TL_ref"] for x in s], axis=0)
                        aBR     = np.mean([x["BR"]     for x in s], axis=0)
                        aB      = np.mean([x["B"]      for x in s], axis=0)
                        aC      = np.mean([x["C"]      for x in s], axis=0)
                        aX_alt  = np.mean([x["X_alt"]  for x in s], axis=0)
                        aL      = np.mean([x["L_A"]    for x in s], axis=0)
                        aR      = np.mean([x["R_A"]    for x in s], axis=0)
                        avg_rot = float(np.mean([x["rot"] for x in s]))
                        avg_Lz  = float(np.mean([x["L_z"] for x in s]))
                        avg_Rz  = float(np.mean([x["R_z"] for x in s]))
                        p1_px = tuple(np.mean([x["p1_px"] for x in s], axis=0).astype(int))
                        p2_px = tuple(np.mean([x["p2_px"] for x in s], axis=0).astype(int))

                        v_raw = aTR - aTL
                        v_len = np.linalg.norm(v_raw)
                        v = v_raw / v_len if v_len > 0 else np.zeros(3)
                        w = aC - aB
                        u = aB - aA
                        den = (np.dot(v,v)*np.dot(w,w)) - (np.dot(v,w)**2)

                        def _perp():
                            if abs(den) > 1e-6:
                                kv_ = ((np.dot(v,w)*np.dot(u,v)) - (np.dot(v,v)*np.dot(u,w))) / den
                                kv_ = float(np.clip(kv_, 0.0, 1.0))
                                aX_ = aB + kv_ * w
                                return aX_, np.linalg.norm(aX_ - aA), kv_
                            return aX_alt, np.linalg.norm(aX_alt - aA), 0.5

                        if app.use_angle_thresh.get():
                            thresh = app.rot_threshold.get()
                            if avg_rot > thresh:
                                aX, dv, kv = aX_alt, np.linalg.norm(aX_alt - aA), 0.5
                            else:
                                aX, dv, kv = _perp()
                        else:
                            aX, dv, kv = _perp()

                        app.last_data[key].update({
                            "A": aA, "X": aX, "TR": aTR, "BR": aBR,
                            "B": aB, "C": aC, "dist": dv, "k": kv,
                            "rot_2d": avg_rot, "L_A": aL, "R_A": aR,
                            "L_z": avg_Lz, "R_z": avg_Rz,
                            "p1_px": p1_px, "p2_px": p2_px,
                        })
                        app.last_data["session_count"] += 1
                        buffers[key].clear()

                        v_vec = aTR - aTL; u_vec = aA; w_vec = aC - aB
                        try:
                            log.record(dv, kv, aA, aX, aTR, aBR, aB, aC, u_vec, v_vec, w_vec, aL, aR)
                        except Exception: pass

                        if app.mv_state in ("collecting_init", "ready", "collecting_final"):
                            try:
                                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
                                img_dir = "captured_images"
                                os.makedirs(img_dir, exist_ok=True)
                                cv.imwrite(os.path.join(img_dir, f"mv_{ts}.jpg"), frame)
                            except Exception: pass
                l_u = curr
        else:
            for key in ["top", "bottom"]:
                app.last_data[key].update({"dist": 0.0, "L_det": False, "R_det": False})

        for key, color in [("top", (0, 165, 255)), ("bottom", (255, 0, 255))]:
            d = app.last_data[key]
            if d["dist"] > 0 and d.get("p1_px") is not None and d.get("p2_px") is not None:
                p1 = d["p1_px"]
                X_3d = np.array([d["X"]], dtype=np.float64).reshape(1, 1, 3)
                if d["X"][2] > 0:
                    p_proj, _ = cv.projectPoints(X_3d, np.zeros(3), np.zeros(3), K, dist_c)
                    p2 = tuple(p_proj[0].ravel().astype(int))
                else:
                    p2 = d["p2_px"]
                cv.line(frame, p1, p2, color, 3)
                cv.circle(frame, p2, 6, (0, 255, 0), -1)
                cv.putText(frame, f"{key.upper()}: {d['dist']:.2f}mm", (p1[0], p1[1]-12), 0, 0.6, color, 2)

        app.current_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        app._cam_preview_frame = app.current_frame
