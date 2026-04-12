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
    STRICT Port of v14.py's math engine with GUI telemetry support (Euler angles & Lighting).
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
    b = {"top": [], "bottom": []}
    l_s = l_u = time.time() * 1000

    while app.is_running:
        K, dist_c = app.load_calib()
        try:
            frame = cv.cvtColor(pc.capture_array(), cv.COLOR_RGB2BGR)
        except Exception:
            continue

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        mean_b = np.mean(gray)
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
            if len(m_data) == 2:
                dy, dx = abs(m_data[0]["y"]-m_data[1]["y"]), abs(m_data[0]["x"]-m_data[1]["x"])
                if dx > dy: top_m = m_data if np.mean([m["y"] for m in m_data]) < RESOLUTION[1]/2 else []; bot_m = m_data if not top_m else []
                else: top_m = [m_data[0]]; bot_m = [m_data[1]]
            elif len(m_data) == 3:
                top_m = m_data[:2]; bot_m = m_data[2:]

            def proc(marker_list, key, size):
                if len(marker_list) < 2:
                    app.last_data[key].update({"dist": 0.0, "L_det": False, "R_det": False})
                    return
                marker_list.sort(key=lambda m: m["x"])
                is_rf = (app.fixed_side.get() == "Right")
                S_m = marker_list[-1] if is_rf else marker_list[0]
                T_m = marker_list[0] if is_rf else marker_list[-1]
                h = size / 2.0
                obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)

                def get_p(c):
                    _, rv, tv = cv.solvePnP(obj, c, K, dist_c)
                    R_m, _ = cv.Rodrigues(rv)
                    sy = math.sqrt(R_m[0,0]**2 + R_m[1,0]**2)
                    singular = sy < 1e-6
                    if not singular:
                        pit = math.degrees(math.atan2(-R_m[2,0], sy))
                        yaw = math.degrees(math.atan2(R_m[2,1], R_m[2,2]))
                        rol = math.degrees(math.atan2(R_m[1,0], R_m[0,0]))
                    else:
                        pit = math.degrees(math.atan2(-R_m[2,0], sy))
                        yaw = 0
                        rol = math.degrees(math.atan2(-R_m[0,1], R_m[1,1]))
                    return np.array([R_m @ pt + tv.ravel() for pt in obj]), rol, pit, yaw, float(tv[2])

                S_pts, S_r, S_p, S_y, S_z = get_p(S_m["c_raw"])
                T_pts, T_r, T_p, T_y, T_z = get_p(T_m["c_raw"])
                
                def i_o(pts, ir):
                    o = pts[:, 0].argsort(); i = pts[o[2:]] if ir else pts[o[:2]]; ot = pts[o[:2]] if ir else pts[o[2:]]
                    return i[i[:,1].argsort()][0], i[i[:,1].argsort()][1], ot[ot[:,1].argsort()][0]

                S_it, S_ib, S_ot = i_o(S_pts, not is_rf); T_it, T_ib, _ = i_o(T_pts, is_rf)
                p1 = tuple(((S_m["c"][1]+S_m["c"][2])/2 if not is_rf else (S_m["c"][0]+S_m["c"][3])/2).astype(int))
                p2 = tuple(((T_m["c"][0]+T_m["c"][3])/2 if not is_rf else (T_m["c"][1]+T_m["c"][2])/2).astype(int))
                si = abs(math.degrees(math.atan2(S_m["c"][1,1]-S_m["c"][0,1], S_m["c"][1,0]-S_m["c"][0,0])))
                ti = abs(math.degrees(math.atan2(T_m["c"][1,1]-T_m["c"][0,1], T_m["c"][1,0]-T_m["c"][0,0])))
                
                b[key].append({
                    "A": (S_it+S_ib)/2.0, "XA": (T_it+T_ib)/2.0, "TR": S_it, "TL": S_ot, "BR": S_ib, "B": T_it, "C": T_ib,
                    "L_A": (S_r, S_p, S_y, S_z), "R_A": (T_r, T_p, T_y, T_z), "rt": max(si, ti), "p1": p1, "p2": p2, "mb": mean_b
                })
                app.last_data[key].update({"L_det": True, "R_det": True})

            if (curr - l_s) >= 100:
                proc(top_m, "top", app.size_top.get())
                proc(bot_m, "bottom", app.size_bot.get())
                l_s = curr

            if (curr - l_u) >= 1000:
                for k in ["top", "bottom"]:
                    if b[k]:
                        s = b[k]; aA = np.mean([x["A"] for x in s],0); aTR = np.mean([x["TR"] for x in s],0); aTL = np.mean([x["TL"] for x in s],0)
                        aB = np.mean([x["B"] for x in s],0); aC = np.mean([x["C"] for x in s],0); aXA = np.mean([x["XA"] for x in s],0)
                        vr = aTR-aTL; v = vr/np.linalg.norm(vr) if np.linalg.norm(vr)>0 else np.zeros(3); w=aC-aB; u=aB-aA; den=np.dot(v,v)*np.dot(w,w)-np.dot(v,w)**2
                        def _perp():
                            if abs(den)>1e-6: kv = np.clip((np.dot(v,w)*np.dot(u,v)-np.dot(v,v)*np.dot(u,w))/den, 0, 1); return aB+kv*w, np.linalg.norm((aB+kv*w)-aA), kv
                            return aXA, np.linalg.norm(aXA-aA), 0.5
                        if (not app.use_v8_only.get()) and np.mean([x["rt"] for x in s]) > app.rot_threshold.get(): aX, dv, kv = aXA, np.linalg.norm(aXA-aA), 0.5
                        else: aX, dv, kv = _perp()
                        
                        L_avg = np.mean([x["L_A"] for x in s], 0); R_avg = np.mean([x["R_A"] for x in s], 0)
                        app.last_data[k].update({
                            "A": aA, "X": aX, "dist": dv, "k": kv, "rot_2d": np.mean([x["rt"] for x in s]),
                            "L_rol": L_avg[0], "L_pit": L_avg[1], "L_yaw": L_avg[2], "L_z": L_avg[3],
                            "R_rol": R_avg[0], "R_pit": R_avg[1], "R_yaw": R_avg[2], "R_z": R_avg[3],
                            "TR": aTR, "BR": np.mean([x["BR"] for x in s],0), "B": aB, "C": aC, "mean_b": np.mean([x["mb"] for x in s]),
                            "p1_px": tuple(np.mean([x["p1"] for x in s],0).astype(int)), "p2_px": tuple(np.mean([x["p2"] for x in s],0).astype(int))
                        })
                        try: log.record(dv, kv, aA, aX, aTR, np.mean([x["BR"] for x in s],0), aB, aC, aA, aTR-aTL, aC-aB, L_avg, R_avg)
                        except: pass
                        b[k].clear()
                app.last_data["session_count"] += 1
                l_u = curr
        else:
            for key in ["top", "bottom"]:
                app.last_data[key].update({"dist": 0.0, "L_det": False, "R_det": False})

        r_th, p_th, y_th = app.rot_threshold.get(), app.pitch_threshold.get(), app.yaw_threshold.get()
        for key, color in [("top", (0, 165, 255)), ("bottom", (255, 0, 255))]:
            d = app.last_data[key]
            issues = 0
            if not d["L_det"]: issues += 1
            if not d["R_det"]: issues += 1
            if d["L_det"]:
                if abs(d["L_pit"]) > p_th: issues += 1
                if abs(d["L_yaw"]) > y_th: issues += 1
            if d["R_det"]:
                if abs(d["R_pit"]) > p_th: issues += 1
                if abs(d["R_yaw"]) > y_th: issues += 1

            if d["dist"] > 0 and d.get("p1_px") is not None:
                p1 = d["p1_px"]
                p2 = tuple(cv.projectPoints(np.array([d["X"]]), np.zeros(3), np.zeros(3), K, dist_c)[0].ravel().astype(int)) if d["X"][2]>0 else d["p2_px"]
                cv.line(frame, p1, p2, color, 3)
                cv.circle(frame, p2, 6, (0, 255, 0), -1)
                
                # Draw Measurement Text
                cv.putText(frame, f"{d['dist']:.2f}mm", (p2[0]+10, p2[1]), 0, 0.6, (255, 0, 255), 2)

                # Draw Issues Box (if any)
                if issues > 0:
                    mid_x = (p1[0] + p2[0]) // 2
                    mid_y = (p1[1] + p2[1]) // 2
                    txt = f"ISSUES: {issues}"
                    (tw, th), _ = cv.getTextSize(txt, 0, 0.5, 2)
                    bx, by, bw, bh = mid_x - tw//2 - 5, mid_y - 30 - th - 5, tw + 10, th + 10
                    cv.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 0, 0), -1)
                    cv.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 165, 255), 2)
                    cv.putText(frame, txt, (bx+5, by+bh-7), 0, 0.5, (0, 255, 255), 2)

        app.current_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        app._cam_preview_frame = app.current_frame
