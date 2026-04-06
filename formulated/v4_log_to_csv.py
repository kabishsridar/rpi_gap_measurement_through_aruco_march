try:
    from picamera2 import Picamera2
except ImportError:
    print("Error: 'picamera2' library not found.")
    Picamera2 = None

import cv2 as cv
import numpy as np
import os
import time
import log  # Importing your new logging module

# --- SETTINGS ---
MARKER_SIZE_MM = 53.8             
ARUCO_DICT = cv.aruco.DICT_4X4_50  
RESOLUTION = (1280, 720)          

def get_dummy_calibration(frame_width, frame_height):
    focal_length = frame_width  
    center = (frame_width / 2, frame_height / 2)
    K = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float32)
    return K, np.zeros(5)

def get_marker_pose(corners, size, K, dist):
    half = size / 2.0
    obj_pts = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
    _, rvec, tvec = cv.solvePnP(obj_pts, corners, K, dist)
    return rvec, tvec

def get_3d_corners(rvec, tvec, size):
    half = size / 2.0
    obj_pts = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
    R, _ = cv.Rodrigues(rvec)
    return np.array([np.dot(R, pt) + tvec.ravel() for pt in obj_pts])

def identify_inner_edges(pts_l, pts_r):
    l_sorted = pts_l[pts_l[:, 0].argsort()[::-1]]
    right_two = l_sorted[:2]
    TR_L = right_two[right_two[:, 1].argsort()][0]
    BR_L = right_two[right_two[:, 1].argsort()][1]
    TL_L = l_sorted[2:][l_sorted[2:][:, 1].argsort()][0]
    r_sorted = pts_r[pts_r[:, 0].argsort()]
    B, C = r_sorted[:2][r_sorted[:2][:, 1].argsort()][0], r_sorted[:2][r_sorted[:2][:, 1].argsort()][1]
    return TL_L, TR_L, BR_L, B, C

def main():
    if Picamera2 is None: return
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": RESOLUTION, "format": "RGB888"}))
    picam2.start()

    K, dist_coeffs = get_dummy_calibration(RESOLUTION[0], RESOLUTION[1])
    detector = cv.aruco.ArucoDetector(cv.aruco.getPredefinedDictionary(ARUCO_DICT), cv.aruco.DetectorParameters())
    
    # Initialize the log file
    log.init_log()
    print("Measurement Started. Data is being logged to gap_measurements.csv")

    try:
        while True:
            frame_rgb = picam2.capture_array()
            frame = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            
            if ids is not None and len(ids) >= 2:
                order = np.argsort([np.mean(c[0], axis=0)[0] for c in corners])
                rv_l, tv_l = get_marker_pose(corners[order[0]][0], MARKER_SIZE_MM, K, dist_coeffs)
                pts_l = get_3d_corners(rv_l, tv_l, MARKER_SIZE_MM)
                rv_r, tv_r = get_marker_pose(corners[order[1]][0], MARKER_SIZE_MM, K, dist_coeffs)
                pts_r = get_3d_corners(rv_r, tv_r, MARKER_SIZE_MM)


                TL_L, TR_L, BR_L, B, C = identify_inner_edges(pts_l, pts_r)
                A = (TR_L + BR_L) / 2.0
                v_dir = TR_L - TL_L
                V_vec = v_dir / np.linalg.norm(v_dir)
                W_vec, U_vec = C - B, B - A
                
                denom = (np.dot(V_vec, V_vec) * np.dot(W_vec, W_vec)) - (np.dot(V_vec, W_vec)**2)
                if abs(denom) > 1e-6:
                    k = ((np.dot(V_vec, W_vec) * np.dot(U_vec, V_vec)) - (np.dot(V_vec, V_vec) * np.dot(U_vec, W_vec))) / denom
                    if 0 <= k <= 1:
                        X = B + k * W_vec
                        dist_val = np.linalg.norm(X - A)
                        
                        # --- DATA LOGGING ---
                        log.record(dist_val, k, A, X, TR_L, BR_L, B, C, U_vec, V_vec, W_vec)

                        # --- LIVE FEED DISPLAY ---
                        if os.environ.get('DISPLAY') is not None:
                            y_off = 35
                            for label, val in [("A", A), ("X", X), ("TR", TR_L), ("BR", BR_L), ("B", B), ("C", C)]:
                                cv.putText(frame, f"{label}:({val[0]:.1f}, {val[1]:.1f})", (15, y_off), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                y_off += 25
                            
                            p_disp, _ = cv.projectPoints(np.array([A, X]), np.zeros(3), np.zeros(3), K, dist_coeffs)
                            pA, pX = tuple(p_disp[0].ravel().astype(int)), tuple(p_disp[1].ravel().astype(int))
                            cv.line(frame, pA, pX, (0, 165, 255), 3)
                            cv.circle(frame, pX, 8, (255, 0, 255), -1)
                            pMid = ((pA[0] + pX[0]) // 2, (pA[1] + pX[1]) // 2)
                            cv.putText(frame, f"{dist_val:.1f} mm", (pMid[0], pMid[1] - 15), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if os.environ.get('DISPLAY') is not None:
                cv.imshow("V3 Pi Measurement", frame)
                if cv.waitKey(1) & 0xFF == ord('q'): break
            else:
                time.sleep(0.01)

    except KeyboardInterrupt: pass
    finally:
        picam2.stop()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
