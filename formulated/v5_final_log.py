try:
    from picamera2 import Picamera2
except ImportError:
    print("Error: 'picamera2' library not found. Run: sudo apt install python3-picamera2")
    Picamera2 = None

import cv2 as cv
import numpy as np
import os
import time
import math
import log  # Import your DB/CSV module

# --- SETTINGS ---
MARKER_SIZE_MM = 53.8             
ARUCO_DICT = cv.aruco.DICT_4X4_50  
RESOLUTION = (1280, 720)          
CALIB_FILE = "camera_params.npz"

# --- TIMING ---
SAMPLE_INTERVAL_MS = 100    # Raw data every 100ms
DISPLAY_UPDATE_MS  = 500    # Filter and Update every 500ms

def load_calibration():
    """Loads camera calibration data from .npz file."""
    if os.path.exists(CALIB_FILE):
        with np.load(CALIB_FILE) as data:
            return data['camera_matrix'], data['dist_coeff']
    else:
        print("WARNING: camera_params.npz NOT FOUND. Using generic matrix.")
        f = RESOLUTION[0]
        K = np.array([[f, 0, RESOLUTION[0]/2], [0, f, RESOLUTION[1]/2], [0, 0, 1]], dtype=np.float32)
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

def calculate_angles(rvec):
    """Calculates Roll (clock-rotation) and Tilt (inclination from camera) in degrees."""
    R, _ = cv.Rodrigues(rvec)
    tilt = math.degrees(math.acos(np.clip(R[2, 2], -1, 1)))
    roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return roll, tilt


def identify_inner_edges(pts_l, pts_r):
    # Left: Right-most two
    l_sorted = pts_l[pts_l[:, 0].argsort()[::-1]]
    right_two = l_sorted[:2]
    TR_L, BR_L = right_two[right_two[:, 1].argsort()][0], right_two[right_two[:, 1].argsort()][1]
    TL_L = l_sorted[2:][l_sorted[2:][:, 1].argsort()][0]
    # Right: Left-most two
    r_sorted = pts_r[pts_r[:, 0].argsort()]
    left_two_r = r_sorted[:2]
    B, C = left_two_r[left_two_r[:, 1].argsort()][0], left_two_r[left_two_r[:, 1].argsort()][1]
    return TL_L, TR_L, BR_L, B, C

def main():
    if Picamera2 is None: return
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": RESOLUTION, "format": "RGB888"}))
    picam2.start()

    K, dist_coeffs = load_calibration()
    detector = cv.aruco.ArucoDetector(cv.aruco.getPredefinedDictionary(ARUCO_DICT), cv.aruco.DetectorParameters())
    log.init_log()

    # --- STATE BUFFERS ---
    sample_buffer = []  
    last_sample_time = 0
    last_display_time = 0
    disp_data = {"A":None} # Screen overlay data

    print(f"--- V5 Final Online (Averaging: {DISPLAY_UPDATE_MS}ms) ---")

    try:
        while True:
            frame_rgb = picam2.capture_array()
            frame = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            curr_time = time.time() * 1000 

            # Step 1: Detect and Sample (100ms)
            if ids is not None and len(ids) >= 2:
                if (curr_time - last_sample_time) >= SAMPLE_INTERVAL_MS:
                    order = np.argsort([np.mean(c[0], axis=0)[0] for c in corners])
                    rv_l, tv_l = get_marker_pose(corners[order[0]][0], MARKER_SIZE_MM, K, dist_coeffs)
                    rv_r, tv_r = get_marker_pose(corners[order[1]][0], MARKER_SIZE_MM, K, dist_coeffs)
                    
                    l_ang = calculate_angles(rv_l)
                    r_ang = calculate_angles(rv_r)
                    pts_l, pts_r = get_3d_corners(rv_l, tv_l, MARKER_SIZE_MM), get_3d_corners(rv_r, tv_r, MARKER_SIZE_MM)
                    TL_L, TR_L, BR_L, B, C = identify_inner_edges(pts_l, pts_r)
                    A = (TR_L + BR_L) / 2.0
                    
                    sample_buffer.append({"A":A, "B":B, "C":C, "TR":TR_L, "BR":BR_L, "TL":TL_L, "L_Ang":l_ang, "R_Ang":r_ang})
                    last_sample_time = curr_time

                # Step 2: Average and Log (500ms)
                if (curr_time - last_display_time) >= DISPLAY_UPDATE_MS and len(sample_buffer) > 0:
                    aA, aB, aC = np.mean([s["A"] for s in sample_buffer], axis=0), np.mean([s["B"] for s in sample_buffer], axis=0), np.mean([s["C"] for s in sample_buffer], axis=0)
                    aTR, aBR, aTL = np.mean([s["TR"] for s in sample_buffer], axis=0), np.mean([s["BR"] for s in sample_buffer], axis=0), np.mean([s["TL"] for s in sample_buffer], axis=0)
                    aL_Ang, aR_Ang = np.mean([s["L_Ang"] for s in sample_buffer], axis=0), np.mean([s["R_Ang"] for s in sample_buffer], axis=0)

                    v_dir = aTR - aTL
                    V_vec = v_dir / np.linalg.norm(v_dir)
                    W_vec, U_vec = aC - aB, aB - aA
                    denom = (np.dot(V_vec, V_vec) * np.dot(W_vec, W_vec)) - (np.dot(V_vec, W_vec)**2)
                    
                    if abs(denom) > 1e-6:
                        k = ((np.dot(V_vec, W_vec) * np.dot(U_vec, V_vec)) - (np.dot(V_vec, V_vec) * np.dot(U_vec, W_vec))) / denom
                        aX = aB + k * W_vec
                        aDist = np.linalg.norm(aX - aA)
                        
                        # Set Display Data
                        disp_data = {"A":aA, "X":aX, "TR":aTR, "BR":aBR, "B":aB, "C":aC, "dist":aDist}
                        # LOG: numerically stable data including rotation
                        log.record(aDist, k, aA, aX, aTR, aBR, aB, aC, U_vec, V_vec, W_vec, aL_Ang, aR_Ang)
                        
                    sample_buffer.clear()
                    last_display_time = curr_time
            else:
                disp_data = {"A":None} # Clear if lost

            # Step 3: Draw Overlay (Before image save)
            if disp_data["A"] is not None:
                y_off = 35
                for label in ["A", "X", "TR", "BR", "B", "C"]:
                    v = disp_data[label]
                    cv.putText(frame, f"{label}:({v[0]:.3f},{v[1]:.3f})", (15, y_off), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    y_off += 25
                
                p_disp, _ = cv.projectPoints(np.array([disp_data["A"], disp_data["X"]]), np.zeros(3), np.zeros(3), K, dist_coeffs)
                pA, pX = tuple(p_disp[0].ravel().astype(int)), tuple(p_disp[1].ravel().astype(int))
                cv.line(frame, pA, pX, (0, 165, 255), 3)
                cv.circle(frame, pX, 8, (255, 0, 255), -1)
                pMid = ((pA[0] + pX[0]) // 2, (pA[1] + pX[1]) // 2)
                cv.putText(frame, f"AVG: {disp_data['dist']:.3f} mm", (pMid[0], pMid[1] - 15), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            log.save_image(frame) # Save screenshot of results

            if os.environ.get('DISPLAY') is not None:
                cv.imshow("ArUco V5 Measurement", frame)
                if cv.waitKey(1) & 0xFF == ord('q'): break
            else:
                time.sleep(0.01)

    except KeyboardInterrupt: pass
    finally:
        picam2.stop()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
