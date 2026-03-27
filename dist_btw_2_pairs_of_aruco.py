import cv2 as cv
import numpy as np
import os
import sys
import time

# --- 1. SETTINGS & LOAD CALIBRATION ---
# Path used on the RPi
CALIB_FILE = "/home/kabish/rpi_gap_measurement_through_aruco_march/calibration_checkerboard/camera_params.npz"
MARKER_SIZE_CM = 10.0  

if not os.path.exists(CALIB_FILE):
    # Local fallback for testing
    CALIB_FILE = "calibration_params.npz"

if not os.path.exists(CALIB_FILE):
    print(f"Error: {CALIB_FILE} not found.")
    sys.exit(1)

calib_data = np.load(CALIB_FILE)
cam_mat = calib_data["mtx"]
dist_coef = calib_data["dist"]
print(f"Loaded calibration from {CALIB_FILE}")

# --- 2. CAMERA SETUP (Picamera2) ---
try:
    from picamera2 import Picamera2
    picam = Picamera2()
    # Explicitly configure for a single RGB stream to minimize metadata errors
    config = picam.create_video_configuration(main={"format": "RGB888", "size": (1280, 720)})
    picam.configure(config)
    picam.start()
    time.sleep(2) # Give sensor time to stabilize PDAF
    USING_PICAM = True
    print("Picamera2 started.")
except Exception as e:
    print(f"Picamera2 init failed: {e}. Falling back to USB Camera.")
    cap = cv.VideoCapture(0)
    USING_PICAM = False

# --- 3. ARUCO SETUP ---
marker_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_250)
param_markers = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(marker_dict, param_markers)

half_s = MARKER_SIZE_CM / 2
obj_points = np.array([
    [-half_s,  half_s, 0],
    [ half_s,  half_s, 0],
    [ half_s, -half_s, 0],
    [-half_s, -half_s, 0]
], dtype=np.float32)

PAIRS = [(1, 2), (3, 4)]

def process_pair(frame, pair, centers, tVecs_dict):
    id1, id2 = pair
    found1 = id1 in centers
    found2 = id2 in centers
    
    y_offset = 420 if pair == (1, 2) else 500
    color = (0, 0, 255) if pair == (1, 2) else (255, 0, 255)

    if found1 and found2:
        pt1, pt2 = centers[id1], centers[id2]
        cv.line(frame, pt1, pt2, color, 2)
        
        vec1, vec2 = tVecs_dict[id1], tVecs_dict[id2]
        mm_distance = np.linalg.norm(vec1 - vec2) * 10 
        
        cv.putText(frame, f"Pair {id1}-{id2} Dist: {round(mm_distance, 1)} mm", (20, y_offset),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        msg = f"Pair {id1}-{id2}: ID {' and '.join([str(i) for i in [id1, id2] if i not in centers])} not detected"
        cv.putText(frame, msg, (20, y_offset), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

print("Starting measurement. Press 'q' to quit.")

try:
    while True:
        if USING_PICAM:
            frame_rgb = picam.capture_array()
            if frame_rgb is None: continue
            frame = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
        else:
            ret, frame = cap.read()
            if not ret: break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        marker_corners, marker_IDs, _ = detector.detectMarkers(gray)

        centers = {}
        tVecs_dict = {}

        if marker_IDs is not None:
            for i in range(len(marker_IDs)):
                m_id = int(marker_IDs[i][0])
                _, rvec, tvec = cv.solvePnP(obj_points, marker_corners[i], cam_mat, dist_coef)
                
                corners = marker_corners[i].reshape(4, 2).astype(int)
                centers[m_id] = (int(np.mean(corners[:, 0])), int(np.mean(corners[:, 1])))
                tVecs_dict[m_id] = tvec.flatten()

                # Visuals
                cv.polylines(frame, [corners], True, (0, 255, 255), 2)
                cv.drawFrameAxes(frame, cam_mat, dist_coef, rvec, tvec, 5)
                cv.putText(frame, f"ID:{m_id}", (centers[m_id][0], centers[m_id][1] - 10),
                           cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)

        for pair in PAIRS:
            process_pair(frame, pair, centers, tVecs_dict)

        cv.imshow("RPi Aruco Pairs", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    if USING_PICAM: picam.stop()
    else: cap.release()
    cv.destroyAllWindows()
