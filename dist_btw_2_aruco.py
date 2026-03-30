import cv2 as cv
import numpy as np
import os
import sys

# --- 1. SETTINGS & LOAD CALIBRATION ---
# Load your Raspberry Pi calibration file
CALIB_FILE = "/home/kabish/rpi_gap_measurement_through_aruco_march/calibration/camera_params.npz"
MARKER_SIZE_CM = 5.0  # Physical size of your marker in CM
ARUCO_DICT = cv.aruco.DICT_5X5_250

if not os.path.exists(CALIB_FILE):
    print(f"Error: {CALIB_FILE} not found. Ensure you ran calibration first!")
    sys.exit(1)

calib_data = np.load(CALIB_FILE)
cam_mat = calib_data["mtx"]
dist_coef = calib_data["dist"]

# --- 2. CAMERA SETUP (Picamera2) ---
try:
    from picamera2 import Picamera2
    picam = Picamera2()
    # 720p is often ideal for detection (high enough for detail, fast enough for FPS)
    config = picam.create_video_configuration(main={"format": "RGB888", "size": (1280, 720)})
    picam.configure(config)
    picam.start()
    USING_PICAM = True
    print("Picamera2 started.")
except ImportError:
    print("Picamera2 not found. Falling back to USB Camera 0.")
    cap = cv.VideoCapture(0)
    USING_PICAM = False

# --- 3. ARUCO & REFINEMENT SETUP ---
marker_dict = cv.aruco.getPredefinedDictionary(ARUCO_DICT)
param_markers = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(marker_dict, param_markers)

# Refinement criteria for sub-pixel accuracy
sub_pixel_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)

# Prepare 3D object points for pose estimation (single marker at origin)
half_s = MARKER_SIZE_CM / 2
obj_points = np.array([
    [-half_s,  half_s, 0],
    [ half_s,  half_s, 0],
    [ half_s, -half_s, 0],
    [-half_s, -half_s, 0]
], dtype=np.float32)

print("Starting Calibrated Measurement. Press 'q' to quit.")

try:
    while True:
        # Capture frame
        if USING_PICAM:
            frame_rgb = picam.capture_array()
            frame = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
        else:
            ret, frame = cap.read()
            if not ret: break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # 1. Detect markers
        marker_corners, marker_IDs, _ = detector.detectMarkers(gray)

        centers = {}
        tVecs_dict = {}

        if marker_IDs is not None:
            # 2. Refine Corners to Sub-Pixel Accuracy (Essential for tilts!)
            for i in range(len(marker_corners)):
                cv.cornerSubPix(gray, marker_corners[i], (5, 5), (-1, -1), sub_pixel_criteria)
                
                # 3. Pose Estimation (calculate 3D X,Y,Z of marker)
                _, rvec, tvec = cv.solvePnP(obj_points, marker_corners[i], cam_mat, dist_coef)
                
                # Reshape for drawing
                corners = marker_corners[i].reshape(4, 2).astype(int)
                center_x = int(np.mean(corners[:, 0]))
                center_y = int(np.mean(corners[:, 1]))
                
                m_id = int(marker_IDs[i][0])
                centers[m_id] = (center_x, center_y)
                tVecs_dict[m_id] = tvec.flatten()

                # --- Draw Info ---
                cv.polylines(frame, [corners], True, (0, 255, 255), 2)
                cv.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
                # Draw 3D axis (5cm length) to verify tilt accuracy
                cv.drawFrameAxes(frame, cam_mat, dist_coef, rvec, tvec, 5)
                cv.putText(frame, f"ID:{m_id}", (center_x, center_y - 10),
                           cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)

            # 4. Calculate Distance between first two IDs found
            if len(centers) >= 2:
                ids = list(centers.keys())[:2]
                pt1, pt2 = centers[ids[0]], centers[ids[1]]
                
                # Pixel Distance
                pixel_dist = int(np.linalg.norm(np.array(pt1) - np.array(pt2)))
                cv.line(frame, pt1, pt2, (0, 0, 255), 2)
                cv.putText(frame, f"Pixel Dist: {pixel_dist} px", (20, 20),
                           cv.FONT_HERSHEY_PLAIN, 1.4, (255, 255, 255), 2)

                # Real 3D World Distance (Euclidean distance between two vectors)
                v1, v2 = tVecs_dict[ids[0]], tVecs_dict[ids[1]]
                mm_dist = np.linalg.norm(v1 - v2) * 10 # cm to mm
                
                # Display result
                text = f"3D DISTANCE: {round(mm_dist, 1)} mm"
                cv.rectangle(frame, (15, 660), (450, 700), (0, 0, 0), -1)
                cv.putText(frame, text, (20, 690),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv.imshow("Calibrated Distance Extractor", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    if USING_PICAM: picam.stop()
    else: cap.release()
    cv.destroyAllWindows()
