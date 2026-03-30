import cv2 as cv
import numpy as np
import os
import sys
import time

# --- 1. SETTINGS & LOAD CALIBRATION ---
MARKER_SIZE_MM = 50.0   # Real size of one side in mm
GAP_CALIB_FACTOR = 0.9845
CALIB_FILE = "/home/kabish/rpi_gap_measurement_through_aruco_march/calibration_checkerboard/camera_params.npz"

if not os.path.exists(CALIB_FILE):
    CALIB_FILE = "calibration_params.npz"

if not os.path.exists(CALIB_FILE):
    cam_mat = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float32)
    dist_coef = np.zeros(5, dtype=np.float32)
else:
    calib_data = np.load(CALIB_FILE)
    cam_mat = calib_data["mtx"] if "mtx" in calib_data else calib_data["camera_matrix"]
    dist_coef = calib_data["dist"] if "dist" in calib_data else calib_data["distortion_coefficients"]

def order_corners(corners):
    pts = np.array(corners, dtype=np.float32)
    idx_by_y = np.argsort(pts[:, 1])
    top2 = pts[idx_by_y][:2]
    bottom2 = pts[idx_by_y][2:]
    top2 = top2[np.argsort(top2[:, 0])]
    bottom2 = bottom2[np.argsort(bottom2[:, 0])]
    return np.array([top2[0], top2[1], bottom2[1], bottom2[0]], dtype=np.float32) # [TL, TR, BR, BL]

def process_pair(frame, corners_list, idx_left, idx_right, y_display, label):
    # Get ordered corners for both markers
    l_corners = order_corners(corners_list[idx_left].reshape((4, 2)))
    r_corners = order_corners(corners_list[idx_right].reshape((4, 2)))
    
    # Calculate px to mm ratio based on left marker (average of top and bottom side)
    side_px = (np.linalg.norm(l_corners[1]-l_corners[0]) + np.linalg.norm(l_corners[2]-l_corners[3])) / 2
    mm_per_px = MARKER_SIZE_MM / side_px

    # Calculate midpoints of facing vertical edges
    # Left Marker: Midpoint of Right Edge (TR and BR)
    mid_left = (l_corners[1] + l_corners[2]) / 2.0
    # Right Marker: Midpoint of Left Edge (TL and BL)
    mid_right = (r_corners[0] + r_corners[3]) / 2.0

    # Draw the single center line
    p1 = (int(mid_left[0]), int(mid_left[1]))
    p2 = (int(mid_right[0]), int(mid_right[1]))
    cv.line(frame, p1, p2, (0, 255, 0), 2)
    cv.circle(frame, p1, 5, (0, 0, 255), -1)
    cv.circle(frame, p2, 5, (255, 0, 0), -1)

    # Gap Calculation (2D geometric distance * calibration)
    gap_px = np.linalg.norm(mid_left - mid_right)
    gap_mm = gap_px * mm_per_px * GAP_CALIB_FACTOR

    # Display results
    cv.putText(frame, f"{label}: {round(gap_mm, 2)} mm", (50, y_display), 
               cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv.putText(frame, f"{label}_acc: {round(gap_px*mm_per_px, 2)} mm", (50, y_display - 30),
		cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

# --- 3. CAMERA SETUP ---
try:
    from picamera2 import Picamera2
    picam = Picamera2()
    picam.configure(picam.create_video_configuration(main={"format": "RGB888", "size": (1280, 720)}))
    picam.start()
    USING_PICAM = True
except:
    cap = cv.VideoCapture(0)
    USING_PICAM = False

detector = cv.aruco.ArucoDetector(cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_250))

print("Starting center-line measurement. Press 'q' to quit.")

while True:
    frame = cv.cvtColor(picam.capture_array(), cv.COLOR_RGB2BGR) if USING_PICAM else cap.read()[1]
    if frame is None: break
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    
    if ids is not None and len(corners) >= 2:
        # Group markers by vertical position
        centers = [(i, np.mean(c[0][:, 0]), np.mean(c[0][:, 1])) for i, c in enumerate(corners)]
        centers.sort(key=lambda x: x[2]) # Sort by Y
        
        # Process Top Pair
        if len(centers) >= 2:
            top_pair = sorted(centers[:2], key=lambda x: x[1]) # Sort by X to find Left and Right
            process_pair(frame, corners, top_pair[0][0], top_pair[1][0], 60, "TOP GAP")

        # Process Bottom Pair
        if len(centers) >= 4:
            bot_pair = sorted(centers[-2:], key=lambda x: x[1]) # Sort by X to find Left and Right
            process_pair(frame, corners, bot_pair[0][0], bot_pair[1][0], 120, "BOTTOM GAP")

    cv.imshow("Single Center Line Gap Measurement", frame)
    if cv.waitKey(1) & 0xFF == ord('q'): break

if USING_PICAM: picam.stop()
else: cap.release()
cv.destroyAllWindows()
