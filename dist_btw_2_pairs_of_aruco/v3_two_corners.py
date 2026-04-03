import cv2 as cv
import numpy as np
import os
import sys

# --- 1. SETTINGS & AUTOMATIC PATH DETECTION ---
MARKER_SIZE_MM = 54.0
# Add your local Windows path to the calibration file here
POSSIBLE_PATHS = [
    "camera_params.npz",
    "calibration_params.npz",
    "../camera_params.npz",
    "../../camera_params.npz",
    "/home/kabish/rpi_gap_measurement_through_aruco_march/calibration_checkerboard/camera_params.npz"
]

CALIB_FILE = None
for path in POSSIBLE_PATHS:
    if os.path.exists(path):
        CALIB_FILE = path
        break

if CALIB_FILE is None:
    print("Error: Calibration file (.npz) not found. Please place it in the same folder as this script.")
    # Fallback to defaults to prevent crash, but MM will be inaccurate
    cam_mat = np.array([[1000,0,640],[0,1000,360],[0,0,1]], dtype=np.float32)
    dist_coef = np.zeros(5, dtype=np.float32)
else:
    print(f"Loading calibration from: {CALIB_FILE}")
    try:
        calib_data = np.load(CALIB_FILE)
        cam_mat = calib_data["mtx"] if "mtx" in calib_data else calib_data["camera_matrix"]
        dist_coef = calib_data["dist"] if "dist" in calib_data else calib_data["distortion_coefficients"]
    except Exception as e:
        print(f"Error loading {CALIB_FILE}: {e}")
        sys.exit(1)

# Define 3D points
h = MARKER_SIZE_MM / 2.0
obj_points = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)

def calculate_3d_dist(id1_l_idx, id2_r_idx, r_l, t_l, r_r, t_r):
    Rl, _ = cv.Rodrigues(r_l)
    Rr, _ = cv.Rodrigues(r_r)
    p_l_3d = np.dot(Rl, obj_points[id1_l_idx]) + t_l.flatten()
    p_r_3d = np.dot(Rr, obj_points[id2_r_idx]) + t_r.flatten()
    return np.linalg.norm(p_l_3d - p_r_3d)

def get_inner_corner_indices(corners_px, is_left_marker=True):
    pts = corners_px.reshape(4, 2)
    # Sort by X to find horizontal boundaries
    sorted_x_idx = np.argsort(pts[:, 0])
    
    if is_left_marker:
        # We need the two right-most points (largest X)
        inner_idx = sorted_x_idx[2:]
    else:
        # We need the two left-most points (smallest X)
        inner_idx = sorted_x_idx[:2]
        
    # Sort these two by Y to distinguish Top from Bottom
    if pts[inner_idx[0]][1] < pts[inner_idx[1]][1]:
        return inner_idx[0], inner_idx[1] # [Top_Index, Bottom_Index]
    else:
        return inner_idx[1], inner_idx[0]

def process_gap(frame, all_corners, i_left, i_right, y_pos, label):
    # Get Pose
    _, r_l, t_l = cv.solvePnP(obj_points, all_corners[i_left], cam_mat, dist_coef)
    _, r_r, t_r = cv.solvePnP(obj_points, all_corners[i_right], cam_mat, dist_coef)

    # Correct indices based on X/Y coordinates
    l_top_i, l_bot_i = get_inner_corner_indices(all_corners[i_left], True)
    r_top_i, r_bot_i = get_inner_corner_indices(all_corners[i_right], False)

    # Accurate 3D distance
    dt = calculate_3d_dist(l_top_i, r_top_i, r_l, t_l, r_r, t_r)
    db = calculate_3d_dist(l_bot_i, r_bot_i, r_l, t_l, r_r, t_r)

    # Draw
    p_l_t = tuple(all_corners[i_left][0][l_top_i].astype(int))
    p_r_t = tuple(all_corners[i_right][0][r_top_i].astype(int))
    p_l_b = tuple(all_corners[i_left][0][l_bot_i].astype(int))
    p_r_b = tuple(all_corners[i_right][0][r_bot_i].astype(int))

    cv.line(frame, p_l_t, p_r_t, (0, 0, 255), 2)
    cv.line(frame, p_l_b, p_r_b, (0, 255, 0), 2)

    cv.putText(frame, f"{label} T: {round(dt, 2)}mm", (50, y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv.putText(frame, f"{label} B: {round(db, 2)}mm", (50, y_pos+30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

# --- CAMERA ---
try:
    from picamera2 import Picamera2
    picam = Picamera2()
    picam.configure(picam.create_video_configuration(main={"format": "RGB888", "size": (1280, 720)}))
    picam.start()
    is_pi = True
except:
    cap = cv.VideoCapture(0)
    is_pi = False

detector = cv.aruco.ArucoDetector(cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250))

while True:
    img = picam.capture_array() if is_pi else cap.read()[1]
    if img is None: break
    frame = cv.cvtColor(img, cv.COLOR_RGB2BGR) if is_pi else img
    corners, ids, _ = detector.detectMarkers(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
    
    if ids is not None and len(corners) >= 2:
        # Pair logic based on vertical height
        pts = [(i, np.mean(c[0][:,0]), np.mean(c[0][:,1])) for i, c in enumerate(corners)]
        pts.sort(key=lambda x: x[2])
        if len(pts) >= 2:
            tp = sorted(pts[:2], key=lambda x: x[1])
            process_gap(frame, corners, tp[0][0], tp[1][0], 60, "TOP PAIR")
        if len(pts) >= 4:
            bp = sorted(pts[-2:], key=lambda x: x[1])
            process_gap(frame, corners, bp[0][0], bp[1][0], 150, "BOT PAIR")

    cv.imshow("Geometric 3D Measurement", frame)
    if cv.waitKey(1) & 0xFF == ord('q'): break

if is_pi: picam.stop()
else: cap.release()
cv.destroyAllWindows()
