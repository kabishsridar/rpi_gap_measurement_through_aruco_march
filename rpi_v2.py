try:
    from picamera2 import Picamera2
    from picamera2.outputs import FileOutput
except ImportError:
    print("Error: 'picamera2' library not found. Please install it on your Raspberry Pi.")
    # In a real RPi environment, you'd run: sudo apt install python3-picamera2
    Picamera2 = None

import cv2 as cv
import numpy as np
import os
import time

# --- SETTINGS ---
MARKER_SIZE_MM = 53.8             # Your marker size
ARUCO_DICT = cv.aruco.DICT_4X4_50  # 4x4 Dictionary
RESOLUTION = (1280, 720)          # Picamera2 handles 720p easily

def get_dummy_calibration(frame_width, frame_height):
    """Creates a generic camera matrix for standard cameras."""
    focal_length = frame_width  
    center = (frame_width / 2, frame_height / 2)
    K = np.array([[focal_length, 0, center[0]],
                  [0, focal_length, center[1]],
                  [0, 0, 1]], dtype=np.float32)
    dist = np.zeros(5) 
    return K, dist

def get_marker_pose(corners, size, K, dist):
    """Computes rvec and tvec using solvePnP."""
    half = size / 2.0
    obj_pts = np.array([
        [-half,  half, 0], # TL
        [ half,  half, 0], # TR
        [ half, -half, 0], # BR
        [-half, -half, 0]  # BL
    ], dtype=np.float32)
    ret, rvec, tvec = cv.solvePnP(obj_pts, corners, K, dist)
    return rvec, tvec

def get_3d_corners(rvec, tvec, size):
    """Transforms local marker corners into 3D camera space (mm)."""
    half = size / 2.0
    obj_pts = np.array([
        [-half,  half, 0], [ half,  half, 0], 
        [ half, -half, 0], [-half, -half, 0]
    ], dtype=np.float32)
    R, _ = cv.Rodrigues(rvec)
    return np.array([np.dot(R, pt) + tvec.ravel() for pt in obj_pts])

def identify_inner_edges(pts_l, pts_r):
    l_sorted = pts_l[pts_l[:, 0].argsort()[::-1]]
    right_two = l_sorted[:2]
    TR_L = right_two[right_two[:, 1].argsort()][0]
    BR_L = right_two[right_two[:, 1].argsort()][1]
    TL_L = l_sorted[2:][l_sorted[2:][:, 1].argsort()][0]

    r_sorted = pts_r[pts_r[:, 0].argsort()]
    left_two_r = r_sorted[:2]
    B = left_two_r[left_two_r[:, 1].argsort()][0]
    C = left_two_r[left_two_r[:, 1].argsort()][1]
    
    return TL_L, TR_L, BR_L, B, C

def main():
    if Picamera2 is None: return

    # --- PICAMERA2 SETUP ---
    picam2 = Picamera2()
    # Configure for the desired resolution and RGB format for processing
    config = picam2.create_video_configuration(main={"size": RESOLUTION, "format": "RGB888"})
    picam2.configure(config)
    picam2.start()

    K, dist = get_dummy_calibration(RESOLUTION[0], RESOLUTION[1])
    
    detector_params = cv.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
    detector = cv.aruco.ArucoDetector(
        cv.aruco.getPredefinedDictionary(ARUCO_DICT), 
        detector_params
    )

    print("--- 3D Measurement System Active (Picamera2 Native) ---")
    print(f"Res: {RESOLUTION[0]}x{RESOLUTION[1]}")
    
    is_headless = os.environ.get('DISPLAY') is None

    try:
        while True:
            # Capture array directly (RGB format)
            frame_rgb = picam2.capture_array()
            
            # Convert RGB to BGR for OpenCV processing/display
            frame = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            
            corners, ids, _ = detector.detectMarkers(gray)
            
            if ids is not None and len(ids) >= 2:
                centers_x = [np.mean(c[0], axis=0)[0] for c in corners]
                order = np.argsort(centers_x)
                
                rv_l, tv_l = get_marker_pose(corners[order[0]][0], MARKER_SIZE_MM, K, dist)
                pts_l = get_3d_corners(rv_l, tv_l, MARKER_SIZE_MM)
                
                rv_r, tv_r = get_marker_pose(corners[order[1]][0], MARKER_SIZE_MM, K, dist)
                pts_r = get_3d_corners(rv_r, tv_r, MARKER_SIZE_MM)

                TL_L, TR_L, BR_L, B, C = identify_inner_edges(pts_l, pts_r)

                # --- VECTOR MATH ---
                A = (TR_L + BR_L) / 2.0
                v_dir = TR_L - TL_L
                V = v_dir / np.linalg.norm(v_dir)
                W, U = C - B, B - A
                
                vv, ww, vw = np.dot(V, V), np.dot(W, W), np.dot(V, W)
                uv, uw = np.dot(U, V), np.dot(U, W)
                
                denom = (vv * ww) - (vw**2)
                if abs(denom) > 1e-6:
                    k = ((vw * uv) - (vv * uw)) / denom
                    if 0 <= k <= 1:
                        X = B + k * W
                        distance = np.linalg.norm(X - A)
                        
                        # --- LOGGING ---
                        print(f"\rDIST: {distance:6.1f} mm | k: {k:.2f} | A: {A[0]:.0f},{A[1]:.0f} | X: {X[0]:.0f},{X[1]:.0f}", end="", flush=True)

                        # --- DISPLAY (Only if not headless) ---
                        if not is_headless:
                            disp_pts, _ = cv.projectPoints(np.array([A, X]), np.zeros(3), np.zeros(3), K, dist)
                            pA, pX = tuple(disp_pts[0].ravel().astype(int)), tuple(disp_pts[1].ravel().astype(int))
                            
                            cv.line(frame, pA, pX, (0, 165, 255), 3)
                            cv.circle(frame, pX, 6, (255, 0, 255), -1)
                            cv.putText(frame, f"{distance:.1f} mm", (pA[0], pA[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if not is_headless:
                cv.imshow("Picamera2 ArUco Gap", frame)
                if cv.waitKey(1) & 0xFF == ord('q'): break
            else:
                # Small sleep to yield to OS
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        picam2.stop()
        if not is_headless:
            cv.destroyAllWindows()

if __name__ == "__main__":
    main()
