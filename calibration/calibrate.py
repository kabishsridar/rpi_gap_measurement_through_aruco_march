import numpy as np
import cv2, glob, os

# --- YOUR SETTINGS ---
CORNERS = (10, 7) 
SQUARE_SIZE_MM = 15.0  # From your board text
IMAGES_DIR = "board_images"
# ---------------------

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((CORNERS[0] * CORNERS[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CORNERS[0], 0:CORNERS[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE_MM

objpoints, imgpoints = [], []
images = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))

print(f"Processing {len(images)} images...")

successful = 0
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CORNERS, None)

    if ret:
        successful += 1
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        print(f"[{successful}] ✓ {os.path.basename(fname)}")

if successful >= 10:
    print(f"\nCalibrating on {successful} images...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    print("-" * 30)
    print(f"Calibration successful!")
    print(f"Reprojection error: {ret:.4f} pixels")
    print(f"Camera Matrix:\n{mtx}")
    
    side = input("\nIs this calibration for the Right side? (y/n): ").strip().lower()
    fn = "camera_params_2.npz" if side == 'y' else "camera_params.npz"
    
    np.savez(fn, mtx=mtx, dist=dist)
    print(f"\nParameters saved to '{fn}'")
else:
    print(f"\nError: Need at least 10 detections (Only found {successful}).")
