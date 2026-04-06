import cv2 as cv
import numpy as np

# --- SETTINGS ---
MARKER_SIZE_MM = 53.8             
ARUCO_DICT = cv.aruco.DICT_4X4_50  
WEBCAM_INDEX = 1                  

def get_dummy_calibration(frame_width, frame_height):
    focal_length = frame_width  
    center = (frame_width / 2, frame_height / 2)
    K = np.array([[focal_length, 0, center[0]],
                  [0, focal_length, center[1]],
                  [0, 0, 1]], dtype=np.float32)
    dist = np.zeros(5) 
    return K, dist

def get_marker_pose(corners, size, K, dist):
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
    half = size / 2.0
    obj_pts = np.array([
        [-half,  half, 0], [ half,  half, 0], 
        [ half, -half, 0], [-half, -half, 0]
    ], dtype=np.float32)
    R, _ = cv.Rodrigues(rvec)
    return np.array([np.dot(R, pt) + tvec.ravel() for pt in obj_pts])

def identify_inner_edges(pts_l, pts_r):
    # Left Marker: Right-most two are TR and BR
    l_sorted = pts_l[pts_l[:, 0].argsort()[::-1]]
    right_two = l_sorted[:2]
    TR_L = right_two[right_two[:, 1].argsort()][0]
    BR_L = right_two[right_two[:, 1].argsort()][1]
    TL_L = l_sorted[2:][l_sorted[2:][:, 1].argsort()][0]

    # Right Marker: Left-most two are B and C
    r_sorted = pts_r[pts_r[:, 0].argsort()]
    left_two_r = r_sorted[:2]
    B = left_two_r[left_two_r[:, 1].argsort()][0]
    C = left_two_r[left_two_r[:, 1].argsort()][1]
    
    return TL_L, TR_L, BR_L, B, C

def main():
    cap = cv.VideoCapture(WEBCAM_INDEX)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    K, dist = get_dummy_calibration(1280, 720)
    
    detector = cv.aruco.ArucoDetector(
        cv.aruco.getPredefinedDictionary(ARUCO_DICT), 
        cv.aruco.DetectorParameters()
    )

    print("--- 3D Measurement System Active ---")
    print("Log Order: TR, BR, B, C | A, X | U, V, W | k, Dist")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
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
                    
                    # --- DISPLAY ---
                    disp_pts, _ = cv.projectPoints(np.array([A, X]), np.zeros(3), np.zeros(3), K, dist)
                    pA, pX = tuple(disp_pts[0].ravel().astype(int)), tuple(disp_pts[1].ravel().astype(int))
                    
                    cv.line(frame, pA, pX, (0, 165, 255), 3)
                    cv.circle(frame, pX, 6, (255, 0, 255), -1)
                    cv.putText(frame, f"{distance:.1f} mm", (pA[0], pA[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # --- FULL TELEMETRY LOG ---
                    corners_str = f"TR:({TR_L[0]:.0f},{TR_L[1]:.0f}) BR:({BR_L[0]:.0f},{BR_L[1]:.0f}) B:({B[0]:.0f},{B[1]:.0f}) C:({C[0]:.0f},{C[1]:.0f})"
                    vectors_str = f"A:({A[0]:.0f},{A[1]:.0f}) X:({X[0]:.0f},{X[1]:.0f}) U:({U[0]:.0f},{U[1]:.0f}) V:({V[0]:.1f},{V[1]:.1f}) W:({W[0]:.1f},{W[1]:.1f})"
                    print(f"\r{corners_str} | {vectors_str} | k:{k:.2f} DIST:{distance:.1f}mm", end="")

        cv.imshow("Dynamic ArUco Gap", frame)
        if cv.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
