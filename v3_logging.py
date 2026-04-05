import cv2 as cv
import numpy as np

# --- SETTINGS ---
MARKER_SIZE_MM = 53.8             
ARUCO_DICT = cv.aruco.DICT_4X4_50  
WEBCAM_INDEX = 0                  

def get_dummy_calibration(frame_width, frame_height):
    focal_length = frame_width  
    center = (frame_width / 2, frame_height / 2)
    K = np.array([[focal_length, 0, center[0]],
                  [0, focal_length, center[1]],
                  [0, 0, 1]], dtype=np.float32)
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
    left_two_r = r_sorted[:2]
    B = left_two_r[left_two_r[:, 1].argsort()][0]
    C = left_two_r[left_two_r[:, 1].argsort()][1]
    return TL_L, TR_L, BR_L, B, C

def main():
    cap = cv.VideoCapture(WEBCAM_INDEX)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    K, dist = get_dummy_calibration(1280, 720)
    detector = cv.aruco.ArucoDetector(cv.aruco.getPredefinedDictionary(ARUCO_DICT), cv.aruco.DetectorParameters())
    
    print(f"--- 3D System Ready --- Streaming to Terminal ---")

    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        
        if ids is not None and len(ids) >= 2:
            order = np.argsort([np.mean(c[0], axis=0)[0] for c in corners])
            rv_l, tv_l = get_marker_pose(corners[order[0]][0], MARKER_SIZE_MM, K, dist)
            pts_l = get_3d_corners(rv_l, tv_l, MARKER_SIZE_MM)
            rv_r, tv_r = get_marker_pose(corners[order[1]][0], MARKER_SIZE_MM, K, dist)
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
                    distance = np.linalg.norm(X - A)
                    
                    # --- TERMINAL OUTPUT ---
                    log = (f"A:({A[0]:.0f},{A[1]:.0f}) X:({X[0]:.0f},{X[1]:.0f}) "
                           f"TR:({TR_L[0]:.0f},{TR_L[1]:.0f}) BR:({BR_L[0]:.0f},{BR_L[1]:.0f}) "
                           f"B:({B[0]:.0f},{B[1]:.0f}) C:({C[0]:.0f},{C[1]:.0f}) | "
                           f"U:({U_vec[0]:.0f},{U_vec[1]:.0f}) V:({V_vec[0]:.1f},{V_vec[1]:.1f}) "
                           f"W:({W_vec[0]:.1f},{W_vec[1]:.1f}) | k:{k:.2f} Dist:{distance:.1f}mm")
                    print(f"\r{log}", end="")

                    # --- LIVE FEED DATA DISPLAY ---
                    y_off = 30
                    for label, val in [("A", A), ("X", X), ("TR", TR_L), ("BR", BR_L), ("B", B), ("C", C)]:
                        text = f"{label}:({val[0]:.0f},{val[1]:.0f},{val[2]:.0f})"
                        cv.putText(frame, text, (10, y_off), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        y_off += 20
                    
                    cv.putText(frame, f"GAP: {distance:.1f} mm", (10, y_off + 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Drawing
                    disp_pts, _ = cv.projectPoints(np.array([A, X]), np.zeros(3), np.zeros(3), K, dist)
                    pA, pX = tuple(disp_pts[0].ravel().astype(int)), tuple(disp_pts[1].ravel().astype(int))
                    cv.line(frame, pA, pX, (0, 165, 255), 3)
                    cv.circle(frame, pX, 6, (255, 0, 255), -1)

        cv.imshow("V3 Display & Log", frame)
        if cv.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
