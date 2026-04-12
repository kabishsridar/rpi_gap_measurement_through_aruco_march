import cv2 as cv
import numpy as np
import math
from constants import *
from utils import rotation_to_euler

class MeasurementEngine:
    def __init__(self):
        self.detector = cv.aruco.ArucoDetector(
            cv.aruco.getPredefinedDictionary(ARUCO_DICT),
            cv.aruco.DetectorParameters())

    def analyze_lighting(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        mean_b = float(np.mean(gray))
        std_b  = float(np.std(gray))
        if mean_b < LIGHT_LOW: status = "dark"
        elif mean_b > LIGHT_HIGH: status = "bright"
        elif std_b < CONTRAST_MIN: status = "flat"
        else: status = "ok"
        return {"status": status, "mean": mean_b, "std": std_b}, gray

    def detect_markers(self, gray):
        corners_raw, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return [], []
            
        m_data = []
        for i in range(len(ids)):
            p = corners_raw[i][0]
            cy = np.mean(p[:, 1])
            cx = np.mean(p[:, 0])
            m_data.append({"c": p, "y": cy, "x": cx, "id": int(ids[i][0])})

        m_data.sort(key=lambda m: m["y"])
        top_pair = m_data[:2]
        bot_pair = m_data[2:4] if len(m_data) >= 4 else []
        
        # Simple logical fallback for 2-marker case
        if len(m_data) == 2:
            dy, dx = abs(m_data[0]["y"]-m_data[1]["y"]), abs(m_data[0]["x"]-m_data[1]["x"])
            if dx > dy: # Horizontal pair
                if m_data[0]["y"] < RESOLUTION[1]/2: top_pair, bot_pair = m_data, []
                else: top_pair, bot_pair = [], m_data
            else: # Vertical pair
                top_pair, bot_pair = [m_data[0]], [m_data[1]]
        elif len(m_data) == 3:
            top_pair, bot_pair = m_data[:2], m_data[2:]

        return top_pair, bot_pair

    def get_pose(self, corners, K, dist_c, size):
        h = size / 2.0
        obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)
        _, rv, tv = cv.solvePnP(obj, corners, K, dist_c)
        R, _ = cv.Rodrigues(rv)
        pts3d = np.array([R @ pt + tv.ravel() for pt in obj])
        roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        pitch, yaw, _ = rotation_to_euler(R)
        return pts3d, roll, float(tv[2, 0]), pitch, yaw

    def inplane_rot(self, c):
        # Angle of top edge relative to horizon
        return abs(math.degrees(math.atan2(c[1, 1] - c[0, 1], c[1, 0] - c[0, 0])))

    def inner_outer_pts(self, pts, is_right_face):
        # Sort by X
        idx = np.argsort(pts[:, 0])
        # Inner edges are the ones closest to the gap
        inner = pts[idx[2:]] if is_right_face else pts[idx[:2]]
        outer = pts[idx[:2]] if is_right_face else pts[idx[2:]]
        # Sort by Y
        inner = inner[np.argsort(inner[:, 1])]
        return inner[0], inner[1], outer[0]
