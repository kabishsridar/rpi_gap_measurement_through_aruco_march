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
            return None, None
            
        m_data = []
        for i in range(len(ids)):
            raw = corners_raw[i][0]
            idx_x = np.argsort(raw[:, 0])
            lp, rp = raw[idx_x[:2]], raw[idx_x[2:]]
            tl, bl = lp[np.argmin(lp[:, 1])], lp[np.argmax(lp[:, 1])]
            tr, br = rp[np.argmin(rp[:, 1])], rp[np.argmax(rp[:, 1])]
            cy, cx = (tl[1]+tr[1]+br[1]+bl[1]) / 4.0, (tl[0]+tr[0]+br[0]+bl[0]) / 4.0
            m_data.append({"c": np.array([tl, tr, br, bl], dtype=np.float32), "c_raw": raw, "y": cy, "x": cx})

        m_data.sort(key=lambda m: m["y"])
        top_m = m_data[:2]
        bot_m = m_data[2:4] if len(m_data) >= 4 else []
        if len(m_data) == 2:
            dy_s, dx_s = abs(m_data[0]["y"] - m_data[1]["y"]), abs(m_data[0]["x"] - m_data[1]["x"])
            if dx_s > dy_s:
                avg_y = (m_data[0]["y"] + m_data[1]["y"]) / 2
                top_m, bot_m = (m_data, []) if avg_y < RESOLUTION[1]/2 else ([], m_data)
            else:
                top_m, bot_m = [m_data[0]], [m_data[1]]
        elif len(m_data) == 3:
            top_m, bot_m = m_data[:2], m_data[2:]
            
        return top_m, bot_m

    def get_pose(self, c_raw, K, dist_c, size):
        h = size / 2.0
        obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)
        _, rv, tv = cv.solvePnP(obj, c_raw, K, dist_c)
        R, _ = cv.Rodrigues(rv)
        pts3d = np.array([R @ pt + tv.ravel() for pt in obj])
        roll_r = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        pitch, yaw, _ = rotation_to_euler(R)
        return pts3d, roll_r, float(tv[2, 0]), pitch, yaw

    def inplane_rot(self, c_sorted):
        return abs(math.degrees(math.atan2(c_sorted[1, 1] - c_sorted[0, 1], c_sorted[1, 0] - c_sorted[0, 0])))

    def inner_outer(self, pts, inner_is_right):
        order = pts[:, 0].argsort()
        inner = pts[order[2:]] if inner_is_right else pts[order[:2]]
        outer = pts[order[:2]] if inner_is_right else pts[order[2:]]
        inner, outer = inner[inner[:, 1].argsort()], outer[outer[:, 1].argsort()]
        return inner[0], inner[1], outer[0]
