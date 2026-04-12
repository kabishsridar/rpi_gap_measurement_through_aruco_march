import cv2 as cv
import numpy as np
import math
from constants import *
from utils import rotation_to_euler

class MeasurementEngine:
    def __init__(self):
        pass

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
        inner = inner[inner[:, 1].argsort()]
        outer = outer[outer[:, 1].argsort()]
        return inner[0], inner[1], outer[0]
