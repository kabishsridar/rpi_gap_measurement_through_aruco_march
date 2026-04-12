import math
import numpy as np
import os

def rotation_to_euler(R):
    """
    Decompose an OpenCV 3×3 rotation matrix into (pitch, yaw, roll) degrees.
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll  = math.degrees(math.atan2( R[2, 1],  R[2, 2]))
        pitch = math.degrees(math.atan2(-R[2, 0],  sy))
        yaw   = math.degrees(math.atan2( R[1, 0],  R[0, 0]))
    else:                               # gimbal-lock fallback
        roll  = math.degrees(math.atan2(-R[1, 2],  R[1, 1]))
        pitch = math.degrees(math.atan2(-R[2, 0],  sy))
        yaw   = 0.0
    return pitch, yaw, roll

def load_calib(fixed_side):
    f = "camera_params_2.npz" if fixed_side == "Right" else "camera_params.npz"
    if os.path.exists(f):
        d = np.load(f)
        return d['camera_matrix'], d['dist_coeff']
    return (np.array([[1280, 0, 640], [0, 1280, 360], [0, 0, 1]], dtype=np.float32),
            np.zeros(5))
