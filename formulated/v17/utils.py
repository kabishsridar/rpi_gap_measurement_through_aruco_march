import numpy as np
import math

def rotation_to_euler(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return math.degrees(x), math.degrees(y), math.degrees(z)

def load_calibration(file):
    try:
        data = np.load(file)
        return data['mtx'], data['dist']
    except:
        return None, None
