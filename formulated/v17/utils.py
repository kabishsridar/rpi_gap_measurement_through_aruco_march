import math
import numpy as np
import os

def rot_to_euler(R):
    """
    Decompose an OpenCV 3x3 rotation matrix into Euler angles (pitch, yaw, roll).
    
    Args:
        R (numpy.ndarray): 3x3 rotation matrix.
        
    Returns:
        tuple: (pitch, yaw, roll) in degrees.
    """
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        return math.degrees(math.atan2(R[2, 1], R[2, 2])), \
               math.degrees(math.atan2(-R[2, 0], sy)), \
               math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return math.degrees(math.atan2(-R[1, 2], R[1, 1])), \
           math.degrees(math.atan2(-R[2, 0], sy)), 0.0

def load_calibration_params(fixed_side):
    """
    Load camera matrix and distortion coefficients from NPZ files.
    
    Args:
        fixed_side (str): "Left" or "Right" to determine which file to load.
        
    Returns:
        tuple: (camera_matrix, dist_coeffs) as numpy arrays.
    """
    filename = "camera_params_2.npz" if fixed_side == "Right" else "camera_params.npz"
    if os.path.exists(filename):
        data = np.load(filename)
        mtx_key = 'camera_matrix' if 'camera_matrix' in data else 'mtx'
        dist_key = 'dist_coeff' if 'dist_coeff' in data else 'dist'
        return data[mtx_key], data[dist_key]
    # Fallback to defaults
    return np.array([[1280, 0, 640], [0, 1280, 360], [0, 0, 1]], dtype=np.float32), np.zeros(5)
