import math

def rotation_to_euler(R):
    """Decompose an OpenCV 3×3 rotation matrix into (pitch, yaw, roll) degrees."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll  = math.degrees(math.atan2( R[2, 1],  R[2, 2]))
        pitch = math.degrees(math.atan2(-R[2, 0],  sy))
        yaw   = math.degrees(math.atan2( R[1, 0],  R[0, 0]))
    else:
        roll  = math.degrees(math.atan2(-R[1, 2],  R[1, 1]))
        pitch = math.degrees(math.atan2(-R[2, 0],  sy))
        yaw   = 0.0
    return pitch, yaw, roll
