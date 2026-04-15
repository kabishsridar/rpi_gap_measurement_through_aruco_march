import cv2 as cv

# ── Marker Settings ──────────────────────────────────────────────────────────
# NEW NAMING CONVENTION (LT1, LT2, LB1, LB2, RT1, RT2, RB1, RB2):
# - L = Left side, R = Right side
# - T = Top position, B = Bottom position
# - This enables "L1-L2 pair" (Left side: LT + LB markers) and "R1-R2 pair" (Right side: RT + RB markers)
DEFAULT_MARKER_SIZE = 53.8
ARUCO_DICT          = cv.aruco.DICT_4X4_50
RESOLUTION          = (1280, 720)
COLLECT_N           = 5

# Marker position constants for new naming convention
MARKER_POSITIONS = ['LT', 'LB', 'RT', 'RB']  # Left-Top, Left-Bottom, Right-Top, Right-Bottom
SIDE_PAIRS = ['L', 'R']  # L1-L2 pair (Left side), R1-R2 pair (Right side)

# ── Aesthetics (Control Room Design) ──────────────────────────────────────────
C_BG      = "#0f172a"
C_PANEL   = "#1e293b"
C_CARD    = "#334155"
C_ACCENT  = "#38bdf8"
C_LEFT    = "#fb923c"  # Orange - Left side (L1-L2 pair)
C_RIGHT   = "#c084fc"  # Purple - Right side (R1-R2 pair)
C_GREEN   = "#4ade80"
C_RED     = "#f87171"
C_TEXT_BRT= "#f8fafc"
C_TEXT_MED= "#94a3b8"
C_AMBER   = "#fbbf24"

# Legacy color names maintained for backward compatibility
C_TOP = C_LEFT   # Top now maps to Left side
C_BOT = C_RIGHT  # Bottom now maps to Right side

# Typography
F_TITLE = ("Inter", 16, "bold")
F_HEAD  = ("Inter", 13, "bold")
F_BODY  = ("Inter", 12)
F_SMALL = ("Inter", 10)
F_DATA  = ("Inter", 32, "bold")
F_MONO  = ("Consolas", 18)
F_BTN   = ("Inter", 15, "bold")
