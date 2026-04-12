import cv2 as cv

# ─── COLORS ───
C_BG       = "#0f172a"
C_PANEL    = "#1e293b"
C_CARD     = "#334155"
C_ACCENT   = "#38bdf8"
C_GREEN    = "#4ade80"
C_RED      = "#f87171"
C_AMBER    = "#fbbf24"
C_TOP      = "#fb923c"
C_BOT      = "#c084fc"
C_TEXT_BRT = "#f8fafc"
C_TEXT_MED = "#94a3b8"

# ─── FONTS ───
F_TITLE = ("Inter", 16, "bold")
F_HEAD  = ("Inter", 13, "bold")
F_BODY  = ("Inter", 12)
F_SMALL = ("Inter", 10)
F_DATA  = ("Inter", 28, "bold")      # Optimized for perfect fit
F_MONO  = ("Consolas", 13)
F_BTN   = ("Inter", 14, "bold")

# ─── CAMERA & ARUCO ───
RESOLUTION = (800, 600)
ARUCO_DICT = cv.aruco.DICT_5X5_100
MARKER_SIZE_TOP = 23.5   # mm
MARKER_SIZE_BOT = 23.5   # mm
CALIB_FILE_1 = "camera_params.npz"
CALIB_FILE_2 = "camera_params_2.npz"

# ─── LOGGING ───
LOG_CSV = "gap_measurements.csv"
LOG_DB  = "gap_measurements.db"

# ─── LOGIC ───
COLLECT_N = 10           # Frames for median
LIGHT_LOW = 40
LIGHT_HIGH = 220
CONTRAST_MIN = 20
PITCH_THRESH = 10.0
YAW_THRESH   = 10.0
ROT_THRESH   = 5.0
