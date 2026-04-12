import cv2 as cv

# ─── Constants ───
DEFAULT_MARKER_SIZE = 53.8
ARUCO_DICT          = cv.aruco.DICT_4X4_50
RESOLUTION          = (1280, 720)
COLLECT_N           = 5

# Lighting thresholds
LIGHT_LOW    = 40
LIGHT_HIGH   = 215
CONTRAST_MIN = 18

AWB_MODES = {
    "Auto":        0,
    "Tungsten":    1,
    "Fluorescent": 2,
    "Indoor":      3,
    "Daylight":    4,
    "Cloudy":      5,
}

# ─── Colors ───
C_BG      = "#0f172a"
C_PANEL   = "#1e293b"
C_CARD    = "#334155"
C_ACCENT   = "#38bdf8"
C_TOP      = "#fb923c"
C_BOT      = "#c084fc"
C_GREEN    = "#4ade80"
C_RED      = "#f87171"
C_TEXT_BRT = "#f8fafc"
C_TEXT_MED = "#94a3b8"
C_AMBER    = "#fbbf24"

# ─── Typography ───
F_TITLE = ("Inter", 16, "bold")
F_HEAD  = ("Inter", 13, "bold")
F_BODY  = ("Inter", 12)
F_SMALL = ("Inter", 10)
F_DATA  = ("Inter", 28, "bold")      # Optimized for perfect fit
F_MONO  = ("Consolas", 13)
F_BTN   = ("Inter", 14, "bold")
