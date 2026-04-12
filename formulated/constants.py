import cv2 as cv

# ── Marker Settings ──────────────────────────────────────────────────────────
DEFAULT_MARKER_SIZE = 53.8
ARUCO_DICT          = cv.aruco.DICT_4X4_50
RESOLUTION          = (1280, 720)
COLLECT_N           = 5

# ── Aesthetics (Control Room Design) ──────────────────────────────────────────
C_BG      = "#0f172a"
C_PANEL   = "#1e293b"
C_CARD    = "#334155"
C_ACCENT  = "#38bdf8"
C_TOP     = "#fb923c"
C_BOT     = "#c084fc"
C_GREEN   = "#4ade80"
C_RED     = "#f87171"
C_TEXT_BRT= "#f8fafc"
C_TEXT_MED= "#94a3b8"
C_AMBER   = "#fbbf24"

# Typography
F_TITLE = ("Inter", 20, "bold")
F_HEAD  = ("Inter", 18, "bold")
F_BODY  = ("Inter", 16)
F_SMALL = ("Inter", 14)
F_DATA  = ("Inter", 32, "bold")
F_MONO  = ("Consolas", 18)
F_BTN   = ("Inter", 14, "bold")
