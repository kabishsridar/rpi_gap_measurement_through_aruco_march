import cv2 as cv

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_MARKER_SIZE = 53.8
ARUCO_DICT          = cv.aruco.DICT_4X4_50
RESOLUTION          = (1280, 720)
COLLECT_N           = 5          # readings to average for initial / final

# Lighting health thresholds (pixel intensity 0-255)
LIGHT_LOW    = 40     # mean brightness below → too dark
LIGHT_HIGH   = 215    # mean brightness above → overexposed
CONTRAST_MIN = 18     # std-dev below → flat / foggy / blown-out

AWB_MODES = {
    "Auto":        0,
    "Tungsten":    1,
    "Fluorescent": 2,
    "Indoor":      3,
    "Daylight":    4,
    "Cloudy":      5,
}

# ── Control Room Aesthetic Palette ───────────────────────────────────────────
C_BG      = "#0f172a"  # Deep slate/navy
C_PANEL   = "#1e293b"  # Rich navy card background
C_CARD    = "#334155"  # Lighter slate for nested cards
C_ACCENT   = "#38bdf8"  # Professional cyan blue
C_TOP      = "#fb923c"  # Vivid orange
C_BOT      = "#c084fc"  # Vivid purple
C_GREEN    = "#4ade80"  # Vibrant green
C_RED      = "#f87171"  # Vibrant red
C_TEXT_BRT = "#f8fafc"  # Off-white
C_TEXT_MED = "#94a3b8"  # Muted blue-gray
C_AMBER    = "#fbbf24"  # Warning amber

# ── Typography ───────────────────────────────────────────────────────────────
F_TITLE = ("Inter", 16, "bold")      # Modern font fallback
F_HEAD  = ("Inter", 13, "bold")
F_BODY  = ("Inter", 12)
F_SMALL = ("Inter", 10)
F_DATA  = ("Inter", 32, "bold")      # Scaled for perfect fit in result cards
F_MONO  = ("Consolas", 13)
F_BTN   = ("Inter", 14, "bold")
