# Detection thresholds
HAND_MOUTH_DISTANCE_RATIO = 0.20   # wrist-to-lip distance as fraction of face height
VAPOR_THRESHOLD = 0.10             # fraction of ROI pixels that must be diffuse bright
VAPOR_WINDOW_FRAMES = 8            # sliding window size
VAPOR_WINDOW_MIN_HITS = 5          # min frames in window to confirm vapor

# Timing
GESTURE_COOLDOWN_SEC = 3.0         # seconds to hold gesture active state
ALERT_COOLDOWN_SEC = 15.0          # min seconds between alerts
CONFIDENCE_WINDOW_SEC = 3.0        # gesture and vapor must fire within this window

# Alert message
ALERT_TITLE = "DETECTED"
ALERT_MESSAGE = "Tu fais quoi fils de pute"

# Camera
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
PROCESS_EVERY_N_FRAMES = 2         # skip frames to save CPU

# Debug overlay
SHOW_DEBUG = True
