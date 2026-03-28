"""
Configuration & Constants
Single source of truth for all tunable parameters.
"""

# ─────────────────────────── Camera ───────────────────────────
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ─────────────────────────── MediaPipe ────────────────────────
MAX_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
MODEL_COMPLEXITY = 1  # 0=fastest, 1=full (more accurate)

# ─────────────────────────── Gesture Thresholds ───────────────
# Finger is "up" when TIP.y < PIP.y by at least this fraction of hand height
FINGER_UP_MARGIN = 0.03
# Thumb is "out" when TIP.x exceeds IP.x laterally by this fraction
THUMB_OUT_MARGIN = 0.04
# Pinch distance (thumb-tip to index-tip) threshold, normalized to hand size
PINCH_THRESHOLD = 0.07
# Debounce: gesture must persist for N consecutive frames to trigger transition
DEBOUNCE_FRAMES = 3
# Minimum confidence for a gesture to be accepted
MIN_GESTURE_CONFIDENCE = 0.65
# Frames the fist must be held to trigger CLEAR
CLEAR_HOLD_FRAMES = 20  # ~0.6s at 30 fps
# Frames the OK sign must be held to trigger SAVE
SAVE_HOLD_FRAMES = 30   # ~1.0s at 30 fps

# ─────────────────────────── Drawing ──────────────────────────
COLOR_PALETTE = [
    (255, 50, 50),     # Blue (BGR)
    (50, 50, 255),     # Red
    (50, 205, 50),     # Green
    (0, 215, 255),     # Gold
    (255, 0, 255),     # Magenta
    (255, 165, 0),     # Light Blue / Cyan-ish
    (255, 255, 255),   # White
    (0, 0, 0),         # Black
]
DEFAULT_COLOR_INDEX = 0
DEFAULT_THICKNESS = 4
MIN_THICKNESS = 2
MAX_THICKNESS = 24
THICKNESS_STEP = 2
ERASER_THICKNESS = 40

# ─────────────────────────── Smoothing ────────────────────────
# Kalman Filter
KALMAN_PROCESS_NOISE = 1e-2
KALMAN_MEASUREMENT_NOISE = 1e-1

# Exponential Moving Average fallback
EMA_ALPHA = 0.45

# ─────────────────────────── UI / HUD ─────────────────────────
HUD_HEIGHT = 60
HUD_BG_COLOR = (30, 30, 30)
HUD_TEXT_COLOR = (220, 220, 220)
HUD_FONT_SCALE = 0.6
HUD_FONT_THICKNESS = 1

TOOLBAR_HEIGHT = 60
TOOLBAR_BG_COLOR = (30, 30, 30)
SWATCH_RADIUS = 18
SWATCH_SPACING = 48

# ─────────────────────────── Toolbar Interaction ──────────────
COLOR_SELECT_COOLDOWN = 15   # frames cooldown between color selections
SIZE_BUTTON_RADIUS = 14
SIZE_CHANGE_COOLDOWN = 12    # frames cooldown between size changes
ACTION_COOLDOWN = 20         # cooldown for Undo/Clear actions

LANDMARK_COLOR = (0, 255, 0)
LANDMARK_CONNECTION_COLOR = (0, 200, 0)
LANDMARK_RADIUS = 3
FINGERTIP_RADIUS = 8
FINGERTIP_DRAW_COLOR = (0, 255, 255)
FINGERTIP_IDLE_COLOR = (200, 200, 200)

INSTRUCTION_FONT_SCALE = 0.45
INSTRUCTION_COLOR = (180, 180, 180)

# ─────────────────────────── Export ───────────────────────────
OUTPUT_DIR = "output"
CANVAS_SAVE_FORMAT = "png"
VIDEO_CODEC = "XVID"
VIDEO_FPS = 20.0

# ─────────────────────────── Landmark Indices ─────────────────
# MediaPipe hand landmark indices
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20
