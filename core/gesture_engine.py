"""
AirCanvas — Gesture Recognition Engine
Classifies hand landmark data into discrete gestures using finger-state analysis.
"""

from enum import Enum
from config import settings
from utils.geometry import distance, hand_bbox_size


class Gesture(Enum):
    """All recognized gesture types."""
    NONE = "none"
    DRAW = "draw"                # Index up, others folded → draw mode
    STOP = "stop"                # Open palm, all 5 fingers up → idle
    ERASE = "erase"              # Index + middle up, others folded → eraser
    COLOR_NEXT = "color_next"    # Thumb + pinky out, others folded → cycle color
    SIZE_UP = "size_up"          # Thumbs up → increase brush
    SIZE_DOWN = "size_down"      # Thumbs down → decrease brush
    CLEAR = "clear"              # Closed fist (held) → clear canvas
    SAVE = "save"                # OK sign: thumb-index pinch + 3 fingers up
    UNDO = "undo"                # Index + pinky up, middle + ring folded


class GestureEngine:
    """
    Classifies a set of hand landmarks into a Gesture.
    
    Design goals:
    - Deterministic: same landmarks → same gesture, every time.
    - Confidence-scored: returns a float [0, 1] indicating classification strength.
    - Robust to hand size variation: distances normalized by hand bounding-box diagonal.
    """

    def __init__(self):
        self._finger_up_margin = settings.FINGER_UP_MARGIN
        self._thumb_out_margin = settings.THUMB_OUT_MARGIN
        self._pinch_threshold = settings.PINCH_THRESHOLD

    def classify(
        self,
        landmarks: list[tuple[float, float, float]],
        handedness: str = "Right",
    ) -> tuple[Gesture, float]:
        """
        Classify the current hand pose.
        
        Args:
            landmarks: 21 normalized (x,y,z) landmarks from MediaPipe.
            handedness: "Left" or "Right" (MediaPipe's camera-perspective label).
            
        Returns:
            (Gesture, confidence) — confidence is 0.0–1.0.
        """
        if not landmarks or len(landmarks) < 21:
            return Gesture.NONE, 0.0

        fingers_up = self._get_fingers_up(landmarks, handedness)
        hand_size = hand_bbox_size(landmarks)

        # ── Priority-ordered classification ──────────────────────────

        # 1. STOP — all 5 fingers up (universal escape)
        if all(fingers_up):
            conf = self._stop_confidence(landmarks)
            return Gesture.STOP, conf

        # 2. SAVE — OK sign: thumb-index pinch + middle/ring/pinky up
        if self._is_ok_sign(landmarks, hand_size, fingers_up):
            return Gesture.SAVE, 0.85

        # 3. ERASE — index + middle up, ring + pinky down
        if (fingers_up[1] and fingers_up[2]
                and not fingers_up[3] and not fingers_up[4]):
            conf = 0.8 if not fingers_up[0] else 0.65
            return Gesture.ERASE, conf

        # 4. DRAW — only index up
        if (fingers_up[1]
                and not fingers_up[2] and not fingers_up[3] and not fingers_up[4]):
            conf = 0.85 if not fingers_up[0] else 0.7
            return Gesture.DRAW, conf

        # 5. COLOR_NEXT — thumb + pinky out, others folded (shaka)
        if (fingers_up[0] and fingers_up[4]
                and not fingers_up[1] and not fingers_up[2] and not fingers_up[3]):
            return Gesture.COLOR_NEXT, 0.9

        # 6. UNDO — index + pinky up, middle + ring folded
        if (fingers_up[1] and fingers_up[4]
                and not fingers_up[2] and not fingers_up[3]):
            conf = 0.8 if not fingers_up[0] else 0.65
            return Gesture.UNDO, conf

        # 7. SIZE_UP — only thumb up, all others down
        if (fingers_up[0]
                and not fingers_up[1] and not fingers_up[2]
                and not fingers_up[3] and not fingers_up[4]):
            if self._is_thumb_up(landmarks):
                return Gesture.SIZE_UP, 0.85
            else:
                return Gesture.SIZE_DOWN, 0.85

        # 8. CLEAR — closed fist (no fingers up)
        if not any(fingers_up):
            return Gesture.CLEAR, 0.9

        return Gesture.NONE, 0.0

    # ── Finger-state helpers ─────────────────────────────────────

    def _get_fingers_up(
        self,
        landmarks: list[tuple[float, float, float]],
        handedness: str,
    ) -> list[bool]:
        """
        Determine which fingers are extended.
        Returns [thumb, index, middle, ring, pinky] as booleans.
        """
        fingers = [False] * 5

        # Thumb: compare TIP.x vs IP.x
        # MediaPipe handedness is camera-relative, so:
        #   "Right" in MediaPipe → user's left hand → thumb TIP.x > IP.x means up
        #   "Left" in MediaPipe → user's right hand → thumb TIP.x < IP.x means up
        thumb_tip = landmarks[settings.THUMB_TIP]
        thumb_ip = landmarks[settings.THUMB_IP]
        thumb_mcp = landmarks[settings.THUMB_MCP]

        if handedness == "Right":
            # Camera-right = user-left hand
            fingers[0] = thumb_tip[0] > thumb_ip[0] + self._thumb_out_margin
        else:
            fingers[0] = thumb_tip[0] < thumb_ip[0] - self._thumb_out_margin

        # Fingers 1–4: compare TIP.y vs PIP.y (lower y = higher on screen)
        tip_pip_pairs = [
            (settings.INDEX_TIP, settings.INDEX_PIP),
            (settings.MIDDLE_TIP, settings.MIDDLE_PIP),
            (settings.RING_TIP, settings.RING_PIP),
            (settings.PINKY_TIP, settings.PINKY_PIP),
        ]
        for i, (tip_idx, pip_idx) in enumerate(tip_pip_pairs, start=1):
            fingers[i] = landmarks[tip_idx][1] < landmarks[pip_idx][1] - self._finger_up_margin

        return fingers

    def _is_ok_sign(
        self,
        landmarks: list[tuple[float, float, float]],
        hand_size: float,
        fingers_up: list[bool],
    ) -> bool:
        """Detect OK sign: thumb tip touching index tip, other 3 fingers up."""
        thumb_tip = landmarks[settings.THUMB_TIP]
        index_tip = landmarks[settings.INDEX_TIP]
        pinch_dist = distance(thumb_tip, index_tip) / hand_size

        return (
            pinch_dist < self._pinch_threshold
            and fingers_up[2] and fingers_up[3] and fingers_up[4]
        )

    def _is_thumb_up(self, landmarks: list[tuple[float, float, float]]) -> bool:
        """Check if thumb is pointing upward (vs downward)."""
        thumb_tip = landmarks[settings.THUMB_TIP]
        thumb_mcp = landmarks[settings.THUMB_MCP]
        # Thumb tip above MCP = thumbs up
        return thumb_tip[1] < thumb_mcp[1]

    def _stop_confidence(self, landmarks: list[tuple[float, float, float]]) -> float:
        """
        Compute confidence for STOP gesture based on how clearly
        all fingertips are above their PIP joints.
        """
        tip_pip_pairs = [
            (settings.INDEX_TIP, settings.INDEX_PIP),
            (settings.MIDDLE_TIP, settings.MIDDLE_PIP),
            (settings.RING_TIP, settings.RING_PIP),
            (settings.PINKY_TIP, settings.PINKY_PIP),
        ]
        margins = []
        for tip_idx, pip_idx in tip_pip_pairs:
            margin = landmarks[pip_idx][1] - landmarks[tip_idx][1]
            margins.append(max(0.0, margin))

        avg_margin = sum(margins) / len(margins)
        # Scale to [0.6, 1.0] range
        return min(1.0, 0.6 + avg_margin * 8.0)
