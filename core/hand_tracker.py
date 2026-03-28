"""
Hand Tracker Module
Wraps MediaPipe Hands to provide clean landmark data.
"""

from dataclasses import dataclass, field
import numpy as np
import cv2
import mediapipe as mp

from config import settings
from utils.geometry import normalize_to_pixel


@dataclass
class HandResult:
    """Processed result for a single detected hand."""
    landmarks: list[tuple[float, float, float]]   # (x, y, z) normalized [0,1]
    pixel_landmarks: list[tuple[int, int]]          # (px, py) in frame coords
    handedness: str                                  # "Left" or "Right"
    confidence: float                                # Detection confidence


class HandTracker:
    """
    MediaPipe Hands wrapper.
    
    Provides a simple interface to detect hands and extract landmarks.
    Handles initialization, processing, and resource cleanup.
    """

    def __init__(
        self,
        max_hands: int = settings.MAX_HANDS,
        detection_confidence: float = settings.MIN_DETECTION_CONFIDENCE,
        tracking_confidence: float = settings.MIN_TRACKING_CONFIDENCE,
        model_complexity: int = settings.MODEL_COMPLEXITY,
    ):
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=model_complexity,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_draw_styles = mp.solutions.drawing_styles

    def process(
        self,
        frame: np.ndarray,
        frame_width: int = settings.FRAME_WIDTH,
        frame_height: int = settings.FRAME_HEIGHT,
    ) -> list[HandResult]:
        """
        Process a BGR frame and return a list of HandResult objects.
        
        Args:
            frame: BGR image from OpenCV (np.uint8 array).
            frame_width: Width used for pixel coordinate conversion.
            frame_height: Height used for pixel coordinate conversion.
            
        Returns:
            List of HandResult (empty list if no hands detected).
        """
        # MediaPipe expects a contiguous RGB array.
        # Slicing with [::-1] creates a non-contiguous array which causes the C++ backend
        # to read garbage memory in the bottom half of the image!
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        hands: list[HandResult] = []

        if results.multi_hand_landmarks is None:
            return hands

        for hand_lms, hand_info in zip(
            results.multi_hand_landmarks,
            results.multi_handedness,
        ):
            # Extract normalized landmarks
            landmarks = [
                (lm.x, lm.y, lm.z) for lm in hand_lms.landmark
            ]

            # Convert to pixel coordinates
            # NOTE: frame is already flipped in main.py, so mirror=False to avoid double-flip
            pixel_landmarks = [
                normalize_to_pixel(lm.x, lm.y, frame_width, frame_height, mirror=False)
                for lm in hand_lms.landmark
            ]

            # MediaPipe labels are relative to the camera's perspective,
            # so "Left" in MediaPipe = user's right hand when mirrored.
            handedness = hand_info.classification[0].label
            confidence = hand_info.classification[0].score

            hands.append(HandResult(
                landmarks=landmarks,
                pixel_landmarks=pixel_landmarks,
                handedness=handedness,
                confidence=confidence,
            ))

        return hands

    def draw_landmarks(self, frame: np.ndarray, results) -> np.ndarray:
        """
        Draw MediaPipe hand landmarks on a frame (for debugging).
        This method is optional — the UIRenderer draws custom landmarks.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_results = self._hands.process(rgb)
        if raw_results.multi_hand_landmarks:
            for hand_lms in raw_results.multi_hand_landmarks:
                self._mp_draw.draw_landmarks(
                    frame,
                    hand_lms,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_draw_styles.get_default_hand_landmarks_style(),
                    self._mp_draw_styles.get_default_hand_connections_style(),
                )
        return frame

    def release(self):
        """Release MediaPipe resources."""
        self._hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
