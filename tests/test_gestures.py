"""
AirCanvas — Unit Tests for Gesture Engine & State Manager

Tests the pure-logic modules that don't require a camera.
Uses synthetic landmark data to verify gesture classification
and FSM transitions.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.gesture_engine import GestureEngine, Gesture
from core.state_manager import StateManager, AppMode, ActionEvent
from core.smoothing import KalmanFilter2D, ExponentialSmoother
from config import settings


# ─────────────────────── Helpers ──────────────────────────────

def make_landmarks(
    thumb_up=False,
    index_up=False,
    middle_up=False,
    ring_up=False,
    pinky_up=False,
    thumb_down=False,
    pinch=False,
) -> list[tuple[float, float, float]]:
    """
    Generate synthetic 21-landmark data for testing.
    
    Each finger "up" means TIP.y < PIP.y.
    Each finger "down" means TIP.y > PIP.y.
    Thumb "up" means TIP.x > IP.x (for "Right" handedness in MediaPipe).
    
    We use "Right" handedness (camera perspective) in all tests, which means
    the thumb extends in the +x direction when it's "up".
    """
    # Base hand centered at (0.5, 0.5)
    # Wrist at (0.5, 0.8)
    landmarks = [(0.0, 0.0, 0.0)] * 21

    # Wrist
    landmarks[settings.WRIST] = (0.5, 0.8, 0.0)

    # Thumb chain
    landmarks[settings.THUMB_CMC] = (0.55, 0.7, 0.0)
    landmarks[settings.THUMB_MCP] = (0.58, 0.65, 0.0)
    landmarks[settings.THUMB_IP] = (0.60, 0.60, 0.0)

    if thumb_up:
        # Thumb tip clearly to the right of IP and above MCP (thumbs up)
        landmarks[settings.THUMB_TIP] = (0.70, 0.50, 0.0)
    elif thumb_down:
        # Thumb tip to the right but below MCP (thumbs down)
        landmarks[settings.THUMB_TIP] = (0.70, 0.75, 0.0)
    else:
        # Folded: tip is at same x or behind IP
        landmarks[settings.THUMB_TIP] = (0.58, 0.62, 0.0)

    # Index finger
    landmarks[settings.INDEX_MCP] = (0.50, 0.60, 0.0)
    landmarks[settings.INDEX_PIP] = (0.48, 0.50, 0.0)
    landmarks[settings.INDEX_DIP] = (0.47, 0.45, 0.0)

    if index_up:
        landmarks[settings.INDEX_TIP] = (0.46, 0.35, 0.0)
    elif pinch:
        # For pinch, index tip goes up to meet thumb at a mid point
        landmarks[settings.INDEX_TIP] = (0.62, 0.45, 0.0)
    else:
        landmarks[settings.INDEX_TIP] = (0.49, 0.58, 0.0)

    # Middle finger
    landmarks[settings.MIDDLE_MCP] = (0.45, 0.58, 0.0)
    landmarks[settings.MIDDLE_PIP] = (0.43, 0.48, 0.0)
    landmarks[settings.MIDDLE_DIP] = (0.42, 0.43, 0.0)

    if middle_up:
        landmarks[settings.MIDDLE_TIP] = (0.41, 0.33, 0.0)
    else:
        landmarks[settings.MIDDLE_TIP] = (0.44, 0.56, 0.0)

    # Ring finger
    landmarks[settings.RING_MCP] = (0.40, 0.60, 0.0)
    landmarks[settings.RING_PIP] = (0.38, 0.50, 0.0)
    landmarks[settings.RING_DIP] = (0.37, 0.45, 0.0)

    if ring_up:
        landmarks[settings.RING_TIP] = (0.36, 0.35, 0.0)
    else:
        landmarks[settings.RING_TIP] = (0.39, 0.58, 0.0)

    # Pinky finger
    landmarks[settings.PINKY_MCP] = (0.35, 0.62, 0.0)
    landmarks[settings.PINKY_PIP] = (0.33, 0.52, 0.0)
    landmarks[settings.PINKY_DIP] = (0.32, 0.47, 0.0)

    if pinky_up:
        landmarks[settings.PINKY_TIP] = (0.31, 0.37, 0.0)
    else:
        landmarks[settings.PINKY_TIP] = (0.34, 0.60, 0.0)

    # Override for pinch (OK sign): thumb tip touches index tip
    # Both are at an intermediate position — thumb is NOT extended
    if pinch:
        landmarks[settings.THUMB_TIP] = (0.625, 0.455, 0.0)
        landmarks[settings.INDEX_TIP] = (0.62, 0.45, 0.0)

    return landmarks


# ─────────────────────── Gesture Engine Tests ─────────────────

class TestGestureEngine:
    """
    All tests use "Right" handedness (MediaPipe camera perspective),
    which means thumb is "up/out" when TIP.x > IP.x.
    Our synthetic data has thumb extending in the +x direction.
    """

    def setup_method(self):
        self.engine = GestureEngine()

    def test_draw_gesture(self):
        """Index up, all others down → DRAW."""
        lm = make_landmarks(index_up=True)
        gesture, conf = self.engine.classify(lm, "Right")
        assert gesture == Gesture.DRAW
        assert conf >= 0.6

    def test_stop_gesture(self):
        """All 5 fingers up → STOP."""
        lm = make_landmarks(
            thumb_up=True, index_up=True, middle_up=True,
            ring_up=True, pinky_up=True
        )
        gesture, conf = self.engine.classify(lm, "Right")
        assert gesture == Gesture.STOP
        assert conf >= 0.6

    def test_erase_gesture(self):
        """Index + middle up → ERASE."""
        lm = make_landmarks(index_up=True, middle_up=True)
        gesture, conf = self.engine.classify(lm, "Right")
        assert gesture == Gesture.ERASE
        assert conf >= 0.6

    def test_color_next_gesture(self):
        """Thumb + pinky out → COLOR_NEXT."""
        lm = make_landmarks(thumb_up=True, pinky_up=True)
        gesture, conf = self.engine.classify(lm, "Right")
        assert gesture == Gesture.COLOR_NEXT
        assert conf >= 0.6

    def test_clear_gesture(self):
        """All fingers down (fist) → CLEAR."""
        lm = make_landmarks()
        gesture, conf = self.engine.classify(lm, "Right")
        assert gesture == Gesture.CLEAR
        assert conf >= 0.6

    def test_size_up_gesture(self):
        """Only thumb up, pointing upward → SIZE_UP."""
        lm = make_landmarks(thumb_up=True)
        gesture, conf = self.engine.classify(lm, "Right")
        assert gesture == Gesture.SIZE_UP
        assert conf >= 0.6

    def test_save_gesture(self):
        """OK sign: thumb-index pinch + 3 fingers up → SAVE."""
        lm = make_landmarks(pinch=True, middle_up=True, ring_up=True, pinky_up=True)
        gesture, conf = self.engine.classify(lm, "Right")
        assert gesture == Gesture.SAVE
        assert conf >= 0.6

    def test_undo_gesture(self):
        """Index + pinky up, middle + ring down → UNDO."""
        lm = make_landmarks(index_up=True, pinky_up=True)
        gesture, conf = self.engine.classify(lm, "Right")
        assert gesture == Gesture.UNDO
        assert conf >= 0.6

    def test_none_with_empty_landmarks(self):
        """Empty landmarks → NONE."""
        gesture, conf = self.engine.classify([], "Right")
        assert gesture == Gesture.NONE


# ─────────────────────── State Manager Tests ──────────────────

class TestStateManager:
    def setup_method(self):
        self.sm = StateManager(debounce_frames=2)

    def _send_gesture(self, gesture, confidence=0.9, times=1):
        """Helper to send a gesture N times."""
        result = None
        for _ in range(times):
            result = self.sm.update(gesture, confidence)
        return result

    def test_initial_state(self):
        assert self.sm.current_mode == AppMode.IDLE

    def test_draw_transition(self):
        """DRAW gesture sustained for debounce threshold → DRAWING mode."""
        self._send_gesture(Gesture.DRAW, times=3)
        assert self.sm.current_mode == AppMode.DRAWING

    def test_stop_returns_to_idle(self):
        """STOP gesture from DRAWING → IDLE."""
        self._send_gesture(Gesture.DRAW, times=3)
        assert self.sm.current_mode == AppMode.DRAWING
        self._send_gesture(Gesture.STOP, times=3)
        assert self.sm.current_mode == AppMode.IDLE

    def test_erase_transition(self):
        """ERASE gesture → ERASING mode."""
        self._send_gesture(Gesture.ERASE, times=3)
        assert self.sm.current_mode == AppMode.ERASING

    def test_debounce_prevents_flicker(self):
        """Single-frame gesture should NOT trigger transition."""
        self._send_gesture(Gesture.DRAW, times=1)
        assert self.sm.current_mode == AppMode.IDLE  # Still idle

    def test_low_confidence_ignored(self):
        """Gestures below confidence threshold are treated as NONE."""
        self._send_gesture(Gesture.DRAW, confidence=0.2, times=5)
        assert self.sm.current_mode == AppMode.IDLE

    def test_color_next_is_instant(self):
        """COLOR_NEXT fires once and doesn't change mode."""
        mode, action = self.sm.update(Gesture.COLOR_NEXT, 0.9)
        assert action == ActionEvent.COLOR_NEXT
        assert mode == AppMode.IDLE

    def test_color_next_no_repeat(self):
        """Sustained COLOR_NEXT should NOT fire repeatedly."""
        _, action1 = self.sm.update(Gesture.COLOR_NEXT, 0.9)
        _, action2 = self.sm.update(Gesture.COLOR_NEXT, 0.9)
        assert action1 == ActionEvent.COLOR_NEXT
        assert action2 == ActionEvent.NONE  # No repeat

    def test_clear_requires_hold(self):
        """CLEAR requires sustained fist hold."""
        for i in range(settings.CLEAR_HOLD_FRAMES - 1):
            _, action = self.sm.update(Gesture.CLEAR, 0.9)
            assert action == ActionEvent.NONE
        # The Nth frame should trigger CLEAR
        _, action = self.sm.update(Gesture.CLEAR, 0.9)
        assert action == ActionEvent.CLEAR

    def test_reset(self):
        """Reset brings state machine back to initial state."""
        self._send_gesture(Gesture.DRAW, times=3)
        assert self.sm.current_mode == AppMode.DRAWING
        self.sm.reset()
        assert self.sm.current_mode == AppMode.IDLE


# ─────────────────────── Smoothing Tests ──────────────────────

class TestKalmanFilter:
    def test_reduces_jitter(self):
        """Kalman filter should smooth out jittery measurements."""
        kf = KalmanFilter2D()
        # Simulate jittery input around (100, 100)
        jittery = [(100 + (i % 3 - 1) * 10, 100 + (i % 3 - 1) * 8) for i in range(20)]
        smoothed = [kf.update(pt) for pt in jittery]

        # After warmup, smoothed values should be closer to center
        last_5 = smoothed[-5:]
        for sx, sy in last_5:
            assert abs(sx - 100) < 15, f"x={sx} too far from 100"
            assert abs(sy - 100) < 15, f"y={sy} too far from 100"

    def test_reset_clears_state(self):
        kf = KalmanFilter2D()
        kf.update((500, 500))
        assert kf.is_initialized
        kf.reset()
        assert not kf.is_initialized


class TestExponentialSmoother:
    def test_converges(self):
        """EMA should converge toward repeated measurements."""
        ema = ExponentialSmoother(alpha=0.5)
        for _ in range(20):
            result = ema.update((200, 300))
        assert abs(result[0] - 200) < 2
        assert abs(result[1] - 300) < 2

    def test_reset(self):
        ema = ExponentialSmoother()
        ema.update((100, 100))
        assert ema.is_initialized
        ema.reset()
        assert not ema.is_initialized


# ─────────────────────── Run ──────────────────────────────────

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
