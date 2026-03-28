"""
AirCanvas — State Manager
Finite State Machine for managing application mode transitions.

Design goals:
- Debounced transitions: gesture must persist N frames before triggering.
- Confidence-gated: low-confidence gestures are ignored.
- Time-gated actions: destructive operations (CLEAR, SAVE) require sustained hold.
- Instant actions: COLOR_NEXT, SIZE_UP, SIZE_DOWN fire once without changing FSM state.
"""

from enum import Enum
from core.gesture_engine import Gesture
from config import settings


class AppMode(Enum):
    """Application modes controlled by the FSM."""
    IDLE = "IDLE"
    DRAWING = "DRAWING"
    ERASING = "ERASING"


class ActionEvent(Enum):
    """One-shot action events that don't change mode."""
    NONE = "none"
    COLOR_NEXT = "color_next"
    SIZE_UP = "size_up"
    SIZE_DOWN = "size_down"
    CLEAR = "clear"
    SAVE = "save"
    UNDO = "undo"


class StateManager:
    """
    Finite State Machine for mode transitions and action events.
    
    The FSM has two output channels:
    1. mode: persistent state (IDLE, DRAWING, ERASING)
    2. action_event: one-shot events fired on transition edges
    """

    def __init__(self, debounce_frames: int = settings.DEBOUNCE_FRAMES):
        self._mode = AppMode.IDLE
        self._debounce_frames = debounce_frames
        self._min_confidence = settings.MIN_GESTURE_CONFIDENCE

        # Debounce tracking
        self._pending_gesture = Gesture.NONE
        self._pending_count = 0

        # Time-gated action tracking
        self._fist_count = 0
        self._ok_count = 0

        # One-shot edge detection for instant actions
        self._last_committed_gesture = Gesture.NONE

    @property
    def current_mode(self) -> AppMode:
        return self._mode

    def update(self, gesture: Gesture, confidence: float) -> tuple[AppMode, ActionEvent]:
        """
        Process a new gesture frame.
        
        Args:
            gesture: Classified gesture from GestureEngine.
            confidence: Classification confidence [0, 1].
            
        Returns:
            (current_mode, action_event) — action_event is NONE most frames.
        """
        action = ActionEvent.NONE

        # Gate on confidence
        if confidence < self._min_confidence:
            gesture = Gesture.NONE

        # ── Time-gated destructive actions ───────────────────────
        if gesture == Gesture.CLEAR:
            self._fist_count += 1
            if self._fist_count >= settings.CLEAR_HOLD_FRAMES:
                action = ActionEvent.CLEAR
                self._fist_count = 0
            # Don't process further — fist hold shouldn't change mode
            self._ok_count = 0
            return self._mode, action
        else:
            self._fist_count = 0

        if gesture == Gesture.SAVE:
            self._ok_count += 1
            if self._ok_count >= settings.SAVE_HOLD_FRAMES:
                action = ActionEvent.SAVE
                self._ok_count = 0
            self._fist_count = 0
            return self._mode, action
        else:
            self._ok_count = 0

        # ── Instant one-shot actions (fire on rising edge) ───────
        if gesture in (Gesture.COLOR_NEXT, Gesture.SIZE_UP, Gesture.SIZE_DOWN, Gesture.UNDO):
            if gesture != self._last_committed_gesture:
                action = self._gesture_to_action(gesture)
                self._last_committed_gesture = gesture
            return self._mode, action

        # ── Debounced mode transitions ───────────────────────────
        if gesture == self._pending_gesture:
            self._pending_count += 1
        else:
            self._pending_gesture = gesture
            self._pending_count = 1

        if self._pending_count >= self._debounce_frames:
            new_mode = self._resolve_mode(gesture)
            if new_mode != self._mode:
                self._mode = new_mode
            self._last_committed_gesture = gesture

        # Reset committed gesture tracking when we see a mode-changing gesture
        if gesture not in (Gesture.COLOR_NEXT, Gesture.SIZE_UP,
                           Gesture.SIZE_DOWN, Gesture.UNDO):
            self._last_committed_gesture = gesture

        return self._mode, action

    def _resolve_mode(self, gesture: Gesture) -> AppMode:
        """Map a debounced gesture to the appropriate mode."""
        mode_map = {
            Gesture.DRAW: AppMode.DRAWING,
            Gesture.STOP: AppMode.IDLE,
            Gesture.ERASE: AppMode.ERASING,
            Gesture.NONE: AppMode.IDLE,
        }
        return mode_map.get(gesture, self._mode)

    @staticmethod
    def _gesture_to_action(gesture: Gesture) -> ActionEvent:
        """Map instant gesture to action event."""
        action_map = {
            Gesture.COLOR_NEXT: ActionEvent.COLOR_NEXT,
            Gesture.SIZE_UP: ActionEvent.SIZE_UP,
            Gesture.SIZE_DOWN: ActionEvent.SIZE_DOWN,
            Gesture.UNDO: ActionEvent.UNDO,
        }
        return action_map.get(gesture, ActionEvent.NONE)

    def reset(self):
        """Reset FSM to initial state."""
        self._mode = AppMode.IDLE
        self._pending_gesture = Gesture.NONE
        self._pending_count = 0
        self._fist_count = 0
        self._ok_count = 0
        self._last_committed_gesture = Gesture.NONE
