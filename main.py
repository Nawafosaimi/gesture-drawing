"""
AirCanvas — Main Application
Real-time gesture-based drawing system.

Pipeline: Camera → HandTracker → GestureEngine → StateManager
         → Smoothing → DrawingEngine → UIRenderer → Display

Usage:
    python main.py
    python main.py --source video.mp4     # Use video file instead of webcam
    python main.py --record               # Record session to video file
    python main.py --width 1920 --height 1080

Controls:
    Q / ESC  — Quit
    R        — Toggle video recording
    U        — Undo last stroke
    C        — Clear canvas (keyboard fallback)
    S        — Save canvas (keyboard fallback)
"""

import argparse
import sys
import time
import cv2
import numpy as np

from config import settings
from core.hand_tracker import HandTracker
from core.gesture_engine import GestureEngine, Gesture
from core.state_manager import StateManager, AppMode, ActionEvent
from core.smoothing import KalmanFilter2D
from core.drawing_engine import DrawingEngine
from core.ui_renderer import UIRenderer
from core.export_manager import ExportManager


def parse_args():
    parser = argparse.ArgumentParser(
        description="AirCanvas — Gesture-Based Smart Drawing System"
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Video file path (default: webcam)"
    )
    parser.add_argument(
        "--width", type=int, default=settings.FRAME_WIDTH,
        help=f"Frame width (default: {settings.FRAME_WIDTH})"
    )
    parser.add_argument(
        "--height", type=int, default=settings.FRAME_HEIGHT,
        help=f"Frame height (default: {settings.FRAME_HEIGHT})"
    )
    parser.add_argument(
        "--record", action="store_true",
        help="Start video recording immediately"
    )
    return parser.parse_args()


class AirCanvas:
    """
    Main application class.
    Orchestrates the entire pipeline from camera capture to display.
    """

    def __init__(self, args):
        self._args = args
        self._frame_width = args.width
        self._frame_height = args.height

        # ── Initialize modules ────────────────────────────────
        self._tracker = HandTracker()
        self._gesture_engine = GestureEngine()
        self._state_manager = StateManager()
        self._smoother = KalmanFilter2D()
        self._drawing_engine = DrawingEngine(self._frame_width, self._frame_height)
        self._ui = UIRenderer()
        self._export = ExportManager()

        # ── Drawing state ─────────────────────────────────────
        self._color_index = settings.DEFAULT_COLOR_INDEX
        self._thickness = settings.DEFAULT_THICKNESS
        self._active_color = settings.COLOR_PALETTE[self._color_index]

        # ── FPS tracking ──────────────────────────────────────
        self._fps = 0.0
        self._frame_times: list[float] = []

        # ── Notification system ───────────────────────────────
        self._notification: str | None = None
        self._notification_color: tuple[int, int, int] = (255, 255, 255)
        self._notification_timer = 0

        # ── Hold progress tracking ────────────────────────────
        self._fist_frames = 0
        self._ok_frames = 0

        # ── Previous tracking state (for stroke lifecycle) ───
        self._was_drawing = False
        self._was_erasing = False
        self._hand_visible = False

    def run(self):
        """Main application loop."""
        # ── Open video source ─────────────────────────────────
        source = self._args.source if self._args.source else settings.CAMERA_INDEX
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"[ERROR] Cannot open video source: {source}")
            sys.exit(1)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_height)

        # Read actual dimensions (camera may not support requested size)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w != self._frame_width or actual_h != self._frame_height:
            print(f"[INFO] Camera resolution: {actual_w}x{actual_h} "
                  f"(requested {self._frame_width}x{self._frame_height})")
            self._frame_width = actual_w
            self._frame_height = actual_h
            self._drawing_engine = DrawingEngine(actual_w, actual_h)

        if self._args.record:
            path = self._export.start_recording(actual_w, actual_h)
            self._show_notification(f"Recording: {path}", (0, 255, 100))

        print(f"[AirCanvas] Started — {actual_w}x{actual_h}")
        print("[AirCanvas] Press Q or ESC to quit")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if self._args.source:
                        break  # End of video file
                    continue

                # Mirror the frame so it feels natural
                frame = cv2.flip(frame, 1)

                # Process the frame through the pipeline
                display = self._process_frame(frame)

                # Record if active
                self._export.write_frame(display)

                # Show result
                cv2.imshow("AirCanvas", display)

                # Handle keyboard input
                if self._handle_keyboard(cv2.waitKey(1) & 0xFF):
                    break

        finally:
            cap.release()
            self._tracker.release()
            self._export.release()
            cv2.destroyAllWindows()

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Full pipeline for one frame.
        
        Steps:
        1. Hand detection
        2. Gesture classification
        3. State machine update
        4. Handle actions (color, size, clear, save, undo)
        5. Smoothing
        6. Drawing logic (start/add/end strokes)
        7. Canvas compositing
        8. UI rendering
        """
        t0 = time.perf_counter()

        # ── 1. Hand tracking ──────────────────────────────────
        hands = self._tracker.process(frame, self._frame_width, self._frame_height)

        gesture = Gesture.NONE
        confidence = 0.0
        pixel_landmarks = None

        if hands:
            hand = hands[0]  # Primary hand
            pixel_landmarks = hand.pixel_landmarks

            # ── 2. Gesture classification ─────────────────────
            gesture, confidence = self._gesture_engine.classify(
                hand.landmarks, hand.handedness
            )

            self._hand_visible = True
        else:
            self._hand_visible = False
            # Hand disappeared — end any active stroke
            if self._was_drawing or self._was_erasing:
                self._drawing_engine.end_stroke()
                self._was_drawing = False
                self._was_erasing = False
            self._smoother.reset()

        # ── 3. State machine update ───────────────────────────
        mode, action = self._state_manager.update(gesture, confidence)

        # ── 4. Handle one-shot actions ────────────────────────
        self._handle_action(action)

        # ── Track hold progress for UI ────────────────────────
        if gesture == Gesture.CLEAR:
            self._fist_frames += 1
        else:
            self._fist_frames = 0

        if gesture == Gesture.SAVE:
            self._ok_frames += 1
        else:
            self._ok_frames = 0

        # ── 5–6. Drawing logic ────────────────────────────────
        if pixel_landmarks:
            # Get the index fingertip position
            raw_tip = pixel_landmarks[settings.INDEX_TIP]

            # Smooth the position
            smooth_tip = self._smoother.update(raw_tip)

            if mode == AppMode.DRAWING:
                if not self._was_drawing:
                    # Start a new stroke
                    self._drawing_engine.start_stroke(
                        smooth_tip, self._active_color, self._thickness
                    )
                    self._was_drawing = True
                    self._was_erasing = False
                else:
                    self._drawing_engine.add_point(smooth_tip)

            elif mode == AppMode.ERASING:
                if not self._was_erasing:
                    self._drawing_engine.start_stroke(
                        smooth_tip, (0, 0, 0), settings.ERASER_THICKNESS,
                        is_eraser=True,
                    )
                    self._was_erasing = True
                    self._was_drawing = False
                else:
                    self._drawing_engine.add_point(smooth_tip)

            else:  # IDLE
                if self._was_drawing or self._was_erasing:
                    self._drawing_engine.end_stroke()
                    self._was_drawing = False
                    self._was_erasing = False

        # ── 7. Canvas compositing ─────────────────────────────
        display = self._drawing_engine.composite(frame)

        # ── 8. UI rendering ───────────────────────────────────
        if pixel_landmarks:
            self._ui.draw_landmarks(display, pixel_landmarks, mode)

        self._ui.draw_hud(
            display, mode, self._active_color, self._thickness,
            self._fps, self._drawing_engine.stroke_count,
        )
        self._ui.draw_toolbar(display, settings.COLOR_PALETTE, self._color_index)
        self._ui.draw_instructions(display)

        # Progress indicators for hold gestures
        if self._fist_frames > 0:
            progress = self._fist_frames / settings.CLEAR_HOLD_FRAMES
            self._ui.draw_clear_progress(display, progress)

        if self._ok_frames > 0:
            progress = self._ok_frames / settings.SAVE_HOLD_FRAMES
            self._ui.draw_save_progress(display, progress)

        # Notification overlay
        if self._notification and self._notification_timer > 0:
            self._ui.draw_notification(display, self._notification, self._notification_color)
            self._notification_timer -= 1

        # Recording indicator
        if self._export.is_recording:
            cv2.circle(display, (display.shape[1] - 30, 30), 8, (0, 0, 255), -1)

        # ── FPS calculation ───────────────────────────────────
        elapsed = time.perf_counter() - t0
        self._frame_times.append(elapsed)
        if len(self._frame_times) > 30:
            self._frame_times.pop(0)
        avg = sum(self._frame_times) / len(self._frame_times)
        self._fps = 1.0 / avg if avg > 0 else 0

        return display

    def _handle_action(self, action: ActionEvent):
        """Process one-shot action events from the state manager."""
        if action == ActionEvent.NONE:
            return

        if action == ActionEvent.COLOR_NEXT:
            self._color_index = (self._color_index + 1) % len(settings.COLOR_PALETTE)
            self._active_color = settings.COLOR_PALETTE[self._color_index]
            self._show_notification("Color changed", self._active_color)

        elif action == ActionEvent.SIZE_UP:
            self._thickness = min(self._thickness + settings.THICKNESS_STEP,
                                   settings.MAX_THICKNESS)
            self._show_notification(f"Brush: {self._thickness}px", (0, 255, 200))

        elif action == ActionEvent.SIZE_DOWN:
            self._thickness = max(self._thickness - settings.THICKNESS_STEP,
                                   settings.MIN_THICKNESS)
            self._show_notification(f"Brush: {self._thickness}px", (0, 255, 200))

        elif action == ActionEvent.CLEAR:
            self._drawing_engine.clear()
            self._was_drawing = False
            self._was_erasing = False
            self._show_notification("Canvas cleared!", (0, 200, 255))

        elif action == ActionEvent.SAVE:
            canvas = self._drawing_engine.get_canvas_bgr()
            path = self._export.save_canvas(canvas)
            self._show_notification(f"Saved!", (0, 255, 100))
            print(f"[AirCanvas] Saved to {path}")

        elif action == ActionEvent.UNDO:
            if self._drawing_engine.undo():
                self._show_notification("Undo", (200, 200, 200))

    def _handle_keyboard(self, key: int) -> bool:
        """
        Handle keyboard input.
        Returns True if the application should exit.
        """
        if key == ord("q") or key == 27:  # Q or ESC
            return True

        if key == ord("u"):
            if self._drawing_engine.undo():
                self._show_notification("Undo", (200, 200, 200))

        if key == ord("c"):
            self._drawing_engine.clear()
            self._was_drawing = False
            self._was_erasing = False
            self._show_notification("Canvas cleared!", (0, 200, 255))

        if key == ord("s"):
            canvas = self._drawing_engine.get_canvas_bgr()
            path = self._export.save_canvas(canvas)
            self._show_notification("Saved!", (0, 255, 100))
            print(f"[AirCanvas] Saved to {path}")

        if key == ord("r"):
            if self._export.is_recording:
                self._export.stop_recording()
                self._show_notification("Recording stopped", (200, 200, 200))
            else:
                path = self._export.start_recording(self._frame_width, self._frame_height)
                self._show_notification("Recording started", (0, 0, 255))
                print(f"[AirCanvas] Recording to {path}")

        return False

    def _show_notification(self, text: str, color: tuple[int, int, int]):
        """Show a temporary notification overlay."""
        self._notification = text
        self._notification_color = color
        self._notification_timer = 40  # ~1.3s at 30fps


def main():
    args = parse_args()
    app = AirCanvas(args)
    app.run()


if __name__ == "__main__":
    main()
