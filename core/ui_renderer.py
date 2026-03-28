"""
AirCanvas — UI Renderer
Draws HUD overlay, toolbar, landmarks, and instructions.
"""

import cv2
import numpy as np

from config import settings
from core.state_manager import AppMode
from core.gesture_engine import Gesture


class UIRenderer:
    """
    Renders all UI elements onto the display frame.
    
    Layers (drawn bottom-to-top):
    1. Camera feed (already present)
    2. Canvas overlay (handled by DrawingEngine.composite)
    3. Hand landmarks
    4. HUD bar (top)
    5. Color toolbar (left)
    6. Instructions panel (bottom-right)
    7. Fingertip cursor
    """

    def __init__(self):
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._progress_flash = 0  # For animated clear/save progress

    # ── HUD Bar ──────────────────────────────────────────────────

    def draw_hud(
        self,
        frame: np.ndarray,
        mode: AppMode,
        active_color: tuple[int, int, int],
        thickness: int,
        fps: float,
        stroke_count: int,
    ):
        """Draw the top HUD bar with mode, color, brush, FPS, stroke count."""
        h = settings.HUD_HEIGHT
        # Semi-transparent black bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], h), settings.HUD_BG_COLOR, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        y = 38
        x = 15

        # Mode indicator with color coding
        mode_colors = {
            AppMode.IDLE: (180, 180, 180),
            AppMode.DRAWING: (0, 255, 100),
            AppMode.ERASING: (80, 130, 255),
        }
        mode_color = mode_colors.get(mode, (180, 180, 180))

        # Mode dot
        cv2.circle(frame, (x, y - 5), 6, mode_color, -1)
        x += 18

        # Mode text
        cv2.putText(
            frame, mode.value, (x, y),
            self._font, settings.HUD_FONT_SCALE + 0.1,
            mode_color, settings.HUD_FONT_THICKNESS + 1,
            cv2.LINE_AA,
        )
        x += 120

        # Divider
        cv2.line(frame, (x, 10), (x, h - 10), (80, 80, 80), 1)
        x += 15

        # Active color swatch
        cv2.putText(frame, "Color:", (x, y), self._font, settings.HUD_FONT_SCALE,
                     settings.HUD_TEXT_COLOR, 1, cv2.LINE_AA)
        x += 65
        cv2.circle(frame, (x, y - 5), 10, active_color, -1)
        cv2.circle(frame, (x, y - 5), 10, (100, 100, 100), 1)
        x += 25

        # Divider
        cv2.line(frame, (x, 10), (x, h - 10), (80, 80, 80), 1)
        x += 15

        # Brush size
        cv2.putText(frame, f"Brush: {thickness}px", (x, y), self._font,
                     settings.HUD_FONT_SCALE, settings.HUD_TEXT_COLOR, 1, cv2.LINE_AA)
        x += 130

        # Divider
        cv2.line(frame, (x, 10), (x, h - 10), (80, 80, 80), 1)
        x += 15

        # Stroke count
        cv2.putText(frame, f"Strokes: {stroke_count}", (x, y), self._font,
                     settings.HUD_FONT_SCALE, settings.HUD_TEXT_COLOR, 1, cv2.LINE_AA)
        x += 140

        # FPS (right-aligned)
        fps_text = f"FPS: {fps:.0f}"
        fps_size = cv2.getTextSize(fps_text, self._font, settings.HUD_FONT_SCALE, 1)[0]
        cv2.putText(
            frame, fps_text,
            (frame.shape[1] - fps_size[0] - 15, y),
            self._font, settings.HUD_FONT_SCALE,
            (0, 200, 255) if fps >= 25 else (0, 100, 255),
            1, cv2.LINE_AA,
        )

    # ── Color Toolbar ────────────────────────────────────────────

    def draw_toolbar(
        self,
        frame: np.ndarray,
        palette: list[tuple[int, int, int]],
        active_index: int,
    ):
        """Draw the color palette toolbar on the left edge."""
        tw = settings.TOOLBAR_WIDTH
        h = frame.shape[0]

        # Semi-transparent toolbar background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, settings.HUD_HEIGHT), (tw, h), settings.TOOLBAR_BG_COLOR, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        start_y = settings.HUD_HEIGHT + 30

        for i, color in enumerate(palette):
            cy = start_y + i * settings.SWATCH_SPACING
            cx = tw // 2

            r = settings.SWATCH_RADIUS
            if i == active_index:
                # Active indicator: larger ring
                cv2.circle(frame, (cx, cy), r + 4, (255, 255, 255), 2)

            cv2.circle(frame, (cx, cy), r, color, -1)
            cv2.circle(frame, (cx, cy), r, (80, 80, 80), 1)

    # ── Hand Landmarks ───────────────────────────────────────────

    def draw_landmarks(
        self,
        frame: np.ndarray,
        pixel_landmarks: list[tuple[int, int]],
        mode: AppMode,
    ):
        """Draw hand skeleton and fingertip cursor."""
        if not pixel_landmarks or len(pixel_landmarks) < 21:
            return

        # Draw connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),       # Index
            (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17),              # Palm
        ]

        for start, end in connections:
            pt1 = pixel_landmarks[start]
            pt2 = pixel_landmarks[end]
            cv2.line(frame, pt1, pt2, settings.LANDMARK_CONNECTION_COLOR, 1, cv2.LINE_AA)

        # Draw landmark dots
        for i, pt in enumerate(pixel_landmarks):
            cv2.circle(frame, pt, settings.LANDMARK_RADIUS, settings.LANDMARK_COLOR, -1)

        # Draw fingertip with mode-specific color
        index_tip = pixel_landmarks[settings.INDEX_TIP]
        if mode == AppMode.DRAWING:
            tip_color = settings.FINGERTIP_DRAW_COLOR
            tip_r = settings.FINGERTIP_RADIUS
        elif mode == AppMode.ERASING:
            tip_color = (80, 130, 255)
            tip_r = settings.FINGERTIP_RADIUS + 6
        else:
            tip_color = settings.FINGERTIP_IDLE_COLOR
            tip_r = settings.FINGERTIP_RADIUS - 2

        cv2.circle(frame, index_tip, tip_r, tip_color, 2, cv2.LINE_AA)
        cv2.circle(frame, index_tip, 2, tip_color, -1)

    # ── Instructions Panel ───────────────────────────────────────

    def draw_instructions(self, frame: np.ndarray):
        """Draw gesture instructions in the bottom-right corner."""
        instructions = [
            "Gestures:",
            "  Index up      -> DRAW",
            "  Open palm     -> STOP",
            "  V sign        -> ERASE",
            "  Shaka         -> Color",
            "  Thumb up/down -> Size",
            "  Fist (hold)   -> Clear",
            "  OK (hold)     -> Save",
            "  Index+Pinky   -> Undo",
            "  Q key         -> Quit",
        ]

        fh = frame.shape[0]
        fw = frame.shape[1]
        line_h = 18
        panel_h = len(instructions) * line_h + 20
        panel_w = 210
        px = fw - panel_w - 10
        py = fh - panel_h - 10

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h),
                       (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        for i, text in enumerate(instructions):
            y = py + 15 + i * line_h
            color = (220, 220, 220) if i == 0 else settings.INSTRUCTION_COLOR
            scale = settings.INSTRUCTION_FONT_SCALE + 0.05 if i == 0 else settings.INSTRUCTION_FONT_SCALE
            cv2.putText(
                frame, text, (px + 10, y),
                self._font, scale, color, 1, cv2.LINE_AA,
            )

    # ── Progress Indicators ──────────────────────────────────────

    def draw_clear_progress(self, frame: np.ndarray, progress: float):
        """Draw a progress ring for the 'clear canvas' hold gesture."""
        if progress <= 0:
            return
        cx, cy = frame.shape[1] // 2, frame.shape[0] // 2
        radius = 60
        angle = int(360 * min(progress, 1.0))

        cv2.ellipse(
            frame, (cx, cy), (radius, radius),
            -90, 0, angle, (0, 0, 255), 4, cv2.LINE_AA,
        )

        cv2.putText(
            frame, "CLEARING...", (cx - 55, cy + radius + 25),
            self._font, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
        )

    def draw_save_progress(self, frame: np.ndarray, progress: float):
        """Draw a progress ring for the 'save' hold gesture."""
        if progress <= 0:
            return
        cx, cy = frame.shape[1] // 2, frame.shape[0] // 2
        radius = 60
        angle = int(360 * min(progress, 1.0))

        cv2.ellipse(
            frame, (cx, cy), (radius, radius),
            -90, 0, angle, (0, 255, 100), 4, cv2.LINE_AA,
        )

        cv2.putText(
            frame, "SAVING...", (cx - 45, cy + radius + 25),
            self._font, 0.6, (0, 255, 100), 2, cv2.LINE_AA,
        )

    # ── Notification Flash ───────────────────────────────────────

    def draw_notification(self, frame: np.ndarray, text: str, color: tuple[int, int, int]):
        """Draw a centered notification message (for save/clear confirmation)."""
        text_size = cv2.getTextSize(text, self._font, 1.0, 2)[0]
        cx = (frame.shape[1] - text_size[0]) // 2
        cy = frame.shape[0] // 2

        # Background box
        pad = 20
        cv2.rectangle(
            frame,
            (cx - pad, cy - text_size[1] - pad),
            (cx + text_size[0] + pad, cy + pad),
            (0, 0, 0), -1,
        )
        cv2.putText(frame, text, (cx, cy), self._font, 1.0, color, 2, cv2.LINE_AA)
