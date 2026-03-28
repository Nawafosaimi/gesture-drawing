"""
UI Renderer
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

        # ── Spring Physics States ────────────────────────────────────
        self._anim_swatches = [0.0] * 8
        self._anim_buttons = {
            "minus": 0.0, "plus": 0.0, 
            "undo": 0.0, "redo": 0.0, "clear": 0.0, "save": 0.0
        }

    # ── Glassmorphism Background ─────────────────────────────────

    def draw_glass_panel(self, frame: np.ndarray):
        """
        Draws a unified frosted glass background over the top menu area.
        Runs a heavy Gaussian blur with a subtle dark tint.
        """
        w = frame.shape[1]
        h = settings.HUD_HEIGHT + settings.TOOLBAR_HEIGHT
        
        roi = frame[0:h, 0:w]
        
        # High radius mathematical blur
        blurred_roi = cv2.GaussianBlur(roi, (55, 55), 0)
        
        # Subtle dark tint to ensure white text is highly legible
        dark_tint = np.full_like(blurred_roi, (20, 20, 20))
        glass_roi = cv2.addWeighted(blurred_roi, 0.65, dark_tint, 0.35, 0)
        
        frame[0:h, 0:w] = glass_roi
        
        # Elegant bottom border and separator
        cv2.line(frame, (0, h), (w, h), (180, 180, 180), 1, cv2.LINE_AA)
        cv2.line(frame, (0, settings.HUD_HEIGHT), (w, settings.HUD_HEIGHT), 
                 (120, 120, 120), 1, cv2.LINE_AA)

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
        hovered_index: int = -1,
        hovered_button: str | None = None,
        thickness: int = 4,
    ):
        """Draw the color palette toolbar with hover effects and brush controls."""
        th = settings.TOOLBAR_HEIGHT
        w = frame.shape[1]

        cy = settings.HUD_HEIGHT + th // 2
        start_x = 50

        # ── Update Spring Physics ────────────────────────────
        anim_speed = 0.25
        for i in range(len(palette)):
            target = 1.0 if (i == hovered_index and i != active_index) else 0.0
            self._anim_swatches[i] += (target - self._anim_swatches[i]) * anim_speed
            
        for btn in self._anim_buttons:
            target = 1.0 if hovered_button == btn else 0.0
            self._anim_buttons[btn] += (target - self._anim_buttons[btn]) * anim_speed

        for i, color in enumerate(palette):
            sx = start_x + i * settings.SWATCH_SPACING
            r = settings.SWATCH_RADIUS
            anim_val = self._anim_swatches[i]

            # Hover glow effect (smoothly expanding ring)
            if anim_val > 0.01:
                extra_r = int(anim_val * 8)
                alpha = int(255 * anim_val)
                # Draw growing circle
                cv2.circle(frame, (sx, cy), r + extra_r, color, 2, cv2.LINE_AA)

            # Active indicator: white ring
            if i == active_index:
                cv2.circle(frame, (sx, cy), r + 5, (255, 255, 255), 2, cv2.LINE_AA)

            # Swatch fill
            cv2.circle(frame, (sx, cy), r, color, -1, cv2.LINE_AA)

            # Border fade from dark to light on hover
            border_c = int(80 + anim_val * 140)
            cv2.circle(frame, (sx, cy), r, (border_c, border_c, border_c), 1, cv2.LINE_AA)

        # ── Brush size controls ──────────────────────────────
        last_sx = start_x + (len(palette) - 1) * settings.SWATCH_SPACING

        # Divider line
        div_x = last_sx + settings.SWATCH_SPACING // 2 + 10
        cv2.line(frame, (div_x, settings.HUD_HEIGHT + 10), (div_x, settings.HUD_HEIGHT + th - 10), (60, 60, 60), 1)

        btn_x = last_sx + 100
        minus_cx = btn_x - 30
        plus_cx = btn_x + 30
        
        # Size label text
        size_text = f"{thickness}px"
        text_size = cv2.getTextSize(size_text, self._font, 0.45, 1)[0]
        cv2.putText(frame, size_text, (btn_x - text_size[0] // 2, settings.HUD_HEIGHT + 20),
                     self._font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # Brush preview circle
        preview_r = max(2, min(thickness // 2, 12))
        active_color = palette[active_index] if active_index < len(palette) else (255, 255, 255)
        cv2.circle(frame, (btn_x, cy + 5), preview_r, active_color, -1, cv2.LINE_AA)

        # ── [−] and [+] buttons ──────────────────────────────
        btn_r = settings.SIZE_BUTTON_RADIUS

        def get_btn_style(btn_id):
            anim_val = self._anim_buttons[btn_id]
            b = int(150 + anim_val * 105)
            g = int(150 + anim_val * 50)
            r = int(150 - anim_val * 50)
            thick = 1 + int(anim_val)
            return (b, g, r), thick

        # Minus button
        mc, mt = get_btn_style("minus")
        cv2.circle(frame, (minus_cx, cy), btn_r, mc, mt, cv2.LINE_AA)
        cv2.line(frame, (minus_cx - 5, cy), (minus_cx + 5, cy), mc, 2, cv2.LINE_AA)

        # Plus button
        pc, pt = get_btn_style("plus")
        cv2.circle(frame, (plus_cx, cy), btn_r, pc, pt, cv2.LINE_AA)
        cv2.line(frame, (plus_cx - 5, cy), (plus_cx + 5, cy), pc, 2, cv2.LINE_AA)
        cv2.line(frame, (plus_cx, cy - 5), (plus_cx, cy + 5), pc, 2, cv2.LINE_AA)

        # ── Action Buttons ───────────────────────────────────
        
        div2_x = plus_cx + settings.SWATCH_SPACING // 2 + 10
        cv2.line(frame, (div2_x, settings.HUD_HEIGHT + 10), (div2_x, settings.HUD_HEIGHT + th - 10), (60, 60, 60), 1)
        
        undo_x = plus_cx + 80
        redo_x = undo_x + 70
        clear_x = redo_x + 70
        save_x = clear_x + 70
        
        action_buttons = [
            ("undo", "Undo", undo_x),
            ("redo", "Redo", redo_x),
            ("clear", "Clear", clear_x),
            ("save", "Save", save_x)
        ]
        
        for b_id, b_label, bx in action_buttons:
            anim_val = self._anim_buttons[b_id]
            
            # Smoothly transition colors
            bc_b = int(80 + anim_val * 175)
            bc_g = int(80 + anim_val * 120)
            bc_r = int(80 + anim_val * 20)
            bc = (bc_b, bc_g, bc_r)
            
            tc_val = int(200 - anim_val * 200)
            tc = (tc_val, tc_val, tc_val)

            # Expand the fill mask outward
            if anim_val > 0.05:
                w_anim = int(30 * anim_val)
                h_anim = int(14 * anim_val)
                cv2.rectangle(frame, (bx - w_anim, cy - h_anim), (bx + w_anim, cy + h_anim), bc, -1, cv2.LINE_AA)
            
            # Static outline
            cv2.rectangle(frame, (bx - 30, cy - 14), (bx + 30, cy + 14), bc, 1 if anim_val < 0.95 else 2, cv2.LINE_AA)
            
            ts = cv2.getTextSize(b_label, self._font, 0.4, 1)[0]
            cv2.putText(frame, b_label, (bx - ts[0] // 2, cy + ts[1] // 2), 
                        self._font, 0.4, tc, 1, cv2.LINE_AA)

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
            "  Fist (hold)   -> Clear",
            "  OK (hold)     -> Save",
            "",
            "Quit:",
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
            if text == "":
                continue
            is_header = text in ("Gestures:", "Keyboard:")
            color = (220, 220, 220) if is_header else settings.INSTRUCTION_COLOR
            scale = settings.INSTRUCTION_FONT_SCALE + 0.05 if is_header else settings.INSTRUCTION_FONT_SCALE
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
        """Draw a centered structured frosted glass notification message."""
        text_size = cv2.getTextSize(text, self._font, 1.0, 2)[0]
        cx = (frame.shape[1] - text_size[0]) // 2
        cy = frame.shape[0] // 2

        pad = 20
        # Determine exact pixel bounds for the notification popup
        x1 = max(0, cx - pad)
        y1 = max(0, cy - text_size[1] - pad)
        x2 = min(frame.shape[1], cx + text_size[0] + pad)
        y2 = min(frame.shape[0], cy + pad + 10)

        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            # High radius mathematical blur
            blurred_roi = cv2.GaussianBlur(roi, (55, 55), 0)
            
            # Subtle dark tint
            dark_tint = np.full_like(blurred_roi, (20, 20, 20))
            glass_roi = cv2.addWeighted(blurred_roi, 0.65, dark_tint, 0.35, 0)
            
            frame[y1:y2, x1:x2] = glass_roi
            
            # Elegant border
            cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1, cv2.LINE_AA)

        cv2.putText(frame, text, (cx, cy), self._font, 1.0, color, 2, cv2.LINE_AA)
