"""
Drawing Engine
Manages strokes, canvas rendering, undo/redo, and eraser.
"""

from dataclasses import dataclass, field
import numpy as np
import cv2

from config import settings


@dataclass
class Stroke:
    """A single stroke (line or erase path)."""
    points: list[tuple[int, int]] = field(default_factory=list)
    color: tuple[int, int, int] = (255, 255, 255)
    thickness: int = 4
    is_eraser: bool = False


class DrawingEngine:
    """
    Canvas and stroke manager.
    
    Design:
    - Strokes are stored as polyline objects for undo/redo.
    - Canvas is re-rendered from the stroke list each frame.
    - For <200 strokes this is fast enough (~1ms); for more,
      an incremental dirty-region approach would be needed.
    - The canvas is a transparent overlay (BGRA) composited onto the camera feed.
    """

    def __init__(
        self,
        width: int = settings.FRAME_WIDTH,
        height: int = settings.FRAME_HEIGHT,
    ):
        self._width = width
        self._height = height

        self._strokes: list[Stroke] = []
        self._redo_stack: list[Stroke] = []
        self._active_stroke: Stroke | None = None

        # Persistent canvas (re-rendered from strokes)
        self._canvas = np.zeros((height, width, 4), dtype=np.uint8)

    @property
    def stroke_count(self) -> int:
        return len(self._strokes)

    @property
    def has_active_stroke(self) -> bool:
        return self._active_stroke is not None

    # ── Stroke lifecycle ─────────────────────────────────────────

    def start_stroke(
        self,
        point: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int,
        is_eraser: bool = False,
    ):
        """Begin a new stroke at the given point."""
        self._active_stroke = Stroke(
            points=[point],
            color=color,
            thickness=thickness,
            is_eraser=is_eraser,
        )
        # Starting a new stroke clears the redo stack
        self._redo_stack.clear()

    def add_point(self, point: tuple[int, int]):
        """Append a point to the active stroke."""
        if self._active_stroke is not None:
            # Only add if point moved enough (avoid duplicate points)
            if self._active_stroke.points:
                last = self._active_stroke.points[-1]
                dx = abs(point[0] - last[0])
                dy = abs(point[1] - last[1])
                if dx < 2 and dy < 2:
                    return
            self._active_stroke.points.append(point)

    def end_stroke(self):
        """Finalize the active stroke and add it to the stroke list."""
        if self._active_stroke is not None:
            if len(self._active_stroke.points) >= 2:
                self._strokes.append(self._active_stroke)
            self._active_stroke = None
            self._render_canvas()

    # ── Undo / Redo ──────────────────────────────────────────────

    def undo(self) -> bool:
        """Undo the last stroke. Returns True if successful."""
        if self._strokes:
            stroke = self._strokes.pop()
            self._redo_stack.append(stroke)
            self._render_canvas()
            return True
        return False

    def redo(self) -> bool:
        """Redo the last undone stroke. Returns True if successful."""
        if self._redo_stack:
            stroke = self._redo_stack.pop()
            self._strokes.append(stroke)
            self._render_canvas()
            return True
        return False

    # ── Clear & Erase Objects ────────────────────────────────────

    def clear(self):
        """Clear all strokes and reset canvas."""
        self._strokes.clear()
        self._redo_stack.clear()
        self._active_stroke = None
        self._canvas = np.zeros(
            (self._height, self._width, 4), dtype=np.uint8
        )

    def erase_at_point(self, point: tuple[int, int], radius: int) -> bool:
        """
        Check if the given point intersects with any stroke. 
        If so, delete that stroke entirely (object erasure).
        Returns True if a stroke was deleted.
        """
        px, py = point
        deleted = False

        # Iterate backwards to erase the top-most stroke and delete one at a time
        for i in range(len(self._strokes) - 1, -1, -1):
            stroke = self._strokes[i]
            if self._stroke_intersects(stroke, px, py, radius):
                # User deleted a stroke; clear the redo stack to prevent timeline anomalies
                del self._strokes[i]
                self._redo_stack.clear()
                deleted = True
                break

        if deleted:
            self._render_canvas()
        
        return deleted

    def _stroke_intersects(self, stroke: Stroke, px: int, py: int, radius: int) -> bool:
        pts = stroke.points
        if not pts:
            return False

        # 1. Fast Bounding Box Check -> reduces computation by 90%
        r_total = radius + stroke.thickness
        min_x = min(p[0] for p in pts) - r_total
        max_x = max(p[0] for p in pts) + r_total
        if not (min_x <= px <= max_x): return False

        min_y = min(p[1] for p in pts) - r_total
        max_y = max(p[1] for p in pts) + r_total
        if not (min_y <= py <= max_y): return False

        # 2. Accurate Distance Check
        r_sq = r_total ** 2
        
        if len(pts) == 1:
            dx = pts[0][0] - px
            dy = pts[0][1] - py
            return (dx*dx + dy*dy) <= r_sq

        for j in range(len(pts) - 1):
            x1, y1 = pts[j]
            x2, y2 = pts[j+1]
            
            dx = x2 - x1
            dy = y2 - y1
            
            px_dx = px - x1
            py_dy = py - y1
            
            if dx == 0 and dy == 0:
                dist_sq = px_dx*px_dx + py_dy*py_dy
            else:
                t = (px_dx * dx + py_dy * dy) / (dx*dx + dy*dy)
                t = max(0, min(1, t))
                
                proj_x = x1 + t * dx
                proj_y = y1 + t * dy
                
                dist_sq = (px - proj_x)**2 + (py - proj_y)**2
                
            if dist_sq <= r_sq:
                return True
                
        return False

    # ── Rendering ────────────────────────────────────────────────

    def _render_canvas(self):
        """Re-render the canvas from the stroke list."""
        self._canvas = np.zeros(
            (self._height, self._width, 4), dtype=np.uint8
        )
        for stroke in self._strokes:
            self._draw_stroke_on_canvas(stroke)

    def _draw_stroke_on_canvas(self, stroke: Stroke):
        """Draw a single stroke onto the canvas."""
        if len(stroke.points) < 2:
            return

        pts = np.array(stroke.points, dtype=np.int32)

        # Draw with full alpha for all standard lines
        # (Legacy opacity eraser support removed)
        bgr = stroke.color
        cv2.polylines(
            self._canvas,
            [pts],
            isClosed=False,
            color=(bgr[0], bgr[1], bgr[2], 255),
            thickness=stroke.thickness,
            lineType=cv2.LINE_AA,
        )

    def render(self) -> np.ndarray:
        """
        Get the current canvas as a BGRA image.
        Includes the active stroke if one is in progress.
        """
        # Start with the committed canvas
        canvas = self._canvas.copy()

        # Draw the active stroke on top
        if self._active_stroke and len(self._active_stroke.points) >= 2:
            self._draw_stroke_on_array(canvas, self._active_stroke)

        return canvas

    def _draw_stroke_on_array(self, img: np.ndarray, stroke: Stroke):
        """Draw a stroke onto an arbitrary BGRA array."""
        pts = np.array(stroke.points, dtype=np.int32)
        bgr = stroke.color
        cv2.polylines(
            img, [pts], isClosed=False,
            color=(bgr[0], bgr[1], bgr[2], 255),
            thickness=stroke.thickness,
            lineType=cv2.LINE_AA,
        )

    def composite(self, background: np.ndarray) -> np.ndarray:
        """
        Composite the canvas onto a BGR background image.
        Uses alpha blending for smooth edges.
        """
        canvas = self.render()

        # Extract alpha channel as float mask [0, 1]
        alpha = canvas[:, :, 3:4].astype(np.float32) / 255.0
        canvas_bgr = canvas[:, :, :3].astype(np.float32)
        bg = background.astype(np.float32)

        # Alpha blend: result = canvas * alpha + background * (1 - alpha)
        blended = canvas_bgr * alpha + bg * (1.0 - alpha)
        return blended.astype(np.uint8)

    def get_canvas_bgr(self) -> np.ndarray:
        """Get the canvas as a BGR image (white background) for export."""
        canvas = self.render()
        # White background
        bg = np.full((self._height, self._width, 3), 255, dtype=np.uint8)

        alpha = canvas[:, :, 3:4].astype(np.float32) / 255.0
        canvas_bgr = canvas[:, :, :3].astype(np.float32)
        result = canvas_bgr * alpha + bg.astype(np.float32) * (1.0 - alpha)
        return result.astype(np.uint8)
