"""
Geometry Utilities
Distance, angle, and landmark helper functions.
"""

import math
import numpy as np


def distance(p1: tuple, p2: tuple) -> float:
    """Euclidean distance between two 2D or 3D points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def distance_2d(p1: tuple, p2: tuple) -> float:
    """Fast 2D Euclidean distance."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def angle_between(p1: tuple, p2: tuple, p3: tuple) -> float:
    """
    Angle at p2 formed by p1-p2-p3, in degrees.
    Useful for measuring finger curl.
    """
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def normalize_to_pixel(
    landmark_x: float,
    landmark_y: float,
    frame_width: int,
    frame_height: int,
    mirror: bool = True,
) -> tuple[int, int]:
    """
    Convert MediaPipe normalized [0,1] coordinates to pixel coordinates.
    Mirrors x-axis by default so the user sees a natural mirror image.
    """
    px = int(landmark_x * frame_width)
    py = int(landmark_y * frame_height)
    if mirror:
        px = frame_width - px
    return px, py


def hand_bbox_size(landmarks: list[tuple]) -> float:
    """
    Estimate the 'size' of the hand from its landmarks.
    Used to normalize distance thresholds across different hand distances from camera.
    Returns the diagonal of the bounding box of all landmarks.
    """
    if not landmarks:
        return 1.0
    xs = [lm[0] for lm in landmarks]
    ys = [lm[1] for lm in landmarks]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    return math.sqrt(w ** 2 + h ** 2) + 1e-6


def midpoint(p1: tuple, p2: tuple) -> tuple:
    """Midpoint of two 2D points."""
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
