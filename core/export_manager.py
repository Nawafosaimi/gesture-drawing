"""
AirCanvas — Export Manager
Handles saving canvas images and optional video recording.
"""

import os
import time
import cv2
import numpy as np

from config import settings


class ExportManager:
    """
    Manages export of canvas images and video recordings.
    
    Features:
    - Auto-creates output directory
    - Timestamped filenames to avoid overwrites
    - PNG export with configurable quality
    - Video recording via cv2.VideoWriter
    """

    def __init__(self, output_dir: str = settings.OUTPUT_DIR):
        self._output_dir = output_dir
        self._video_writer: cv2.VideoWriter | None = None
        os.makedirs(output_dir, exist_ok=True)

    def save_canvas(self, canvas_bgr: np.ndarray, prefix: str = "drawing") -> str:
        """
        Save the canvas as a PNG file.
        
        Args:
            canvas_bgr: BGR image to save.
            prefix: Filename prefix.
            
        Returns:
            Absolute path to the saved file.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.{settings.CANVAS_SAVE_FORMAT}"
        filepath = os.path.join(self._output_dir, filename)

        cv2.imwrite(filepath, canvas_bgr)
        return os.path.abspath(filepath)

    # ── Video Recording ──────────────────────────────────────────

    def start_recording(
        self,
        width: int = settings.FRAME_WIDTH,
        height: int = settings.FRAME_HEIGHT,
        fps: float = settings.VIDEO_FPS,
        prefix: str = "recording",
    ) -> str:
        """
        Start recording frames to a video file.
        
        Returns:
            Path to the video file being written.
        """
        if self._video_writer is not None:
            self.stop_recording()

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.avi"
        filepath = os.path.join(self._output_dir, filename)

        fourcc = cv2.VideoWriter_fourcc(*settings.VIDEO_CODEC)
        self._video_writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

        return os.path.abspath(filepath)

    def write_frame(self, frame: np.ndarray):
        """Write a single frame to the video. No-op if not recording."""
        if self._video_writer is not None:
            self._video_writer.write(frame)

    def stop_recording(self):
        """Stop recording and release the video writer."""
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None

    @property
    def is_recording(self) -> bool:
        return self._video_writer is not None

    def release(self):
        """Release all resources."""
        self.stop_recording()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
