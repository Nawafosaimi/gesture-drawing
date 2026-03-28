"""
Coordinate Smoothing Filters
Kalman Filter (primary) and Exponential Moving Average (fallback)
for jitter-free fingertip tracking.
"""

import numpy as np
from config import settings


class KalmanFilter2D:
    """
    2D Kalman Filter for (x, y) position tracking.
    
    State vector: [x, y, vx, vy] (position + velocity)
    Measurement: [x, y]
    
    The velocity components allow the filter to predict motion,
    reducing lag compared to pure smoothing methods.
    """

    def __init__(
        self,
        process_noise: float = settings.KALMAN_PROCESS_NOISE,
        measurement_noise: float = settings.KALMAN_MEASUREMENT_NOISE,
    ):
        self._process_noise = process_noise
        self._measurement_noise = measurement_noise

        # State: [x, y, vx, vy]
        self._x = np.zeros(4, dtype=np.float64)  # State estimate
        self._P = np.eye(4, dtype=np.float64) * 1000  # Covariance (high = uncertain)

        # State transition matrix (constant velocity model)
        self._F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        # Measurement matrix (we observe x, y only)
        self._H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Process noise covariance
        self._Q = np.eye(4, dtype=np.float64) * process_noise

        # Measurement noise covariance
        self._R = np.eye(2, dtype=np.float64) * measurement_noise

        self._initialized = False

    def update(self, measurement: tuple[int, int]) -> tuple[int, int]:
        """
        Feed a new measurement and return the filtered position.
        
        Args:
            measurement: (x, y) pixel coordinates.
            
        Returns:
            Filtered (x, y) as integers.
        """
        z = np.array([measurement[0], measurement[1]], dtype=np.float64)

        if not self._initialized:
            self._x[:2] = z
            self._x[2:] = 0  # Zero velocity initially
            self._initialized = True
            return int(z[0]), int(z[1])

        # ── Predict ──
        x_pred = self._F @ self._x
        P_pred = self._F @ self._P @ self._F.T + self._Q

        # ── Update ──
        y = z - self._H @ x_pred  # Innovation
        S = self._H @ P_pred @ self._H.T + self._R  # Innovation covariance
        K = P_pred @ self._H.T @ np.linalg.inv(S)  # Kalman gain

        self._x = x_pred + K @ y
        self._P = (np.eye(4) - K @ self._H) @ P_pred

        return int(round(self._x[0])), int(round(self._x[1]))

    def reset(self):
        """Reset filter state (call when hand disappears/reappears)."""
        self._x = np.zeros(4, dtype=np.float64)
        self._P = np.eye(4, dtype=np.float64) * 1000
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized


class ExponentialSmoother:
    """
    Exponential Moving Average (EMA) filter for 2D coordinates.
    
    Simpler fallback when Kalman is not needed.
    smoothed = alpha * new + (1 - alpha) * smoothed_prev
    """

    def __init__(self, alpha: float = settings.EMA_ALPHA):
        self._alpha = alpha
        self._smoothed: tuple[float, float] | None = None

    def update(self, measurement: tuple[int, int]) -> tuple[int, int]:
        """
        Feed a new measurement and return the smoothed position.
        """
        if self._smoothed is None:
            self._smoothed = (float(measurement[0]), float(measurement[1]))
        else:
            self._smoothed = (
                self._alpha * measurement[0] + (1 - self._alpha) * self._smoothed[0],
                self._alpha * measurement[1] + (1 - self._alpha) * self._smoothed[1],
            )
        return int(round(self._smoothed[0])), int(round(self._smoothed[1]))

    def reset(self):
        """Reset smoother state."""
        self._smoothed = None

    @property
    def is_initialized(self) -> bool:
        return self._smoothed is not None
