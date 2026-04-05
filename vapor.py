import cv2
import numpy as np
from collections import deque
from typing import Optional, Tuple
import config

class VaporDetector:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200,
            varThreshold=50,
            detectShadows=False
        )
        self.window = deque(maxlen=config.VAPOR_WINDOW_FRAMES)

    def process(
        self,
        frame_bgr: np.ndarray,
        mouth_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> float:
        """
        Returns a vapor score between 0.0 and 1.0.
        """
        h, w = frame_bgr.shape[:2]
        fg_mask = self.bg_subtractor.apply(frame_bgr)

        if mouth_bbox:
            bx, by, bw, bh = mouth_bbox
            # Expand ROI upward (vapor rises) by 3x the height
            roi_y = max(0, by - bh * 3)
            roi_h = min(bh * 4, h - roi_y)
            roi_x = max(0, bx - bw // 4)
            roi_w = min(bw + bw // 2, w - roi_x)
        else:
            # No face detected: use center portion of frame
            roi_x = w // 4
            roi_y = 0
            roi_w = w // 2
            roi_h = h * 2 // 3

        roi = frame_bgr[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        fg_roi = fg_mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        if roi.size == 0:
            self.window.append(0.0)
            return 0.0

        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        s_channel = hsv[:, :, 1]

        # Vapor signature: bright (V > 200) + desaturated (S < 50) + moving
        bright_mask = (v_channel > 200).astype(np.uint8)
        desat_mask = (s_channel < 50).astype(np.uint8)
        motion_mask = (fg_roi > 128).astype(np.uint8)

        vapor_mask = bright_mask & desat_mask & motion_mask

        # Diffuseness: blur and check low variance (vapor is fuzzy)
        blurred = cv2.GaussianBlur(roi, (21, 21), 0)
        diff = cv2.absdiff(roi, blurred)
        texture_var = float(np.std(diff))
        diffuse_bonus = 1.0 if texture_var < 15.0 else 0.5

        total_pixels = roi_h * roi_w
        vapor_pixels = int(np.sum(vapor_mask))
        score = (vapor_pixels / total_pixels) * diffuse_bonus

        self.window.append(score)

        # Sliding window: require consistent detection
        hits = sum(1 for s in self.window if s > config.VAPOR_THRESHOLD)
        smoothed = hits / config.VAPOR_WINDOW_FRAMES

        return smoothed

    def get_roi_rect(
        self,
        frame_shape: Tuple[int, int],
        mouth_bbox: Optional[Tuple[int, int, int, int]]
    ) -> Tuple[int, int, int, int]:
        """Return the current ROI rect for debug drawing."""
        h, w = frame_shape[:2]
        if mouth_bbox:
            bx, by, bw, bh = mouth_bbox
            roi_y = max(0, by - bh * 3)
            roi_h = min(bh * 4, h - roi_y)
            roi_x = max(0, bx - bw // 4)
            roi_w = min(bw + bw // 2, w - roi_x)
            return roi_x, roi_y, roi_w, roi_h
        return w // 4, 0, w // 2, h * 2 // 3
