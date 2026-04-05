import time
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from dataclasses import dataclass
from typing import Optional, Tuple
import config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class GestureState:
    gesture_active: bool = False
    last_gesture_time: float = 0.0
    mouth_region_bbox: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h

class GestureDetector:
    def __init__(self):
        hand_model_path = os.path.join(BASE_DIR, "hand_landmarker.task")
        face_model_path = os.path.join(BASE_DIR, "face_landmarker.task")

        hand_options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=hand_model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        face_options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=face_model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)
        self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(face_options)
        self.state = GestureState()

    def process(self, frame_rgb: np.ndarray) -> GestureState:
        h, w = frame_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        face_result = self.face_landmarker.detect(mp_image)
        hand_result = self.hand_landmarker.detect(mp_image)

        mouth_center = None
        mouth_bbox = None
        face_h = h * 0.5  # fallback

        if face_result.face_landmarks:
            lm = face_result.face_landmarks[0]
            # Lip landmarks: 13=upper lip, 14=lower lip, 61/291=corners
            lip_indices = [13, 14, 61, 291, 0, 17]
            lip_pts = [lm[i] for i in lip_indices if i < len(lm)]
            xs = [p.x * w for p in lip_pts]
            ys = [p.y * h for p in lip_pts]
            mx = int(np.mean(xs))
            my = int(np.mean(ys))
            mouth_center = (mx, my)

            all_xs = [p.x * w for p in lm]
            all_ys = [p.y * h for p in lm]
            face_h = max(all_ys) - min(all_ys)

            bw = int(face_h * 0.6)
            bh = int(face_h * 0.4)
            mouth_bbox = (
                max(0, mx - bw // 2),
                max(0, my - bh // 4),
                min(bw, w),
                min(bh, h)
            )
            self.state.mouth_region_bbox = mouth_bbox

        if hand_result.hand_landmarks and mouth_center:
            mx, my = mouth_center
            for hand_lm in hand_result.hand_landmarks:
                tip_indices = [4, 8, 12, 16, 20]
                tips = [hand_lm[i] for i in tip_indices if i < len(hand_lm)]
                tip_x = np.mean([p.x * w for p in tips])
                tip_y = np.mean([p.y * h for p in tips])

                dist = np.sqrt((tip_x - mx) ** 2 + (tip_y - my) ** 2)
                norm_dist = dist / (face_h if face_h > 0 else 1)

                if norm_dist < config.HAND_MOUTH_DISTANCE_RATIO:
                    self.state.gesture_active = True
                    self.state.last_gesture_time = time.time()
                    return self.state

        elapsed = time.time() - self.state.last_gesture_time
        if elapsed > config.GESTURE_COOLDOWN_SEC:
            self.state.gesture_active = False

        return self.state

    def close(self):
        self.hand_landmarker.close()
        self.face_landmarker.close()
