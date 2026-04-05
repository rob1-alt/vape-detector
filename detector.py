#!/usr/bin/env python3
"""
Vape Detector — détecte si tu vapes devant ta webcam et t'envoie un message.
Usage: python3 detector.py
Quit: press 'q' in the camera window
"""

import cv2
import time
import sys
import config
from gesture import GestureDetector
from vapor import VaporDetector
from alert import AlertManager

def main():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("Impossible d'ouvrir la camera. Verifie les permissions dans Reglages > Confidentialite > Camera.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    gesture_detector = GestureDetector()
    vapor_detector = VaporDetector()
    alert_manager = AlertManager()

    print("Vape Detector actif. Appuie sur 'q' pour quitter.")
    frame_count = 0

    gesture_state = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur de lecture camera.")
            break

        frame_count += 1
        display = frame.copy()

        # Run detection every N frames to save CPU
        if frame_count % config.PROCESS_EVERY_N_FRAMES == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gesture_state = gesture_detector.process(frame_rgb)

        vapor_score = vapor_detector.process(frame, gesture_state.mouth_region_bbox if gesture_state else None)

        # Compute combined confidence
        confidence = 0.0
        if gesture_state and gesture_state.gesture_active:
            confidence += 0.4
        if vapor_score > config.VAPOR_THRESHOLD:
            confidence += 0.6

        alert_manager.maybe_alert(confidence, gesture_state.gesture_active if gesture_state else False, vapor_score)

        # Debug overlay
        if config.SHOW_DEBUG:
            # Vapor ROI box
            if gesture_state:
                rx, ry, rw, rh = vapor_detector.get_roi_rect(frame.shape, gesture_state.mouth_region_bbox)
                cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 1)

                # Mouth bbox
                if gesture_state.mouth_region_bbox:
                    bx, by, bw, bh = gesture_state.mouth_region_bbox
                    cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (0, 200, 0), 2)

            # Scores
            gesture_active = gesture_state.gesture_active if gesture_state else False
            g_color = (0, 255, 0) if gesture_active else (80, 80, 80)
            v_color = (0, 255, 0) if vapor_score > config.VAPOR_THRESHOLD else (80, 80, 80)
            c_color = (0, 0, 255) if confidence >= 0.6 else (200, 200, 200)

            cv2.putText(display, f"Gesture: {'OUI' if gesture_active else 'non'}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, g_color, 2)
            cv2.putText(display, f"Vapeur:  {vapor_score:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, v_color, 2)
            cv2.putText(display, f"Confiance: {confidence:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, c_color, 2)
            cv2.putText(display, "q = quitter", (10, config.FRAME_HEIGHT - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("Vape Detector", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    gesture_detector.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Vape Detector arrete.")

if __name__ == "__main__":
    main()
