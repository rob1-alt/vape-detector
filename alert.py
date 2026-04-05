import time
import threading
import subprocess
import config

class AlertManager:
    def __init__(self):
        self.last_alert_time = 0.0

    def maybe_alert(self, confidence: float, gesture_active: bool, vapor_score: float):
        now = time.time()
        if now - self.last_alert_time < config.ALERT_COOLDOWN_SEC:
            return
        if confidence < 0.6:
            return

        self.last_alert_time = now
        print(f"[VAPE DETECTED] {time.strftime('%H:%M:%S')} | confidence={confidence:.2f} | gesture={gesture_active} | vapor={vapor_score:.2f}")

        with open("detections.log", "a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} confidence={confidence:.2f} gesture={gesture_active} vapor={vapor_score:.2f}\n")

        threading.Thread(target=self._send_alert, daemon=True).start()

    def _send_alert(self):
        msg = config.ALERT_MESSAGE.replace('"', '\\"')
        title = config.ALERT_TITLE.replace('"', '\\"')

        # macOS native dialog — always on top, no thread restrictions
        script = f'display dialog "{msg}" with title "{title}" buttons {{"OK"}} default button "OK" with icon stop'
        subprocess.Popen(["osascript", "-e", script])
