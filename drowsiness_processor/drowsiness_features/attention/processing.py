import time
from typing import Dict, Any
from drowsiness_processor.drowsiness_features.processor import DrowsinessProcessor

class GazeAttentionEstimator(DrowsinessProcessor):
    def __init__(self, screen_center_x=0.5, tolerance=0.2):
        self.screen_center_x = screen_center_x  # Normalizado (0 a 1)
        self.tolerance = tolerance  # Cuánto puede desviarse la mirada del centro
        self.attention_log = []  # Guarda (timestamp, attention: bool)
        self.start_time = None

    def process(self, face_points: dict) -> Dict[str, Any]:
        # Suponemos que face_points tiene la clave 'gaze_x' normalizada (0 a 1)
        if self.start_time is None:
            self.start_time = time.time()
        current_time = time.time() - self.start_time
        gaze_x = face_points.get('gaze_x', 0.5)
        in_attention = abs(gaze_x - self.screen_center_x) <= self.tolerance
        self.attention_log.append((current_time, in_attention))
        return {
            'timestamp': current_time,
            'in_attention': in_attention
        }

    def export_attention_log(self):
        return self.attention_log 