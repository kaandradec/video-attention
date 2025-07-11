import time
from typing import Dict, Any
from drowsiness_processor.drowsiness_features.processor import DrowsinessProcessor

class GazeAttentionEstimator(DrowsinessProcessor):
    def __init__(self, screen_center_x=0.5, screen_center_y=0.5, tolerance_x=0.2, tolerance_y=0.2, eyes_closed_threshold=1.5):
        self.screen_center_x = screen_center_x
        self.screen_center_y = screen_center_y
        self.tolerance_x = tolerance_x
        self.tolerance_y = tolerance_y
        self.attention_log = []
        self.start_time = None
        self.eyes_closed_threshold = eyes_closed_threshold
        self.eyes_closed_start = None
        self.eyes_closed_long = False

    def process(self, face_points: dict) -> Dict[str, Any]:
        if self.start_time is None:
            self.start_time = time.time()
        current_time = time.time() - self.start_time

        # Obtener variables necesarias
        head_pitch = abs(face_points.get('head_pitch', 0.0))
        head_yaw = abs(face_points.get('head_yaw', 0.0))
        head_roll = abs(face_points.get('head_roll', 0.0))
        eyes_closed = face_points.get('eyes_closed', False)
        face_detected = face_points.get('face_detected', True)

        # Calcular duración de ojos cerrados
        if eyes_closed:
            if self.eyes_closed_start is None:
                self.eyes_closed_start = time.time()
            eyes_closed_duration = time.time() - self.eyes_closed_start
            eyes_closed_long = eyes_closed_duration >= self.eyes_closed_threshold
        else:
            self.eyes_closed_start = None
            eyes_closed_duration = 0.0
            eyes_closed_long = False

        # Lógica de atención igual que en frontend
        pitch_threshold = 12.0  # igual que en el frontend
        pitch_valid = abs(head_pitch) < pitch_threshold
        eyes_valid = not eyes_closed_long
        face_valid = face_detected
        in_attention = pitch_valid and eyes_valid and face_valid

        self.attention_log.append((current_time, in_attention))
        return {
            'timestamp': current_time,
            'in_attention': in_attention,
            'head_pitch': head_pitch,
            'head_yaw': head_yaw,
            'head_roll': head_roll,
            'eyes_closed': eyes_closed,
            'face_detected': face_detected,
            'eyes_closed_long': eyes_closed_long,
            'eyes_closed_duration': eyes_closed_duration
        }

    def export_attention_log(self):
        return self.attention_log 