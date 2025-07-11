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
        eyes_closed = face_points.get('eyes_closed', False)
        face_detected = face_points.get('face_detected', True)
        head_yaw = abs(face_points.get('head_yaw', 0.0))
        head_pitch = abs(face_points.get('head_pitch', 0.0))
        head_roll = abs(face_points.get('head_roll', 0.0))
        if eyes_closed:
            if self.eyes_closed_start is None:
                self.eyes_closed_start = time.time()
            elif (time.time() - self.eyes_closed_start) >= self.eyes_closed_threshold:
                self.eyes_closed_long = True
        else:
            self.eyes_closed_start = None
            self.eyes_closed_long = False
        max_angle = 20.0
        in_attention = (
            head_yaw < max_angle and
            head_pitch < max_angle and
            head_roll < max_angle and
            not self.eyes_closed_long and
            face_detected
        )
        self.attention_log.append((current_time, in_attention))
        return {
            'timestamp': current_time,
            'in_attention': in_attention,
            'head_yaw': head_yaw,
            'head_pitch': head_pitch,
            'head_roll': head_roll,
            'eyes_closed': eyes_closed,
            'face_detected': face_detected,
            'eyes_closed_long': self.eyes_closed_long,
            'eyes_closed_duration': (time.time() - self.eyes_closed_start) if self.eyes_closed_start else 0.0
        }

    def export_attention_log(self):
        return self.attention_log 