import numpy as np
import base64
import cv2

from drowsiness_processor.extract_points.point_extractor import PointsExtractor
from drowsiness_processor.data_processing.main import PointsProcessing
from drowsiness_processor.drowsiness_features.processing import FeaturesDrowsinessProcessing
from drowsiness_processor.visualization.main import ReportVisualizer
from drowsiness_processor.reports.main import DrowsinessReports
from drowsiness_processor.drowsiness_features.attention.processing import GazeAttentionEstimator


class DrowsinessDetectionSystem:
    def __init__(self):
        self.points_extractor = PointsExtractor()
        self.points_processing = PointsProcessing()
        self.features_processing = FeaturesDrowsinessProcessing()
        self.visualizer = ReportVisualizer()
        self.reports = DrowsinessReports('drowsiness_processor/reports/august/drowsiness_report.csv')
        self.json_report: dict = {}
        self.attention_estimator = GazeAttentionEstimator()
        self.last_attention = None

    def run(self, picture_base64: str):
        # decode base64
        picture_bytes = base64.b64decode(picture_base64)
        # convert bytes to OpenCV image
        picture = cv2.imdecode(np.frombuffer(picture_bytes, np.uint8), cv2.IMREAD_COLOR)
        return self.frame_processing(picture)

    def frame_processing(self, face_image: np.ndarray):
        key_points, control_process, sketch = self.points_extractor.process(face_image)
        if control_process:
            points_processed = self.points_processing.main(key_points)
            drowsiness_features_processed = self.features_processing.main(points_processed)
            sketch = self.visualizer.visualize_all_reports(sketch, drowsiness_features_processed)
            self.reports.main(drowsiness_features_processed)
            self.json_report = self.reports.generate_json_report(drowsiness_features_processed)
            # --- NUEVO: Determinar eyes_closed y face_detected ---
            eyes_closed = False
            if 'flicker_and_micro_sleep' in drowsiness_features_processed:
                flicker_estimator = self.features_processing.features_drowsiness['flicker_and_micro_sleep']
                eyes_distances = points_processed.get('eyes', {})
                if hasattr(flicker_estimator, 'micro_sleep_detector'):
                    eyes_closed = flicker_estimator.micro_sleep_detector.closed_eyes(eyes_distances)
            face_detected = control_process
            # Usar solo los valores de rotación
            head_yaw = points_processed.get('head_yaw', key_points.get('head_yaw', 0.0))
            head_pitch = points_processed.get('head_pitch', key_points.get('head_pitch', 0.0))
            head_roll = points_processed.get('head_roll', key_points.get('head_roll', 0.0))
            attention_input = dict(key_points)
            attention_input['eyes_closed'] = eyes_closed
            attention_input['face_detected'] = face_detected
            attention_input['head_yaw'] = head_yaw
            attention_input['head_pitch'] = head_pitch
            attention_input['head_roll'] = head_roll
            attention_result = self.attention_estimator.process(attention_input)
            # --- NUEVO: Agregar features de drowsiness al JSON de atención ---
            # (solo los campos principales, puedes agregar más si lo deseas)
            attention_result['flicker'] = drowsiness_features_processed.get('flicker_and_micro_sleep', {})
            attention_result['micro_sleep'] = drowsiness_features_processed.get('flicker_and_micro_sleep', {})
            attention_result['pitch'] = drowsiness_features_processed.get('pitch', {})
            attention_result['yawn'] = drowsiness_features_processed.get('yawn', {})
            attention_result['eye_rub_first_hand'] = drowsiness_features_processed.get('eye_rub_first_hand', {})
            attention_result['eye_rub_second_hand'] = drowsiness_features_processed.get('eye_rub_second_hand', {})
            self.last_attention = attention_result
        return face_image, sketch, self.json_report, self.last_attention
