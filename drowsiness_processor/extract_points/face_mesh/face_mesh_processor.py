import mediapipe as mp
import numpy as np
import cv2
from typing import Tuple, Any, List, Dict


class FaceMeshInference:
    def __init__(self, min_detection_confidence=0.6, min_tracking_confidence=0.6):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process(self, image: np.ndarray) -> Tuple[bool, Any]:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_mesh = self.face_mesh.process(rgb_image)
        return bool(face_mesh.multi_face_landmarks), face_mesh


class FaceMeshExtractor:
    def __init__(self):
        self.points: dict = {
            'eyes': {'distances': []},
            'mouth': {'distances': []},
            'head': {'distances': []},
        }

    def extract_points(self, face_image: np.ndarray, face_mesh_info: Any) -> List[List[int]]:
        h, w, _ = face_image.shape
        mesh_points = [
            [i, int(pt.x * w), int(pt.y * h)]
            for face in face_mesh_info.multi_face_landmarks
            for i, pt in enumerate(face.landmark)
        ]
        return mesh_points

    def extract_feature_points(self, face_points: List[List[int]], feature_indices: dict):
        for feature, indices in feature_indices.items():
            for sub_feature, sub_indices in indices.items():
                self.points[feature][sub_feature] = [face_points[i][1:] for i in sub_indices]

    def get_eyes_points(self, face_points: List[List[int]]) -> Dict[str, List[List[int]]]:
        feature_indices = {
            'eyes': {
                'distances': [159, 145, 385, 374, 468, 472, 473, 477, 468, 473],
            }
        }
        self.extract_feature_points(face_points, feature_indices)
        return self.points['eyes']

    def get_mouth_points(self, face_points: List[List[int]]) -> Dict[str, List[List[int]]]:
        feature_indices = {
            'mouth': {
                'distances': [13, 14, 17, 199]
            }
        }
        self.extract_feature_points(face_points, feature_indices)
        return self.points['mouth']

    def get_head_points(self, face_points: List[List[int]]) -> Dict[str, List[List[int]]]:
        feature_indices = {
            'head': {
                'distances': [1, 0, 1, 5, 4, 205, 425]
            }
        }
        self.extract_feature_points(face_points, feature_indices)
        return self.points['head']


class FaceMeshDrawer:
    def __init__(self, color: Tuple[int, int, int] = (255, 255, 0)):
        self.mp_draw = mp.solutions.drawing_utils
        self.config_draw = self.mp_draw.DrawingSpec(color=color, thickness=1, circle_radius=1)

    def draw(self, face_image: np.ndarray, face_mesh_info: Any):
        for face_mesh in face_mesh_info.multi_face_landmarks:
            self.mp_draw.draw_landmarks(face_image, face_mesh, mp.solutions.face_mesh.FACEMESH_TESSELATION,
                                        self.config_draw, self.config_draw)

    def draw_sketch(self, face_image: np.ndarray, face_mesh_info: Any):
        h, w, _ = face_image.shape
        black_image = np.zeros((h, w, 3), dtype=np.uint8)
        for face_mesh in face_mesh_info.multi_face_landmarks:
            for pt in face_mesh.landmark:
                x = int(pt.x * w)
                y = int(pt.y * h)
                z = int(pt.z * 50)
                cv2.circle(black_image, (x, y), 1, (255 - z, 255 - z, 0 - z), -1)
        return black_image


class FaceMeshProcessor:
    def __init__(self):
        self.inference = FaceMeshInference()
        self.extractor = FaceMeshExtractor()
        self.drawer = FaceMeshDrawer()

    def process(self, face_image: np.ndarray, draw: bool = True) -> Tuple[dict, bool, np.ndarray]:
        h, w, _ = face_image.shape
        sketch = np.zeros((h, w, 3), dtype=np.uint8)
        success, face_mesh_info = self.inference.process(face_image)
        if not success:
            points = {'face_detected': False}
            return points, success, sketch

        face_points = self.extractor.extract_points(face_image, face_mesh_info)
        
        # Verificar si hay suficientes landmarks para considerar un rostro válido
        if len(face_points) < 473:  # Necesitamos al menos 473 landmarks para un rostro completo
            points = {'face_detected': False}
            return points, False, sketch
        
        # Verificar landmarks específicos que deben estar presentes en un rostro real
        # Landmarks clave: nariz (1), ojos (33, 263), boca (61, 291), mentón (152)
        required_landmarks = [1, 33, 263, 61, 291, 152]
        for landmark_id in required_landmarks:
            if landmark_id >= len(face_points):
                points = {'face_detected': False}
                return points, False, sketch
        
        # Verificar que los landmarks estén en posiciones razonables (dentro de la imagen)
        for landmark_id in required_landmarks:
            x, y = face_points[landmark_id][1], face_points[landmark_id][2]
            if x < 0 or x >= w or y < 0 or y >= h:
                points = {'face_detected': False}
                return points, False, sketch
        
        # Verificar que la distancia entre ojos sea razonable (indicador de rostro real)
        left_eye = face_points[33][1:]  # Ojo izquierdo
        right_eye = face_points[263][1:]  # Ojo derecho
        eye_distance = np.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)
        
        # La distancia entre ojos debe ser al menos 50 píxeles para un rostro real
        if eye_distance < 50:
            points = {'face_detected': False}
            return points, False, sketch
            
        points = {
            'eyes': self.extractor.get_eyes_points(face_points),
            'mouth': self.extractor.get_mouth_points(face_points),
            'head': self.extractor.get_head_points(face_points),
        }

        # --- NUEVO: Calcular gaze_x (posición horizontal) y gaze_y (vertical) de la pupila respecto al rostro ---
        # Usar landmarks de los ojos para estimar la dirección de la mirada
        # Ejemplo simple: usar el landmark 468 (ojo derecho) y 473 (ojo izquierdo) para estimar el centro de la mirada
        # y normalizar respecto al ancho y alto de la cara
        if len(face_points) > 473:
            left_eye_x = face_points[468][1]
            right_eye_x = face_points[473][1]
            left_eye_y = face_points[468][2]
            right_eye_y = face_points[473][2]
            # Centro de la mirada
            gaze_x_pixel = (left_eye_x + right_eye_x) / 2
            gaze_y_pixel = (left_eye_y + right_eye_y) / 2
            gaze_x_norm = gaze_x_pixel / w  # Normalizado entre 0 y 1
            gaze_y_norm = gaze_y_pixel / h  # Normalizado entre 0 y 1
            points['gaze_x'] = gaze_x_norm
            points['gaze_y'] = gaze_y_norm
        else:
            points['gaze_x'] = 0.5  # Valor por defecto (centro)
            points['gaze_y'] = 0.5  # Valor por defecto (centro)

        # --- NUEVO: Calcular ángulos de rotación de la cabeza (yaw, pitch, roll) ---
        # Usar landmarks de la mesh facial para estimar orientación de la cabeza
        # Yaw: rotación izquierda/derecha
        # Pitch: arriba/abajo
        # Roll: inclinación lateral
        if len(face_points) > 473:
            # Nariz (landmark 1), mentón (152), ojo izquierdo (33), ojo derecho (263),
            # comisura izq boca (61), comisura der boca (291)
            nose = np.array(face_points[1][1:])
            chin = np.array(face_points[152][1:])
            left_eye = np.array(face_points[33][1:])
            right_eye = np.array(face_points[263][1:])
            left_mouth = np.array(face_points[61][1:])
            right_mouth = np.array(face_points[291][1:])
            # Vectores
            eye_line = right_eye - left_eye
            mouth_line = right_mouth - left_mouth
            nose_chin = chin - nose
            # Roll: ángulo entre la línea de los ojos y el eje horizontal
            roll = np.degrees(np.arctan2(eye_line[1], eye_line[0]))
            # Yaw: ángulo entre la línea de los ojos y la línea de la boca (proyección horizontal)
            # Si la nariz se desplaza lateralmente respecto al centro de la línea de los ojos
            eye_center = (left_eye + right_eye) / 2
            mouth_center = (left_mouth + right_mouth) / 2
            face_center = (eye_center + mouth_center) / 2
            yaw = np.degrees(np.arctan2(nose[0] - face_center[0], nose[1] - face_center[1]))
            # Pitch: ángulo entre la línea nariz-mentón y el eje vertical
            pitch = np.degrees(np.arctan2(nose_chin[1], nose_chin[0]))
            points['head_yaw'] = float(yaw)
            points['head_pitch'] = float(pitch)
            points['head_roll'] = float(roll)
        else:
            points['head_yaw'] = 0.0
            points['head_pitch'] = 0.0
            points['head_roll'] = 0.0

        if draw:
            sketch = self.drawer.draw_sketch(face_image, face_mesh_info)
            points['face_detected'] = True
            return points, True, sketch

        points['face_detected'] = True
        return points, True, sketch
