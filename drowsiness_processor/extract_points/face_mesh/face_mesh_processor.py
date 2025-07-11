import mediapipe as mp
import numpy as np
import cv2
from typing import Tuple, Any, List, Dict
import math

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
        return bool(face_mesh and face_mesh.multi_face_landmarks), face_mesh

class FaceMeshExtractor:
    def __init__(self):
        self.points: dict = {
            'eyes': {'distances': []},
            'mouth': {'distances': []},
            'head': {'distances': []},
        }

    def extract_points(self, face_image: np.ndarray, face_mesh_info: Any) -> List[List[int]]:
        """Extrae todos los landmarks de la malla facial en coordenadas de imagen [id, x, y, z]."""
        h, w, _ = face_image.shape
        mesh_points = [
            [i, int(pt.x * w), int(pt.y * h), pt.z]
            for face in face_mesh_info.multi_face_landmarks
            for i, pt in enumerate(face.landmark)
        ]
        return mesh_points

    def extract_feature_points(self, face_points: List[List[int]], feature_indices: dict):
        for feature, indices in feature_indices.items():
            for sub_feature, sub_indices in indices.items():
                self.points[feature][sub_feature] = [face_points[i][1:3] for i in sub_indices if i < len(face_points)]

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
        if face_mesh_info and face_mesh_info.multi_face_landmarks:
            for face_mesh in face_mesh_info.multi_face_landmarks:
                self.mp_draw.draw_landmarks(face_image, face_mesh, mp.solutions.face_mesh.FACEMESH_TESSELATION,
                                            self.config_draw, self.config_draw)

    def draw_sketch(self, face_image: np.ndarray, face_mesh_info: Any):
        h, w, _ = face_image.shape
        black_image = np.zeros((h, w, 3), dtype=np.uint8)
        if face_mesh_info and face_mesh_info.multi_face_landmarks:
            for face_mesh in face_mesh_info.multi_face_landmarks:
                for pt in face_mesh.landmark:
                    x = int(pt.x * w)
                    y = int(pt.y * h)
                    z_scaled = int(pt.z * w/4)
                    color_intensity = max(0, min(255, 128 - z_scaled))
                    cv2.circle(black_image, (x, y), 1, (color_intensity, color_intensity, 0), -1)
        return black_image

    def draw_pose_axes(self, image, nose_tip_2d, yaw_end_2d, pitch_end_2d, roll_end_2d):
        """Dibuja los ejes de pose (Yaw, Pitch, Roll) proyectados en la imagen."""
        cv2.line(image, nose_tip_2d, yaw_end_2d, (0, 0, 255), 2)  # Eje Z (azul)
        cv2.line(image, nose_tip_2d, pitch_end_2d, (0, 255, 0), 2)  # Eje Y (verde)
        cv2.line(image, nose_tip_2d, roll_end_2d, (255, 0, 0), 2)  # Eje X (rojo)

class FaceMeshProcessor:
    def __init__(self):
        self.inference = FaceMeshInference()
        self.extractor = FaceMeshExtractor()
        self.drawer = FaceMeshDrawer()

        # Puntos 3D del modelo de cabeza genérica (en milímetros)
        # Estos puntos corresponden a los landmarks específicos de MediaPipe
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # 1. Punta de la nariz
            (0.0, -330.0, -65.0),        # 152. Mentón
            (-225.0, 170.0, -135.0),     # 33. Esquina interna ojo izq
            (225.0, 170.0, -135.0),      # 263. Esquina interna ojo der
            (-150.0, -150.0, -125.0),    # 61. Comisura izq boca
            (150.0, -150.0, -125.0)      # 291. Comisura der boca
        ], dtype=np.float32)

        # Índices de MediaPipe correspondientes a los model_points definidos
        self.pnp_landmark_indices = [1, 152, 33, 263, 61, 291]

        # Coeficientes de distorsión (asumimos cero si no se calibra la cámara)
        self.dist_coeffs = np.zeros((4, 1))

    def process(self, face_image: np.ndarray, draw: bool = True) -> Tuple[dict, bool, np.ndarray]:
        h, w, _ = face_image.shape
        sketch = np.zeros((h, w, 3), dtype=np.uint8)
        points = {'face_detected': False, 'head_yaw': 0.0, 'head_pitch': 0.0, 'head_roll': 0.0}

        success, face_mesh_info = self.inference.process(face_image)

        if not success:
            return points, False, sketch

        all_face_points = self.extractor.extract_points(face_image, face_mesh_info)

        # Verificar si tenemos suficientes landmarks para PnP
        if len(all_face_points) < max(self.pnp_landmark_indices) + 1:
            return points, False, sketch

        # Preparar puntos 2D para PnP
        image_points = np.array([
            all_face_points[idx][1:3] for idx in self.pnp_landmark_indices
        ], dtype=np.float32)

        # Verificar si algún punto PnP está fuera de la imagen
        if np.any(image_points < 0) or np.any(image_points[:, 0] >= w) or np.any(image_points[:, 1] >= h):
            return points, False, sketch

        # Calcular la distancia entre ojos como indicador de rostro válido
        left_eye = np.array(all_face_points[33][1:3])
        right_eye = np.array(all_face_points[263][1:3])
        eye_distance = np.linalg.norm(left_eye - right_eye)

        if eye_distance < 20:
            return points, False, sketch

        # Estimar Matriz de Cámara
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)

        # Resolver PnP
        success_pnp, rvec, tvec = cv2.solvePnP(self.model_points, image_points, camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_DLS)

        if success_pnp:
            # Convertir rvec a ángulos de Euler (Yaw, Pitch, Roll)
            R, _ = cv2.Rodrigues(rvec)

            try:
                # Extraer ángulos de Euler de la matriz de rotación R
                sin_pitch = -R[2, 0]
                cos_pitch = math.sqrt(R[2, 1]**2 + R[2, 2]**2)

                if cos_pitch < 1e-6:  # Manejar gimbal lock
                    pitch = math.atan2(sin_pitch, 0)
                    if sin_pitch > 0:  # pitch = +90
                        yaw = math.atan2(R[0, 1], R[0, 2])
                        roll = 0
                    else:  # pitch = -90
                        yaw = math.atan2(-R[0, 1], -R[0, 2])
                        roll = 0
                else:
                    pitch = math.atan2(sin_pitch, cos_pitch)
                    yaw = math.atan2(R[1, 0], R[0, 0])
                    roll = math.atan2(R[2, 1], R[2, 2])

                # Convertir a grados
                yaw_deg = math.degrees(yaw)
                pitch_deg = math.degrees(pitch)
                roll_deg = math.degrees(roll)

                # Asignar los valores de rotación a gaze_x y gaze_y
                points['head_yaw'] = float(yaw_deg)
                points['head_pitch'] = float(pitch_deg)
                points['head_roll'] = float(roll_deg)
                points['face_detected'] = True
                print(f"PnP ROTATION: yaw={yaw_deg:.2f}, pitch={pitch_deg:.2f}, roll={roll_deg:.2f}")

            except Exception as e:
                print(f"Error calculando ángulos de Euler: {e}")
                points['face_detected'] = True
                points['head_yaw'] = 0.0
                points['head_pitch'] = 0.0
                points['head_roll'] = 0.0
        else:
            points['face_detected'] = False

        # Agregar los puntos de características necesarios para el resto del pipeline
        if points['face_detected']:
            # Extraer puntos de características usando el extractor
            face_points = self.extractor.extract_points(face_image, face_mesh_info)
            points['eyes'] = self.extractor.get_eyes_points(face_points)
            points['mouth'] = self.extractor.get_mouth_points(face_points)
            points['head'] = self.extractor.get_head_points(face_points)
        
        # Dibujar malla facial
        if draw:
            sketch = self.drawer.draw_sketch(face_image, face_mesh_info)

        return points, points['face_detected'], sketch
