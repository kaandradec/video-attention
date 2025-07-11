from flet import *
import asyncio
import websockets
import json
import cv2
import numpy as np
import threading
import base64
import requests
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
from flet import Image, Dropdown, dropdown
import base64
import csv
import os
from flet import DataTable, DataColumn, DataRow, DataCell, IconButton, icons

from gui.resources.resources_path import (ImagePaths, FontsPath)


class Monitor:
    def __init__(self, page):
        self.page = page

        self.running = False
        self.video_thread = None
        self.sketch_image_control = None
        self.original_image_control = None

        self.images = ImagePaths()
        self.fonts = FontsPath()

        self.attention_text = None
        self.last_attention = {"timestamp": 0, "in_attention": False}
        self.toggle_button = None
        self.download_file_picker = FilePicker(on_result=self.on_download_location_selected)
        self.file_to_download = None
        self.file_to_download_backend_name = None
        
        # Debug simple
        self.debug_text = None
        self.debug_timestamp = None
        self.debug_gaze = None
        self.debug_eyes = None
        self.debug_face = None
        self.debug_eyes_duration = None
        
        # Variables para cálculo de umbral de ojos cerrados en frontend
        self.eyes_closed_start = None
        self.eyes_closed_threshold = 1.5  # segundos

    def main(self):
        self.attention_text = Text("Atención: --", size=32, color="#FFFFFF", weight="bold")
        self.head_rotation_status_text = Text("", size=18, color="#FFFFFF")
        self.eyes_status_text = Text("", size=18, color="#FFFFFF")
        self.face_status_text = Text("", size=18, color="#FFFFFF")
        self.original_image_control = Image(
            fit=ImageFit.CONTAIN,
            src_base64=self.get_placeholder_image(),
            expand=True
        )
        self.sketch_image_control = Image(
            fit=ImageFit.CONTAIN,
            src_base64=self.get_placeholder_image(),
            expand=True
        )
        self.toggle_button = ElevatedButton(
            text="Iniciar",
            on_click=self.toggle_detection,
            bgcolor="#613bbb",
            color="#FFFFFF",
            width=180,
            height=50,
            style=ButtonStyle(),
        )
        
        # Debug simple
        self.debug_text = Text("Debug: No hay datos", size=14, color="#FFFF00")
        self.debug_timestamp = Text("Timestamp: --", size=14, color="#FFFF00")
        self.debug_gaze = Text("Rotación 3D: --", size=14, color="#FFFF00")
        self.debug_eyes = Text("Eyes: --", size=14, color="#FFFF00")
        self.debug_face = Text("Face: --", size=14, color="#FFFF00")
        self.debug_eyes_duration = Text("Eyes Duration: --", size=14, color="#FFFF00")
        
        # --- Mejora visual historial ---
        self.status_text = Text("", size=18, color="#FFFFFF")
        self.plot_image = None
        self.report_dropdown = None
        self.selected_report_title = Text("", size=20, color="#FFFFFF", weight="bold")
        self.report_label = Text("Selecciona un reporte para visualizar:", size=18, color="#FFFFFF", weight="bold")
        history_tab = Column([
            self.report_label,
            Row([
                # Eliminar botón morado vacío, solo mostrar dropdown cuando corresponda
                self.status_text
            ], alignment="center", spacing=20),
            # Dropdown y gráfico se agregan dinámicamente
            self.selected_report_title,
        ], alignment="start", horizontal_alignment="center", spacing=20, expand=True)
        self.history_tab = history_tab
        # --- Fin mejora visual ---
        self.tabs = Tabs([
            Tab(
                text="Monitoreo",
                content=Column([
                    Row([
                        self.original_image_control,
                        self.sketch_image_control
                    ], alignment='center', vertical_alignment='center', spacing=40, expand=True),
                    self.toggle_button,
                    self.attention_text,
                    self.head_rotation_status_text,
                    self.eyes_status_text,
                    self.face_status_text,
                    self.debug_text,
                    self.debug_timestamp,
                    self.debug_gaze,
                    self.debug_eyes,
                    self.debug_face,
                    self.debug_eyes_duration,
                ], alignment='center', horizontal_alignment='center', spacing=10, expand=True)
            ),
            Tab(
                text="Historial de Reportes",
                content=self.history_tab
            ),
            Tab(
                text="Todos los Reportes",
                content=Column([Text("Cargando...", size=18, color="#FFFFFF")], alignment="center", horizontal_alignment="center", expand=True)
            )
        ], expand=True, on_change=self.on_tab_change)
        elements = Container(
            content=Column([
                self.download_file_picker,  # FilePicker oculto para descargas
                self.tabs
            ], expand=True),
            bgcolor="#807da6",
            padding=30,
            expand=True
        )
        return elements

    def toggle_detection(self, e):
        if not self.running:
            self.start_detection(e)
            self.toggle_button.text = "Detener"
            self.toggle_button.bgcolor = "#e03851"
        else:
            self.stop_detection(e)
            self.toggle_button.text = "Iniciar"
            self.toggle_button.bgcolor = "#613bbb"

    def start_detection(self, e):
        if not self.running:
            self.running = True
            self.video_thread = threading.Thread(target=self.run_detection, daemon=True)
            self.video_thread.start()

    def stop_detection(self, e):
        self.running = False
        self.original_image_control.src_base64 = self.get_placeholder_image()
        self.sketch_image_control.src_base64 = self.get_placeholder_image()
        self.attention_text.value = "Atención: --"
        self.attention_text.color = "#FFFFFF"
        self.debug_text.value = "Debug: No hay datos"
        self.debug_timestamp.value = "Timestamp: --"
        self.debug_gaze.value = "Rotación 3D: --"
        self.debug_eyes.value = "Eyes: --"
        self.debug_face.value = "Face: --"
        self.debug_eyes_duration.value = "Eyes Duration: --"
        
        # Resetear variables de cálculo de ojos cerrados
        self.eyes_closed_start = None
        
        self.page.update()

    def run_detection(self):
        uri = "ws://localhost:8000/ws"
        # cap = cv2.VideoCapture(0)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 
        # camera_index = 0
        # cap = cv2.VideoCapture(camera_index)
        
        # if not cap.isOpened():
        #     # Si la cámara 0 no funciona, probar con otras
        #     for i in range(5):  # Probar índices 0-4
        #         cap = cv2.VideoCapture(i)
        #         if cap.isOpened():
        #             camera_index = i
        #             break
        #         cap.release()
        
        # if not cap.isOpened():
        #     print("Error: No se pudo abrir ninguna cámara")
        #     return
            
        # print(f"Usando cámara con índice: {camera_index}")
# 



        try:
            asyncio.run(self.process_video(uri, cap))
        finally:
            cap.release()

    def get_placeholder_image(self):
        drowsiness_image = cv2.imread(self.images.image_5)
        _, buffer = cv2.imencode('.jpg', drowsiness_image)
        blank_base64 = base64.b64encode(buffer).decode('utf-8')
        return blank_base64

    def cv2_to_base64(self, image):
        _, img_buffer = cv2.imencode(".jpg", image)
        return base64.b64encode(img_buffer).decode('utf-8')

    async def process_video(self, uri, cap):
        async with websockets.connect(uri) as websocket:
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # encode the frame
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')

                # send frame
                await websocket.send(frame_base64)

                # receive response
                response = await websocket.recv()
                response_data = json.loads(response)

                # sketch image
                sketch_base64 = response_data.get("sketch_image")
                sketch_data = base64.b64decode(sketch_base64)
                nparr_sketch = np.frombuffer(sketch_data, np.uint8)
                sketch_image = cv2.imdecode(nparr_sketch, cv2.IMREAD_COLOR)
                # Flip horizontal
                sketch_image = cv2.flip(sketch_image, 1)

                # image original
                original_base64 = response_data.get("original_image")
                original_data = base64.b64decode(original_base64)
                nparr_original = np.frombuffer(original_data, np.uint8)
                original_image = cv2.imdecode(nparr_original, cv2.IMREAD_COLOR)
                # Flip horizontal
                original_image = cv2.flip(original_image, 1)

                # atención
                attention = response_data.get("attention")
                if attention is not None:
                    self.last_attention = attention
                    # --- NUEVO: Mostrar detalles de atención ---
                    # Por compatibilidad, si el backend no envía detalles, los valores por defecto serán None
                    head_yaw = attention.get('head_yaw', None)  # Rotación horizontal 3D (grados)
                    head_pitch = attention.get('head_pitch', None)  # Rotación vertical 3D (grados)
                    eyes_closed = attention.get('eyes_closed', None)
                    face_detected = attention.get('face_detected', None)
                    # Parámetros de tolerancia para rotación (grados)
                    tolerance_rotation = 20.0  # grados
                    # Estado de rotación de la cabeza
                    gaze_msg = ""
                    if head_yaw is not None and head_pitch is not None:
                        if abs(head_yaw) > tolerance_rotation:
                            if head_yaw < 0:
                                gaze_msg = f"Cabeza rotada IZQUIERDA ({head_yaw:.1f}°)"
                            else:
                                gaze_msg = f"Cabeza rotada DERECHA ({head_yaw:.1f}°)"
                        elif abs(head_pitch) > tolerance_rotation:
                            if head_pitch < 0:
                                gaze_msg = f"Cabeza rotada ARRIBA ({head_pitch:.1f}°)"
                            else:
                                gaze_msg = f"Cabeza rotada ABAJO ({head_pitch:.1f}°)"
                        else:
                            gaze_msg = f"Cabeza centrada (Yaw: {head_yaw:.1f}°, Pitch: {head_pitch:.1f}°)"
                    else:
                        gaze_msg = "No se puede determinar la rotación de la cabeza"
                    self.head_rotation_status_text.value = gaze_msg
                    # Estado de ojos
                    if eyes_closed is not None:
                        if eyes_closed:
                            self.eyes_status_text.value = "Ojos cerrados (no atención)"
                        else:
                            self.eyes_status_text.value = "Ojos abiertos"
                    else:
                        self.eyes_status_text.value = "No se puede determinar el estado de los ojos"
                    # Estado de rostro
                    if face_detected is not None:
                        if not face_detected:
                            self.face_status_text.value = "No se detecta persona en cámara (no atención)"
                        else:
                            self.face_status_text.value = "Persona detectada"
                    else:
                        self.face_status_text.value = "No se puede determinar si hay persona"
                    
                    # Debug simple
                    self.debug_text.value = "Debug: JSON recibido"
                    
                    # Timestamp con 2 decimales
                    timestamp = attention.get('timestamp', 'N/A')
                    if isinstance(timestamp, (int, float)):
                        self.debug_timestamp.value = f"Timestamp: {timestamp:.2f}"
                    else:
                        self.debug_timestamp.value = f"Timestamp: {timestamp}"
                    
                    # Gaze con 2 decimales (ahora son rotaciones 3D en grados)
                    head_yaw = attention.get('head_yaw', 'N/A')
                    head_pitch = attention.get('head_pitch', 'N/A')
                    if isinstance(head_yaw, (int, float)) and isinstance(head_pitch, (int, float)):
                        self.debug_gaze.value = f"Rotación 3D: Yaw={head_yaw:.1f}°, Pitch={head_pitch:.1f}°"
                    else:
                        self.debug_gaze.value = f"Rotación 3D: ({head_yaw}, {head_pitch})"
                    
                    # Eyes
                    eyes_closed = attention.get('eyes_closed', 'N/A')
                    self.debug_eyes.value = f"Eyes: {eyes_closed}"
                    
                    # Cálculo de duración de ojos cerrados en frontend
                    import time
                    if eyes_closed == True:
                        if self.eyes_closed_start is None:
                            self.eyes_closed_start = time.time()
                        eyes_duration = time.time() - self.eyes_closed_start
                        eyes_closed_long = eyes_duration >= self.eyes_closed_threshold
                    else:
                        self.eyes_closed_start = None
                        eyes_duration = 0.0
                        eyes_closed_long = False
                    
                    self.debug_eyes_duration.value = f"Eyes Duration: {eyes_duration:.2f}s (Umbral: {eyes_closed_long})"
                    
                    # Face - verificar si realmente hay datos válidos
                    face_detected = attention.get('face_detected', 'N/A')
                    if face_detected == 'N/A' or face_detected is None:
                        self.debug_face.value = "Face: No detectado"
                    else:
                        self.debug_face.value = f"Face: {face_detected}"
                    
                    # --- CALCULAR ATENCIÓN GENERAL EN FRONTEND ---
                    # Obtener todas las variables del JSON
                    head_yaw = abs(attention.get('head_yaw', 0.0))  # Rotación horizontal
                    head_pitch = abs(attention.get('head_pitch', 0.0))  # Rotación vertical
                    head_roll = abs(attention.get('head_roll', 0.0))
                    
                    # Criterios para determinar atención
                    max_angle = 20.0  # grados (mismo que backend)
                    face_ok = face_detected == True
                    head_ok = head_yaw < max_angle and head_pitch < max_angle and head_roll < max_angle
                    eyes_ok = not eyes_closed_long  # Usar nuestro cálculo del umbral
                    
                    # Atención general
                    in_attention = face_ok and head_ok and eyes_ok
                    
                    # Actualizar texto de atención
                    if in_attention:
                        self.attention_text.value = f"Atención: SÍ ({attention.get('timestamp', 0):.1f}s)"
                        self.attention_text.color = "#00FF00"
                    else:
                        self.attention_text.value = f"Atención: NO ({attention.get('timestamp', 0):.1f}s)"
                        self.attention_text.color = "#FF0000"
                # update image in Flet
                self.original_image_control.src_base64 = self.cv2_to_base64(original_image)
                self.sketch_image_control.src_base64 = self.cv2_to_base64(sketch_image)
                # update UI
                self.page.update()
                await asyncio.sleep(0.01)

    def get_last_attention(self):
        return self.last_attention

    def show_reports_history(self, e):
        backend_url = os.environ.get("BACKEND_URL", "http://localhost:8000")
        try:
            response = requests.get(f"{backend_url}/list_reports")
            if response.status_code == 200:
                files = response.json()
                if not files:
                    self.status_text.value = "No hay reportes guardados."
                    self.page.update()
                    return
                options = [dropdown.Option(f) for f in files]
                self.report_dropdown = Dropdown(options=options, on_change=self.on_report_selected, width=400)
                # Eliminar dropdown anterior si existe
                for c in list(self.history_tab.controls):
                    if isinstance(c, Dropdown):
                        self.history_tab.controls.remove(c)
                self.history_tab.controls.insert(2, self.report_dropdown)
                self.status_text.value = ""
                self.selected_report_title.value = ""
                self.page.update()
            else:
                self.status_text.value = f"Error obteniendo historial: {response.text}"
                self.page.update()
        except Exception as ex:
            self.status_text.value = f"Error conectando al backend: {ex}"
            self.page.update()

    def on_report_selected(self, e):
        filename = e.control.value
        backend_url = os.environ.get("BACKEND_URL", "http://localhost:8000")
        try:
            response = requests.get(f"{backend_url}/get_report", params={"filename": filename})
            if response.status_code == 200:
                # Guardar temporalmente el archivo descargado
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
                    f.write(response.content)
                    temp_path = f.name
                self.show_attention_plot(temp_path)
                os.remove(temp_path)
                self.selected_report_title.value = f"Mostrando gráfico para: {filename}"
                self.status_text.value = ""
            else:
                self.status_text.value = f"Error descargando reporte: {response.text}"
                self.selected_report_title.value = ""
            self.page.update()
        except Exception as ex:
            self.status_text.value = f"Error conectando al backend: {ex}"
            self.selected_report_title.value = ""
            self.page.update()

    def show_attention_plot(self, csv_path):
        timestamps = []
        attention = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamps.append(float(row["timestamp_s"]))
                attention.append(int(row["in_attention"]))
        plt.figure(figsize=(8, 3))
        plt.step(timestamps, attention, where="post", label="Atención (1=Sí, 0=No)", color="#613bbb")
        plt.fill_between(timestamps, 0, attention, step="post", alpha=0.2, color="#613bbb")
        plt.xlabel("Tiempo (s)", fontsize=12)
        plt.ylabel("Atención", fontsize=12)
        plt.title("Atención a lo largo del video", fontsize=14, weight="bold")
        plt.ylim(-0.1, 1.1)
        plt.yticks([0, 1])
        plt.grid(True, axis="x", color="#cccccc")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        img_data = buf.read()
        img_base64 = base64.b64encode(img_data).decode("utf-8")
        img = Image(src_base64=img_base64, width=700, height=260)
        # Eliminar gráfico anterior si existe
        for c in list(self.history_tab.controls):
            if isinstance(c, Image):
                self.history_tab.controls.remove(c)
        self.history_tab.controls.append(img)
        self.page.update()

    def on_tab_change(self, e):
        # 1: Historial de Reportes, 2: Todos los Reportes
        if self.tabs.selected_index == 1:
            self.show_reports_history(None)
        elif self.tabs.selected_index == 2:
            self.show_reports_grid()

    def show_reports_grid(self):
        backend_url = os.environ.get("BACKEND_URL", "http://localhost:8000")
        try:
            response = requests.get(f"{backend_url}/list_reports")
            if response.status_code == 200:
                files = response.json()
                if not files:
                    grid = Column([Text("No hay reportes guardados.", size=18, color="#FFFFFF")], alignment="center", horizontal_alignment="center", expand=True)
                else:
                    rows = []
                    for f in files:
                        rows.append(DataRow(cells=[
                            DataCell(Text(f, color="#FFFFFF")),
                            DataCell(IconButton(icon=icons.DOWNLOAD, tooltip="Descargar", on_click=lambda e, filename=f: self.download_report(filename)))
                        ]))
                    grid = DataTable(
                        columns=[
                            DataColumn(Text("Nombre de Reporte", color="#FFFFFF")),
                            DataColumn(Text("Descargar", color="#FFFFFF")),
                        ],
                        rows=rows,
                        heading_row_color="#6a6791",
                        data_row_color={"hovered": "#a3a1c2"},
                        expand=True
                    )
                # Reemplazar contenido de la tab
                self.tabs.tabs[2].content.controls.clear()
                self.tabs.tabs[2].content.controls.append(grid)
                self.page.update()
            else:
                self.tabs.tabs[2].content.controls.clear()
                self.tabs.tabs[2].content.controls.append(Text(f"Error obteniendo reportes: {response.text}", size=18, color="#FFFFFF"))
                self.page.update()
        except Exception as ex:
            self.tabs.tabs[2].content.controls.clear()
            self.tabs.tabs[2].content.controls.append(Text(f"Error conectando al backend: {ex}", size=18, color="#FFFFFF"))
            self.page.update()

    def download_report(self, filename):
        # Guardar el nombre del archivo a descargar
        self.file_to_download = filename
        self.file_to_download_backend_name = filename
        # Abrir FilePicker para elegir ubicación
        self.download_file_picker.save_file(file_name=filename, allowed_extensions=["csv"])

    def on_download_location_selected(self, e):
        if not e.path or not self.file_to_download_backend_name:
            return
        backend_url = os.environ.get("BACKEND_URL", "http://localhost:8000")
        try:
            response = requests.get(f"{backend_url}/get_report", params={"filename": self.file_to_download_backend_name})
            if response.status_code == 200:
                with open(e.path, "wb") as f:
                    f.write(response.content)
                self.page.snack_bar = SnackBar(Text(f"Reporte guardado en: {e.path}", color="#fff"), bgcolor="#613bbb")
                self.page.snack_bar.open = True
                self.page.update()
            else:
                self.page.snack_bar = SnackBar(Text(f"Error descargando reporte: {response.text}", color="#fff"), bgcolor="#e03851")
                self.page.snack_bar.open = True
                self.page.update()
        except Exception as ex:
            self.page.snack_bar = SnackBar(Text(f"Error conectando al backend: {ex}", color="#fff"), bgcolor="#e03851")
            self.page.snack_bar.open = True
            self.page.update()
        finally:
            self.file_to_download = None
            self.file_to_download_backend_name = None
