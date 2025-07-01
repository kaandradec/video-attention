from flet import *
import asyncio
import websockets
import json
import cv2
import numpy as np
import threading
import base64

from gui.resources.resources_path import (ImagePaths, FontsPath)


class Drowsiness:
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

    def main(self):
        self.attention_text = Text("Atención: --", size=32, color="#FFFFFF", weight="bold")
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
            style=ButtonStyle(shape=RoundedRectangleBorder(radius=10)),
        )
        content = Column([
            Row([
                self.original_image_control,
                self.sketch_image_control
            ], alignment='center', vertical_alignment='center', spacing=40, expand=True),
            self.toggle_button,
            self.attention_text
        ], alignment='center', horizontal_alignment='center', spacing=30, expand=True)
        elements = Container(
            content=content,
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

                # image original
                original_base64 = response_data.get("original_image")
                original_data = base64.b64decode(original_base64)
                nparr_original = np.frombuffer(original_data, np.uint8)
                original_image = cv2.imdecode(nparr_original, cv2.IMREAD_COLOR)

                # atención
                attention = response_data.get("attention")
                if attention is not None:
                    self.last_attention = attention
                    if attention.get("in_attention"):
                        self.attention_text.value = f"Atención: SÍ ({attention.get('timestamp'):.1f}s)"
                        self.attention_text.color = "#00FF00"
                    else:
                        self.attention_text.value = f"Atención: NO ({attention.get('timestamp'):.1f}s)"
                        self.attention_text.color = "#FF0000"

                # update image in Flet
                self.original_image_control.src_base64 = self.cv2_to_base64(original_image)
                self.sketch_image_control.src_base64 = self.cv2_to_base64(sketch_image)

                # update UI
                self.page.update()
                await asyncio.sleep(0.01)

    def get_last_attention(self):
        return self.last_attention
