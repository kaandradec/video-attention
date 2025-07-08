from flet import *
import threading
import asyncio
import websockets
import os
import datetime
import csv
from flet import Video, VideoMedia
import time
import requests
import tempfile

class VideoAttentionPage:
    def __init__(self, page):
        self.page = page
        self.video_path = None
        self.playing = False
        self.websocket_uri = "ws://localhost:8765"
        self.toggle_play_button = None  # Botón único para Play/Detener
        self.select_button = None
        self.status_text = None
        self.current_time_text = None
        self.attention_log = []
        self.file_picker = FilePicker(on_result=self.on_file_selected)
        self.timer = None
        self.video_duration = 0
        self.video_control = None
        self.start_time = None  # Para llevar el tiempo desde que inicia el video
        self.timer_running = False  # Para el contador de tiempo

    def main(self):
        self.status_text = Text("Selecciona un video para comenzar.")
        self.current_time_text = Text("00:00 / 00:00", size=16)
        self.select_button = ElevatedButton(
            text="Seleccionar Video",
            on_click=self.select_video,
            bgcolor="#3f64c1",
            color="#FFFFFF",
        )
        self.toggle_play_button = ElevatedButton(
            text="Play",
            on_click=self.toggle_play,
            bgcolor="#613bbb",
            color="#FFFFFF",
            disabled=True
        )
        self.file_picker = FilePicker(on_result=self.on_file_selected)
        self.video_control = Video(
            playlist=[],
            width=640,
            height=360,
            autoplay=False,
            show_controls=True,
            volume=100,
        )
        self.controls_column = Column([
            self.video_control,                # 1. Video primero
            self.current_time_text,            # 2. Contador de tiempo
            self.status_text,                  # 3. Texto de estado/log
            Row([
                self.select_button,
                self.toggle_play_button
            ], alignment="center", spacing=20), # 4. Botones en una fila
            self.file_picker,
        ], alignment="center", horizontal_alignment="center", spacing=20)
        return Container(content=self.controls_column, alignment=alignment.center, expand=True)

    def select_video(self, e):
        self.file_picker.pick_files(allow_multiple=False, file_type=FilePickerFileType.VIDEO)

    def on_file_selected(self, e):
        if e.files:
            self.video_path = e.files[0].path
            self.status_text.value = f"Video seleccionado: {os.path.basename(self.video_path)}"
            self.toggle_play_button.disabled = False
            self.toggle_play_button.text = "Play"
            # Ocultar botón seleccionar video
            if self.select_button in self.controls_column.controls:
                self.controls_column.controls.remove(self.select_button)
            # Crear un nuevo control Video con el video seleccionado
            new_video_control = Video(
                playlist=[VideoMedia(self.video_path)],
                width=640,
                height=360,
                autoplay=False,
                show_controls=True,
                volume=100,
            )
            parent = self.video_control.parent
            idx = parent.controls.index(self.video_control)
            parent.controls[idx] = new_video_control
            self.video_control = new_video_control
            self.page.update()

    def toggle_play(self, e):
        if not self.playing:
            self.play_video(e)
        else:
            self.stop_video(e)

    def play_video(self, e):
        if self.video_path:
            self.start_time = time.time()
            self.status_text.value = "Reproduciendo video..."
            self.toggle_play_button.text = "Detener"
            self.toggle_play_button.bgcolor = "#e03851"
            self.toggle_play_button.disabled = False
            self.attention_log = []
            self.page.update()
            self.video_control.play()
            self.page.update()
            threading.Thread(target=self.notify_monitor, daemon=True).start()
            self.playing = True
            self.timer_running = True
            threading.Thread(target=self.update_time_loop, daemon=True).start()

    def stop_video(self, e):
        self.status_text.value = "Video detenido. Guardando log de atención..."
        self.toggle_play_button.text = "Play"
        self.toggle_play_button.bgcolor = "#613bbb"
        self.toggle_play_button.disabled = False
        self.video_control.pause()
        self.page.update()
        self.playing = False
        self.timer_running = False
        self.save_attention_log()
        # Mostrar botón seleccionar video
        if self.select_button not in self.controls_column.controls:
            self.controls_column.controls.insert(2, self.select_button)
        self.page.update()

    def on_video_loaded(self, e):
        self.video_duration = self.video_control.duration
        self.update_time_text()

    def on_time_update(self, e):
        self.update_time_text()
        # Aquí podrías recibir la atención desde el monitor vía websocket o archivo compartido
        # Por ahora, solo simula que se recibe atención cada segundo
        if self.playing:
            # Aquí deberías obtener el valor real de atención del monitor
            # Simulación: atención aleatoria
            # from random import choice
            # in_attention = choice([True, False])
            # self.attention_log.append((self.video_control.current_time, in_attention))
            pass

    def on_video_ended(self, e):
        self.status_text.value = "Video finalizado. Guardando log de atención..."
        self.toggle_play_button.disabled = False
        self.toggle_play_button.text = "Play"
        self.toggle_play_button.bgcolor = "#613bbb"
        self.page.update()
        self.playing = False
        self.save_attention_log()

    def update_time_text(self):
        if self.start_time is not None and self.playing:
            current = self.format_time(time.time() - self.start_time)
        else:
            current = "00:00"
        total = self.format_time(self.video_duration)
        self.current_time_text.value = f"{current} / {total}"
        self.page.update()

    def format_time(self, seconds):
        if seconds is None:
            return "00:00"
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"

    def notify_monitor(self):
        # Notifica al proceso de monitoreo que debe iniciar el procesamiento
        try:
            asyncio.run(self._notify_ws())
        except Exception as ex:
            print(f"Error notificando al monitor: {ex}")

    async def _notify_ws(self):
        async with websockets.connect(self.websocket_uri) as ws:
            await ws.send("play")
            # Recibe atención en tiempo real
            while self.playing:
                try:
                    attention_data = await ws.recv()
                    # Espera que el monitor envíe un JSON con {"timestamp": ..., "in_attention": ...}
                    import json
                    attention = json.loads(attention_data)
                    # Usar contador propio de tiempo
                    if self.start_time is not None:
                        timestamp = time.time() - self.start_time
                    else:
                        timestamp = 0
                    self.attention_log.append((timestamp, attention.get("in_attention", False)))
                except Exception as ex:
                    print(f"Error recibiendo atención: {ex}")
                    break

    def save_attention_log(self):
        if not self.attention_log:
            self.status_text.value = "No se registró atención."
            self.page.update()
            return
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"attention_{video_name}_{now}.csv"
        # Guardar temporalmente el archivo CSV
        with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8", delete=False, suffix=".csv") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_s", "in_attention"])
            for t, att in self.attention_log:
                writer.writerow([f"{t:.2f}", int(att)])
            temp_path = f.name
        # Enviar al backend
        try:
            backend_url = os.environ.get("BACKEND_URL", "http://localhost:8000")
            with open(temp_path, "rb") as f:
                files = {"file": (filename, f, "text/csv")}
                response = requests.post(f"{backend_url}/save_attention_log", files=files)
            if response.status_code == 200:
                self.status_text.value = f"Log de atención guardado en HDFS: {filename}"
            else:
                self.status_text.value = f"Error guardando en HDFS: {response.text}"
        except Exception as e:
            self.status_text.value = f"Error conectando al backend: {e}"
        finally:
            os.remove(temp_path)
        self.page.update()

    def update_time_loop(self):
        while self.timer_running:
            self.update_time_text()
            time.sleep(0.5)

    # def start_ws_server(self):
    #     def on_play():
    #         if hasattr(self, 'drowsiness_page'):
    #             self.drowsiness_page.start_detection(None)
    #     print("Iniciando servidor websocket en ws://localhost:8765")
    #     asyncio.run(monitor_ws_server(on_play)) 