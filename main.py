import sys
from flet import *
from gui.pages.start_page import Start
from gui.pages.selection_interface_page import SelectionInterface
from gui.pages.monitor_page import Monitor
from gui.pages.video_attention_page import VideoAttentionPage
import threading
import asyncio
import websockets
import json
import time

# Servidor websocket para sincronización
def start_ws_server(on_play_callback, get_attention_callback=None):
    async def handler(websocket):
        print("Cliente conectado al websocket")
        async for message in websocket:
            print(f"Mensaje recibido: {message}")
            if message == "play":
                on_play_callback()
                # Enviar datos de atención en tiempo real mientras el monitoreo esté activo
                while True:
                    if get_attention_callback is not None:
                        attention = get_attention_callback()
                        await websocket.send(json.dumps(attention))
                    await asyncio.sleep(1)  # Enviar cada segundo
    async def run_server():
        print("Servidor websocket escuchando en ws://localhost:8765")
        async with websockets.serve(handler, "localhost", 8765):
            await asyncio.Future()  # run forever
    asyncio.run(run_server())

class MainApp:
    def __init__(self, page: Page, mode: str):
        self.page = page
        self.mode = mode
        self.page.title = "Sistema de Detección de Atención"
        self.page.bgcolor = "#fffffe"
        self.page.window.resizable = False
        self.page.padding = 0
        self.page.window.width = 1280
        self.page.window.height = 720
        self.page.vertical_alignment = "center"
        self.page.horizontal_alignment = "center"

        self.page.theme = Theme(
            page_transitions=PageTransitionsTheme(
                android=PageTransitionTheme.FADE_UPWARDS,
                ios=PageTransitionTheme.CUPERTINO,
                macos=PageTransitionTheme.ZOOM,
                linux=PageTransitionTheme.ZOOM,
                windows=PageTransitionTheme.FADE_UPWARDS,
            ),
        )

        if self.mode == "monitor":
            self.drowsiness_page = Monitor(page)
            self.page.views.append(View(route="/monitor", controls=[self.drowsiness_page.main()]))
            self.page.update()
            # Iniciar servidor websocket para sincronización en un hilo aparte
            threading.Thread(target=self.launch_ws_server, daemon=True).start()
        elif self.mode == "video":
            self.video_page = VideoAttentionPage(page)
            self.page.views.append(View(route="/video", controls=[self.video_page.main()]))
            self.page.update()
        else:
            self.start_page = Start(page)
            self.page.views.append(View(route="/", controls=[self.start_page.main()]))
            self.page.update()

    def launch_ws_server(self):
        def on_play():
            print("Recibido 'play' desde el video. Iniciando procesamiento en monitor.")
            if hasattr(self, 'drowsiness_page'):
                self.drowsiness_page.start_detection(None)
        def get_attention():
            # Obtener el último valor de atención del monitor
            if hasattr(self, 'drowsiness_page') and hasattr(self.drowsiness_page, 'get_last_attention'):
                return self.drowsiness_page.get_last_attention()
            return {"timestamp": 0, "in_attention": False}
        start_ws_server(on_play, get_attention)

def main(page: Page):
    mode = "monitor"
    if len(sys.argv) > 1:
        if sys.argv[1] == "--mode=video":
            mode = "video"
        elif sys.argv[1] == "--mode=monitor":
            mode = "monitor"
    MainApp(page, mode)

if __name__ == "__main__":
    app(target=main)
