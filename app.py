import cv2
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from drowsiness_processor.main import DrowsinessDetectionSystem
from hdfs import InsecureClient
import os

# Configuración de HDFS (puedes ajustar la URL y usuario según tu entorno)
HDFS_URL = os.environ.get("HDFS_URL", "http://namenode:9870")
HDFS_USER = os.environ.get("HDFS_USER", "hadoop")
hdfs_client = InsecureClient(HDFS_URL, user=HDFS_USER)


app = FastAPI()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    drowsiness_detection_system = DrowsinessDetectionSystem()

    await websocket.accept()
    try:
        while True:
            # read data
            data = await websocket.receive_text()

            # decode data
            original_image, sketch, json_report, attention = drowsiness_detection_system.run(data)

            _, buffer_sketch = cv2.imencode('.jpg', sketch)
            sketch_base64 = base64.b64encode(buffer_sketch).decode('utf-8')

            _, buffer_original_image = cv2.imencode('.jpg', original_image)
            original_image_base64 = base64.b64encode(buffer_original_image).decode('utf-8')

            # send answer
            await websocket.send_json({
                "json_report": json_report,
                "sketch_image": sketch_base64,
                "original_image": original_image_base64,
                "attention": attention,
            })

    except WebSocketDisconnect:
        print("disconnect client")


@app.post("/save_attention_log")
async def save_attention_log(file: UploadFile = File(...)):
    """Recibe un archivo CSV y lo guarda en HDFS en /reports/"""
    try:
        # Nombre destino en HDFS
        filename = file.filename
        hdfs_path = f"/reports/{filename}"
        # Leer el archivo recibido
        contents = await file.read()
        # Guardar en HDFS
        with hdfs_client.write(hdfs_path, overwrite=True) as writer:
            writer.write(contents)
        return {"status": "ok", "hdfs_path": hdfs_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

