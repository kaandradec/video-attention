import cv2
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from drowsiness_processor.main import DrowsinessDetectionSystem
import os
import boto3
from botocore.client import Config
from fastapi.responses import StreamingResponse
from typing import List

# Configuración de MinIO/S3
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "reports")

s3_client = boto3.client(
    's3',
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    config=Config(signature_version='s3v4'),
    region_name='us-east-1'
)

# Crear el bucket si no existe
try:
    s3_client.head_bucket(Bucket=MINIO_BUCKET)
except Exception:
    s3_client.create_bucket(Bucket=MINIO_BUCKET)


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
    """Recibe un archivo CSV y lo guarda en MinIO/S3 en el bucket reports"""
    try:
        filename = file.filename
        contents = await file.read()
        s3_client.put_object(Bucket=MINIO_BUCKET, Key=filename, Body=contents, ContentType='text/csv')
        return {"status": "ok", "s3_path": f"s3://{MINIO_BUCKET}/{filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list_reports")
def list_reports() -> List[str]:
    """Lista los archivos CSV en el bucket de MinIO (reports)."""
    try:
        response = s3_client.list_objects_v2(Bucket=MINIO_BUCKET)
        files = [item['Key'] for item in response.get('Contents', []) if item['Key'].endswith('.csv')]
        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_report")
def get_report(filename: str):
    """Descarga un archivo CSV específico desde MinIO."""
    try:
        obj = s3_client.get_object(Bucket=MINIO_BUCKET, Key=filename)
        return StreamingResponse(obj['Body'], media_type='text/csv', headers={
            'Content-Disposition': f'attachment; filename={filename}'
        })
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

