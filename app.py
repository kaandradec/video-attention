import cv2
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from drowsiness_processor.main import DrowsinessDetectionSystem
import os
import boto3
from botocore.client import Config
from fastapi.responses import StreamingResponse
from typing import List
from hdfs import InsecureClient
import logging
from drowsiness_processor.analytics.hdfs_analytics import HDFSAnalytics

# Configuración de MinIO/S3
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "reports")

# Configuración de Hadoop HDFS
HDFS_URL = os.environ.get("HDFS_URL", "http://namenode:9870")
HDFS_USER = os.environ.get("HDFS_USER", "root")
HDFS_REPORTS_PATH = os.environ.get("HDFS_REPORTS_PATH", "/video-attention/reports")
USE_HDFS = os.environ.get("USE_HDFS", "false").lower() == "true"

# Cliente HDFS
hdfs_client = None
if USE_HDFS:
    try:
        hdfs_client = InsecureClient(HDFS_URL, user=HDFS_USER)
        if not hdfs_client.status(HDFS_REPORTS_PATH, strict=False):
            hdfs_client.makedirs(HDFS_REPORTS_PATH)
    except Exception as e:
        logging.error(f"Error conectando a HDFS: {e}")
        hdfs_client = None

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
    """Recibe un archivo CSV y lo guarda en MinIO/S3 y HDFS"""
    try:
        filename = file.filename
        contents = await file.read()
        
        # Guardar en MinIO
        s3_client.put_object(Bucket=MINIO_BUCKET, Key=filename, Body=contents, ContentType='text/csv')
        result = {"status": "ok", "s3_path": f"s3://{MINIO_BUCKET}/{filename}"}
        
        # Guardar en HDFS si está habilitado
        if USE_HDFS and hdfs_client:
            try:
                hdfs_path = f"{HDFS_REPORTS_PATH}/{filename}"
                with hdfs_client.write(hdfs_path, overwrite=True) as writer:
                    writer.write(contents)
                result["hdfs_path"] = f"hdfs://{HDFS_REPORTS_PATH}/{filename}"
                logging.info(f"Archivo guardado en HDFS: {hdfs_path}")
            except Exception as e:
                logging.error(f"Error guardando en HDFS: {e}")
                result["hdfs_error"] = str(e)
        
        return result
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


@app.get("/list_hdfs_reports")
def list_hdfs_reports() -> List[str]:
    """Lista los archivos CSV en HDFS."""
    if not USE_HDFS or not hdfs_client:
        raise HTTPException(status_code=503, detail="HDFS no está habilitado o disponible")
    
    try:
        files = hdfs_client.list(HDFS_REPORTS_PATH)
        csv_files = [f for f in files if f.endswith('.csv')]
        return csv_files
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_hdfs_report")
def get_hdfs_report(filename: str):
    """Descarga un archivo CSV específico desde HDFS."""
    if not USE_HDFS or not hdfs_client:
        raise HTTPException(status_code=503, detail="HDFS no está habilitado o disponible")
    
    try:
        hdfs_path = f"{HDFS_REPORTS_PATH}/{filename}"
        with hdfs_client.read(hdfs_path) as reader:
            content = reader.read()
        
        from io import BytesIO
        return StreamingResponse(BytesIO(content), media_type='text/csv', headers={
            'Content-Disposition': f'attachment; filename={filename}'
        })
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/storage_stats")
def get_storage_stats():
    """Obtiene estadísticas de almacenamiento de MinIO y HDFS."""
    stats = {
        "minio": {
            "available": True,
            "bucket": MINIO_BUCKET,
            "endpoint": MINIO_ENDPOINT
        },
        "hdfs": {
            "available": USE_HDFS and hdfs_client is not None,
            "path": HDFS_REPORTS_PATH if USE_HDFS else None,
            "url": HDFS_URL if USE_HDFS else None
        }
    }
    
    # Contar archivos en MinIO
    try:
        response = s3_client.list_objects_v2(Bucket=MINIO_BUCKET)
        stats["minio"]["file_count"] = len(response.get('Contents', []))
    except Exception:
        stats["minio"]["file_count"] = 0
    
    # Contar archivos en HDFS
    if USE_HDFS and hdfs_client:
        try:
            files = hdfs_client.list(HDFS_REPORTS_PATH)
            stats["hdfs"]["file_count"] = len([f for f in files if f.endswith('.csv')])
        except Exception:
            stats["hdfs"]["file_count"] = 0
    
    return stats


# Inicializar analizador HDFS
hdfs_analytics = HDFSAnalytics() if USE_HDFS else None

@app.get("/analytics/attention_patterns")
def get_attention_patterns(days_back: int = 7):
    """Obtiene análisis de patrones de atención de los últimos N días"""
    if not hdfs_analytics:
        raise HTTPException(status_code=503, detail="Analytics no disponible - HDFS no configurado")
    
    try:
        analysis = hdfs_analytics.analyze_attention_patterns(days_back)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/user_report")
def get_user_report(user_id: str = None):
    """Genera reporte detallado para un usuario específico"""
    if not hdfs_analytics:
        raise HTTPException(status_code=503, detail="Analytics no disponible - HDFS no configurado")
    
    try:
        report = hdfs_analytics.generate_user_report(user_id)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analytics/export")
def export_analysis(filename: str, days_back: int = 7):
    """Exporta análisis a HDFS como archivo JSON"""
    if not hdfs_analytics:
        raise HTTPException(status_code=503, detail="Analytics no disponible - HDFS no configurado")
    
    try:
        analysis = hdfs_analytics.analyze_attention_patterns(days_back)
        hdfs_path = hdfs_analytics.export_analysis_to_hdfs(analysis, filename)
        return {"status": "exported", "hdfs_path": hdfs_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

