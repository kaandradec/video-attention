import csv
import os
import json
from datetime import datetime
import boto3
from botocore.client import Config
from hdfs import InsecureClient
import logging

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
        # Crear directorio si no existe
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


class DrowsinessReports:
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.fields = ['timestamp', 'eye_rub_first_hand_report', 'eye_rub_first_hand_count',
                       'eye_rub_first_hand_durations', '|',
                       'eye_rub_second_hand_report', 'eye_rub_second_hand_count', 'eye_rub_second_hand_durations', '|',
                       'flicker_report', 'flicker_count', '|',
                       'micro_sleep_report', 'micro_sleep_count', 'micro_sleep_durations', '|',
                       'pitch_report', 'pitch_count', 'pitch_durations', '|',
                       'yawn_report', 'yawn_count', 'yawn_durations']

        if not os.path.exists(self.file_name):
            self.create_csv_file()

    def create_csv_file(self):
        with open(self.file_name, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fields)
            writer.writeheader()
        
        # Subir a MinIO
        with open(self.file_name, 'rb') as file:
            s3_client.put_object(Bucket=MINIO_BUCKET, Key=os.path.basename(self.file_name), Body=file, ContentType='text/csv')
        
        # Subir a HDFS si está habilitado
        if USE_HDFS and hdfs_client:
            try:
                hdfs_path = f"{HDFS_REPORTS_PATH}/{os.path.basename(self.file_name)}"
                with open(self.file_name, 'rb') as local_file:
                    hdfs_client.write(hdfs_path, local_file, overwrite=True)
                logging.info(f"Archivo guardado en HDFS: {hdfs_path}")
            except Exception as e:
                logging.error(f"Error guardando en HDFS: {e}")

    def main(self, report_data: dict):
        if (report_data['eye_rub_first_hand']['eye_rub_report'] or
                report_data['eye_rub_second_hand']['eye_rub_report'] or
                report_data['flicker_and_micro_sleep']['flicker_report'] or
                report_data['flicker_and_micro_sleep']['micro_sleep_report'] or
                report_data['pitch']['pitch_report'] or
                report_data['yawn']['yawn_report']):
            row = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'eye_rub_first_hand_report': report_data.get('eye_rub_first_hand', {}).get('eye_rub_report', False),
                'eye_rub_first_hand_count': report_data.get('eye_rub_first_hand', {}).get('eye_rub_count', 0),
                'eye_rub_first_hand_durations': report_data.get('eye_rub_first_hand', {}).get('eye_rub_durations', []),
                '|': '|',
                'eye_rub_second_hand_report': report_data.get('eye_rub_second_hand', {}).get('eye_rub_report', False),
                'eye_rub_second_hand_count': report_data.get('eye_rub_second_hand', {}).get('eye_rub_count', 0),
                'eye_rub_second_hand_durations': report_data.get('eye_rub_second_hand', {}).get('eye_rub_durations', []),
                '|': '|',
                'flicker_report': report_data.get('flicker_and_micro_sleep', {}).get('flicker_report', False),
                'flicker_count': report_data.get('flicker_and_micro_sleep', {}).get('flicker_count', 0),
                '|': '|',
                'micro_sleep_report': report_data.get('flicker_and_micro_sleep', {}).get('micro_sleep_report', False),
                'micro_sleep_count': report_data.get('flicker_and_micro_sleep', {}).get('micro_sleep_count', 0),
                'micro_sleep_durations': report_data.get('flicker_and_micro_sleep', {}).get('micro_sleep_durations', []),
                '|': '|',
                'pitch_report': report_data.get('pitch', {}).get('pitch_report', False),
                'pitch_count': report_data.get('pitch', {}).get('pitch_count', 0),
                'pitch_durations': report_data.get('pitch', {}).get('pitch_durations', []),
                '|': '|',
                'yawn_report': report_data.get('yawn', {}).get('yawn_report', False),
                'yawn_count': report_data.get('yawn', {}).get('yawn_count', 0),
                'yawn_durations': report_data.get('yawn', {}).get('yawn_durations', [])
            }
            with open(self.file_name, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=self.fields)
                writer.writerow(row)
            # Subir a MinIO
            with open(self.file_name, 'rb') as file:
                s3_client.put_object(Bucket=MINIO_BUCKET, Key=os.path.basename(self.file_name), Body=file, ContentType='text/csv')

    def generate_json_report(self, report_data: dict) -> str:
        report_json = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'eye_rub_first_hand': {
                'report': report_data.get('eye_rub_first_hand', {}).get('eye_rub_report', False),
                'count': report_data.get('eye_rub_first_hand', {}).get('eye_rub_count', 0),
                'durations': report_data.get('eye_rub_first_hand', {}).get('eye_rub_durations', [])
            },
            'eye_rub_second_hand': {
                'report': report_data.get('eye_rub_second_hand', {}).get('eye_rub_report', False),
                'count': report_data.get('eye_rub_second_hand', {}).get('eye_rub_count', 0),
                'durations': report_data.get('eye_rub_second_hand', {}).get('eye_rub_durations', [])
            },
            'flicker': {
                'report': report_data.get('flicker_and_micro_sleep', {}).get('flicker_report', False),
                'count': report_data.get('flicker_and_micro_sleep', {}).get('flicker_count', 0)
            },
            'micro_sleep': {
                'report': report_data.get('flicker_and_micro_sleep', {}).get('micro_sleep_report', False),
                'count': report_data.get('flicker_and_micro_sleep', {}).get('micro_sleep_count', 0),
                'durations': report_data.get('flicker_and_micro_sleep', {}).get('micro_sleep_durations', [])
            },
            'pitch': {
                'report': report_data.get('pitch', {}).get('pitch_report', False),
                'count': report_data.get('pitch', {}).get('pitch_count', 0),
                'durations': report_data.get('pitch', {}).get('pitch_durations', [])
            },
            'yawn': {
                'report': report_data.get('yawn', {}).get('yawn_report', False),
                'count': report_data.get('yawn', {}).get('yawn_count', 0),
                'durations': report_data.get('yawn', {}).get('yawn_durations', [])
            }
        }
        return json.dumps(report_json)
