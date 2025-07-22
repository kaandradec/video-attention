"""
Módulo para análisis de big data usando Hadoop HDFS
Procesa grandes volúmenes de datos de atención almacenados en HDFS
"""
import os
import pandas as pd
from hdfs import InsecureClient
from typing import List, Dict, Any
import logging
from datetime import datetime, timedelta
import json

class HDFSAnalytics:
    def __init__(self):
        self.hdfs_url = os.environ.get("HDFS_URL", "http://namenode:9870")
        self.hdfs_user = os.environ.get("HDFS_USER", "root")
        self.hdfs_reports_path = os.environ.get("HDFS_REPORTS_PATH", "/video-attention/reports")
        
        try:
            self.client = InsecureClient(self.hdfs_url, user=self.hdfs_user)
        except Exception as e:
            logging.error(f"Error conectando a HDFS: {e}")
            self.client = None

    def list_attention_reports(self, date_filter: str = None) -> List[str]:
        """Lista archivos de atención en HDFS, opcionalmente filtrados por fecha"""
        if not self.client:
            return []
        
        try:
            files = self.client.list(self.hdfs_reports_path)
            attention_files = [f for f in files if f.startswith('attention_') and f.endswith('.csv')]
            
            if date_filter:
                # Filtrar por fecha (formato: YYYYMMDD)
                attention_files = [f for f in attention_files if date_filter in f]
            
            return attention_files
        except Exception as e:
            logging.error(f"Error listando archivos: {e}")
            return []

    def analyze_attention_patterns(self, days_back: int = 7) -> Dict[str, Any]:
        """Analiza patrones de atención de los últimos N días"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        all_data = []
        date_range = [(start_date + timedelta(days=i)).strftime("%Y%m%d") 
                     for i in range(days_back + 1)]
        
        for date_str in date_range:
            files = self.list_attention_reports(date_filter=date_str)
            for file in files:
                try:
                    data = self._read_csv_from_hdfs(f"{self.hdfs_reports_path}/{file}")
                    if data is not None:
                        data['date'] = date_str
                        data['file'] = file
                        all_data.append(data)
                except Exception as e:
                    logging.error(f"Error procesando {file}: {e}")
        
        if not all_data:
            return {"error": "No se encontraron datos para analizar"}
        
        # Combinar todos los DataFrames
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Análisis de patrones
        analysis = {
            "total_sessions": len(all_data),
            "total_records": len(combined_df),
            "attention_rate": {
                "overall": float(combined_df['in_attention'].mean()),
                "by_date": combined_df.groupby('date')['in_attention'].mean().to_dict()
            },
            "session_durations": {
                "average": float(combined_df.groupby('file')['timestamp_s'].max().mean()),
                "total": float(combined_df.groupby('file')['timestamp_s'].max().sum())
            },
            "attention_loss_events": self._analyze_attention_loss(combined_df),
            "peak_hours": self._analyze_peak_hours(combined_df),
            "date_range": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        }
        
        return analysis

    def generate_user_report(self, user_id: str = None) -> Dict[str, Any]:
        """Genera reporte detallado para un usuario específico"""
        files = self.list_attention_reports()
        user_files = files
        
        if user_id:
            # Filtrar archivos por usuario si se especifica
            user_files = [f for f in files if user_id in f]
        
        user_data = []
        for file in user_files:
            try:
                data = self._read_csv_from_hdfs(f"{self.hdfs_reports_path}/{file}")
                if data is not None:
                    user_data.append(data)
            except Exception as e:
                logging.error(f"Error procesando {file}: {e}")
        
        if not user_data:
            return {"error": f"No se encontraron datos para el usuario {user_id}"}
        
        combined_df = pd.concat(user_data, ignore_index=True)
        
        report = {
            "user_id": user_id or "all_users",
            "total_sessions": len(user_data),
            "total_time_minutes": float(combined_df['timestamp_s'].max()),
            "attention_metrics": {
                "overall_rate": float(combined_df['in_attention'].mean()),
                "best_session": float(combined_df.groupby('file')['in_attention'].mean().max()),
                "worst_session": float(combined_df.groupby('file')['in_attention'].mean().min())
            },
            "recommendations": self._generate_recommendations(combined_df)
        }
        
        return report

    def _read_csv_from_hdfs(self, hdfs_path: str) -> pd.DataFrame:
        """Lee un archivo CSV desde HDFS y retorna DataFrame"""
        try:
            with self.client.read(hdfs_path) as reader:
                content = reader.read().decode('utf-8')
                from io import StringIO
                df = pd.read_csv(StringIO(content))
                return df
        except Exception as e:
            logging.error(f"Error leyendo {hdfs_path}: {e}")
            return None

    def _analyze_attention_loss(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analiza eventos de pérdida de atención"""
        # Encontrar secuencias de no-atención
        df['attention_change'] = df['in_attention'].diff()
        loss_events = df[df['attention_change'] == -1]
        
        return {
            "total_events": len(loss_events),
            "average_duration": float(df[df['in_attention'] == 0].groupby('file').size().mean()) if len(loss_events) > 0 else 0,
            "most_common_times": loss_events['timestamp_s'].quantile([0.25, 0.5, 0.75]).to_dict()
        }

    def _analyze_peak_hours(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analiza horas pico de atención (simulado con timestamp)"""
        # Simular horas basado en timestamp dentro de sesión
        df['hour_simulation'] = (df['timestamp_s'] % 86400) // 3600  # Simular hora del día
        hourly_attention = df.groupby('hour_simulation')['in_attention'].mean()
        
        return {
            "best_hour": int(hourly_attention.idxmax()),
            "worst_hour": int(hourly_attention.idxmin()),
            "hourly_rates": hourly_attention.to_dict()
        }

    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Genera recomendaciones basadas en los datos"""
        recommendations = []
        attention_rate = df['in_attention'].mean()
        
        if attention_rate < 0.6:
            recommendations.append("Considera tomar descansos más frecuentes")
            recommendations.append("Evalúa la ergonomía de tu espacio de trabajo")
        elif attention_rate < 0.8:
            recommendations.append("Buen nivel de atención, mantén el ritmo")
            recommendations.append("Identifica los momentos de mayor distracción")
        else:
            recommendations.append("Excelente nivel de atención sostenida")
            recommendations.append("Considera compartir tus técnicas con otros")
        
        return recommendations

    def export_analysis_to_hdfs(self, analysis: Dict[str, Any], filename: str):
        """Exporta análisis a HDFS como JSON"""
        try:
            hdfs_path = f"{self.hdfs_reports_path}/analytics/{filename}"
            
            # Crear directorio si no existe
            analytics_dir = f"{self.hdfs_reports_path}/analytics"
            if not self.client.status(analytics_dir, strict=False):
                self.client.makedirs(analytics_dir)
            
            # Escribir análisis
            with self.client.write(hdfs_path, overwrite=True) as writer:
                writer.write(json.dumps(analysis, indent=2).encode('utf-8'))
            
            logging.info(f"Análisis exportado a HDFS: {hdfs_path}")
            return hdfs_path
        except Exception as e:
            logging.error(f"Error exportando análisis: {e}")
            return None
