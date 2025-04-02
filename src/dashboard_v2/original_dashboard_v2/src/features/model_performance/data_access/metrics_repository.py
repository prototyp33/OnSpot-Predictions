"""
Repository class for handling database interactions with metrics data
"""

from typing import List, Dict, Optional
from datetime import datetime
import asyncpg
import logging
from pydantic import BaseModel

from src.core.config import settings

logger = logging.getLogger(__name__)

class MetricsRepository:
    def __init__(self, db_settings):
        """Initialize the metrics repository with database settings"""
        self.db_settings = db_settings
        self._pool = None

    async def _get_pool(self):
        """Get or create the database connection pool"""
        if self._pool is None:
            try:
                self._pool = await asyncpg.create_pool(
                    dsn=str(self.db_settings.url),
                    min_size=self.db_settings.pool_size,
                    max_size=self.db_settings.max_connections
                )
            except Exception as e:
                logger.error(f"Failed to create database pool: {str(e)}")
                raise

        return self._pool

    async def store_metric(self, model_id: str, metric_name: str, metric_result: Dict):
        """Store a single metric result in the database"""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            try:
                await conn.execute("""
                    INSERT INTO model_metrics (
                        model_id, metric_name, value, timestamp,
                        confidence_interval_lower, confidence_interval_upper,
                        metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, model_id, metric_name, metric_result['value'],
                    metric_result.get('timestamp', datetime.now()),
                    metric_result.get('confidence_interval', (None, None))[0],
                    metric_result.get('confidence_interval', (None, None))[1],
                    metric_result.get('metadata', {})
                )
            except Exception as e:
                logger.error(f"Failed to store metric: {str(e)}")
                raise

    async def get_metric_history(
        self,
        model_id: str,
        start_time: datetime,
        end_time: datetime,
        metric_names: Optional[List[str]] = None
    ) -> List[Dict]:
        """Retrieve metric history for a model within a time range"""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            try:
                query = """
                    SELECT model_id, metric_name, value, timestamp,
                           confidence_interval_lower, confidence_interval_upper,
                           metadata
                    FROM model_metrics
                    WHERE model_id = $1
                    AND timestamp BETWEEN $2 AND $3
                """
                params = [model_id, start_time, end_time]

                if metric_names:
                    query += " AND metric_name = ANY($4)"
                    params.append(metric_names)

                query += " ORDER BY timestamp DESC"
                
                rows = await conn.fetch(query, *params)
                
                return [
                    {
                        'model_id': row['model_id'],
                        'metric_name': row['metric_name'],
                        'value': row['value'],
                        'timestamp': row['timestamp'],
                        'confidence_interval': (
                            row['confidence_interval_lower'],
                            row['confidence_interval_upper']
                        ) if row['confidence_interval_lower'] is not None else None,
                        'metadata': row['metadata']
                    }
                    for row in rows
                ]
            except Exception as e:
                logger.error(f"Failed to fetch metric history: {str(e)}")
                raise

    async def get_latest_quality_metrics(self, model_id: str) -> Dict:
        """Retrieve the latest data quality metrics for a model"""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            try:
                row = await conn.fetchrow("""
                    SELECT model_id, timestamp, missing_rate, out_of_range_rate,
                           correlation_changes, distribution_metrics, sample_size
                    FROM data_quality_metrics
                    WHERE model_id = $1
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, model_id)

                if not row:
                    return None

                return {
                    'model_id': row['model_id'],
                    'timestamp': row['timestamp'],
                    'missing_rate': row['missing_rate'],
                    'out_of_range_rate': row['out_of_range_rate'],
                    'correlation_changes': row['correlation_changes'],
                    'distribution_metrics': row['distribution_metrics'],
                    'sample_size': row['sample_size']
                }
            except Exception as e:
                logger.error(f"Failed to fetch data quality metrics: {str(e)}")
                raise

    async def get_latest_health_metrics(self, model_id: str) -> Dict:
        """Retrieve the latest health metrics for a model"""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            try:
                row = await conn.fetchrow("""
                    SELECT model_id, timestamp, status, metrics, alerts
                    FROM health_metrics
                    WHERE model_id = $1
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, model_id)

                if not row:
                    return None

                return {
                    'model_id': row['model_id'],
                    'timestamp': row['timestamp'],
                    'status': row['status'],
                    'metrics': row['metrics'],
                    'alerts': row['alerts']
                }
            except Exception as e:
                logger.error(f"Failed to fetch health metrics: {str(e)}")
                raise

    async def get_monitored_models(self) -> List[str]:
        """Get a list of all monitored model IDs"""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            try:
                rows = await conn.fetch("""
                    SELECT DISTINCT model_id
                    FROM model_metrics
                    ORDER BY model_id
                """)
                return [row['model_id'] for row in rows]
            except Exception as e:
                logger.error(f"Failed to fetch monitored models: {str(e)}")
                raise

    async def cleanup_old_metrics(self, older_than: datetime):
        """Delete metrics older than the specified date"""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            try:
                async with conn.transaction():
                    await conn.execute("""
                        DELETE FROM model_metrics
                        WHERE timestamp < $1
                    """, older_than)
                    
                    await conn.execute("""
                        DELETE FROM data_quality_metrics
                        WHERE timestamp < $1
                    """, older_than)
                    
                    await conn.execute("""
                        DELETE FROM health_metrics
                        WHERE timestamp < $1
                    """, older_than)
            except Exception as e:
                logger.error(f"Failed to cleanup old metrics: {str(e)}")
                raise

    async def close(self):
        """Close the database connection pool"""
        if self._pool:
            await self._pool.close()
            self._pool = None 