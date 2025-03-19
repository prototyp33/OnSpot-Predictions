"""
Tests for the MetricsRepository class
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from src.features.model_performance.data_access.metrics_repository import MetricsRepository
from src.core.config import DatabaseSettings

# Test database settings
TEST_DB_SETTINGS = DatabaseSettings(
    url="postgresql://test_user:test_pass@localhost:5432/test_db",
    pool_size=5,
    max_connections=10
)

@pytest.fixture
async def repository():
    """Create a test repository instance"""
    repo = MetricsRepository(TEST_DB_SETTINGS)
    yield repo
    await repo.close()

@pytest.fixture
async def sample_metrics():
    """Create sample metric data"""
    now = datetime.now()
    return [
        {
            'model_id': 'test_model',
            'metric_name': 'rmse',
            'value': 0.15,
            'timestamp': now,
            'confidence_interval': (0.12, 0.18),
            'metadata': {'sample_size': 1000}
        },
        {
            'model_id': 'test_model',
            'metric_name': 'mae',
            'value': 0.10,
            'timestamp': now - timedelta(hours=1),
            'confidence_interval': (0.08, 0.12),
            'metadata': {'sample_size': 1000}
        }
    ]

@pytest.mark.asyncio
async def test_store_and_retrieve_metrics(repository, sample_metrics):
    """Test storing and retrieving metrics"""
    # Store sample metrics
    for metric in sample_metrics:
        await repository.store_metric(
            metric['model_id'],
            metric['metric_name'],
            metric
        )

    # Retrieve metrics history
    start_time = datetime.now() - timedelta(days=1)
    end_time = datetime.now() + timedelta(days=1)
    
    metrics = await repository.get_metric_history(
        model_id='test_model',
        start_time=start_time,
        end_time=end_time
    )

    assert len(metrics) == 2
    assert metrics[0]['model_id'] == 'test_model'
    assert metrics[0]['metric_name'] in ['rmse', 'mae']
    assert isinstance(metrics[0]['value'], float)
    assert isinstance(metrics[0]['timestamp'], datetime)
    assert len(metrics[0]['confidence_interval']) == 2
    assert isinstance(metrics[0]['metadata'], dict)

@pytest.mark.asyncio
async def test_get_latest_quality_metrics(repository):
    """Test retrieving latest data quality metrics"""
    model_id = 'test_model'
    quality_metrics = await repository.get_latest_quality_metrics(model_id)
    
    if quality_metrics:
        assert quality_metrics['model_id'] == model_id
        assert isinstance(quality_metrics['missing_rate'], float)
        assert isinstance(quality_metrics['out_of_range_rate'], float)
        assert isinstance(quality_metrics['correlation_changes'], dict)
        assert isinstance(quality_metrics['distribution_metrics'], dict)
        assert isinstance(quality_metrics['sample_size'], int)

@pytest.mark.asyncio
async def test_get_latest_health_metrics(repository):
    """Test retrieving latest health metrics"""
    model_id = 'test_model'
    health_metrics = await repository.get_latest_health_metrics(model_id)
    
    if health_metrics:
        assert health_metrics['model_id'] == model_id
        assert isinstance(health_metrics['status'], str)
        assert isinstance(health_metrics['metrics'], dict)
        assert isinstance(health_metrics['alerts'], list)

@pytest.mark.asyncio
async def test_get_monitored_models(repository, sample_metrics):
    """Test retrieving list of monitored models"""
    # Store sample metrics first
    for metric in sample_metrics:
        await repository.store_metric(
            metric['model_id'],
            metric['metric_name'],
            metric
        )

    models = await repository.get_monitored_models()
    assert 'test_model' in models
    assert isinstance(models, list)
    assert all(isinstance(model_id, str) for model_id in models)

@pytest.mark.asyncio
async def test_cleanup_old_metrics(repository, sample_metrics):
    """Test cleaning up old metrics"""
    # Store sample metrics
    for metric in sample_metrics:
        await repository.store_metric(
            metric['model_id'],
            metric['metric_name'],
            metric
        )

    # Clean up metrics older than current time
    cleanup_time = datetime.now() + timedelta(hours=1)
    await repository.cleanup_old_metrics(cleanup_time)

    # Verify metrics were cleaned up
    start_time = datetime.now() - timedelta(days=1)
    end_time = datetime.now() + timedelta(days=1)
    
    metrics = await repository.get_metric_history(
        model_id='test_model',
        start_time=start_time,
        end_time=end_time
    )

    assert len(metrics) == 0

@pytest.mark.asyncio
async def test_connection_error_handling(repository):
    """Test handling of database connection errors"""
    # Set invalid database settings
    repository.db_settings.url = "postgresql://invalid:invalid@localhost:5432/invalid"
    
    with pytest.raises(Exception):
        await repository.get_monitored_models()

@pytest.mark.asyncio
async def test_concurrent_operations(repository, sample_metrics):
    """Test concurrent operations on the repository"""
    async def store_metric(metric):
        await repository.store_metric(
            metric['model_id'],
            metric['metric_name'],
            metric
        )

    # Create multiple concurrent store operations
    tasks = [
        store_metric(metric)
        for metric in sample_metrics * 5  # Create 10 concurrent operations
    ]

    # Run operations concurrently
    await asyncio.gather(*tasks)

    # Verify all metrics were stored
    start_time = datetime.now() - timedelta(days=1)
    end_time = datetime.now() + timedelta(days=1)
    
    metrics = await repository.get_metric_history(
        model_id='test_model',
        start_time=start_time,
        end_time=end_time
    )

    assert len(metrics) == 10  # Should have 10 metrics (5 * 2 original metrics) 