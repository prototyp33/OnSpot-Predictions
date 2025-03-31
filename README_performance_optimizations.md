# Supabase Performance Optimizations

This document outlines the performance optimizations made to the Supabase integration components.

## Overview of Optimizations

1. **Batch Inserts for Drift Analysis**
   - Replaced individual record inserts with batch inserts
   - Reduced number of API calls
   - Decreased latency and database load

2. **Health Metrics Caching**
   - Added time-based caching for health metrics queries
   - Configurable TTL (time-to-live) for cache entries
   - Reduced database load during frequent health checks

## Implementation Details

### Batch Inserts

The `store_drift_analysis` method previously inserted one record per feature, making `N` separate API calls for `N` features. The optimized implementation:

1. Collects all feature records into a batch array
2. Performs a single API call to insert all records
3. Properly handles success/failure for the entire batch

This change significantly reduces:
- Number of network round-trips
- Connection overhead
- Overall operation latency

#### Before vs After:

**Before**: For 10 features, 10 separate API calls
```
Feature 1 → API call
Feature 2 → API call
...
Feature 10 → API call
```

**After**: For 10 features, 1 API call
```
[Feature 1, Feature 2, ..., Feature 10] → Single API call
```

### Health Metrics Caching

The `get_latest_supabase_health` method is frequently called for monitoring and dashboard displays. We've implemented:

1. Time-based caching with configurable TTL (default: 60 seconds)
2. Cache invalidation mechanism for when immediate fresh data is needed
3. Seamless integration into existing code paths

This optimization reduces:
- Redundant database queries
- Network traffic
- CPU load from repetitive calculations

## Configuration Options

### Batch Insert Behavior
No additional configuration needed - batch inserts are now the default behavior for the `store_drift_analysis` method.

### Cache Configuration
Caching behavior can be configured through:

```python
# Get the extension
extension = integrate_supabase_monitoring()

# Set custom cache TTL (in seconds)
extension.set_cache_ttl(30)  # 30 seconds cache TTL

# Manually invalidate cache when needed
extension.invalidate_health_cache()
```

The cache TTL can also be configured via the `retraining_config.json` file:

```json
{
  "supabase_thresholds": {
    "cache_ttl_seconds": 120,
    "latency_threshold_ms": 500,
    ...
  }
}
```

## Impact Analysis

### Performance Improvement Estimates

**Batch Inserts**:
- API Calls: Reduced by factor of N (number of features)
- Latency: ~40-60% reduction for multiple features
- Database Load: Reduced transaction overhead

**Health Metrics Caching**:
- Query Reduction: Up to 98% with 60-second TTL in high-frequency scenarios
- Dashboard Responsiveness: Significantly improved for real-time dashboards
- Backend Load: Reduced CPU and database connection usage

### Monitoring Cache Performance

You can monitor cache performance by checking the logs at the DEBUG level. The following log messages are added:
- `"Using cached Supabase health metrics (age: {age}s)"` - When a cached result is used
- `"Cache miss/expired, fetching fresh Supabase health metrics"` - When fresh metrics are fetched
- `"Supabase health metrics cache invalidated"` - When cache is manually invalidated

## Implementation Notes

1. The batch insert implementation still maintains proper error handling and retries for transient errors.
2. The caching system is thread-safe and works correctly in concurrent environments.
3. Cache invalidation is automatic when TTL expires, or can be triggered manually.
4. These optimizations are compatible with existing monitoring and dashboard code. 