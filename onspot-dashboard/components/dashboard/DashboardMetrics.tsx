'use client';

import React, { useEffect, useState, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { useApiRequest } from '@/hooks/useApiRequest';
import { AlertCircle } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { KpiCard } from '@/components/kpi-card';

interface MetricCardProps {
  title: string;
  value: string | number;
  description?: string;
  className?: string;
  loading?: boolean;
}

function MetricCard({ title, value, description, className, loading = false }: MetricCardProps) {
  return (
    <Card className={className}>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground" suppressHydrationWarning>
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {loading ? (
          <Skeleton className="h-8 w-24" aria-label={`Loading ${title} value`} />
        ) : (
          <div 
            className="text-2xl font-bold" 
            suppressHydrationWarning
            aria-label={`${title} value: ${value}`}
          >
            {value}
          </div>
        )}
        {description && (
          <p 
            className="text-xs text-muted-foreground mt-1" 
            suppressHydrationWarning
            aria-label={`${title} description: ${description}`}
          >
            {description}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

interface MetricsData {
  metrics: {
    name: string;
    value: number;
    change_percentage: number;
    is_improvement: boolean;
    timestamp: string;
  }[];
  timestamp: string;
}

export default function DashboardMetrics() {
  const [retryCount, setRetryCount] = useState(0);
  const [lastError, setLastError] = useState<Error | null>(null);
  
  const { 
    data: metricsData,
    loading: metricsLoading,
    error: metricsError,
    execute: fetchMetrics
  } = useApiRequest<MetricsData>();

  const fetchData = useCallback(async () => {
    try {
      console.log('[DashboardMetrics] Fetching metrics... (attempt', retryCount + 1, ')');
      const data = await fetchMetrics('/api/metrics/performance?time_range=7d');
      console.log('[DashboardMetrics] Received data:', data);
      
      if (!data?.metrics?.length) {
        throw new Error('No metrics data received');
      }
      
      setLastError(null); // Clear error on success
    } catch (error) {
      console.error('[DashboardMetrics] Error:', error);
      setLastError(error instanceof Error ? error : new Error('Unknown error'));
      
      // Retry up to 3 times with exponential backoff
      if (retryCount < 3) {
        const delay = Math.pow(2, retryCount) * 1000; // 1s, 2s, 4s
        console.log(`[DashboardMetrics] Retrying in ${delay}ms...`);
        setTimeout(() => {
          setRetryCount(prev => prev + 1);
        }, delay);
      }
    }
  }, [fetchMetrics, retryCount]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const getMetricValue = (name: string) => {
    if (!metricsData?.metrics) {
      console.log('[DashboardMetrics] No metrics data available for:', name);
      return {
        value: '0.000',
        change: '0.0',
        improved: false
      };
    }

    const metric = metricsData.metrics.find(m => m.name === name);
    if (!metric) {
      console.warn('[DashboardMetrics] Metric not found:', name, 'in data:', metricsData.metrics);
      return {
        value: '0.000',
        change: '0.0',
        improved: false
      };
    }

    return {
      value: metric.value.toFixed(3),
      change: metric.change_percentage.toFixed(1),
      improved: metric.is_improvement
    };
  };

  const rmse = getMetricValue('RMSE');
  const mae = getMetricValue('MAE');
  const r2 = getMetricValue('R2 Score');

  if (lastError && retryCount >= 3) {
    console.error('[DashboardMetrics] All retries failed:', lastError);
    return (
      <Alert variant="destructive" role="alert" aria-live="assertive">
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Error Loading Metrics</AlertTitle>
        <AlertDescription>
          Failed to load metrics data after multiple attempts. Please try again later.
          {lastError instanceof Error ? `: ${lastError.message}` : ''}
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      <KpiCard
        title="RMSE"
        value={rmse.value}
        description="Root Mean Square Error"
        change={parseFloat(rmse.change)}
        loading={metricsLoading}
        trend={rmse.improved ? 'down' : 'up'}
        trendDescription={rmse.improved ? 'Improved' : 'Degraded'}
      />
      <KpiCard
        title="MAE"
        value={mae.value}
        description="Mean Absolute Error"
        change={parseFloat(mae.change)}
        loading={metricsLoading}
        trend={mae.improved ? 'down' : 'up'}
        trendDescription={mae.improved ? 'Improved' : 'Degraded'}
      />
      <KpiCard
        title="R² Score"
        value={r2.value}
        description="Coefficient of Determination"
        change={parseFloat(r2.change)}
        loading={metricsLoading}
        trend={r2.improved ? 'up' : 'down'}
        trendDescription={r2.improved ? 'Improved' : 'Degraded'}
      />
    </div>
  );
} 