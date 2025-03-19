'use client';

import { useState, useEffect } from 'react';

interface SystemHealthData {
  cpu: {
    usage: number;
    trend: number;
  };
  memory: {
    usage: number;
    trend: number;
  };
  disk: {
    usage: number;
    trend: number;
  };
  api: {
    responseTime: number;
    trend: number;
    availability: number;
  };
  database: {
    connections: number;
    queryTime: number;
    trend: number;
  };
  services: {
    name: string;
    status: 'healthy' | 'degraded' | 'down';
    responseTime: number;
  }[];
  logs: {
    level: 'info' | 'warning' | 'error' | 'critical';
    message: string;
    timestamp: string;
    service: string;
  }[];
}

interface UseSystemHealthOptions {
  timeRange?: string;
  refreshInterval?: number;
}

export function useSystemHealth(options: UseSystemHealthOptions = {}) {
  const { timeRange = '1d', refreshInterval = 60000 } = options;
  
  const [data, setData] = useState<SystemHealthData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [errorCode, setErrorCode] = useState<number | null>(null);
  
  const fetchData = async () => {
    setLoading(true);
    
    try {
      // In a real app, this would be an API call
      // For now, we'll just simulate a delay and return mock data
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Mock data
      setData({
        cpu: {
          usage: 45,
          trend: -2,
        },
        memory: {
          usage: 62,
          trend: 5,
        },
        disk: {
          usage: 78,
          trend: 1,
        },
        api: {
          responseTime: 120,
          trend: -15,
          availability: 99.95,
        },
        database: {
          connections: 24,
          queryTime: 45,
          trend: 3,
        },
        services: [
          { name: 'API Gateway', status: 'healthy', responseTime: 45 },
          { name: 'Authentication', status: 'healthy', responseTime: 32 },
          { name: 'Prediction Engine', status: 'healthy', responseTime: 150 },
          { name: 'Data Processing', status: 'degraded', responseTime: 320 },
          { name: 'Notification', status: 'healthy', responseTime: 28 },
        ],
        logs: [
          { level: 'error', message: 'Failed to process batch job', timestamp: '2023-06-15T10:23:45Z', service: 'Data Processing' },
          { level: 'warning', message: 'High memory usage detected', timestamp: '2023-06-15T09:45:12Z', service: 'Prediction Engine' },
          { level: 'info', message: 'System backup completed', timestamp: '2023-06-15T08:30:00Z', service: 'Database' },
          { level: 'warning', message: 'Slow query detected', timestamp: '2023-06-15T07:15:22Z', service: 'Database' },
          { level: 'info', message: 'User authentication spike', timestamp: '2023-06-15T06:45:10Z', service: 'Authentication' },
        ],
      });
      
      setError(null);
      setErrorCode(null);
    } catch (err) {
      console.error('Error fetching system health data:', err);
      setError(err instanceof Error ? err : new Error('Unknown error'));
      setErrorCode(500);
      setData(null);
    } finally {
      setLoading(false);
    }
  };
  
  useEffect(() => {
    fetchData();
    
    // Set up polling if refreshInterval is provided
    if (refreshInterval > 0) {
      const intervalId = setInterval(fetchData, refreshInterval);
      return () => clearInterval(intervalId);
    }
  }, [timeRange, refreshInterval]);
  
  return {
    data,
    loading,
    error,
    errorCode,
    isError: !!error,
    refresh: fetchData,
  };
} 