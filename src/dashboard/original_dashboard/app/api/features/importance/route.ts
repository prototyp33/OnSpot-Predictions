import { NextRequest, NextResponse } from 'next/server';
import { API_CONFIG } from '@/lib/config';
import { createErrorResponse, createTimeoutController } from '@/lib/api-utils';

const BACKEND_URL = process.env.BACKEND_API_URL || 'http://localhost:8000';

export async function GET(request: NextRequest) {
  const { signal, cleanup } = createTimeoutController(API_CONFIG.timeoutMs);
  
  try {
    // Extract query parameters
    const searchParams = request.nextUrl.searchParams;
    const timeRange = searchParams.get('time_range');
    
    // Make request to backend metrics endpoint to get feature importance
    const response = await fetch(`${BACKEND_URL}/metrics?metric_names=feature_importance`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      signal,
    });
    
    cleanup(); // Clear timeout
    
    if (!response.ok) {
      // If the backend doesn't support feature importance yet, return mock data
      const mockFeatureImportance = {
        features: [
          { name: 'Temperature', importance: 0.85 },
          { name: 'Time of Day', importance: 0.75 },
          { name: 'Day of Week', importance: 0.65 },
          { name: 'Precipitation', importance: 0.45 },
          { name: 'Wind Speed', importance: 0.35 }
        ]
      };
      
      return NextResponse.json(mockFeatureImportance);
    }
    
    const data = await response.json();
    
    // Transform data if needed
    const transformedData = {
      features: data.metrics
        .filter((m: any) => m.name === 'feature_importance')
        .map((m: any) => ({
          name: m.feature_name,
          importance: m.value
        }))
    };
    
    return NextResponse.json(transformedData);
  } catch (error) {
    cleanup(); // Ensure timeout is cleared
    
    console.error('Feature importance error:', error);
    
    // Return mock data on error for now
    const mockFeatureImportance = {
      features: [
        { name: 'Temperature', importance: 0.85 },
        { name: 'Time of Day', importance: 0.75 },
        { name: 'Day of Week', importance: 0.65 },
        { name: 'Precipitation', importance: 0.45 },
        { name: 'Wind Speed', importance: 0.35 }
      ]
    };
    
    return NextResponse.json(mockFeatureImportance);
  }
} 