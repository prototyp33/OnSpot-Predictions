import { NextRequest, NextResponse } from 'next/server';
import { API_CONFIG } from '@/lib/config';
import { createErrorResponse, createTimeoutController } from '@/lib/api-utils';

const BACKEND_URL = process.env.BACKEND_API_URL || 'http://localhost:8000';

export async function GET(request: NextRequest) {
  const { signal, cleanup } = createTimeoutController(30000); // 30 second timeout
  
  try {
    // Extract query parameters
    const searchParams = request.nextUrl.searchParams;
    const timeRange = searchParams.get('time_range') || '7d';
    
    const url = `${BACKEND_URL}/metrics?time_range=${timeRange}`;
    console.log('[Metrics API] Fetching from:', url);
    
    // Make request to backend metrics endpoint
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      signal,
      next: { revalidate: 0 }, // Disable cache
    });
    
    cleanup(); // Clear timeout
    
    if (!response.ok) {
      console.error('[Metrics API] Request failed:', {
        status: response.status,
        statusText: response.statusText,
        url: url
      });
      
      // Try to get error details
      let errorDetail;
      try {
        const errorJson = await response.json();
        errorDetail = errorJson.detail || errorJson.message || response.statusText;
      } catch (e) {
        errorDetail = response.statusText;
      }
      
      throw new Error(`Failed to fetch metrics data: ${errorDetail}`);
    }
    
    const data = await response.json();
    console.log('[Metrics API] Received data:', data);
    
    if (!data?.metrics?.length) {
      console.warn('[Metrics API] No metrics in response:', data);
      return NextResponse.json({
        metrics: [],
        timestamp: new Date().toISOString()
      });
    }
    
    return NextResponse.json(data);
  } catch (error) {
    cleanup(); // Ensure timeout is cleared
    
    console.error('[Metrics API] Error:', error);
    
    // Check for timeout error
    if (error instanceof Error && error.name === 'AbortError') {
      return createErrorResponse(
        new Error('Request timed out while fetching performance metrics'),
        { metrics: [], timestamp: new Date().toISOString() },
        408 // Request Timeout
      );
    }
    
    return createErrorResponse(
      error,
      { metrics: [], timestamp: new Date().toISOString() },
      500
    );
  }
} 