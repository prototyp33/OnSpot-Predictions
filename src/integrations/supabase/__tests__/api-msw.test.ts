import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { systemHealthApi, metricsApi, businessContinuityApi } from '../api';
import { server } from '@/test/mocks/server';
import { http, HttpResponse } from 'msw';
import { supabaseUrl } from '@/integrations/supabase/client';

vi.mock('@/hooks/use-toast', () => ({ toast: vi.fn() }));

// Log URL at test file level
console.log('ℹ️ [api-msw.test.ts] Supabase URL used:', supabaseUrl);

describe('Supabase API with MSW - Handler Diagnosis', () => {
  it('TEST 1: should hit the /metrics handler and return 4 items', async () => {
    console.log('🧪 Running test: should hit /metrics handler');
    console.log('⏳ Test: Calling getOverviewMetrics...');
    const data = await metricsApi.getOverviewMetrics();
    console.log('🔍 Test Received metrics data:', data);
    expect(Array.isArray(data)).toBe(true);
    expect(data.length).toBe(4); // Focus on length first
  });

  it('TEST 2: should hit the /business_sla handler and return 4 items', async () => {
    console.log('🧪 Running test: should hit /business_sla handler');
    console.log('⏳ Test: Calling getSlaData...');
    const data = await businessContinuityApi.getSlaData();
    console.log('🔍 Test Received SLA data:', data);
    expect(Array.isArray(data)).toBe(true);
    expect(data.length).toBe(4); // Focus on length first
  });

  it('TEST 3: should handle API errors via MSW 500 response', async () => {
    console.log('🧪 Running test: should handle errors gracefully');
    server.use(
      http.get(`${supabaseUrl}/rest/v1/system_health*`, () => {
        console.log('🔥 MSW TEST OVERRIDE: Matched system_health, returning 500');
        return new HttpResponse(
          JSON.stringify({ message: 'Test Error via 500' }),
          { status: 500, headers: { 'Content-Type': 'application/json' } }
        );
      })
    );
    console.log('⏳ Test: Calling getHealthComponents expecting rejection...');
    try {
      await systemHealthApi.getHealthComponents(new Date());
      throw new Error('Expected getHealthComponents to throw');
    } catch (error: any) {
      expect(error.message).toBe('Failed to fetch system_health data');
    }
  });

  it('TEST 4: should correctly override /business_sla handler', async () => {
    console.log('🧪 Running test: should override SLA mock data');
    const customData = [{ id: 'custom-sla', service: 'Override Worked', target: 1, current: 1, status: 'healthy' }];
    server.use(
      http.get(`${supabaseUrl}/rest/v1/business_sla*`, () => {
        console.log('🔥 MSW TEST OVERRIDE: Matched business_sla, returning custom SLA data');
        return HttpResponse.json(customData);
      })
    );
    console.log('⏳ Test: Calling getSlaData (expecting override)...');
    const data = await businessContinuityApi.getSlaData();
    console.log('🔍 Test Received SLA data (from override):', data);
    expect(data).toEqual(customData); // Check if override worked
    expect(data.length).toBe(1);
  });
});