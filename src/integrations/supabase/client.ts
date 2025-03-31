import { createClient } from '@supabase/supabase-js';
import type { Tables } from '../types/supabase';

// Get environment variables
export const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
export const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

// Validate environment variables
if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error('Missing Supabase environment variables. Please check your .env file.');
}

// Create Supabase client with types
export const supabase = createClient<Tables>(supabaseUrl, supabaseAnonKey);

// Helper function to handle Supabase errors
export const handleSupabaseError = (error: any) => {
  console.error('Supabase error:', error);
  throw new Error(error.message || 'An error occurred while accessing the database');
};

// Typed query helpers
export const fetchModels = async () => {
  const { data, error } = await supabase
    .from('models')
    .select('*');
  
  if (error) handleSupabaseError(error);
  return data;
};

export const fetchPredictions = async (modelId?: string) => {
  let query = supabase
    .from('predictions')
    .select('*');
  
  if (modelId) {
    query = query.eq('model_id', modelId);
  }
  
  const { data, error } = await query;
  if (error) handleSupabaseError(error);
  return data;
};

export const fetchDriftAnalysis = async (modelId?: string) => {
  let query = supabase
    .from('drift_analysis')
    .select('*');
  
  if (modelId) {
    query = query.eq('model_id', modelId);
  }
  
  const { data, error } = await query;
  if (error) handleSupabaseError(error);
  return data;
};

export const fetchModelMetrics = async (modelId?: string) => {
  let query = supabase
    .from('model_metrics')
    .select('*');
  
  if (modelId) {
    query = query.eq('model_id', modelId);
  }
  
  const { data, error } = await query;
  if (error) handleSupabaseError(error);
  return data;
};

if (!response.ok) {
  console.error(`❌ MOCK fetch failed with status: ${response.status}`);
  let errorBody = { message: `Failed to fetch ${table.replace('_', ' ')} data`, status: response.status };
  try {
    const parsedBody = await response.json();
    errorBody = { ...errorBody, ...parsedBody };
  } catch (e) {
    console.warn('⚠️ MOCK could not parse error body:', e);
  }
  throw new Error(`Failed to fetch ${table.replace('_', ' ')} data`);
} 