#!/usr/bin/env ts-node
/**
 * Script to generate TypeScript types from the FastAPI OpenAPI schema
 * 
 * Usage:
 * npm run generate-api-types
 * 
 * This script fetches the OpenAPI schema from the FastAPI backend and generates
 * TypeScript types using openapi-typescript.
 */

import fs from 'fs';
import path from 'path';
import { exec } from 'child_process';
import fetch from 'node-fetch';
import { API_CONFIG } from '../lib/config';

const OPENAPI_URL = `${API_CONFIG.baseUrl}/openapi.json`;
const OUTPUT_PATH = path.resolve(__dirname, '../lib/api-types.ts');
const TEMP_JSON_PATH = path.resolve(__dirname, '../temp-openapi.json');

async function generateApiTypes() {
  try {
    console.log(`Fetching OpenAPI schema from ${OPENAPI_URL}...`);
    
    // Fetch the OpenAPI schema
    const response = await fetch(OPENAPI_URL);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch OpenAPI schema: ${response.statusText}`);
    }
    
    const schema = await response.json();
    
    // Save the schema to a temporary file
    fs.writeFileSync(TEMP_JSON_PATH, JSON.stringify(schema, null, 2));
    
    console.log('OpenAPI schema fetched successfully.');
    console.log(`Generating TypeScript types to ${OUTPUT_PATH}...`);
    
    // Generate TypeScript types using openapi-typescript
    exec(`npx openapi-typescript ${TEMP_JSON_PATH} --output ${OUTPUT_PATH}`, (error, stdout, stderr) => {
      if (error) {
        console.error(`Error generating TypeScript types: ${error.message}`);
        return;
      }
      
      if (stderr) {
        console.error(`stderr: ${stderr}`);
        return;
      }
      
      console.log(`TypeScript types generated successfully to ${OUTPUT_PATH}`);
      
      // Add export statements for commonly used types
      const exportStatements = `
// Export commonly used types
export type Metric = components['schemas']['Metric'];
export type Alert = components['schemas']['Alert'];
export type DriftResult = components['schemas']['DriftResult'];
export type HealthResponse = components['schemas']['HealthResponse'];
export type MetricsResponse = components['schemas']['MetricsResponse'];
export type AlertsResponse = components['schemas']['AlertsResponse'];
export type DriftResponse = components['schemas']['DriftResponse'];
export type PredictionInput = components['schemas']['PredictionInput'];
export type PredictionOutput = components['schemas']['PredictionOutput'];
export type BatchPredictionInput = components['schemas']['BatchPredictionInput'];
export type BatchPredictionOutput = components['schemas']['BatchPredictionOutput'];
`;
      
      // Append export statements to the generated file
      fs.appendFileSync(OUTPUT_PATH, exportStatements);
      
      // Clean up temporary file
      fs.unlinkSync(TEMP_JSON_PATH);
      
      console.log('Added export statements for commonly used types.');
      console.log('Done!');
    });
  } catch (error) {
    console.error('Error generating API types:', error);
    process.exit(1);
  }
}

generateApiTypes(); 