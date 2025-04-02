# Enhanced Type Safety for OnSpot Dashboard

This document describes the enhanced type safety features implemented in the OnSpot Dashboard application.

## Overview

The OnSpot Dashboard now includes two major type safety enhancements:

1. **Generated TypeScript Types from FastAPI OpenAPI Schema**: Automatically generates TypeScript types from the FastAPI backend's OpenAPI schema.
2. **Runtime Validation with Zod**: Validates API responses at runtime using Zod schemas.

These enhancements provide several benefits:

- **Type Safety**: Ensures that API responses match expected types.
- **Error Detection**: Catches type errors at compile time and runtime.
- **Documentation**: Provides self-documenting types for API responses.
- **Developer Experience**: Improves autocomplete and type hints in your IDE.

## Generated TypeScript Types

### How It Works

1. The `generate-api-types.ts` script fetches the OpenAPI schema from the FastAPI backend.
2. It uses `openapi-typescript` to generate TypeScript types from the schema.
3. The generated types are saved to `lib/api-types.ts`.

### Usage

To generate the TypeScript types:

```bash
npm run generate-api-types
```

This should be run whenever the FastAPI backend schema changes.

### Using the Generated Types

Import the types from `lib/api-types.ts`:

```typescript
import { Metric, Alert, DriftResult } from '@/lib/api-types';

// Use the types in your components
const metrics: Metric[] = [...];
```

## Runtime Validation with Zod

### How It Works

1. Zod schemas are defined in `lib/api-schemas.ts` based on the expected API responses.
2. The `useApiRequest` hook accepts a Zod schema for validating API responses.
3. API responses are validated against the schema at runtime.
4. Validation errors are logged and reported to Sentry.

### Usage

#### Defining Schemas

Schemas are defined in `lib/api-schemas.ts`:

```typescript
import { z } from 'zod';

export const metricSchema = z.object({
  timestamp: z.string().datetime(),
  name: z.string(),
  value: z.number(),
  // ...
});
```

#### Using Schemas in API Requests

Use the schemas with the `useApiRequest` hook:

```typescript
import { useApiRequest } from '@/hooks/useApiRequest';
import { metricResponseSchema } from '@/lib/api-schemas';
import { z } from 'zod';

// In your component
const { 
  data, 
  loading, 
  error 
} = useApiRequest<z.infer<typeof metricResponseSchema>>({
  schema: metricResponseSchema,
  endpoint: 'metrics'
});
```

#### Validating API Responses Manually

You can also validate API responses manually using the validation utilities:

```typescript
import { validateApiResponse } from '@/lib/api-validation';
import { metricResponseSchema } from '@/lib/api-schemas';

// Validate a response
const validatedData = validateApiResponse(
  responseData,
  metricResponseSchema,
  { endpoint: 'metrics' }
);

if (validatedData !== null) {
  // Use the validated data
} else {
  // Handle validation error
}
```

## API Routes

API routes in `app/api/` use Zod validation to validate responses from the FastAPI backend:

```typescript
import { metricsResponseSchema } from '@/lib/api-schemas';
import { validateApiResponse } from '@/lib/api-validation';

// In your API route handler
const data = await response.json();

// Validate response data
const validatedData = validateApiResponse(
  data,
  metricsResponseSchema,
  { endpoint: 'metrics' }
);

// Return the validated data
return NextResponse.json(validatedData || data);
```

## Error Handling

Validation errors are:

1. Logged to the console
2. Reported to Sentry with context
3. Gracefully handled by falling back to the original data

This ensures that the application continues to function even if the API response doesn't match the expected schema.

## Best Practices

1. **Always define schemas for API responses**: This ensures that your application is resilient to API changes.
2. **Use the generated types for type annotations**: This ensures consistency between the backend and frontend.
3. **Run `generate-api-types` regularly**: Keep your types in sync with the backend.
4. **Handle validation errors gracefully**: Provide fallbacks for when validation fails.
5. **Monitor validation errors in Sentry**: This helps identify API changes that break your application.

## Troubleshooting

### Validation Errors

If you're seeing validation errors in the console or Sentry:

1. Check if the API response has changed
2. Update the Zod schema to match the new response format
3. Regenerate the TypeScript types

### Type Errors

If you're seeing TypeScript errors:

1. Make sure you've run `generate-api-types`
2. Check if you're using the correct types
3. Update your type annotations to match the generated types 