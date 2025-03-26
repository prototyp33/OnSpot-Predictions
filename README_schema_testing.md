# Database Schema Testing Guide

This guide explains how to test your database schema to ensure it works as expected for the OnSpot Predictive Model project.

## Overview

The schema testing system validates three key aspects of your database:

1. **Table structure** - Confirms tables exist and have the expected columns
2. **CRUD operations** - Tests SELECT, INSERT, UPDATE, and DELETE for each table
3. **Relationships** - Verifies foreign key relationships between tables

## Prerequisites

Before running the tests, ensure you have:

1. Supabase URL and API key in your environment variables:
   ```
   SUPABASE_URL=your_url
   SUPABASE_KEY=your_key
   ```

2. Required Python packages:
   ```
   supabase
   python-dotenv
   ```

## Setup

### 1. Create RPC Functions (Recommended)

For the most comprehensive testing experience, run the included SQL file in your Supabase SQL editor to create helper functions:

1. Navigate to the Supabase dashboard
2. Go to the SQL Editor
3. Copy and paste the contents of `scripts/create_rpc_functions.sql` 
4. Run the SQL to create the functions

These functions provide detailed schema information, but the testing script will still work without them by using fallback methods.

### 2. Run the Tests

Execute the test script with:

```bash
python scripts/test_database_schema.py
```

Additional options:
```bash
# Test specific tables only
python scripts/test_database_schema.py --tables models predictions

# Save results to a custom file
python scripts/test_database_schema.py --output my_results.json

# Skip cleanup of test records (for debugging)
python scripts/test_database_schema.py --skip-cleanup
```

## Test Process

The script performs the following for each table:

1. **Connection Test** - Verifies connectivity to Supabase
2. **Table Discovery** - Identifies available tables
3. **Schema Analysis** - Retrieves column information
4. **SELECT Test** - Verifies read access
5. **INSERT Test** - Creates a test record
6. **UPDATE Test** - Modifies the test record
7. **DELETE Test** - Removes the test record
8. **Relationship Tests** - Verifies foreign key relationships

## Understanding Results

Results are saved to `schema_test_results.json` and logged to the console. The summary shows:

- Passed tests
- Failed tests
- Warnings (non-critical issues)

Example output:
```
2023-11-12 14:35:27 - __main__ - INFO - Schema tests completed.
2023-11-12 14:35:27 - __main__ - INFO - Summary: 35 passed, 2 failed, 5 warnings
```

### Common Issues

1. **Missing RPC Functions**: You'll see warnings if you haven't created the helper functions.
2. **Schema Inference**: Without RPC functions, schema information is inferred from sample records.
3. **INSERT Failures**: May indicate missing required fields or constraint violations.
4. **Relationship Test Failures**: Could signal missing data or incorrect relationship definitions.

## Next Steps

After running the tests, consider:

1. **Fixing Failed Tests**: Address any issues discovered during testing
2. **Schema Documentation**: Update documentation based on actual schema
3. **Data Validation**: Add validation rules to your application code
4. **Indexing**: Create appropriate indexes based on query patterns

## Customizing Tests

You can extend the testing script by:

1. **Adding Custom Tables**: Update the `known_tables` list
2. **Table-Specific Records**: Add cases to `_generate_test_record()`
3. **Expected Relationships**: Extend the `expected_relationships` list
4. **Custom Validation**: Add additional validation logic to the script

## Troubleshooting

- **Connection Issues**: Check your environment variables and network connectivity
- **Permission Errors**: Ensure your API key has sufficient permissions
- **Schema Changes**: If schema has changed, update expected relationships and test records

## Best Practices

1. **Run Tests Regularly**: Especially after schema migrations
2. **Test in Development**: Never run with `--skip-cleanup` in production
3. **Version Control Results**: Track schema_test_results.json over time to see changes
4. **Expand Test Coverage**: Add more specific tests for critical tables 