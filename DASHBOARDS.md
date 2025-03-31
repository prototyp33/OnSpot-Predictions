# OnSpot Dashboard Architecture

This document outlines the dashboard architecture in the OnSpot Predictive Model project. We maintain two distinct dashboards, each serving different purposes.

## 1. ML Monitoring Dashboard (`dashboard_2.0/`)

### Purpose
The ML Monitoring Dashboard is dedicated to monitoring and analyzing the performance of our machine learning models in production.

### Key Features
- Real-time model performance monitoring
- Data quality and drift detection
- Intelligent alerting system
- Automated reporting
- API endpoints for data integration
- Configurable thresholds and settings

### Technical Stack
- Backend: FastAPI
- Data Visualization: Plotly, Dash
- Database Integration: Supabase

### Usage
1. Start the API server:
   ```bash
   python -m src.core.api.main
   ```

2. Launch the dashboard:
   ```bash
   python -m src.features.model_performance.dashboard
   ```

## 2. Business Metrics Dashboard (`onspot-dashboard/`)

### Purpose
The Business Metrics Dashboard focuses on displaying and analyzing business-related KPIs and operational metrics.

### Key Features
- Real-time KPI monitoring (RMSE, MAE, R² Score)
- Type-safe implementation
- Modern UI components
- Error handling with retry mechanisms
- Integration with FastAPI backend

### Technical Stack
- Frontend: React/Next.js with TypeScript
- UI Components: Modern React components
- Type Safety: Generated TypeScript types from OpenAPI
- Runtime Validation: Zod schemas

### Key Metrics
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- R² Score (Coefficient of Determination)

## Archived Dashboards

The following dashboard implementations have been archived and are no longer maintained:

1. `onspot-dashboard-new/` - Deprecated in favor of the current implementations
2. `scripts/model_dashboard.py` - Legacy script-based dashboard
3. `scripts/model_dashboard_enhanced.py` - Enhanced version of legacy dashboard
4. `scripts/supabase_dashboard.py` - Old Supabase-specific dashboard

These implementations can be found in the `archive/` directory for reference purposes only.

## Development Guidelines

1. **New Features**
   - Add ML monitoring features to `dashboard_2.0/`
   - Add business/operational features to `onspot-dashboard/`
   - Avoid creating new dashboard implementations

2. **Maintenance**
   - Keep dependencies up to date in both dashboards
   - Maintain type safety in the business dashboard
   - Follow the modular architecture in both implementations

3. **Testing**
   - Write tests for new features
   - Maintain end-to-end tests for critical paths
   - Test both dashboards before deployment

## Integration Points

The two dashboards share some integration points:

1. **Data Sources**
   - Both connect to the same Supabase backend
   - Share common database tables (see DATABASE.md)

2. **Authentication**
   - Use common authentication mechanisms
   - Share user roles and permissions

3. **API Endpoints**
   - Some endpoints may be used by both dashboards
   - Maintain API compatibility for shared endpoints

## Future Improvements

1. **ML Monitoring Dashboard**
   - Add more advanced drift detection features
   - Implement automated model retraining triggers
   - Enhance visualization capabilities

2. **Business Dashboard**
   - Add more business-specific KPIs
   - Implement custom reporting features
   - Add data export capabilities 