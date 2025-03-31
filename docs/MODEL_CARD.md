# Parking Occupancy Duration Prediction Model Card

## 1. Title & Version Control

- **Model Name**: Parking Occupancy Duration Prediction (Regression Ensemble)
- **Version**: v1.2.3
- **Release Date**: 2024-03-29
- **Authors**: OnSpot Data Science Team
- **Repository**: https://github.com/OnSpot/Predictive_Model
- **License**: MIT
- **Contact**: support@onspot.ai

## 2. Overview

### 2.1 Business Objective

- Reduce urban traffic congestion by enabling dynamic pricing and real-time parking recommendations
- Support city planners in infrastructure optimization (e.g., identifying over/underutilized zones)
- Enable better parking space utilization through predictive analytics

### 2.2 Technical Scope

- **Input**: Time-series sensor data, weather conditions, event calendars
- **Output**: Predicted occupancy duration (minutes) ± confidence interval
- **Supported Regions**: Urban zones with IoT-enabled parking infrastructure

### 2.3 Key Assumptions

- Sensor data is sampled at 5-minute intervals
- Weather data is sourced from OpenWeatherMap API
- No sudden infrastructure changes during prediction windows
- Minimum sensor uptime of 95% required for reliable predictions

### 2.4 Limitations

- **Temporal**: Accuracy degrades beyond 24-hour forecasts
- **Geospatial**: Requires pre-mapped zone IDs; unsupported zones return `null`
- **Data Dependency**: Fails if sensor uptime < 95%
- **Weather Impact**: Extreme weather conditions may affect prediction accuracy

## 3. Data Pipeline

### 3.1 Data Sources

| Source | Example | Update Frequency |
|--------|----------|------------------|
| IoT Sensors | `{"zone_id": "A1", "occupancy_start": "2024-03-29T08:15:00", "occupancy_end": "2024-03-29T09:30:00"}` | Real-time |
| Weather API | `{"temp": 18, "precipitation": 0.2}` | Hourly |
| Event Calendar | `{"event": "Music Festival", "attendees": 5000}` | Daily |

### 3.2 Feature Engineering

- **Derived Features**:
  - `peak_hour`: Binary (1 if 8–10 AM or 5–7 PM)
  - `proximity_to_transit`: Distance to nearest subway/bus stop (meters)
  - `rolling_avg_duration`: 7-day moving average of occupancy per zone
  - `day_type`: Categorical (weekday, weekend, holiday)
  
- **Handling Time**:
  - Cyclical encoding for `hour` (sin/cos transformations)
  - Holiday flags using public calendar APIs
  - Time-based features (hour of day, day of week, month)

### 3.3 Preprocessing Workflow

1. **Impute Missing Data**:
   - Time-based linear interpolation for sensor gaps
   - Weather data: Forward-fill for <2-hour gaps; else, use regional averages
   - Event data: Nearest-neighbor imputation

2. **Outlier Removal**:
   - Drop durations > 12 hours (assumed sensor errors)
   - IQR-based outlier detection for numerical features
   - Domain-specific rules for weather data

3. **Scaling & Encoding**:
   - Min-Max scaling for `temperature`, `precipitation`
   - One-hot encoding for categorical variables
   - Label encoding for high-cardinality features

### 3.4 Dataset Splits

- **Training**: 70% (stratified by `zone_id` and `day_of_week`)
- **Validation**: 15% (used for hyperparameter tuning)
- **Test**: 15% (held out for final evaluation)
- **Time-based split**: Data after 2024-02-01 reserved for testing

## 4. Model Architecture

### 4.1 Base Models

| Model | Library | Key Hyperparameters | Role |
|-------|---------|---------------------|------|
| Linear Regression | Scikit-learn | `fit_intercept=True` | Baseline for trend capture |
| XGBoost | XGBoost | `max_depth=6`, `subsample=0.8` | Non-linear pattern detection |
| Random Forest | Scikit-learn | `n_estimators=200`, `max_features='sqrt'` | Variance reduction |

### 4.2 Ensemble Design

- **Stacking Architecture**:
  1. **Level 1 (Base Models)**: Train Linear Regression, XGBoost, and Random Forest
  2. **Level 2 (Meta-Learner)**: ElasticNet combines base model predictions
  
- **Rationale**:
  - ElasticNet (α=0.1, L1_ratio=0.5) balances sparsity and multi-collinearity
  - K-fold prediction stacking prevents overfitting
  - Weighted average based on model performance on validation set

### 4.3 Hyperparameter Optimization

- **Method**: Bayesian Optimization (10 iterations)
- **Search Space**:
  - XGBoost: `learning_rate` (0.01–0.3), `max_depth` (3–10)
  - ElasticNet: `alpha` (0.001–1.0), `L1_ratio` (0–1)
- **Best Configuration**: Stored in `config/hyperparameters.yaml`

## 5. Training & Validation

### 5.1 Training Environment

- **Cloud**: AWS SageMaker (ml.m5.xlarge instances)
- **Dependencies**: Python 3.9, requirements pinned in `requirements.txt`
- **Training Time**: ~2 hours for full pipeline
- **Resource Usage**: 16GB RAM, 4 vCPUs

### 5.2 Loss Function

- **Primary Metric**: Mean Absolute Error (MAE)
- **Secondary Metrics**:
  - R²: Track model explainability
  - RMSLE: Penalize underprediction of long durations
  - MAPE: Percentage error for business interpretation

### 5.3 Regularization

- **Early Stopping**: Monitor validation MAE with 10-epoch patience
- **Feature Dropout**: 20% chance to mask `event_attendance`
- **L1/L2 Regularization**: Applied in ElasticNet meta-learner

### 5.4 Cross-Validation

- **Method**: TimeSeriesSplit (5 folds)
- **Results**:
  - Avg. MAE: 8.2 ± 1.3 minutes
  - Avg. R²: 0.89 ± 0.04
  - Avg. MAPE: 12.5% ± 2.1%

## 6. Evaluation & Interpretability

### 6.1 Performance Summary

| Metric | Train | Validation | Test |
|--------|--------|------------|------|
| MAE (minutes) | 6.1 | 8.2 | 8.5 |
| RMSE (minutes) | 10.3 | 12.5 | 13.0 |
| R² | 0.92 | 0.89 | 0.88 |
| MAPE (%) | 10.2 | 12.5 | 13.1 |

### 6.2 Error Analysis

- **Worst-Performing Scenarios**:
  - Public holidays with unannounced events (MAE > 15 minutes)
  - Zones near stadiums during games (data sparsity)
  - Extreme weather conditions (outlier predictions)
  
- **Mitigation Strategies**:
  - Enhanced event detection using social media data
  - Weather-specific model variants
  - Increased data collection for problematic zones

### 6.3 Explainability

- **SHAP Values**:
  - Top 3 Features: `hour`, `rolling_avg_duration`, `precipitation`
  - Feature importance plots in `docs/feature_importance/`
  
- **Partial Dependence Plots**:
  - Non-linear relationships documented in `docs/pdp_analysis/`
  - Key insights for business stakeholders

## 7. Deployment

### 7.1 API Specification

- **Endpoint**: `POST /predict`
- **Input Schema**:
```json
{
  "zone_id": "A1",
  "timestamp": "2024-03-29T08:00:00Z",
  "weather": {
    "temp": 15,
    "precipitation": 0.0
  }
}
```

- **Output Schema**:
```json
{
  "predicted_duration": 45,
  "confidence_interval": [30, 60],
  "model_version": "v1.2.3"
}
```

### 7.2 Infrastructure

- **CI/CD**: GitHub Actions for automated testing
- **Monitoring**:
  - Grafana dashboards for latency and accuracy
  - Data drift monitoring via statistical tests
  - Resource utilization tracking
  
- **Alerting**:
  - Slack notifications for drift detection
  - PagerDuty for critical failures
  - Daily performance reports

### 7.3 Security

- **Authentication**: API keys via AWS Cognito
- **Data Privacy**: GDPR-compliant data handling
- **Access Control**: Role-based access to model endpoints
- **Audit Logging**: All predictions logged for compliance

## 8. Maintenance

### 8.1 Retraining Policy

- **Scheduled**: Quarterly retraining with new data
- **Trigger-Based**:
  - MAE increase > 15%
  - Feature drift (PSI > 0.2)
  - Significant zone changes

### 8.2 Versioning

- **Model Registry**: MLflow tracking
- **Artifacts**: Stored in S3 with versioning
- **Rollback**: 15-minute recovery SLA
- **Change Log**: Maintained in `CHANGELOG.md`

### 8.3 User Feedback

- **Collection**: Mobile app feedback integration
- **Storage**: Snowflake data warehouse
- **Analysis**: Weekly feedback review process
- **Action Items**: Tracked in JIRA

## 9. Ethical & Compliance

### 9.1 Bias Audits

- **Fairness Metrics**: Equal MAE across zones
- **Monitoring**: Monthly bias assessment
- **Mitigation**: Data reweighting strategies
- **Reporting**: Quarterly fairness reports

### 9.2 Regulatory Compliance

- **Data Laws**: GDPR, CCPA compliant
- **Documentation**: Public model cards
- **Retention**: Data retention policies
- **Access**: Subject access request process

## 10. Appendices

### 10.1 Code Snippets

```python
# Feature Engineering Example
def create_time_features(df):
    df["peak_hour"] = df["hour"].apply(lambda x: 1 if x in [8, 9, 17, 18] else 0)
    df["day_type"] = df["date"].apply(get_day_type)
    return df

# Model Training Example
def train_ensemble(X_train, y_train, config):
    models = {
        "lr": LinearRegression(),
        "xgb": XGBRegressor(**config["xgb_params"]),
        "rf": RandomForestRegressor(**config["rf_params"])
    }
    return StackingRegressor(models, meta_learner=ElasticNet())
```

### 10.2 Architecture Diagrams

```mermaid
graph LR
    A[Sensors] --> B[Kafka]
    B --> C[Feature Store]
    C --> D[Model API]
    D --> E[Dashboard]
```

### 10.3 References

1. "Urban Parking Prediction: A Temporal Ensemble Approach" (IEEE 2024)
2. XGBoost Documentation: [https://xgboost.readthedocs.io](https://xgboost.readthedocs.io)
3. MLflow Documentation: [https://www.mlflow.org/docs/latest/index.html](https://www.mlflow.org/docs/latest/index.html)
4. Scikit-learn Documentation: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/) 