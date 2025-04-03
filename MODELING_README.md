# Parking Occupancy Prediction Model

## Model Summary
- **Final Model:** Tuned Random Forest Regressor
- **Primary Target:** Parking occupancy (as percentage)
- **Test Set Performance:** 
  - R² Score: 0.8163
  - MAE: 9.6630
  - RMSE: 11.8293
- **Optimal Hyperparameters:**
  - max_depth: None (unlimited)
  - max_features: 'sqrt'
  - min_samples_leaf: 1
  - min_samples_split: 2
  - n_estimators: 300

## Data Pipeline

The full modeling pipeline consists of several stages:

### 1. Data Preparation (`prepare_data.py`)
- **Input:** Raw or partially processed parking data (e.g., `data/raw/prepared_sample_data_features.csv`)
- **Process:**
  - Handles timestamp consistency and ensures regular intervals 
  - Normalizes occupancy values
  - Generates temporal features (hour, day of week, month, etc.)
  - Creates cyclical encodings (sin/cos transformations)
  - Adds lag features (1hr, 2hr, ..., 168hr)
  - Creates rolling window features (mean, std, min, max) for various windows
  - Fills missing values strategically
- **Output:** Feature-rich dataset ready for splitting (`data/processed/prepared_full_features.csv`)

### 2. Data Splitting (`split_data.py`)
- **Input:** Prepared dataset with engineered features
- **Process:**
  - Performs time-based train/validation/test split (70%/15%/15%)
  - Fits a preprocessor (StandardScaler, OneHotEncoder) on training data
  - Saves the fitted preprocessor for consistent transformation
- **Output:**
  - `data/splits_full/train.csv`: 70% of data (chronologically earliest)
  - `data/splits_full/validation.csv`: 15% of data (middle period)
  - `data/splits_full/test.csv`: 15% of data (most recent)
  - `data/splits_full/preprocessor.pkl`: Fitted ColumnTransformer for feature scaling/encoding

### 3. Model Tuning (`tune_random_forest.py`)
- **Input:** Train/validation data and preprocessor
- **Process:**
  - Applies preprocessor to train data
  - Performs TimeSeriesSplit cross-validation (5 folds)
  - Uses GridSearchCV to evaluate 324 hyperparameter combinations
  - Identifies optimal parameters based on R² score
  - Refits best model on full training data
  - Evaluates on validation set
- **Output:**
  - `models/tuned_rf_model_{timestamp}.pkl`: Best model fitted on training data
  - `rf_tuning_results/rf_tuning_summary_{timestamp}.json`: Tuning results and metrics

### 4. Direct Model Training (`train_tuned_rf.py`)
- **Input:** Train/validation data and preprocessor
- **Process:**
  - Applies preprocessor to train data
  - Trains a Random Forest with pre-determined optimal parameters
  - Evaluates on validation set
- **Output:**
  - `models/tuned_rf_model_{timestamp}.pkl`: Model fitted with optimal parameters
  - `models/model_metrics_{timestamp}.json`: Training details and validation metrics

### 5. Model Evaluation (`evaluate_tuned_model.py`)
- **Input:** Test data, preprocessor, and trained model
- **Process:**
  - Applies preprocessor to test data
  - Makes predictions using trained model
  - Calculates comprehensive performance metrics
- **Output:**
  - `model_evaluation_results/test_evaluation_{timestamp}.txt`: Final test set performance metrics

## Model Performance

Comparing the performance across different datasets:

| Metric | Training (CV) | Validation | Test |
|--------|---------------|------------|------|
| R²     | 0.7959        | 0.8424     | 0.8163 |
| MAE    | -             | 8.6895     | 9.6630 |
| RMSE   | -             | 10.6995    | 11.8293 |

The consistent performance across validation and test sets indicates good generalization and robustness of the model.

## Usage Instructions

### Process New Data with Complete Pipeline

```bash
# 1. Fix data issues (if needed)
python scripts/fix_merged_data.py --input data/raw/new_data.csv --output data/processed/fixed_new_data.csv

# 2. Prepare data
python scripts/prepare_data.py --input data/processed/fixed_new_data.csv --output data/processed/prepared_new_features.csv

# 3. Split data (optional if you just want predictions)
python scripts/split_data.py --input-path data/processed/prepared_new_features.csv --output-dir data/splits_new/

# 4. Train model with optimal parameters (if needed)
python scripts/train_tuned_rf.py --data-dir data/splits_new/ --output-dir models/

# 5. Evaluate model
python scripts/evaluate_tuned_model.py --data-dir data/splits_new/ --model-dir models/ --model-path models/tuned_rf_model_TIMESTAMP.pkl
```

### Make Predictions on New Data

```bash
# For batch predictions
python scripts/batch_predict.py --model models/tuned_rf_model_TIMESTAMP.pkl --input data/processed/prepared_new_features.csv --output model_predictions/predictions_TIMESTAMP.csv

# Analyze model performance
python scripts/analyze_model.py --model models/tuned_rf_model_TIMESTAMP.pkl --data data/splits_full/test.csv --output model_analysis_results/
```

## Future Enhancements

- **Feature Importance Analysis:** Investigate which features contribute most to prediction accuracy
- **Model Ensemble:** Combine Random Forest with other algorithms (e.g., Gradient Boosting, Neural Networks)
- **Real-time API:** Implement a REST API for on-demand predictions
- **Monitoring System:** Track model performance and data drift over time
- **Automated Retraining:** Implement pipeline for periodic model updates with new data

## Files Structure

```
onspot-core/
├── data/
│   ├── raw/                              # Raw input data
│   │   └── merged_parking_data.csv
│   ├── processed/                        # Cleaned/processed data
│   │   ├── fixed_merged_parking_data.csv
│   │   └── prepared_full_features.csv
│   └── splits_full/                      # Train/val/test splits
│       ├── train.csv
│       ├── validation.csv
│       ├── test.csv
│       └── preprocessor.pkl
├── scripts/
│   ├── fix_merged_data.py               # Fix data issues
│   ├── prepare_data.py                  # Feature engineering
│   ├── split_data.py                    # Data splitting
│   ├── tune_random_forest.py            # Hyperparameter tuning
│   ├── train_tuned_rf.py                # Direct model training
│   ├── evaluate_tuned_model.py          # Model evaluation
│   ├── batch_predict.py                 # Batch prediction script
│   └── analyze_model.py                 # Model analysis & visualization
├── models/                              # Trained models
│   ├── tuned_rf_model_TIMESTAMP.pkl
│   └── model_metrics_TIMESTAMP.json
├── rf_tuning_results/                   # Tuning results
│   └── rf_tuning_summary_TIMESTAMP.json
├── model_evaluation_results/            # Evaluation results
│   └── test_evaluation_TIMESTAMP.txt
├── model_predictions/                   # Prediction outputs
│   └── predictions_TIMESTAMP.csv
└── model_analysis_results/              # Analysis outputs
    ├── feature_importance_mdi.png
    ├── feature_importance_permutation.png
    ├── predicted_vs_actual.png
    └── model_analysis_summary.md
``` 