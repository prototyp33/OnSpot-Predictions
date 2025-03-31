# Docstring and Comment Guidelines

This document provides guidelines for writing clear, consistent docstrings and comments throughout the OnSpot Predictive Model codebase.

## General Principles

1. **Document why, not what**: Code shows what is being done; documentation should explain why it's being done.
2. **Keep it current**: Outdated documentation is worse than none at all.
3. **Be concise but complete**: Include necessary details without unnecessary verbosity.
4. **Use consistent style**: Follow the conventions in this document throughout the codebase.

## Docstring Format

We use Google-style docstrings for Python code. This style is well-supported by documentation generators and focuses on readability.

### Module Docstrings

Every Python module (.py file) should start with a docstring:

```python
"""
Module for feature engineering functions related to parking data.

This module provides functions to generate, transform, and select features
for the parking occupancy prediction model.
"""
```

### Class Docstrings

```python
class ModelTrainer:
    """
    Trains and evaluates machine learning models for parking prediction.
    
    This class encapsulates the training process, including data preparation,
    model initialization, training, and evaluation. It supports various 
    algorithms and hyperparameter tuning.
    
    Attributes:
        model_type (str): Type of model to train (e.g., "gbm", "xgboost").
        hyperparams (dict): Hyperparameters for the model.
        trained_model: The trained model object after calling train().
    """
```

### Function/Method Docstrings

```python
def calculate_feature_importance(model, feature_names):
    """
    Calculate and return feature importance scores from the trained model.
    
    Args:
        model: Trained model object with feature_importance_ attribute.
        feature_names (list): List of feature names matching the order used in training.
        
    Returns:
        pandas.DataFrame: DataFrame with feature names and their importance scores,
                          sorted by importance in descending order.
        
    Raises:
        ValueError: If the model doesn't have a feature_importance_ attribute.
        ValueError: If the length of feature_names doesn't match the number of features.
    
    Example:
        >>> model = train_gradient_boosting(X_train, y_train)
        >>> importances = calculate_feature_importance(model, X_train.columns)
        >>> print(importances.head())
    """
```

### Property Docstrings

```python
@property
def feature_names(self):
    """List of feature names used by the model."""
    return self._feature_names
```

## Comments

Comments within the code should explain "why" rather than "what" when the code isn't self-explanatory.

### Good Comments

```python
# Use exponential backoff for API retries to avoid rate limiting
retry_delay = min(2 ** num_retries, 60)

# Normalize values to 0-1 range for better model convergence
normalized_values = (values - min_values) / (max_values - min_values)

# Filter outliers using IQR method to improve model robustness
q1, q3 = np.percentile(values, [25, 75])
iqr = q3 - q1
mask = (values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)
```

### Avoid Unnecessary Comments

```python
# Increment counter (unnecessary - code is self-explanatory)
counter += 1

# Loop through values (unnecessary)
for value in values:
    # Process the value (unnecessary)
    result = process(value)
```

## Sections and TODO Notes

Use section headers in longer files to help navigate:

```python
# ------------------------------
# Data Preprocessing Functions
# ------------------------------

# ------------------------------
# Model Training Functions
# ------------------------------
```

For incomplete work, use standardized TODO format:

```python
# TODO(username): Add support for categorical features in the preprocessing step
# FIXME(username): This function doesn't handle missing values correctly
# NOTE(username): This approach could be optimized for larger datasets
```

## Final Thoughts

- Use docstrings for all modules, classes, methods, and functions.
- Update documentation when changing code.
- Run documentation linters (like pydocstyle) to ensure consistency.
- When in doubt, add more documentation rather than less. 