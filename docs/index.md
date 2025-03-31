# OnSpot Predictive Model

![OnSpot Logo](assets/logo.png){ align=right width="150" }

A machine learning system for predicting parking occupancy based on historical data and contextual features.

## Overview

The OnSpot Predictive Model is a comprehensive solution for:

- **Predicting parking occupancy** at different locations and times
- **Analyzing parking patterns** using historical data
- **Providing real-time availability information** for parking areas

It combines multiple data sources, advanced feature engineering, and state-of-the-art machine learning algorithms to deliver accurate predictions.

## Key Features

- 🧠 **Advanced Modeling** - Uses gradient boosting, ensemble methods, and specialized time series approaches
- 📊 **Comprehensive Data Pipeline** - From raw data to engineered features to predictions
- 🔄 **Automatic Retraining** - Keeps models up-to-date with the latest parking patterns
- 📈 **Performance Monitoring** - Tracks model accuracy and data drift over time
- 🔌 **REST API** - Easy integration with front-end applications

## Quick Links

- [Getting Started](user_guide/getting_started.md) - How to start using the system
- [API Reference](api_reference/index.md) - Detailed code documentation
- [Architecture](architecture/index.md) - System design and components
- [Developer Guide](developer_guide/index.md) - For contributors and developers
- [Examples](examples/index.md) - Code examples for common tasks

## Project Structure

```
onspot/                          # Main package
├── data/                        # Data processing modules
├── models/                      # ML model implementations
├── pipeline/                    # End-to-end pipelines
├── monitoring/                  # Model monitoring tools
├── api/                         # REST API implementation
└── utils/                       # Utility functions
```

## Requirements

- Python 3.8+
- Pandas, NumPy, Scikit-learn
- XGBoost, LightGBM, CatBoost (optional)
- PostgreSQL (for data storage)
- Supabase (for hosted database)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 