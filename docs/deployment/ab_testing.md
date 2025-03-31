# A/B Testing System

The OnSpot Predictive Model includes a comprehensive A/B testing system for evaluating model variants in production. This document describes how to use the system effectively.

## Overview

The A/B testing system allows you to:
- Compare different model versions
- Test new feature engineering approaches
- Evaluate performance improvements
- Make data-driven decisions about deployments

## Components

1. **Core A/B Testing Module** (`scripts/deployment/ab_testing.py`)
   - Handles variant assignment
   - Collects metrics
   - Performs statistical analysis
   - Determines winners

2. **Experiment Manager** (`scripts/deployment/experiment_manager.py`)
   - Manages multiple experiments
   - Handles experiment lifecycle
   - Loads configurations
   - Tracks completion criteria

3. **CLI Tool** (`scripts/deployment/manage_experiments.py`)
   - Creates and manages experiments
   - Shows experiment status
   - Exports results

## Configuration

Experiments are configured in `config/ab_testing.yaml`. Example configuration:

```yaml
defaults:
  min_sample_size: 1000
  confidence_level: 0.95
  max_duration_days: 30

experiments:
  model_v2_evaluation:
    name: "model_v2_evaluation"
    description: "Evaluating v2 model performance"
    variants:
      - name: "control"
        model_version: "v1.0.0"
      - name: "treatment"
        model_version: "v2.0.0"
    traffic_split: [0.8, 0.2]
    success_criteria:
      - metric: "prediction_error"
        improvement_threshold: 0.1
      - metric: "response_time"
        improvement_threshold: 0.2
```

## Usage

### Command Line Interface

1. **Create an Experiment**
   ```bash
   python -m scripts.deployment.manage_experiments create model_v2_evaluation
   ```

2. **List Active Experiments**
   ```bash
   python -m scripts.deployment.manage_experiments list
   ```

3. **Show Experiment Details**
   ```bash
   python -m scripts.deployment.manage_experiments show model_v2_evaluation
   ```

4. **End an Experiment**
   ```bash
   python -m scripts.deployment.manage_experiments end model_v2_evaluation --output results.json
   ```

5. **Check All Experiments**
   ```bash
   python -m scripts.deployment.manage_experiments check --output completed.json
   ```

### Python API

1. **Initialize Experiment Manager**
   ```python
   from scripts.deployment.experiment_manager import ExperimentManager
   
   manager = ExperimentManager()
   ```

2. **Create and Run Experiment**
   ```python
   # Create experiment
   test = manager.create_experiment("model_v2_evaluation")
   
   # Record observations
   user_id = "user123"
   variant = test.assign_variant(user_id)
   
   if variant == "control":
       model = load_model("v1.0.0")
   else:
       model = load_model("v2.0.0")
       
   prediction = model.predict(data)
   test.record_observation(
       variant=variant,
       predicted=prediction,
       actual=actual_value,
       response_time=response_time
   )
   ```

3. **Monitor Progress**
   ```python
   # Get statistics
   stats = test.get_statistics()
   
   # Check significance
   significance = test.calculate_significance()
   
   # End experiment
   results = manager.end_experiment("model_v2_evaluation")
   ```

## Statistical Analysis

The system performs several statistical tests:

1. **Prediction Accuracy**
   - Mean Absolute Error (MAE)
   - Root Mean Square Error (RMSE)
   - Statistical significance using t-tests

2. **Response Time**
   - Mean response time
   - 95th percentile latency
   - Statistical significance of differences

3. **Sample Size**
   - Minimum required samples per variant
   - Confidence level adjustments
   - Power analysis

## Best Practices

1. **Experiment Design**
   - Start with a small traffic percentage for new variants
   - Use meaningful success criteria
   - Set appropriate minimum sample sizes

2. **Monitoring**
   - Regularly check experiment progress
   - Monitor for unexpected behavior
   - Document observations

3. **Analysis**
   - Wait for sufficient data before drawing conclusions
   - Consider multiple metrics
   - Account for business impact

4. **Implementation**
   - Use consistent user IDs for variant assignment
   - Handle edge cases gracefully
   - Log all relevant metrics

## Integration with Model Deployment

The A/B testing system integrates with the model deployment pipeline:

1. **Automated Deployment**
   ```python
   from scripts.deployment import deploy_model
   
   # Deploy winning variant
   if results["winner"] == "treatment":
       deploy_model(version="v2.0.0")
   ```

2. **Rollback Handling**
   ```python
   # Automatic rollback if issues detected
   if error_rate > threshold:
       manager.end_experiment("model_v2_evaluation")
       deploy_model(version="v1.0.0")  # Rollback
   ```

## Monitoring and Alerts

The system includes monitoring capabilities:

1. **Traffic Monitoring**
   - Minimum requests per hour
   - Traffic split adherence
   - Error rate thresholds

2. **Performance Alerts**
   - Response time degradation
   - Error rate spikes
   - Sample size notifications

3. **Completion Criteria**
   - Maximum duration reached
   - Statistical significance achieved
   - Success criteria met

## Troubleshooting

Common issues and solutions:

1. **Uneven Traffic Split**
   - Check user ID consistency
   - Verify hash distribution
   - Monitor assignment rates

2. **Inconclusive Results**
   - Increase sample size
   - Adjust confidence levels
   - Review success criteria

3. **Performance Issues**
   - Check resource utilization
   - Monitor response times
   - Verify model loading

## Future Improvements

Planned enhancements:

1. **Multi-armed Bandit**
   - Dynamic traffic allocation
   - Thompson sampling
   - Epsilon-greedy strategies

2. **Advanced Analytics**
   - Bayesian analysis
   - Sequential testing
   - Covariate analysis

3. **Enhanced Monitoring**
   - Real-time dashboards
   - Automated reports
   - Alert integration 