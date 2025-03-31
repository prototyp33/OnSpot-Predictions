resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "${var.project_name}-${var.environment}-dashboard"
  
  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        
        properties = {
          metrics = [
            ["AWS/ECS", "CPUUtilization", "ClusterName", var.ecs_cluster_name],
            [".", "MemoryUtilization", ".", "."]
          ]
          period = 300
          stat   = "Average"
          region = var.aws_region
          title  = "ECS Cluster Utilization"
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6
        
        properties = {
          metrics = [
            ["AWS/ApplicationELB", "RequestCount", "LoadBalancer", var.alb_name],
            [".", "HTTPCode_Target_4XX_Count", ".", "."],
            [".", "HTTPCode_Target_5XX_Count", ".", "."]
          ]
          period = 300
          stat   = "Sum"
          region = var.aws_region
          title  = "ALB Requests"
        }
      }
    ]
  })
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "cpu_high" {
  alarm_name          = "${var.project_name}-${var.environment}-cpu-utilization-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name        = "CPUUtilization"
  namespace          = "AWS/ECS"
  period             = "300"
  statistic          = "Average"
  threshold          = "85"
  alarm_description  = "Average ECS CPU utilization is too high"
  alarm_actions      = [aws_sns_topic.alerts.arn]
  
  dimensions = {
    ClusterName = var.ecs_cluster_name
  }
  
  tags = var.tags
}

resource "aws_cloudwatch_metric_alarm" "memory_high" {
  alarm_name          = "${var.project_name}-${var.environment}-memory-utilization-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name        = "MemoryUtilization"
  namespace          = "AWS/ECS"
  period             = "300"
  statistic          = "Average"
  threshold          = "85"
  alarm_description  = "Average ECS memory utilization is too high"
  alarm_actions      = [aws_sns_topic.alerts.arn]
  
  dimensions = {
    ClusterName = var.ecs_cluster_name
  }
  
  tags = var.tags
}

resource "aws_cloudwatch_metric_alarm" "alb_5xx" {
  alarm_name          = "${var.project_name}-${var.environment}-alb-5xx-error-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name        = "HTTPCode_Target_5XX_Count"
  namespace          = "AWS/ApplicationELB"
  period             = "300"
  statistic          = "Sum"
  threshold          = "10"
  alarm_description  = "Number of 5XX errors is too high"
  alarm_actions      = [aws_sns_topic.alerts.arn]
  
  dimensions = {
    LoadBalancer = var.alb_name
  }
  
  tags = var.tags
}

# SNS Topic for Alerts
resource "aws_sns_topic" "alerts" {
  name = "${var.project_name}-${var.environment}-alerts"
  
  tags = var.tags
}

# CloudWatch Log Metric Filters
resource "aws_cloudwatch_log_metric_filter" "model_drift" {
  name           = "${var.project_name}-${var.environment}-model-drift"
  pattern        = "[timestamp, level=ERROR, message=*model drift*]"
  log_group_name = "/ecs/${var.project_name}-${var.environment}"
  
  metric_transformation {
    name      = "ModelDriftErrors"
    namespace = "${var.project_name}/${var.environment}"
    value     = "1"
  }
}

resource "aws_cloudwatch_log_metric_filter" "prediction_errors" {
  name           = "${var.project_name}-${var.environment}-prediction-errors"
  pattern        = "[timestamp, level=ERROR, message=*prediction error*]"
  log_group_name = "/ecs/${var.project_name}-${var.environment}"
  
  metric_transformation {
    name      = "PredictionErrors"
    namespace = "${var.project_name}/${var.environment}"
    value     = "1"
  }
}

# Alarms for Custom Metrics
resource "aws_cloudwatch_metric_alarm" "model_drift_high" {
  alarm_name          = "${var.project_name}-${var.environment}-model-drift-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name        = "ModelDriftErrors"
  namespace          = "${var.project_name}/${var.environment}"
  period             = "300"
  statistic          = "Sum"
  threshold          = "5"
  alarm_description  = "Number of model drift errors is too high"
  alarm_actions      = [aws_sns_topic.alerts.arn]
  
  tags = var.tags
}

resource "aws_cloudwatch_metric_alarm" "prediction_errors_high" {
  alarm_name          = "${var.project_name}-${var.environment}-prediction-errors-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name        = "PredictionErrors"
  namespace          = "${var.project_name}/${var.environment}"
  period             = "300"
  statistic          = "Sum"
  threshold          = "10"
  alarm_description  = "Number of prediction errors is too high"
  alarm_actions      = [aws_sns_topic.alerts.arn]
  
  tags = var.tags
} 