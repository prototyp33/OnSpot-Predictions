variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name (e.g., dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "onspot"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["us-west-2a", "us-west-2b", "us-west-2c"]
}

variable "model_registry_path" {
  description = "Path to model registry in S3"
  type        = string
  default     = "s3://onspot-models/registry"
}

variable "production_models_path" {
  description = "Path to production models in S3"
  type        = string
  default     = "s3://onspot-models/production"
}

variable "prediction_api_desired_count" {
  description = "Desired number of prediction API tasks"
  type        = number
  default     = 2
}

variable "gateway_desired_count" {
  description = "Desired number of gateway tasks"
  type        = number
  default     = 2
}

variable "enable_monitoring" {
  description = "Enable CloudWatch monitoring"
  type        = bool
  default     = true
}

variable "tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default     = {
    Project     = "OnSpot"
    ManagedBy   = "Terraform"
  }
} 