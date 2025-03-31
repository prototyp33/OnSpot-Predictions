aws_region = "eu-west-1"
environment = "prod"
project_name = "onspot"
vpc_cidr = "10.0.0.0/16"
availability_zones = ["eu-west-1a", "eu-west-1b", "eu-west-1c"]

model_registry_path = "s3://onspot-models/registry"
production_models_path = "s3://onspot-models/production"

prediction_api_desired_count = 2
gateway_desired_count = 2
enable_monitoring = true

tags = {
  Project     = "OnSpot"
  ManagedBy   = "Terraform"
  Environment = "prod"
} 