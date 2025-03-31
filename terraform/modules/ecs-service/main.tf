resource "aws_ecs_task_definition" "service" {
  family                   = "${var.project_name}-${var.environment}-${var.service_name}"
  requires_compatibilities = ["FARGATE"]
  network_mode            = "awsvpc"
  cpu                     = var.cpu
  memory                  = var.memory
  execution_role_arn      = var.task_execution_role_arn
  task_role_arn          = var.task_role_arn
  
  container_definitions = jsonencode([
    {
      name  = var.service_name
      image = "${var.ecr_repository}:latest"
      
      portMappings = [
        {
          containerPort = var.container_port
          protocol      = "tcp"
        }
      ]
      
      environment = [
        for key, value in var.environment_variables : {
          name  = key
          value = value
        }
      ]
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = "/ecs/${var.project_name}-${var.environment}"
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = var.service_name
        }
      }
      
      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:${var.container_port}/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])
  
  tags = var.tags
}

resource "aws_ecs_service" "service" {
  name            = "${var.project_name}-${var.environment}-${var.service_name}"
  cluster         = var.ecs_cluster_id
  task_definition = aws_ecs_task_definition.service.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"
  
  network_configuration {
    subnets          = var.private_subnets
    security_groups  = [aws_security_group.service.id]
    assign_public_ip = false
  }
  
  load_balancer {
    target_group_arn = var.alb_target_group
    container_name   = var.service_name
    container_port   = var.container_port
  }
  
  deployment_controller {
    type = "ECS"
  }
  
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
  
  deployment_configuration {
    deployment_minimum_healthy_percent = 50
    deployment_maximum_percent        = 200
  }
  
  capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight           = 1
    base            = 1
  }
  
  enable_execute_command = true
  
  tags = var.tags
}

resource "aws_security_group" "service" {
  name_prefix = "${var.project_name}-${var.environment}-${var.service_name}-"
  vpc_id      = var.vpc_id
  
  ingress {
    from_port       = var.container_port
    to_port         = var.container_port
    protocol        = "tcp"
    security_groups = [var.alb_security_group_id]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = merge(
    var.tags,
    {
      Name = "${var.project_name}-${var.environment}-${var.service_name}-sg"
    }
  )
}

# Auto Scaling
resource "aws_appautoscaling_target" "service" {
  max_capacity       = var.max_capacity
  min_capacity       = var.min_capacity
  resource_id        = "service/${var.ecs_cluster_name}/${aws_ecs_service.service.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "cpu" {
  name               = "${var.project_name}-${var.environment}-${var.service_name}-cpu"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.service.resource_id
  scalable_dimension = aws_appautoscaling_target.service.scalable_dimension
  service_namespace  = aws_appautoscaling_target.service.service_namespace
  
  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = 70.0
  }
}

resource "aws_appautoscaling_policy" "memory" {
  name               = "${var.project_name}-${var.environment}-${var.service_name}-memory"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.service.resource_id
  scalable_dimension = aws_appautoscaling_target.service.scalable_dimension
  service_namespace  = aws_appautoscaling_target.service.service_namespace
  
  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageMemoryUtilization"
    }
    target_value = 70.0
  }
} 