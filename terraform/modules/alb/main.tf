resource "aws_lb" "main" {
  name               = "${var.project_name}-${var.environment}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets           = var.public_subnets
  
  enable_deletion_protection = true
  
  tags = merge(
    var.tags,
    {
      Name = "${var.project_name}-${var.environment}-alb"
    }
  )
}

# Security Group for ALB
resource "aws_security_group" "alb" {
  name_prefix = "${var.project_name}-${var.environment}-alb-"
  vpc_id      = var.vpc_id
  
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
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
      Name = "${var.project_name}-${var.environment}-alb-sg"
    }
  )
}

# Target Groups
resource "aws_lb_target_group" "prediction_api" {
  name        = "${var.project_name}-${var.environment}-pred-api"
  port        = 5000
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"
  
  health_check {
    enabled             = true
    healthy_threshold   = 3
    interval           = 30
    matcher            = "200"
    path               = "/health"
    port               = "traffic-port"
    protocol           = "HTTP"
    timeout            = 5
    unhealthy_threshold = 3
  }
  
  tags = var.tags
}

resource "aws_lb_target_group" "gateway" {
  name        = "${var.project_name}-${var.environment}-gateway"
  port        = 8080
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"
  
  health_check {
    enabled             = true
    healthy_threshold   = 3
    interval           = 30
    matcher            = "200"
    path               = "/health"
    port               = "traffic-port"
    protocol           = "HTTP"
    timeout            = 5
    unhealthy_threshold = 3
  }
  
  tags = var.tags
}

resource "aws_lb_target_group" "monitoring" {
  name        = "${var.project_name}-${var.environment}-monitoring"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"
  
  health_check {
    enabled             = true
    healthy_threshold   = 3
    interval           = 30
    matcher            = "200"
    path               = "/health"
    port               = "traffic-port"
    protocol           = "HTTP"
    timeout            = 5
    unhealthy_threshold = 3
  }
  
  tags = var.tags
}

# Listeners
resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"
  
  default_action {
    type = "redirect"
    
    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }
}

resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.main.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-2016-08"
  certificate_arn   = var.certificate_arn
  
  default_action {
    type = "fixed-response"
    
    fixed_response {
      content_type = "text/plain"
      message_body = "Not Found"
      status_code  = "404"
    }
  }
}

# Listener Rules
resource "aws_lb_listener_rule" "prediction_api" {
  listener_arn = aws_lb_listener.https.arn
  priority     = 100
  
  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.prediction_api.arn
  }
  
  condition {
    path_pattern {
      values = ["/api/v1/predict*"]
    }
  }
}

resource "aws_lb_listener_rule" "gateway" {
  listener_arn = aws_lb_listener.https.arn
  priority     = 200
  
  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.gateway.arn
  }
  
  condition {
    path_pattern {
      values = ["/gateway/*"]
    }
  }
}

resource "aws_lb_listener_rule" "monitoring" {
  listener_arn = aws_lb_listener.https.arn
  priority     = 300
  
  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.monitoring.arn
  }
  
  condition {
    path_pattern {
      values = ["/monitoring/*", "/metrics/*"]
    }
  }
} 