resource "aws_s3_bucket" "models" {
  bucket = "${var.project_name}-${var.environment}-models"
  
  tags = merge(
    var.tags,
    {
      Name = "${var.project_name}-${var.environment}-models"
    }
  )
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "models" {
  bucket = aws_s3_bucket.models.id
  
  rule {
    id     = "archive-old-versions"
    status = "Enabled"
    
    transition {
      days          = 90
      storage_class = "STANDARD_IA"
    }
    
    noncurrent_version_transition {
      noncurrent_days = 30
      storage_class   = "STANDARD_IA"
    }
    
    noncurrent_version_expiration {
      noncurrent_days = 365
    }
  }
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket = aws_s3_bucket.models.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# IAM Policy for ECS Tasks to access S3
resource "aws_iam_policy" "s3_access" {
  name        = "${var.project_name}-${var.environment}-s3-access"
  description = "Policy for ECS tasks to access model files in S3"
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:DeleteObject"
        ]
        Resource = [
          aws_s3_bucket.models.arn,
          "${aws_s3_bucket.models.arn}/*"
        ]
      }
    ]
  })
}

# Bucket for monitoring data
resource "aws_s3_bucket" "monitoring" {
  bucket = "${var.project_name}-${var.environment}-monitoring"
  
  tags = merge(
    var.tags,
    {
      Name = "${var.project_name}-${var.environment}-monitoring"
    }
  )
}

resource "aws_s3_bucket_versioning" "monitoring" {
  bucket = aws_s3_bucket.monitoring.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "monitoring" {
  bucket = aws_s3_bucket.monitoring.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "monitoring" {
  bucket = aws_s3_bucket.monitoring.id
  
  rule {
    id     = "cleanup-old-data"
    status = "Enabled"
    
    expiration {
      days = 90
    }
    
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "monitoring" {
  bucket = aws_s3_bucket.monitoring.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
} 