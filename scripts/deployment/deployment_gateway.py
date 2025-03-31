#!/usr/bin/env python3
"""
Deployment Gateway for OnSpot Predictive Model.

This gateway handles traffic routing for different deployment patterns:
- A/B testing - route traffic to different model versions based on defined rules
- Canary deployment - gradually increase traffic to new model version
- Shadow deployment - route requests to both production and shadow models
- Production - default routing to production models

The gateway acts as a proxy in front of the prediction services.
"""

from flask import Flask, request, jsonify, Response
import os
import logging
import sys
import json
import time
import random
import requests
import threading
from datetime import datetime
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Set up logging
log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Service endpoints
PRODUCTION_API = os.environ.get('PRODUCTION_API', 'http://prediction-api:5000')
SHADOW_API = os.environ.get('SHADOW_API', 'http://shadow-api:5001')
CANARY_API = os.environ.get('CANARY_API', 'http://canary-api:5002')
AB_TEST_VARIANT_B_API = os.environ.get('AB_TEST_VARIANT_B_API', 'http://canary-api:5002')

# Deployment configuration
DEPLOYMENT_STRATEGY = os.environ.get('DEPLOYMENT_STRATEGY', 'production')
CANARY_TRAFFIC_PERCENTAGE = float(os.environ.get('CANARY_TRAFFIC_PERCENTAGE', '10.0'))
SHADOW_TRAFFIC_PERCENTAGE = float(os.environ.get('SHADOW_TRAFFIC_PERCENTAGE', '100.0'))
AB_TEST_B_TRAFFIC_PERCENTAGE = float(os.environ.get('AB_TEST_B_TRAFFIC_PERCENTAGE', '50.0'))

# Metrics
ENABLE_METRICS = os.environ.get('ENABLE_METRICS', 'true').lower() == 'true'
METRICS_PORT = int(os.environ.get('METRICS_PORT', '8001'))

if ENABLE_METRICS:
    # Initialize metrics
    gateway_requests = Counter('gateway_requests_total', 'Total requests processed by gateway', 
                            ['endpoint', 'strategy', 'destination'])
    gateway_errors = Counter('gateway_errors_total', 'Total errors encountered by gateway', 
                            ['endpoint', 'strategy', 'error_type'])
    gateway_latency = Histogram('gateway_latency_seconds', 'Gateway request latency in seconds', 
                                ['endpoint', 'strategy', 'destination'])
    strategy_traffic = Gauge('strategy_traffic_percentage', 'Traffic percentage for deployment strategies', 
                            ['strategy', 'destination'])
    
    # Set initial values for traffic percentages
    strategy_traffic.labels(strategy='canary', destination='canary').set(CANARY_TRAFFIC_PERCENTAGE)
    strategy_traffic.labels(strategy='canary', destination='production').set(100.0 - CANARY_TRAFFIC_PERCENTAGE)
    strategy_traffic.labels(strategy='ab_test', destination='variant_b').set(AB_TEST_B_TRAFFIC_PERCENTAGE)
    strategy_traffic.labels(strategy='ab_test', destination='variant_a').set(100.0 - AB_TEST_B_TRAFFIC_PERCENTAGE)
    strategy_traffic.labels(strategy='shadow', destination='shadow').set(SHADOW_TRAFFIC_PERCENTAGE)
    strategy_traffic.labels(strategy='shadow', destination='production').set(100.0)
    
    # Start metrics server
    start_http_server(METRICS_PORT)
    logger.info(f"Prometheus metrics server started on port {METRICS_PORT}")

def log_metrics(endpoint: str, strategy: str, destination: str, latency: float, error: Optional[str] = None):
    """Log metrics for gateway requests."""
    if not ENABLE_METRICS:
        return
    
    gateway_requests.labels(endpoint=endpoint, strategy=strategy, destination=destination).inc()
    gateway_latency.labels(endpoint=endpoint, strategy=strategy, destination=destination).observe(latency)
    
    if error:
        error_type = type(error).__name__ if isinstance(error, Exception) else 'generic'
        gateway_errors.labels(endpoint=endpoint, strategy=strategy, error_type=error_type).inc()

def update_traffic_percentages():
    """Update traffic percentage gauges based on current settings."""
    if not ENABLE_METRICS:
        return
    
    strategy_traffic.labels(strategy='canary', destination='canary').set(CANARY_TRAFFIC_PERCENTAGE)
    strategy_traffic.labels(strategy='canary', destination='production').set(100.0 - CANARY_TRAFFIC_PERCENTAGE)
    strategy_traffic.labels(strategy='ab_test', destination='variant_b').set(AB_TEST_B_TRAFFIC_PERCENTAGE)
    strategy_traffic.labels(strategy='ab_test', destination='variant_a').set(100.0 - AB_TEST_B_TRAFFIC_PERCENTAGE)
    strategy_traffic.labels(strategy='shadow', destination='shadow').set(SHADOW_TRAFFIC_PERCENTAGE)
    strategy_traffic.labels(strategy='shadow', destination='production').set(100.0)

def should_route_to_canary() -> bool:
    """Determine if request should be routed to canary based on percentage."""
    return random.random() * 100 < CANARY_TRAFFIC_PERCENTAGE

def should_route_to_variant_b() -> bool:
    """Determine if request should be routed to variant B in A/B testing."""
    return random.random() * 100 < AB_TEST_B_TRAFFIC_PERCENTAGE

def should_route_to_shadow() -> bool:
    """Determine if request should be routed to shadow based on percentage."""
    return random.random() * 100 < SHADOW_TRAFFIC_PERCENTAGE

def forward_request(endpoint: str, target_url: str, data: dict = None, params: dict = None, headers: dict = None) -> Tuple[Response, float]:
    """
    Forward a request to the target service.
    
    Args:
        endpoint: The endpoint path
        target_url: The full target URL
        data: Optional request data
        params: Optional query parameters
        headers: Optional request headers
        
    Returns:
        Tuple of (Response, latency)
    """
    start_time = time.time()
    
    # Prepare request
    url = f"{target_url}{endpoint}"
    method = request.method
    request_headers = dict(request.headers)
    
    # Add request ID if not present
    if 'X-Request-ID' not in request_headers:
        request_headers['X-Request-ID'] = str(uuid.uuid4())
    
    # Remove host header to avoid conflicts
    if 'Host' in request_headers:
        del request_headers['Host']
    
    if headers:
        request_headers.update(headers)
    
    # Set content type for POST/PUT requests
    if method in ['POST', 'PUT'] and 'Content-Type' not in request_headers:
        request_headers['Content-Type'] = 'application/json'
    
    # Forward the request
    try:
        if method == 'GET':
            resp = requests.get(url, params=params, headers=request_headers, timeout=30)
        elif method == 'POST':
            resp = requests.post(url, json=data, params=params, headers=request_headers, timeout=30)
        elif method == 'PUT':
            resp = requests.put(url, json=data, params=params, headers=request_headers, timeout=30)
        elif method == 'DELETE':
            resp = requests.delete(url, json=data, params=params, headers=request_headers, timeout=30)
        else:
            logger.error(f"Unsupported method: {method}")
            return Response(
                json.dumps({"error": f"Unsupported method: {method}"}),
                status=400,
                mimetype='application/json'
            ), time.time() - start_time
        
        # Create Flask response
        response = Response(
            resp.content,
            status=resp.status_code,
            mimetype=resp.headers.get('Content-Type', 'application/json')
        )
        
        # Copy headers from target response
        for header, value in resp.headers.items():
            if header.lower() not in ['content-length', 'connection', 'content-encoding']:
                response.headers[header] = value
        
        return response, time.time() - start_time
    
    except requests.RequestException as e:
        logger.error(f"Error forwarding request to {url}: {str(e)}")
        return Response(
            json.dumps({"error": f"Error forwarding request: {str(e)}"}),
            status=500,
            mimetype='application/json'
        ), time.time() - start_time

def send_to_shadow(endpoint: str, data: dict = None, params: dict = None):
    """
    Send request to shadow deployment without waiting for response.
    
    Args:
        endpoint: The endpoint path
        data: Optional request data
        params: Optional query parameters
    """
    url = f"{SHADOW_API}{endpoint}"
    method = request.method
    request_headers = dict(request.headers)
    
    # Add request ID and shadow flag
    request_headers['X-Request-ID'] = str(uuid.uuid4())
    request_headers['X-Shadow-Request'] = 'true'
    
    # Remove host header to avoid conflicts
    if 'Host' in request_headers:
        del request_headers['Host']
    
    # Set content type for POST/PUT requests
    if method in ['POST', 'PUT'] and 'Content-Type' not in request_headers:
        request_headers['Content-Type'] = 'application/json'
    
    def _send_shadow_request():
        try:
            if method == 'GET':
                requests.get(url, params=params, headers=request_headers, timeout=5)
            elif method == 'POST':
                requests.post(url, json=data, params=params, headers=request_headers, timeout=5)
            elif method == 'PUT':
                requests.put(url, json=data, params=params, headers=request_headers, timeout=5)
            elif method == 'DELETE':
                requests.delete(url, json=data, params=params, headers=request_headers, timeout=5)
            
            logger.debug(f"Shadow request sent to {url}")
        
        except requests.RequestException as e:
            logger.error(f"Error sending shadow request to {url}: {str(e)}")
    
    # Send shadow request in background thread
    threading.Thread(target=_send_shadow_request).start()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    health_info = {
        'status': 'healthy',
        'deployment_strategy': DEPLOYMENT_STRATEGY,
        'services': {
            'production_api': check_service_health(PRODUCTION_API),
            'shadow_api': check_service_health(SHADOW_API) if DEPLOYMENT_STRATEGY in ['shadow', 'all'] else 'not_checked',
            'canary_api': check_service_health(CANARY_API) if DEPLOYMENT_STRATEGY in ['canary', 'ab_test', 'all'] else 'not_checked',
        },
        'configuration': {
            'canary_traffic_percentage': CANARY_TRAFFIC_PERCENTAGE,
            'shadow_traffic_percentage': SHADOW_TRAFFIC_PERCENTAGE,
            'ab_test_b_traffic_percentage': AB_TEST_B_TRAFFIC_PERCENTAGE,
        },
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
    }
    
    # Check overall status
    if health_info['services']['production_api'] != 'healthy':
        health_info['status'] = 'degraded'
    
    if DEPLOYMENT_STRATEGY == 'canary' and health_info['services']['canary_api'] != 'healthy':
        health_info['status'] = 'degraded'
    
    if DEPLOYMENT_STRATEGY == 'ab_test' and health_info['services']['canary_api'] != 'healthy':
        health_info['status'] = 'degraded'
    
    return jsonify(health_info)

def check_service_health(service_url: str) -> str:
    """Check health of a service."""
    try:
        resp = requests.get(f"{service_url}/health", timeout=5)
        if resp.status_code == 200:
            return 'healthy'
        return f'unhealthy: status_code={resp.status_code}'
    except requests.RequestException as e:
        return f'unreachable: {str(e)}'

@app.route('/config', methods=['GET', 'POST'])
def manage_config():
    """Get or update gateway configuration."""
    global DEPLOYMENT_STRATEGY, CANARY_TRAFFIC_PERCENTAGE, SHADOW_TRAFFIC_PERCENTAGE, AB_TEST_B_TRAFFIC_PERCENTAGE
    
    if request.method == 'GET':
        # Return current configuration
        config = {
            'deployment_strategy': DEPLOYMENT_STRATEGY,
            'canary_traffic_percentage': CANARY_TRAFFIC_PERCENTAGE,
            'shadow_traffic_percentage': SHADOW_TRAFFIC_PERCENTAGE,
            'ab_test_b_traffic_percentage': AB_TEST_B_TRAFFIC_PERCENTAGE,
            'services': {
                'production_api': PRODUCTION_API,
                'shadow_api': SHADOW_API,
                'canary_api': CANARY_API,
                'ab_test_variant_b_api': AB_TEST_VARIANT_B_API,
            }
        }
        return jsonify(config)
    
    elif request.method == 'POST':
        # Update configuration
        data = request.json
        
        if 'deployment_strategy' in data:
            strategy = data['deployment_strategy']
            if strategy in ['production', 'shadow', 'canary', 'ab_test', 'all']:
                DEPLOYMENT_STRATEGY = strategy
                logger.info(f"Deployment strategy updated to {strategy}")
            else:
                return jsonify({
                    'error': f"Invalid deployment strategy: {strategy}. Must be one of: production, shadow, canary, ab_test, all"
                }), 400
        
        if 'canary_traffic_percentage' in data:
            try:
                percentage = float(data['canary_traffic_percentage'])
                if 0 <= percentage <= 100:
                    CANARY_TRAFFIC_PERCENTAGE = percentage
                    logger.info(f"Canary traffic percentage updated to {percentage}%")
                else:
                    return jsonify({
                        'error': "Canary traffic percentage must be between 0 and 100"
                    }), 400
            except ValueError:
                return jsonify({
                    'error': "Canary traffic percentage must be a number"
                }), 400
        
        if 'shadow_traffic_percentage' in data:
            try:
                percentage = float(data['shadow_traffic_percentage'])
                if 0 <= percentage <= 100:
                    SHADOW_TRAFFIC_PERCENTAGE = percentage
                    logger.info(f"Shadow traffic percentage updated to {percentage}%")
                else:
                    return jsonify({
                        'error': "Shadow traffic percentage must be between 0 and 100"
                    }), 400
            except ValueError:
                return jsonify({
                    'error': "Shadow traffic percentage must be a number"
                }), 400
        
        if 'ab_test_b_traffic_percentage' in data:
            try:
                percentage = float(data['ab_test_b_traffic_percentage'])
                if 0 <= percentage <= 100:
                    AB_TEST_B_TRAFFIC_PERCENTAGE = percentage
                    logger.info(f"A/B test variant B traffic percentage updated to {percentage}%")
                else:
                    return jsonify({
                        'error': "A/B test variant B traffic percentage must be between 0 and 100"
                    }), 400
            except ValueError:
                return jsonify({
                    'error': "A/B test variant B traffic percentage must be a number"
                }), 400
        
        # Update metrics
        update_traffic_percentages()
        
        return jsonify({
            'message': 'Configuration updated successfully',
            'deployment_strategy': DEPLOYMENT_STRATEGY,
            'canary_traffic_percentage': CANARY_TRAFFIC_PERCENTAGE,
            'shadow_traffic_percentage': SHADOW_TRAFFIC_PERCENTAGE,
            'ab_test_b_traffic_percentage': AB_TEST_B_TRAFFIC_PERCENTAGE,
        })

@app.route('/<path:endpoint>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy(endpoint):
    """Main proxy endpoint that handles routing based on deployment strategy."""
    path = f"/{endpoint}"
    data = request.json if request.is_json else None
    params = request.args.to_dict() if request.args else None
    
    # Handle routing based on deployment strategy
    if DEPLOYMENT_STRATEGY == 'production':
        # Route to production only
        response, latency = forward_request(path, PRODUCTION_API, data, params)
        log_metrics(path, 'production', 'production', latency)
        return response
    
    elif DEPLOYMENT_STRATEGY == 'shadow':
        # Route to production and shadow
        response, latency = forward_request(path, PRODUCTION_API, data, params)
        log_metrics(path, 'shadow', 'production', latency)
        
        # Send to shadow if percentage allows
        if should_route_to_shadow():
            send_to_shadow(path, data, params)
            log_metrics(path, 'shadow', 'shadow', 0)  # No latency as we don't wait for response
        
        return response
    
    elif DEPLOYMENT_STRATEGY == 'canary':
        # Route to canary or production based on percentage
        if should_route_to_canary():
            # Route to canary
            response, latency = forward_request(path, CANARY_API, data, params)
            log_metrics(path, 'canary', 'canary', latency)
            return response
        else:
            # Route to production
            response, latency = forward_request(path, PRODUCTION_API, data, params)
            log_metrics(path, 'canary', 'production', latency)
            return response
    
    elif DEPLOYMENT_STRATEGY == 'ab_test':
        # Route to variant A or B based on percentage
        if should_route_to_variant_b():
            # Route to variant B
            response, latency = forward_request(
                path, AB_TEST_VARIANT_B_API, data, params, 
                headers={'X-Variant': 'B'}
            )
            log_metrics(path, 'ab_test', 'variant_b', latency)
            return response
        else:
            # Route to variant A (production)
            response, latency = forward_request(
                path, PRODUCTION_API, data, params,
                headers={'X-Variant': 'A'}
            )
            log_metrics(path, 'ab_test', 'variant_a', latency)
            return response
    
    elif DEPLOYMENT_STRATEGY == 'all':
        # Special mode that routes to all services and compares responses
        # This is for testing and development
        responses = {}
        
        # Get production response
        prod_response, prod_latency = forward_request(path, PRODUCTION_API, data, params)
        log_metrics(path, 'all', 'production', prod_latency)
        
        try:
            prod_data = json.loads(prod_response.get_data(as_text=True))
            responses['production'] = {
                'data': prod_data,
                'latency': prod_latency,
                'status_code': prod_response.status_code
            }
        except:
            responses['production'] = {
                'error': 'Failed to parse response',
                'latency': prod_latency,
                'status_code': prod_response.status_code
            }
        
        # Get shadow response
        shadow_response, shadow_latency = forward_request(path, SHADOW_API, data, params)
        log_metrics(path, 'all', 'shadow', shadow_latency)
        
        try:
            shadow_data = json.loads(shadow_response.get_data(as_text=True))
            responses['shadow'] = {
                'data': shadow_data,
                'latency': shadow_latency,
                'status_code': shadow_response.status_code
            }
        except:
            responses['shadow'] = {
                'error': 'Failed to parse response',
                'latency': shadow_latency,
                'status_code': shadow_response.status_code
            }
        
        # Get canary response
        canary_response, canary_latency = forward_request(path, CANARY_API, data, params)
        log_metrics(path, 'all', 'canary', canary_latency)
        
        try:
            canary_data = json.loads(canary_response.get_data(as_text=True))
            responses['canary'] = {
                'data': canary_data,
                'latency': canary_latency,
                'status_code': canary_response.status_code
            }
        except:
            responses['canary'] = {
                'error': 'Failed to parse response',
                'latency': canary_latency,
                'status_code': canary_response.status_code
            }
        
        # Return comparison results
        return jsonify({
            'responses': responses,
            'request': {
                'path': path,
                'method': request.method,
                'timestamp': datetime.now().isoformat()
            }
        })
    
    else:
        # Invalid strategy, default to production
        logger.error(f"Invalid deployment strategy: {DEPLOYMENT_STRATEGY}")
        response, latency = forward_request(path, PRODUCTION_API, data, params)
        log_metrics(path, 'invalid', 'production', latency)
        return response

if __name__ == '__main__':
    # Log startup info
    logger.info(f"Starting deployment gateway with strategy: {DEPLOYMENT_STRATEGY}")
    logger.info(f"Production API: {PRODUCTION_API}")
    logger.info(f"Shadow API: {SHADOW_API}")
    logger.info(f"Canary API: {CANARY_API}")
    logger.info(f"A/B Test Variant B API: {AB_TEST_VARIANT_B_API}")
    logger.info(f"Canary traffic percentage: {CANARY_TRAFFIC_PERCENTAGE}%")
    logger.info(f"Shadow traffic percentage: {SHADOW_TRAFFIC_PERCENTAGE}%")
    logger.info(f"A/B test variant B traffic percentage: {AB_TEST_B_TRAFFIC_PERCENTAGE}%")
    
    # Start the Flask app
    app.run(debug=False, host='0.0.0.0', port=8080) 