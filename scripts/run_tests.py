#!/usr/bin/env python
"""Run the test suite for the parking simulation."""

import pytest
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run parking simulation tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--coverage', '-c', action='store_true', help='Generate coverage report')
    parser.add_argument('--pattern', '-p', type=str, default='', help='Test pattern to run')
    args = parser.parse_args()
    
    pytest_args = []
    
    if args.verbose:
        pytest_args.append('-v')
    
    if args.coverage:
        pytest_args.extend(['--cov=parking_sim', '--cov-report=html'])
    
    if args.pattern:
        pytest_args.append(args.pattern)
    
    exit_code = pytest.main(pytest_args)
    sys.exit(exit_code)

if __name__ == '__main__':
    main() 