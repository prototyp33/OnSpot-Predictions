# Contributing to OnSpot Predictive Model

Thank you for your interest in contributing to the OnSpot Predictive Model! This document provides guidelines and instructions for contributing to the project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Pull Request Process](#pull-request-process)
- [Continuous Integration](#continuous-integration)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/OnSpot_Predictive_Model.git`
3. Set up a development environment: See [Developer Setup Guide](docs/developer_guide/setting_up.md)
4. Create a new branch for your work: `git checkout -b feature/your-feature-name`

## Development Process

1. Check the [Issues](https://github.com/yourusername/OnSpot_Predictive_Model/issues) for tasks to work on
2. Assign yourself to an issue or create a new one
3. Follow the [branching strategy](#branching-strategy)
4. Make your changes in your branch
5. Write or update tests as needed
6. Ensure all tests pass locally
7. Submit a pull request

### Branching Strategy

- `main` - Production-ready code
- `develop` - Integration branch for features
- `feature/name` - Feature development
- `bugfix/name` - Bug fixes
- `release/version` - Release preparation
- `hotfix/name` - Urgent production fixes

## Pull Request Process

1. Update the README.md or documentation with details of changes
2. Update the version numbers in version files, if applicable
3. Ensure all CI checks pass
4. Get at least one code review approval
5. Your PR will be merged by a maintainer

## Continuous Integration

We use GitHub Actions for continuous integration. Every pull request and push goes through automated checks.

### Understanding CI Workflows

#### Main Test Suite (`ci.yml`)
- Runs unit, integration, and property-based tests
- Performs code linting
- Generates coverage reports
- Runs on multiple Python versions

**What to do if it fails:**
1. Click on the failing workflow for details
2. Check the specific step that failed
3. Fix the issues in your branch
4. Push the changes to automatically trigger a new run

#### Data Validation (`data-validation.yml`)
Automatically validates new or changed data files against schema definitions.

**What to do if it fails:**
1. Check validation errors in the workflow output
2. Fix the data issues or update the validation rules
3. Rerun the workflow

#### Security Scanning (`codeql.yml`)
Performs static code analysis to detect security issues.

**What to do if it fails:**
1. Review the security alerts
2. Address the identified vulnerabilities
3. Rerun the scan

### Manually Triggering Workflows

Some workflows can be triggered manually:

1. Go to the Actions tab in GitHub
2. Select the workflow you want to run
3. Click "Run workflow"
4. Configure any input parameters
5. Click "Run workflow" button

## Coding Standards

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code
- Use type hints in function signatures
- Format code with Black (100 character line length)
- Sort imports with isort
- Run flake8 before committing

Automated formatting:
```bash
# Run the pre-commit hooks
pre-commit run --all-files
```

## Testing Guidelines

See our [Testing Strategy](tests/README.md) for details.

- Write unit tests for all new code
- Keep test coverage high (aim for >80%)
- Write integration tests for features that span multiple components
- Include property-based tests for validation logic

Running tests:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=onspot

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/property/
pytest tests/performance/
```

## Documentation

- Update or create documentation for all changes
- Follow the existing documentation style
- Use docstrings for all functions and classes
- Update the README.md when needed

Building documentation:
```bash
# Build the documentation
mkdocs build

# Serve documentation locally
mkdocs serve
```

## Questions?

Feel free to create an issue or contact the maintainers if you have questions.

Thank you for contributing to the OnSpot Predictive Model! 