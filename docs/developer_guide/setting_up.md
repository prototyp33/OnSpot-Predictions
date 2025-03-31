# Setting Up Your Development Environment

This guide will help you set up a development environment for contributing to the OnSpot Predictive Model project.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8+**: The project is developed using Python 3.8 or newer
- **Git**: For version control
- **pip and virtualenv**: For package management and isolation
- **Docker** (optional): For containerized development and testing

## Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/OnSpot_Predictive_Model.git
cd OnSpot_Predictive_Model
```

## Step 2: Set Up Virtual Environment

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Unix/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

## Step 3: Install Dependencies

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install the package in development mode
pip install -e .
```

## Step 4: Set Up Pre-commit Hooks

Pre-commit hooks help ensure code quality by running checks before each commit.

```bash
# Install pre-commit
pip install pre-commit

# Set up the hooks
pre-commit install
```

## Step 5: Install Additional Tools

Consider installing these additional tools for development:

```bash
# Install documentation tools
pip install mkdocs mkdocs-material mkdocstrings

# Install testing tools
pip install pytest pytest-cov

# Install linting tools
pip install flake8 mypy pydocstyle
```

## Step 6: Configure Environment Variables

Create a `.env` file for local development:

```
# .env file
ONSPOT_ENV=dev
SUPABASE_URL=https://your-supabase-url.supabase.co
SUPABASE_KEY=your-supabase-key
```

## Step 7: Set Up Database (Optional)

If you need a local database for development:

### Using Docker

```bash
# Start a PostgreSQL container
docker run --name onspot-postgres -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d postgres
```

### Using Local PostgreSQL

1. Install PostgreSQL for your operating system
2. Create a database:
   ```bash
   createdb onspot_dev
   ```

## Step 8: Run Tests

Verify your setup by running the test suite:

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=onspot
```

## Step 9: Build Documentation

```bash
# Build the documentation
mkdocs build

# Serve the documentation locally
mkdocs serve
```

## Project Structure

The project is organized into the following structure:

```
OnSpot_Predictive_Model/
├── config/                 # Configuration files for different environments
├── data/                   # Data files (not stored in repository)
├── docs/                   # Documentation
├── logs/                   # Log files
├── models/                 # Trained models
├── notebooks/              # Jupyter notebooks for exploration
├── results/                # Results from experiments
│   ├── cross_validation_results/
│   ├── hyperparameter_tuning_results/
│   └── ...
├── scripts/                # Utility scripts
│   ├── data/              # Data processing scripts
│   ├── models/            # Model training scripts
│   ├── pipeline/          # Pipeline scripts
│   ├── monitoring/        # Monitoring scripts
│   ├── api/               # API scripts
│   ├── db/                # Database scripts
│   └── utils/             # Utility scripts
├── tests/                  # Test suite
└── .venv/                  # Virtual environment (not stored in repository)
```

## Code Style and Conventions

Please adhere to the following conventions:

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code style
- Write docstrings in Google style format (see [Docstring Guidelines](../DOCSTRING_GUIDELINES.md))
- Use type hints where appropriate
- Follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages

## Troubleshooting

### Common Issues

#### Package Import Errors

If you encounter import errors, make sure:
- Your virtual environment is activated
- The package is installed in development mode (`pip install -e .`)
- Your `PYTHONPATH` includes the project root

#### Database Connection Issues

- Check that your database is running
- Verify your connection credentials in the `.env` file
- Make sure the database exists and has the correct schema

#### Pre-commit Hooks Failing

- Run `pre-commit run --all-files` to see detailed errors
- Update pre-commit hooks with `pre-commit autoupdate`

### Getting Help

If you need additional help:
- Check the [issue tracker](https://github.com/yourusername/OnSpot_Predictive_Model/issues) for known issues
- Join our community chat
- Reach out to the maintainers 