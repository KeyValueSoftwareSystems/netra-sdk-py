# 🤝 Contributing

We welcome contributions from the community! This guide will help you get started with contributing to the Netra SDK.

## 🚀 Getting Started

### Development Setup

You can set up your development environment using the provided setup script:

```bash
./setup_dev.sh
```

This script will:

1. Install all Python dependencies in development mode
2. Set up pre-commit hooks for code quality
3. Configure commit message formatting

### Manual Setup

If you prefer to set up manually:

```bash
# Install dependencies
pip install -e ".[dev,test]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install --install-hooks
pre-commit install --hook-type commit-msg
pre-commit install --hook-type pre-push
```

## 🧪 Testing

Run the test suite with:

```bash
poetry run pytest
```

To run a specific test file:

```bash
poetry run pytest tests/test_netra_init.py
```

### Test Coverage

Generate a test coverage report with:

```bash
poetry run pytest --cov=netra --cov-report=html
```

This creates an `htmlcov` directory with a detailed report.

### Running Specific Test Categories

Tests are organized using `pytest` markers. Run specific categories with:

```bash
# Unit tests
poetry run pytest -m unit

# Integration tests
poetry run pytest -m integration

# Thread safety tests
poetry run pytest -m thread_safety

# Slow tests (run last)
poetry run pytest -m slow
```

## 📝 Commit Message Format

We use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Commit Types

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that do not affect the meaning of the code
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **perf**: A code change that improves performance
- **test**: Adding missing tests or correcting existing tests
- **chore**: Changes to the build process or auxiliary tools

### Examples

```
feat: add support for Claude AI instrumentation
fix(pii): resolve masking issue with nested objects
docs: update installation instructions
```

### Scope

Use the scope to specify the area of change (e.g., `pii`, `instrumentation`, `decorators`).

### Body

Include the motivation for the change and contrast with previous behavior.

### Footer

Use for "BREAKING CHANGE:" or issue references.

## 🔄 Pull Request Process

1. Fork the repository and create your feature branch (`git checkout -b feature/AmazingFeature`)
2. Commit your changes following the commit message guidelines
3. Push to the branch (`git push origin feature/AmazingFeature`)
4. Open a Pull Request

## 📜 Code of Conduct

Please note we have a [Code of Conduct](CODE_OF_CONDUCT.md), please follow it in all your interactions with the project.
