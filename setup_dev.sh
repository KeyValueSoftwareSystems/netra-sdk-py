#!/bin/bash
set -e

echo "Setting up development environment for Combat SDK..."

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -e ".[dev,test]"

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pip install pre-commit
pre-commit install --install-hooks

# Install commit-msg hook for commitizen
pre-commit install --hook-type commit-msg
pre-commit install --hook-type pre-push

echo "✅ Development environment setup complete!"
echo ""
echo "You can now start developing for the Combat SDK."
echo "Run 'pre-commit run --all-files' to verify your setup."
