# Combat SDK

## Development Setup

To set up your development environment for the Combat SDK, run the provided setup script:

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

## Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```text
<type>[optional scope]: <description>

[optional body]

[optional footer]
```

- **`type`** must be one of: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`.
- **`scope`** is optional: a label indicating a section of the code (e.g. `api`, `cli`, `auth`).
- **Short summary** must be written in the imperative (“add feature”, not “added feature”).
- **Body** (if present) should explain _why_ and _what_, not _how_.
- **Footer** can be used for “BREAKING CHANGE:” or issue references.

---
