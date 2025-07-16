# Netra SDK Tests

This directory contains comprehensive unit tests for the Netra SDK, following global standards for SDK testing.

## Test Structure

```
tests/
├── __init__.py                 # Tests package initialization
├── conftest.py                 # Shared fixtures and configuration
├── test_netra_init.py         # Tests for netra/__init__.py
└── README.md                  # This file
```

## Running Tests

### Run All Tests
```bash
poetry run pytest
```

### Run Specific Test File
```bash
poetry run pytest tests/test_netra_init.py
```

### Run with Verbose Output
```bash
poetry run pytest -v
```

### Run with Coverage (if pytest-cov is installed)
```bash
poetry run pytest --cov=netra --cov-report=html
```

## Test Categories

Tests are organized using pytest markers:

- `@pytest.mark.unit` - Unit tests (default)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.thread_safety` - Thread safety tests
- `@pytest.mark.slow` - Slow running tests

### Run Specific Test Categories
```bash
# Run only unit tests
poetry run pytest -m unit

# Run only integration tests
poetry run pytest -m integration

# Run only thread safety tests
poetry run pytest -m thread_safety
```

## Test Coverage

The tests aim for comprehensive coverage of:

### `test_netra_init.py` - Main SDK Class Tests

#### TestNetraInitialization
- ✅ Initial state verification
- ✅ Successful initialization with default parameters
- ✅ Successful initialization with custom parameters
- ✅ Multiple initialization attempts (idempotency)
- ✅ Thread safety during initialization
- ✅ Logging behavior

#### TestNetraSessionManagement
- ✅ Session ID management
- ✅ User ID management
- ✅ Tenant ID management
- ✅ Custom attributes with various data types
- ✅ Custom events with complex attributes

#### TestNetraClassProperties
- ✅ Class attribute verification
- ✅ Initial state consistency
- ✅ Thread lock functionality

#### TestNetraIntegration
- ✅ Complete workflow testing
- ✅ Session methods without initialization

#### TestNetraErrorHandling
- ✅ Tracer initialization failures
- ✅ Instrumentation failures
- ✅ Session management exceptions
- ✅ Custom event exceptions

## Testing Standards

### Code Quality
- **100% test coverage** for critical paths
- **Comprehensive edge case testing**
- **Thread safety verification**
- **Error handling validation**
- **Mock isolation** for external dependencies

### Test Organization
- **Clear test class grouping** by functionality
- **Descriptive test method names** following `test_<action>_<expected_result>` pattern
- **Comprehensive docstrings** for all test methods
- **Setup/teardown methods** for clean test isolation

### Mocking Strategy
- **Mock external dependencies** (Tracer, Config, SessionManager)
- **Preserve original behavior** where possible
- **Verify mock interactions** with proper assertions
- **Use fixtures** for reusable mock configurations

### Assertions
- **Specific assertions** with clear error messages
- **State verification** after operations
- **Mock call verification** with exact parameters
- **Exception testing** with proper context managers

## Fixtures (conftest.py)

### Automatic Fixtures
- `reset_netra_state` - Automatically resets SDK state before/after each test

### Mock Fixtures
- `mock_config` - Mock Config class
- `mock_tracer` - Mock Tracer class
- `mock_init_instrumentations` - Mock instrumentation function
- `mock_session_manager` - Mock SessionManager class

### Data Fixtures
- `sample_config_params` - Sample configuration parameters
- `sample_session_data` - Sample session data
- `sample_custom_attributes` - Sample custom attributes
- `sample_event_data` - Sample event data

### Utility Fixtures
- `clean_environment` - Clean environment variables
- `thread_barrier` - Threading synchronization
- `mock_logger` - Mock logger for testing logging behavior

## Best Practices

### Writing New Tests
1. **Use descriptive names** that explain what is being tested
2. **Follow the AAA pattern** (Arrange, Act, Assert)
3. **Test one thing at a time** - single responsibility per test
4. **Use appropriate fixtures** to reduce code duplication
5. **Mock external dependencies** to ensure test isolation
6. **Include both positive and negative test cases**
7. **Test edge cases and error conditions**

### Test Maintenance
1. **Keep tests simple and readable**
2. **Update tests when code changes**
3. **Remove obsolete tests**
4. **Refactor common test code into fixtures**
5. **Document complex test scenarios**

## Contributing

When adding new functionality to the SDK:

1. **Write tests first** (TDD approach recommended)
2. **Ensure 100% coverage** for new code
3. **Add integration tests** for complex workflows
4. **Update this README** if adding new test categories
5. **Run full test suite** before submitting changes

## Continuous Integration

Tests are designed to run in CI/CD environments:

- **No external dependencies** required
- **Deterministic results** through proper mocking
- **Fast execution** through efficient test design
- **Clear failure reporting** with descriptive error messages
