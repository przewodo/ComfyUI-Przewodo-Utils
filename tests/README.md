# Testing Guide for ComfyUI-Przewodo-Utils

This directory contains comprehensive tests for the ComfyUI-Przewodo-Utils custom nodes.

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Shared fixtures and configuration
├── unit/                       # Unit tests for individual components
│   ├── test_wan_image_to_video_advanced_sampler.py
│   ├── test_cache_manager.py
│   └── test_utility_nodes.py
├── integration/                # Integration tests
│   └── test_wan_integration.py
├── workflows/                  # End-to-end workflow tests
│   └── test_complete_workflows.py
└── fixtures/                   # Test data and utilities
    └── test_fixtures.py
```

## Quick Start

### 1. Install Test Dependencies

```bash
# Install test requirements
pip install -r test-requirements.txt

# Or use the test runner to install dependencies
python run_tests.py --install-deps
```

### 2. Run Tests

```bash
# Run all tests
python run_tests.py

# Run specific test types
python run_tests.py unit          # Unit tests only
python run_tests.py integration   # Integration tests only  
python run_tests.py workflow      # Workflow tests only

# Run with coverage
python run_tests.py all --coverage

# Run with verbose output
python run_tests.py all --verbose

# Include GPU tests (requires CUDA)
python run_tests.py all --gpu

# Include slow tests
python run_tests.py all --slow
```

### 3. Alternative: Direct pytest

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/unit/test_wan_image_to_video_advanced_sampler.py

# Run with specific markers
pytest -m "unit and not gpu"
pytest -m "integration"
pytest -m "workflow"

# Run with coverage
pytest --cov=. --cov-report=html tests/
```

## Test Categories

### Unit Tests (`tests/unit/`)
- Test individual functions and methods in isolation
- Use mocks for external dependencies
- Fast execution, no GPU required
- Focus on correctness of individual components

### Integration Tests (`tests/integration/`)
- Test interactions between components
- Test with real ComfyUI node structure
- May require GPU for some tests
- Verify end-to-end functionality

### Workflow Tests (`tests/workflows/`)
- Test complete ComfyUI workflows
- Simulate real user scenarios
- Test node chaining and data flow
- Performance and memory efficiency tests

## Test Markers

Use pytest markers to run specific subsets of tests:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests  
- `@pytest.mark.workflow` - Workflow tests
- `@pytest.mark.gpu` - Tests requiring CUDA
- `@pytest.mark.slow` - Tests that take longer to run
- `@pytest.mark.memory_intensive` - Tests using significant memory

Examples:
```bash
# Run only unit tests
pytest -m unit

# Run tests that don't require GPU
pytest -m "not gpu"

# Run fast tests only
pytest -m "not slow and not memory_intensive"
```

## Writing New Tests

### Unit Test Example

```python
import pytest
from unittest.mock import Mock, patch

@pytest.mark.unit
class TestMyNode:
    def test_basic_functionality(self):
        from my_node import MyNode
        node = MyNode()
        result = node.run(input_param="test")
        assert result == expected_output
```

### Integration Test Example

```python
@pytest.mark.integration
class TestNodeIntegration:
    def test_with_dependencies(self, mock_dependencies):
        # Test with mocked ComfyUI dependencies
        pass
```

### Workflow Test Example

```python
@pytest.mark.workflow
class TestCompleteWorkflow:
    def test_end_to_end(self, sample_workflow):
        # Test complete workflow execution
        pass
```

## Test Configuration

### Environment Variables

Set these environment variables to customize test behavior:

```bash
export COMFYUI_TEST_GPU=1          # Enable GPU tests
export COMFYUI_TEST_SLOW=1         # Enable slow tests  
export COMFYUI_TEST_MODELS_PATH=/path/to/test/models
```

### Pytest Configuration

Tests are configured via `pytest.ini` in the project root:

- Coverage settings
- Test discovery patterns
- Marker definitions
- Warning filters

### Fixtures

Common test fixtures are defined in `conftest.py`:

- `test_device` - CPU/GPU device for tests
- `sample_image_tensor` - Sample image data
- `sample_latent_tensor` - Sample latent data
- `mock_comfy_nodes` - Mocked ComfyUI nodes
- `basic_node_inputs` - Default node parameters

## Continuous Integration

For CI/CD pipelines, use:

```bash
# Fast test suite (no GPU, no slow tests)
pytest -m "not gpu and not slow" tests/

# Full test suite with coverage
pytest --cov=. --cov-report=xml -m "not gpu" tests/
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure ComfyUI dependencies are available or properly mocked
2. **CUDA Errors**: Skip GPU tests if CUDA not available: `pytest -m "not gpu"`
3. **Memory Issues**: Run tests sequentially: `pytest --maxfail=1`
4. **Slow Tests**: Skip slow tests: `pytest -m "not slow"`

### Debug Mode

Run tests with extra debugging:

```bash
# Verbose output with full tracebacks
pytest -vvv --tb=long tests/

# Stop on first failure
pytest -x tests/

# Drop into debugger on failure
pytest --pdb tests/
```

### Coverage Reports

Generate detailed coverage reports:

```bash
# HTML report
pytest --cov=. --cov-report=html tests/
# View: open htmlcov/index.html

# Terminal report with missing lines
pytest --cov=. --cov-report=term-missing tests/

# XML report (for CI)
pytest --cov=. --cov-report=xml tests/
```

## Best Practices

1. **Isolation**: Each test should be independent and not rely on other tests
2. **Mocking**: Mock external dependencies to ensure test reliability
3. **Parametrization**: Use `@pytest.mark.parametrize` for testing multiple inputs
4. **Fixtures**: Use fixtures for common test data and setup
5. **Markers**: Use appropriate markers to categorize tests
6. **Documentation**: Document complex test scenarios and expected behaviors
7. **Performance**: Keep unit tests fast, put slow tests in integration/workflow categories

## Contributing

When adding new nodes or features:

1. Add unit tests for core functionality
2. Add integration tests for ComfyUI interaction
3. Add workflow tests for user scenarios
4. Update this README if needed
5. Ensure all tests pass before submitting PRs

For questions or issues with testing, please create an issue in the repository.
