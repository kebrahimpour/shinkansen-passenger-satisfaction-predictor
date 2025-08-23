# Contributing to Shinkansen Passenger Satisfaction Predictor

Thank you for your interest in contributing to this project! We welcome contributions from the community and are pleased to have you aboard.

## 🧪 Testing Matrix

Our CI system tests against the following environments:

| Python Version | OS | uv Version | Status |
|---------------|----|-----------|---------|
| 3.10 | Ubuntu Latest | Latest | ✅ Supported |
| 3.11 | Ubuntu Latest | Latest | ✅ Supported |
| 3.12 | Ubuntu Latest | Latest | ✅ Supported |

### Test Coverage Requirements
- Minimum test coverage: 80%
- All new features must include comprehensive tests
- Tests must pass on all supported Python versions
- Integration tests are required for API endpoints

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- Git

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/shinkansen-passenger-satisfaction-predictor.git
   cd shinkansen-passenger-satisfaction-predictor
   ```

2. **Install uv (if not already installed)**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Set up the development environment**
   ```bash
   uv sync --dev
   ```

4. **Install pre-commit hooks**
   ```bash
   uv run pre-commit install
   ```

5. **Verify installation**
   ```bash
   uv run pytest tests/
   ```

## 🔄 Development Workflow

### Making Changes

1. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Make your changes**
   - Write clean, documented code
   - Follow existing code style and conventions
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests locally**
   ```bash
   # Run all tests
   uv run pytest tests/

   # Run with coverage
   uv run pytest tests/ --cov=src --cov-report=term-missing

   # Run pre-commit checks
   uv run pre-commit run --all-files
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature" # Use conventional commits
   ```

5. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📝 Code Style

We use automated code formatting and linting:

- **Black** for code formatting
- **Flake8** for linting
- **Ruff** for additional linting and import sorting
- **Type hints** are encouraged using `mypy`

All formatting is enforced by pre-commit hooks and CI.

## 🧪 Testing Guidelines

### Writing Tests

1. **Unit Tests**: Test individual functions and classes
   ```python
   # tests/test_predictor.py
   def test_satisfaction_predictor_init():
       predictor = SatisfactionPredictor()
       assert predictor is not None
   ```

2. **Integration Tests**: Test API endpoints and workflows
   ```python
   # tests/test_api.py
   def test_predict_endpoint(client):
       response = client.post("/predict", json={"data": "sample"})
       assert response.status_code == 200
   ```

3. **Fixtures**: Use pytest fixtures for setup
   ```python
   @pytest.fixture
   def sample_data():
       return {"feature1": 1.0, "feature2": 2.0}
   ```

### Running Specific Tests

```bash
# Run specific test file
uv run pytest tests/test_predictor.py

# Run tests with specific marker
uv run pytest -m "not slow"

# Run tests with verbose output
uv run pytest -v
```

## 📋 Pull Request Process

1. **Ensure CI passes**: All tests must pass and code quality checks must succeed
2. **Update documentation**: Update README, docstrings, or other docs as needed
3. **Add changelog entry**: Add a brief description to CHANGELOG.md
4. **Request review**: Tag maintainers for review
5. **Address feedback**: Make requested changes and re-request review

### Pull Request Checklist

- [ ] Tests pass locally (`uv run pytest`)
- [ ] Pre-commit hooks pass (`uv run pre-commit run --all-files`)
- [ ] Code coverage meets requirements (≥80%)
- [ ] Documentation updated (if applicable)
- [ ] Changelog updated (if applicable)
- [ ] Commit messages follow [conventional commits](https://www.conventionalcommits.org/)

## 🐛 Reporting Issues

When reporting issues:

1. Use the provided issue templates
2. Include Python version, OS, and uv version
3. Provide minimal reproducible example
4. Include relevant error messages and stack traces

## 💡 Feature Requests

For new features:

1. Check existing issues to avoid duplicates
2. Describe the use case and benefit
3. Consider implementation complexity
4. Discuss with maintainers before starting work

## 📞 Getting Help

- **Issues**: Use GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub discussions for questions and general discussion
- **Email**: Contact maintainers for security issues

## 🎯 Areas for Contribution

We especially welcome contributions in:

- **Model improvements**: Better algorithms, feature engineering
- **Testing**: Expand test coverage, add edge cases
- **Documentation**: Improve guides, add examples
- **Performance**: Optimize prediction speed, memory usage
- **CI/CD**: Improve build pipeline, add deployment automation

## 📄 Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you are expected to uphold this code.

## 📜 License

By contributing to this project, you agree that your contributions will be licensed under the CC0-1.0 license.

---

Thank you for contributing! 🚅✨
