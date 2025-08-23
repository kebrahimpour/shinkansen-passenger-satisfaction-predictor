# Shinkansen Passenger Satisfaction Predictor

[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](http://creativecommons.org/publicdomain/zero/1.0/) [![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)](https://www.python.org/downloads/releases/3.10.0/) [![Machine Learning](https://img.shields.io/badge/ML-scikit--learn-orange)](https://scikit-learn.org/) [![Hackathon](https://img.shields.io/badge/project-hackathon-brightgreen)](https://github.com/kebrahimpour/shinkansen-passenger-satisfaction-predictor) [![uv](https://img.shields.io/badge/package_manager-uv-blue)](https://github.com/astral-sh/uv) [![CI - Lint](https://github.com/kebrahimpour/shinkansen-passenger-satisfaction-predictor/workflows/Lint/badge.svg)](https://github.com/kebrahimpour/shinkansen-passenger-satisfaction-predictor/actions/workflows/lint.yml) [![CI - Test](https://github.com/kebrahimpour/shinkansen-passenger-satisfaction-predictor/workflows/Test/badge.svg)](https://github.com/kebrahimpour/shinkansen-passenger-satisfaction-predictor/actions/workflows/test.yml) [![CI - Build](https://github.com/kebrahimpour/shinkansen-passenger-satisfaction-predictor/workflows/Build/badge.svg)](https://github.com/kebrahimpour/shinkansen-passenger-satisfaction-predictor/actions/workflows/build.yml)

## 🚅 Project Overview

This machine learning project predicts passenger satisfaction on Japan's Shinkansen (bullet train) system using travel and survey data. Developed as part of the **Shinkansen Travel Experience Hackathon**, this solution combines data science, embedded systems insights, and functional safety principles to create a comprehensive passenger satisfaction prediction model.

**⚠️ IMPORTANT: This project is for NON-COMMERCIAL USE ONLY**

## 🎯 Hackathon Context

This project was created for the Shinkansen Travel Experience hackathon competition, focusing on:

- Predicting passenger satisfaction based on travel patterns
- Analyzing survey data and travel metrics
- Implementing machine learning models for real-time predictions
- Considering embedded systems and functional safety aspects of railway operations

## ✨ Key Features

- **Predictive Modeling**: Advanced ML algorithms to predict passenger satisfaction
- **Multi-source Data Integration**: Combines travel data, survey responses, and operational metrics
- **Real-time Analysis**: Capable of processing live data streams
- **Embedded Systems Ready**: Lightweight models suitable for on-board systems
- **Functional Safety Compliance**: Adheres to railway industry safety standards
- **Interactive Visualizations**: Comprehensive data analysis and result visualization

## 🔧 Technical Stack

- **Programming Language**: Python 3.10+
- **Package Manager**: uv (modern Python package and project manager)
- **Machine Learning**: scikit-learn, pandas, numpy
- **Data Visualization**: matplotlib, seaborn, plotly
- **Data Processing**: pandas, numpy
- **Model Deployment**: FastAPI (for API endpoints)
- **Development Tools**: pytest, black, flake8

## 🧪 Tested Environments

This project is continuously tested against the following environments:

| Python Version | OS | Status |
|---------------|----|---------|
| 3.10 | Ubuntu Latest | ✅ Supported |
| 3.11 | Ubuntu Latest | ✅ Supported |
| 3.12 | Ubuntu Latest | ✅ Supported |

### CI/CD Pipeline

Our continuous integration includes:
- **Linting**: Code quality checks with flake8, ruff, and black
- **Testing**: Comprehensive test suite with pytest and coverage reporting
- **Pre-commit Hooks**: Automated code formatting and quality checks
- **Multi-version Testing**: Ensuring compatibility across Python versions

## 🤝 Contributing

We welcome contributions from the community! Whether you're fixing bugs, adding features, improving documentation, or enhancing tests, your help is appreciated.

### Quick Start for Contributors

1. **Fork the repository** and clone your fork
2. **Install dependencies** with `uv sync --dev`
3. **Set up pre-commit** with `uv run pre-commit install`
4. **Run tests** with `uv run pytest tests/`
5. **Make your changes** and create a pull request

### Contribution Guidelines

Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for detailed information on:
- Development setup and workflow
- Testing requirements and guidelines
- Code style and quality standards
- Pull request process
- Areas where we need help

### Areas We Need Help With

- 🧠 **Model Improvements**: Better algorithms and feature engineering
- 🧪 **Testing**: Expanding test coverage and adding edge cases
- 📖 **Documentation**: Improving guides and adding examples
- ⚡ **Performance**: Optimizing prediction speed and memory usage
- 🔄 **CI/CD**: Enhancing build pipeline and deployment automation

---

*Ready for community contributions! 🚅✨*
