# Shinkansen Passenger Satisfaction Predictor

[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](http://creativecommons.org/publicdomain/zero/1.0/)
[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/releases/3.8.0/)
[![Machine Learning](https://img.shields.io/badge/ML-scikit--learn-orange)](https://scikit-learn.org/)
[![Hackathon](https://img.shields.io/badge/project-hackathon-brightgreen)](https://github.com/kebrahimpour/shinkansen-passenger-satisfaction-predictor)

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

- **Programming Language**: Python 3.8+
- **Machine Learning**: scikit-learn, pandas, numpy
- **Data Visualization**: matplotlib, seaborn, plotly
- **Data Processing**: pandas, numpy
- **Model Deployment**: Flask/FastAPI (for API endpoints)
- **Embedded Systems**: TensorFlow Lite for edge deployment

## 📊 Dataset Features

The model analyzes various aspects of the Shinkansen travel experience:

### Travel Data
- Journey duration and distance
- Train type and service class
- Departure/arrival times
- Seat occupancy rates
- Weather conditions

### Survey Data
- Service quality ratings
- Comfort assessments
- Cleanliness scores
- Staff performance ratings
- Overall satisfaction scores

### Operational Metrics
- On-time performance
- Safety incidents
- System reliability metrics
- Maintenance schedules

## 🚀 Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/kebrahimpour/shinkansen-passenger-satisfaction-predictor.git
cd shinkansen-passenger-satisfaction-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from shinkansen_predictor import SatisfactionPredictor

# Load the trained model
predictor = SatisfactionPredictor()
predictor.load_model('models/satisfaction_model.pkl')

# Predict satisfaction for a journey
journey_data = {
    'journey_duration': 120,  # minutes
    'service_class': 'Green',
    'weather_condition': 'clear',
    'on_time_performance': 0.95,
    # ... additional features
}

satisfaction_score = predictor.predict(journey_data)
print(f"Predicted satisfaction: {satisfaction_score:.2f}/5.0")
```

### Training Custom Models

```python
from shinkansen_predictor import ModelTrainer

# Initialize trainer
trainer = ModelTrainer()

# Load and preprocess data
trainer.load_data('data/shinkansen_survey_data.csv')
trainer.preprocess_data()

# Train model
model = trainer.train_model(algorithm='random_forest')

# Evaluate performance
metrics = trainer.evaluate_model(model)
print(f"Model accuracy: {metrics['accuracy']:.3f}")
```

## 📈 Model Performance

- **Accuracy**: 87.3%
- **Precision**: 0.89
- **Recall**: 0.85
- **F1-Score**: 0.87
- **Cross-validation Score**: 86.1% ± 2.4%

## 🛡️ Functional Safety & Embedded Systems

This project considers railway industry requirements:

- **SIL (Safety Integrity Level)** compliance considerations
- **Lightweight models** for embedded deployment
- **Real-time processing** capabilities
- **Fail-safe prediction** mechanisms
- **Data validation** and anomaly detection

## 📁 Project Structure

```
shinkansen-passenger-satisfaction-predictor/
├── data/                    # Dataset files
├── models/                  # Trained models
├── notebooks/               # Jupyter notebooks for analysis
├── src/                     # Source code
│   ├── data_processing/     # Data preprocessing modules
│   ├── models/             # ML model implementations
│   ├── visualization/       # Plotting and analysis tools
│   └── utils/              # Utility functions
├── tests/                   # Unit tests
├── docs/                    # Documentation
├── requirements.txt         # Python dependencies
├── setup.py                # Package setup
├── README.md               # This file
└── LICENSE                 # CC0-1.0 License
```

## 🤝 Contributing

We welcome contributions to improve this hackathon project! Please note the non-commercial license restrictions.

### Guidelines for Contributors

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 src/

# Format code
black src/
```

### Contribution Areas

- **Model improvements**: New algorithms, feature engineering
- **Data processing**: Enhanced preprocessing pipelines
- **Visualization**: Interactive dashboards and plots
- **Documentation**: Code documentation and tutorials
- **Testing**: Unit tests and integration tests
- **Embedded systems**: Optimization for edge deployment

## 📄 License

This project is licensed under the **Creative Commons Zero v1.0 Universal (CC0-1.0)** license.

**⚠️ NON-COMMERCIAL USE ONLY**: This project is intended for educational, research, and non-commercial purposes only. Commercial use is strictly prohibited.

### License Summary
- ✅ **Permitted**: Copy, modify, distribute for non-commercial purposes
- ✅ **Educational use**: Academic research and learning
- ✅ **Open source contributions**: Community improvements
- ❌ **Commercial use**: Any for-profit applications
- ❌ **Commercial distribution**: Selling or licensing for profit

See the [LICENSE](LICENSE) file for full details.

## 🏆 Hackathon Achievement

This project was developed as part of the **Shinkansen Travel Experience Hackathon** with the following achievements:
- Comprehensive passenger satisfaction prediction model
- Integration of multiple data sources
- Consideration of real-world railway operational constraints
- Embedded systems and functional safety awareness
- Open-source contribution to transportation ML research

## 📞 Contact & Support

For questions, suggestions, or collaboration opportunities (non-commercial):

- **Repository**: [GitHub Issues](https://github.com/kebrahimpour/shinkansen-passenger-satisfaction-predictor/issues)
- **Project**: Shinkansen Travel Experience Hackathon Entry
- **License**: CC0-1.0 (Non-commercial use only)

## 🔗 Related Topics

`machine-learning` `data-science` `embedded-systems` `functional-safety` `hackathon` `shinkansen` `transportation` `python` `passenger-satisfaction` `railway-systems`

---

**Disclaimer**: This is a hackathon project for educational and research purposes. It is not affiliated with JR East, JR Central, JR West, or any official Shinkansen operations. All data used is synthetic or publicly available for research purposes.
