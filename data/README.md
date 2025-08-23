# Shinkansen Data Directory

This directory contains data files for the Shinkansen passenger satisfaction prediction project.

## Data Sources

- `sample_data.csv` - Sample training data for model development
- `test_data.csv` - Test data for model validation
- `models/` - Directory for saved model files

## Data Format

The expected data format includes the following features:

- `duration`: Journey duration in minutes (float)
- `service_class`: Service class (Ordinary, Green, GranClass)
- `on_time_performance`: On-time performance ratio 0-1 (float)
- `weather_condition`: Weather condition (clear, rain, cloudy, etc.)
- `seat_occupancy`: Seat occupancy ratio 0-1 (float)
- `satisfaction_score`: Target satisfaction score 0-5 (float)

## Usage

Place your data files in this directory and update the model training scripts accordingly.
