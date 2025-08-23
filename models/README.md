# Trained Models Directory

This directory contains saved machine learning models for the Shinkansen passenger satisfaction prediction project.

## Model Files

- `satisfaction_model.pkl` - Main trained satisfaction prediction model
- `model_metadata.json` - Model metadata and performance metrics

## Usage

Models are automatically loaded by the API server on startup. To train a new model, run:

```bash
uv run python scripts/train.py
```

The trained model will be saved here and can be loaded by the API.
