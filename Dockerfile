# General application container for Shinkansen ML project
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Install dependencies
RUN uv sync --extra dev

# Set environment variables
ENV PYTHONPATH="/app/src"
ENV PATH="/app/.venv/bin:$PATH"

# Default command
CMD ["bash"]
