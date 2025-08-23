# Development container for Shinkansen ML project
FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    cp /root/.local/bin/uv /usr/local/bin/uv && \
    chmod +x /usr/local/bin/uv

# Set the working directory
WORKDIR /workspaces/shinkansen-passenger-satisfaction-predictor

# Make sure uv is in PATH for all users
ENV PATH="/usr/local/bin:$PATH"

# Default CMD for development container
CMD ["sleep", "infinity"]
