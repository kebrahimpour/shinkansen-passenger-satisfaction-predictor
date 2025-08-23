# Use the official Python 3.12 dev container base image
FROM mcr.microsoft.com/devcontainers/python:3.12
# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*
# Install uv via official script and copy to /usr/local/bin with chmod +x
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && cp /root/.local/bin/uv /usr/local/bin/uv && chmod +x /usr/local/bin/uv
# Set the working directory
WORKDIR /workspace
# Default CMD for container
CMD ["sleep", "infinity"]
