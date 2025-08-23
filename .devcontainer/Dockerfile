# Use the official Python 3.10 dev container base image
FROM mcr.microsoft.com/devcontainers/python:3.10

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    ca-certificates \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Install uv - fast Python package installer
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH for all users
ENV PATH="/home/vscode/.cargo/bin:$PATH"

# Set the working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Set proper ownership for vscode user
RUN chown -R vscode:vscode /workspace

# Switch to vscode user
USER vscode

# Create virtual environment and install dependencies
RUN uv venv && uv sync --extra dev

# Keep container running
CMD ["sleep", "infinity"]
