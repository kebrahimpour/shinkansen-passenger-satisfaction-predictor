# Use the official Python 3.10 dev container base image
FROM mcr.microsoft.com/devcontainers/python:3.10

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv via official script and symlink to /usr/local/bin
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.cargo/bin/uv /usr/local/bin/uv

# Copy and setup initialize script
COPY initialize.sh /usr/local/bin/initialize.sh
RUN chmod +x /usr/local/bin/initialize.sh

# Set the working directory
WORKDIR /workspace

# Set ENTRYPOINT to handle permissions, then CMD for default behavior
ENTRYPOINT ["/usr/local/bin/initialize.sh"]
CMD ["sleep", "infinity"]
