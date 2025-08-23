#!/bin/bash

# Initialize script to fix permissions for Dev Container
# This script ensures vscode user owns the necessary directories

set -e

echo "[initialize.sh] Starting permission fix for Dev Container..."

# Ensure vscode user owns /home/vscode
if [ -d "/home/vscode" ]; then
    echo "[initialize.sh] Fixing ownership of /home/vscode"
    chown -R vscode:vscode /home/vscode
fi

# Ensure vscode user owns /workspaces
if [ -d "/workspaces" ]; then
    echo "[initialize.sh] Fixing ownership of /workspaces"
    chown -R vscode:vscode /workspaces
fi

echo "[initialize.sh] Permission fix completed successfully"

# Execute the original CMD if provided
if [ $# -gt 0 ]; then
    echo "[initialize.sh] Executing: $@"
    exec "$@"
else
    echo "[initialize.sh] No CMD provided, starting sleep infinity"
    exec sleep infinity
fi
