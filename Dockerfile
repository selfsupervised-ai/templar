# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install git curl build-essential && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    rm -rf /var/lib/apt/lists/*

# Clone your GitHub repository
RUN git clone https://github.com/<your-username>/<your-repo>.git /app

# Install your Python package (editable install if desired)
WORKDIR /app
RUN uv sync

# Set the default command
CMD ["python3", "your_entrypoint.py"]