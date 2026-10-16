# Slim image for the Aider polyglot benchmark, Python and C++ exercises only (docs/GLM53F_EVAL.md).
# Build context: an Aider checkout (github.com/Aider-AI/aider). The upstream benchmark/Dockerfile also installs
# Java, Go, Rust and Node for the other four languages; this one keeps what pytest and the C++ cmake tests need.
FROM python:3.11-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends cmake libboost-dev \
    && rm -rf /var/lib/apt/lists/*
COPY . /aider
RUN pip install --no-cache-dir --upgrade pip uv && uv pip install --system --no-cache-dir -e '/aider[dev]' pytest
RUN git config --global --add safe.directory '*'
WORKDIR /aider
