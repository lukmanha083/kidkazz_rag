# Stage 1: Copy pre-built helix-container binary from local image
# This image is built by `kidkazz db deploy` (helix push dev)
# and contains the compiled HelixQL queries baked in.
FROM helix-kidkazz_rag-dev:dev AS helix

# Stage 2: Python + helix binary + MCP server
# NOTE: Cannot use Alpine — helix-container is linked against glibc (Debian).
# python:3.13-slim-bookworm is ~150MB, the smallest glibc-compatible option.
FROM python:3.13-slim-bookworm

# Install runtime deps for helix-container (OpenSSL 3.x) + tini + netcat
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libssl3 tini netcat-openbsd && \
    rm -rf /var/lib/apt/lists/*

# Copy helix-container binary
COPY --from=helix /usr/local/bin/helix-container /usr/local/bin/helix-container

# Create data directory for Helix-DB (Fly volume mounts here)
RUN mkdir -p /data

# Set up Python app
WORKDIR /app

# Install Python dependencies first (cache layer)
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[mcp,helixdb,openai,concepts,extractive]"

# Copy source code
COPY src/ src/
COPY db/ db/

# Copy deploy-specific config
COPY deploy/entrypoint.sh /entrypoint.sh
COPY deploy/kidkazz.toml /app/.kidkazz.toml

RUN chmod +x /entrypoint.sh

# Helix-DB env vars
ENV HELIX_PORT=6969
ENV HELIX_DATA_DIR=/data
ENV HELIX_INSTANCE=prod
ENV HELIX_PROJECT=kidkazz_rag
ENV HELIX_CORES_OVERRIDE=2
ENV TOKIO_WORKER_THREADS=2
ENV RAYON_NUM_THREADS=2

# MCP server env vars (defaults, override with Fly secrets)
ENV KIDKAZZ_TRANSPORT=streamable-http
ENV KIDKAZZ_HOST=0.0.0.0
ENV KIDKAZZ_PORT=8080

EXPOSE 8080

# tini handles PID 1 + signal forwarding
ENTRYPOINT ["tini", "--"]
CMD ["/entrypoint.sh"]
