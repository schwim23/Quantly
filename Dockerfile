# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies in a separate layer so Docker can cache them.
# Stub the package so hatchling can resolve it during dep install;
# real source is copied below.
COPY pyproject.toml .
RUN mkdir -p quantly && touch quantly/__init__.py
RUN uv pip install --system --no-cache -e .

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY quantly/ quantly/
COPY pyproject.toml .

# Data directory — SQLite DBs are stored here and mounted as a volume
RUN mkdir -p data

# Non-root user for security
RUN useradd -m -u 1000 quantly && chown -R quantly:quantly /app
USER quantly

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "-m", "quantly", "serve", "--host", "0.0.0.0", "--port", "8000"]
