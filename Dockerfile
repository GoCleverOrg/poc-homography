# ---- build stage ----
FROM python:3.14-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install uv (official installer)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy only dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install production dependencies (no dev group, no editable install)
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source
COPY poc_homography/ poc_homography/
COPY api/             api/
COPY webapp/          webapp/
COPY data/            data/
COPY alembic/         alembic/
COPY alembic.ini      alembic.ini

# Install the project itself (into the same venv)
RUN uv sync --frozen --no-dev


# ---- runtime stage ----
FROM python:3.14-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# OpenCV / tifffile runtime libraries
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgl1-mesa-glx \
       libglib2.0-0 \
       libtiff-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the virtual-env and application code from the builder
COPY --from=builder /app /app

# Put the venv on PATH so `uvicorn` is found directly
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
