FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

COPY corpus/ ./corpus/
COPY eval/ ./eval/

# Non-root: the container has no business writing to its own image.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Render and most PaaS inject $PORT; default to 8000 locally.
CMD ["sh", "-c", "uvicorn sharia_agent.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
