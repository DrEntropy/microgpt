FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

EXPOSE 5001

CMD ["uv", "run", "gunicorn", "--bind", "0.0.0.0:5001", "app:app"]
