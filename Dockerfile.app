FROM python:3.12.3-slim

WORKDIR /app

COPY pyproject.toml .
COPY uv.lock .
RUN pip install --no-cache-dir uv
RUN uv sync --frozen --only-group client
COPY app.py .


CMD ["uv", "run", "streamlit", "run", "app.py"]