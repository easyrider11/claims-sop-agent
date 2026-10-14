FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY app ./app
COPY static ./static
COPY fixtures ./fixtures
COPY tests ./tests
COPY pytest.ini .env.example ./
COPY scripts ./scripts
# run as an unprivileged user; outbox/ and logs/ are the only writable paths
RUN useradd --create-home --uid 10001 agent && mkdir -p outbox logs && chown agent:agent outbox logs
USER agent
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
# API key is passed at run time: docker run -e ANTHROPIC_API_KEY=... (or OPENAI_API_KEY=...)
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
