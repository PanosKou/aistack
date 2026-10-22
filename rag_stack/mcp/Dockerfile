FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install --requirement requirements.txt

RUN addgroup --system recitals \
    && adduser \
        --system \
        --ingroup recitals \
        --home /app \
        recitals

COPY --chown=recitals:recitals server.py .

USER recitals

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["python", "server.py"]
