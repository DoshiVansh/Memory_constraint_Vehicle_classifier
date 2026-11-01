# ── Stage 1: Build Dependencies Stage ──
FROM python:3.10-slim@sha256:6ff506466f8b1e981719b468cc0023b8ef000b983cdec0eafe119d01f379e31d AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY requirements-cpu.txt /build/requirements-cpu.txt
COPY requirements-api.txt /build/requirements-api.txt

RUN python -m pip install --no-cache-dir --upgrade pip==25.0.1 \
    && python -m pip install --no-cache-dir --prefix=/install --extra-index-url https://pypi.org/simple -r requirements-cpu.txt \
    && python -m pip install --no-cache-dir --prefix=/install Pillow==11.1.0 PyYAML==6.0.2 \
    && python -m pip install --no-cache-dir --prefix=/install -r requirements-api.txt

# ── Stage 2: Minimal Serving Runtime Stage ──
FROM python:3.10-slim@sha256:6ff506466f8b1e981719b468cc0023b8ef000b983cdec0eafe119d01f379e31d

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/artifacts/student_model.pth \
    MODEL_METADATA_PATH=/app/artifacts/student_model.pth.json \
    MAX_UPLOAD_BYTES=8388608 \
    MAX_IMAGE_PIXELS=20000000

WORKDIR /app

# Copy pure pre-built dependencies from builder without wheel build caches
COPY --from=builder /install /usr/local

# Copy application source code
COPY vehicle_classifier_core /app/vehicle_classifier_core
COPY vehicle_classifier.py /app/vehicle_classifier.py
COPY configs /app/configs
COPY scripts/smoke_test_api.py /app/scripts/smoke_test_api.py

# Decoupled default model weights (can be mounted or overridden via MODEL_PATH)
COPY student_model.pth /app/artifacts/student_model.pth
COPY student_model.pth.json /app/artifacts/student_model.pth.json

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)" || exit 1

CMD ["uvicorn", "vehicle_classifier_core.api:app", "--host", "0.0.0.0", "--port", "8000"]
