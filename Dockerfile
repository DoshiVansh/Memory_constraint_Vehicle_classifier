FROM python:3.10-slim@sha256:6ff506466f8b1e981719b468cc0023b8ef000b983cdec0eafe119d01f379e31d

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/artifacts/student_model.pth \
    MODEL_METADATA_PATH=/app/artifacts/student_model.pth.json \
    MAX_UPLOAD_BYTES=8388608 \
    MAX_IMAGE_PIXELS=20000000

WORKDIR /app

# Install CPU-only wheels. The service constructs ShuffleNet with weights=None
# and loads only the packaged checkpoint; it never fetches pretrained weights.
COPY requirements-cpu.txt /app/requirements-cpu.txt
COPY requirements-api.txt /app/requirements-api.txt
RUN python -m pip install --no-cache-dir --upgrade pip==25.0.1
RUN python -m pip install --no-cache-dir --extra-index-url https://pypi.org/simple -r requirements-cpu.txt
RUN python -m pip install --no-cache-dir Pillow==11.1.0 PyYAML==6.0.2
RUN python -m pip install --no-cache-dir -r requirements-api.txt

COPY vehicle_classifier_core /app/vehicle_classifier_core
COPY vehicle_classifier.py /app/vehicle_classifier.py
COPY scripts/smoke_test_api.py /app/scripts/smoke_test_api.py
COPY student_model.pth /app/artifacts/student_model.pth
COPY student_model.pth.json /app/artifacts/student_model.pth.json

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)" || exit 1

CMD ["uvicorn", "vehicle_classifier_core.api:app", "--host", "0.0.0.0", "--port", "8000"]
