"""Tests for OpenTelemetry distributed request tracing across inference stages."""

import asyncio
import io
import pytest
import httpx
from PIL import Image

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from vehicle_classifier_core import api
from vehicle_classifier_core.telemetry import (
    extract_trace_context,
    format_span_id,
    format_trace_id,
    setup_telemetry,
)


def sample_png_bytes(width=32, height=24):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (120, 160, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_telemetry_predict_spans_and_trace_hierarchy():
    exporter = InMemorySpanExporter()
    setup_telemetry(service_name="vehicle-classifier-test", exporter=exporter, force_reinit=True)

    async def exercise():
        async with api.app.router.lifespan_context(api.app):
            transport = httpx.ASGITransport(app=api.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/predict", files={"file": ("test_car.png", sample_png_bytes(), "image/png")}
                )
                assert response.status_code == 200
                assert "X-Trace-ID" in response.headers
                trace_id_header = response.headers["X-Trace-ID"]
                assert len(trace_id_header) == 32

                spans = exporter.get_finished_spans()
                span_names = {s.name for s in spans}
                expected_spans = {
                    "predict_request",
                    "read_upload_payload",
                    "validate_and_decode_image",
                    "convert_image_rgb",
                    "model_inference",
                    "tensor_preprocessing",
                    "pytorch_forward_pass",
                }
                assert expected_spans.issubset(span_names), f"Missing spans: {expected_spans - span_names}"

                # Find root span
                root_span = next(s for s in spans if s.name == "predict_request")
                assert format_trace_id(root_span) == trace_id_header
                assert root_span.attributes["http.method"] == "POST"
                assert root_span.attributes["http.route"] == "/predict"
                assert root_span.attributes["upload.filename"] == "test_car.png"
                assert "prediction.class_name" in root_span.attributes

                # Verify payload span attributes
                payload_span = next(s for s in spans if s.name == "read_upload_payload")
                assert payload_span.parent.span_id == root_span.context.span_id
                assert payload_span.attributes["payload.bytes"] > 0
                assert payload_span.attributes["payload.chunks"] >= 1

                # Verify image decode span attributes
                decode_span = next(s for s in spans if s.name == "validate_and_decode_image")
                assert decode_span.parent.span_id == root_span.context.span_id
                assert decode_span.attributes["image.format"] == "PNG"
                assert decode_span.attributes["image.width"] == 32
                assert decode_span.attributes["image.height"] == 24

                # Verify model inference span & child spans
                infer_span = next(s for s in spans if s.name == "model_inference")
                assert infer_span.parent.span_id == root_span.context.span_id

                preprocess_span = next(s for s in spans if s.name == "tensor_preprocessing")
                assert preprocess_span.parent.span_id == infer_span.context.span_id
                assert "tensor.shape" in preprocess_span.attributes

                forward_span = next(s for s in spans if s.name == "pytorch_forward_pass")
                assert forward_span.parent.span_id == infer_span.context.span_id
                assert "prediction.class_name" in forward_span.attributes

    asyncio.run(exercise())


def test_distributed_trace_context_propagation():
    exporter = InMemorySpanExporter()
    setup_telemetry(service_name="vehicle-classifier-test", exporter=exporter, force_reinit=True)

    trace_id_hex = "4bf92f3577b34da6a3ce929d0e0e4736"
    parent_span_id_hex = "00f067aa0ba902b7"
    traceparent_header = f"00-{trace_id_hex}-{parent_span_id_hex}-01"

    async def exercise():
        async with api.app.router.lifespan_context(api.app):
            transport = httpx.ASGITransport(app=api.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/predict",
                    files={"file": ("test.png", sample_png_bytes(), "image/png")},
                    headers={"traceparent": traceparent_header},
                )
                assert response.status_code == 200
                assert response.headers["X-Trace-ID"] == trace_id_hex

                spans = exporter.get_finished_spans()
                root_span = next(s for s in spans if s.name == "predict_request")
                assert format_trace_id(root_span) == trace_id_hex
                assert format_span_id(root_span) != parent_span_id_hex
                assert format(root_span.parent.span_id, "016x") == parent_span_id_hex

    asyncio.run(exercise())


def test_telemetry_records_exception_on_bad_image():
    exporter = InMemorySpanExporter()
    setup_telemetry(service_name="vehicle-classifier-test", exporter=exporter, force_reinit=True)

    async def exercise():
        async with api.app.router.lifespan_context(api.app):
            transport = httpx.ASGITransport(app=api.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/predict",
                    files={"file": ("corrupt.png", b"not-an-image", "image/png")},
                )
                assert response.status_code == 400

                spans = exporter.get_finished_spans()
                root_span = next(s for s in spans if s.name == "predict_request")
                assert len(root_span.events) > 0
                assert root_span.events[0].name == "exception"

    asyncio.run(exercise())
