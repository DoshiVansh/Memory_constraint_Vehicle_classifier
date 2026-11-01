import io
import asyncio

import pytest
import httpx
from PIL import Image

from vehicle_classifier_core import api
from vehicle_classifier_core.artifact_validation import validate_inference_package
from vehicle_classifier_core.model import CLASS_IDX
from vehicle_classifier_core.api import RequestBodySizeLimit


def png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), (90, 120, 160)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_api_contract_health_prediction_and_rejections(monkeypatch):
    async def exercise():
        async with api.app.router.lifespan_context(api.app):
            transport = httpx.ASGITransport(app=api.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                live = await client.get("/health/live")
                assert live.status_code == 200
                assert live.json() == {"status": "alive"}
                ready = await client.get("/health/ready")
                assert ready.status_code == 200
                assert len(ready.json()["model_sha256"]) == 64

                response = await client.post(
                    "/predict", files={"file": ("sample.png", png_bytes(), "image/png")}
                )
                assert response.status_code == 200
                body = response.json()
                assert set(body) == {"predicted_index", "class_name", "model_sha256", "confidence", "probabilities"}
                assert body["predicted_index"] in CLASS_IDX
                assert body["class_name"] == CLASS_IDX[body["predicted_index"]]
                assert len(body["model_sha256"]) == 64
                assert 0.0 <= body["confidence"] <= 1.0
                assert set(body["probabilities"]) == {"Bus", "Truck", "Car", "Bike", "None"}

                metrics_resp = await client.get("/metrics", follow_redirects=True)
                assert metrics_resp.status_code == 200
                metrics_text = metrics_resp.text
                assert "vehicle_classifier_prediction_confidence" in metrics_text
                assert "vehicle_classifier_predictions_total" in metrics_text

                # Test OpenTelemetry trace endpoints
                traces_resp = await client.get("/api/traces")
                assert traces_resp.status_code == 200
                traces_data = traces_resp.json()
                assert "traces" in traces_data
                assert len(traces_data["traces"]) > 0
                sample_trace_id = traces_data["traces"][0]["trace_id"]

                single_trace = await client.get(f"/api/traces/{sample_trace_id}")
                assert single_trace.status_code == 200
                assert single_trace.json()["trace_id"] == sample_trace_id
                assert "spans" in single_trace.json()

                html_resp = await client.get("/traces")
                assert html_resp.status_code == 200
                assert "Distributed Tracing" in html_resp.text

                malformed = await client.post(
                    "/predict", files={"file": ("bad.png", b"not an image", "image/png")}
                )
                assert malformed.status_code == 400
                assert "valid supported image" in malformed.json()["detail"]

                corrupt_png = bytearray(png_bytes())
                corrupt_png[-5] ^= 0xFF
                corrupt = await client.post(
                    "/predict", files={"file": ("corrupt.png", bytes(corrupt_png), "image/png")}
                )
                assert corrupt.status_code == 400

                buffer = io.BytesIO()
                Image.new("RGB", (4, 4)).save(buffer, format="GIF")
                unsupported = await client.post(
                    "/predict", files={"file": ("sample.gif", buffer.getvalue(), "image/gif")}
                )
                assert unsupported.status_code == 415

                monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 16)
                oversized = await client.post(
                    "/predict", files={"file": ("large.png", b"x" * 17, "image/png")}
                )
                assert oversized.status_code == 413

                monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 8 * 1024 * 1024)
                monkeypatch.setattr(api, "MAX_IMAGE_PIXELS", 100)
                large_pixels = io.BytesIO()
                Image.new("RGB", (11, 10)).save(large_pixels, format="PNG")
                too_many_pixels = await client.post(
                    "/predict", files={"file": ("large.png", large_pixels.getvalue(), "image/png")}
                )
                assert too_many_pixels.status_code == 413

                schema = (await client.get("/openapi.json")).json()
                assert "/predict" in schema["paths"]
                assert "multipart/form-data" in schema["paths"]["/predict"]["post"]["requestBody"]["content"]
                response_schema = schema["paths"]["/predict"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
                response_schema = schema["components"]["schemas"][response_schema["$ref"].rsplit("/", 1)[-1]]
                assert set(response_schema["required"]) == {"predicted_index", "class_name", "model_sha256", "confidence", "probabilities"}
                documented_statuses = schema["paths"]["/predict"]["post"]["responses"]
                assert {"200", "400", "413", "415", "500"}.issubset(documented_statuses)

    asyncio.run(exercise())


def test_artifact_package_rejects_tampered_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"tampered")
    metadata = tmp_path / "checkpoint.json"
    metadata.write_text(
        '{"architecture":"shufflenet_v2_x1_0","class_mapping":{"0":"Bus","1":"Truck",'
        '"2":"Car","3":"Bike","4":"None"},"preprocessing":{"color":"RGB",'
        '"size":[224,224],"mean":[0.485,0.456,0.406],"std":[0.229,0.224,0.225]},'
        '"size_bytes":8,"sha256":"wrong"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SHA-256"):
        validate_inference_package(checkpoint, metadata)


def test_request_body_limit_applies_to_streamed_requests_without_content_length():
    async def exercise():
        messages = [
            {"type": "http.request", "body": b"123456", "more_body": True},
            {"type": "http.request", "body": b"78901", "more_body": True},
        ]
        sent = []

        async def receive():
            return messages.pop(0)

        async def send(message):
            sent.append(message)

        async def app(scope, receive, send):
            await receive()
            await receive()

        middleware = RequestBodySizeLimit(app, max_body_bytes=10)
        await middleware(
            {"type": "http", "method": "POST", "path": "/predict", "headers": []},
            receive,
            send,
        )
        assert sent[0]["status"] == 413

    asyncio.run(exercise())
