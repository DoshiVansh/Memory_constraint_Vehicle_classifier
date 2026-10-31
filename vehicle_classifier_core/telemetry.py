"""OpenTelemetry distributed tracing configuration and span utilities."""

import logging
import os
from typing import Any, Mapping, Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.trace import Span, Status, StatusCode, Tracer
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

logger = logging.getLogger(__name__)

TRACER_NAME = "vehicle_classifier"
TRACER_VERSION = "1.0.0"


def setup_telemetry(
    service_name: str = "vehicle-classifier",
    exporter: Optional[SpanExporter] = None,
    force_reinit: bool = False,
) -> TracerProvider:
    """Initialize and register the global TracerProvider if not already configured."""
    current_provider = trace.get_tracer_provider()
    if not force_reinit and isinstance(current_provider, TracerProvider):
        if exporter is not None:
            current_provider.add_span_processor(SimpleSpanProcessor(exporter))
        return current_provider

    service_name = os.getenv("OTEL_SERVICE_NAME", service_name)
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    else:
        exporter_type = os.getenv("OTEL_TRACES_EXPORTER", "").lower().strip()
        if exporter_type == "console":
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        elif exporter_type == "otlp":
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

                endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
                otlp_exporter = OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()
                provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            except ImportError:
                logger.warning("opentelemetry-exporter-otlp not installed; falling back to ConsoleSpanExporter")
                provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    if force_reinit:
        trace._TRACER_PROVIDER = provider
        if hasattr(trace, "_TRACER_PROVIDER_SET_ONCE"):
            trace._TRACER_PROVIDER_SET_ONCE._done = True
    else:
        trace.set_tracer_provider(provider)

    return provider


class DynamicTracer:
    """Tracer proxy that always delegates to the currently active global TracerProvider."""

    def __init__(self, name: str = TRACER_NAME, version: str = TRACER_VERSION):
        self.name = name
        self.version = version

    @property
    def _tracer(self) -> Tracer:
        provider = trace.get_tracer_provider()
        return provider.get_tracer(self.name, self.version)

    def start_span(self, *args, **kwargs) -> Span:
        return self._tracer.start_span(*args, **kwargs)

    def start_as_current_span(self, *args, **kwargs):
        return self._tracer.start_as_current_span(*args, **kwargs)


def get_tracer(name: str = TRACER_NAME, version: str = TRACER_VERSION) -> Tracer:
    """Return a dynamic OpenTelemetry Tracer instance."""
    return DynamicTracer(name, version)


def extract_trace_context(headers: Any) -> Any:
    """Extract W3C traceparent context from ASGI or HTTP headers."""
    carrier: dict[str, str] = {}
    if headers:
        if hasattr(headers, "items"):
            for k, v in headers.items():
                carrier[k.lower()] = str(v)
        elif isinstance(headers, (list, tuple)):
            for item in headers:
                if len(item) == 2:
                    k = item[0].decode("latin-1").lower() if isinstance(item[0], bytes) else str(item[0]).lower()
                    v = item[1].decode("latin-1") if isinstance(item[1], bytes) else str(item[1])
                    carrier[k] = v
    return TraceContextTextMapPropagator().extract(carrier)


def format_trace_id(span: Span) -> str:
    """Return the 32-character hex trace ID for a span."""
    ctx = span.get_span_context()
    if ctx and ctx.trace_id:
        return format(ctx.trace_id, "032x")
    return "0" * 32


def format_span_id(span: Span) -> str:
    """Return the 16-character hex span ID for a span."""
    ctx = span.get_span_context()
    if ctx and ctx.span_id:
        return format(ctx.span_id, "016x")
    return "0" * 16
