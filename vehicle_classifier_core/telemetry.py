"""OpenTelemetry distributed tracing configuration and span utilities."""

import collections
import datetime
import logging
import os
import threading
from typing import Any, Optional, Sequence

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Span, Tracer
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

logger = logging.getLogger(__name__)

TRACER_NAME = "vehicle_classifier"
TRACER_VERSION = "1.0.0"


class InMemoryTraceStore(SpanExporter):
    """Circular in-memory store for recent OpenTelemetry distributed traces and span hierarchies."""

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self._lock = threading.Lock()
        self._traces: collections.OrderedDict[str, list[dict[str, Any]]] = collections.OrderedDict()

    def export(self, spans: Sequence[Any]) -> SpanExportResult:
        with self._lock:
            for s in spans:
                trace_id = format_trace_id(s)
                if trace_id not in self._traces:
                    if len(self._traces) >= self.capacity:
                        self._traces.popitem(last=False)
                    self._traces[trace_id] = []

                span_id = format_span_id(s)
                parent_id = (
                    format(s.parent.span_id, "016x")
                    if (s.parent and hasattr(s.parent, "span_id") and s.parent.span_id)
                    else None
                )
                duration_ms = (
                    round((s.end_time - s.start_time) / 1_000_000, 2)
                    if (s.start_time and s.end_time)
                    else 0.0
                )
                status_name = "OK"
                if hasattr(s, "status") and hasattr(s.status, "status_code"):
                    status_name = s.status.status_code.name

                span_dict = {
                    "name": s.name,
                    "span_id": span_id,
                    "parent_span_id": parent_id,
                    "start_time_nano": s.start_time or 0,
                    "end_time_nano": s.end_time or 0,
                    "duration_ms": duration_ms,
                    "status": status_name,
                    "attributes": {
                        str(k): (float(v) if isinstance(v, (int, float)) else str(v))
                        for k, v in (s.attributes or {}).items()
                    },
                }
                self._traces[trace_id].append(span_dict)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def get_recent_traces(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            result = []
            for trace_id, spans in reversed(list(self._traces.items())):
                if not spans:
                    continue
                root = next((s for s in spans if s.get("parent_span_id") is None), spans[-1])
                valid_starts = [s["start_time_nano"] for s in spans if s["start_time_nano"] > 0]
                t_start = min(valid_starts) if valid_starts else 0
                t_end = max(s["end_time_nano"] for s in spans) if spans else 0
                total_duration = (
                    round((t_end - t_start) / 1_000_000, 2)
                    if (t_end and t_start)
                    else root["duration_ms"]
                )

                class_name = None
                confidence = None
                for s in spans:
                    attrs = s.get("attributes", {})
                    if "prediction.class_name" in attrs:
                        class_name = attrs["prediction.class_name"]
                    if "prediction.confidence" in attrs:
                        try:
                            confidence = float(attrs["prediction.confidence"])
                        except (ValueError, TypeError):
                            pass

                spans_sorted = sorted(spans, key=lambda x: x["start_time_nano"])
                spans_with_offset = []
                for s in spans_sorted:
                    cp = dict(s)
                    cp["offset_ms"] = (
                        round((s["start_time_nano"] - t_start) / 1_000_000, 2)
                        if t_start
                        else 0.0
                    )
                    spans_with_offset.append(cp)

                iso_time = (
                    datetime.datetime.fromtimestamp(
                        t_start / 1_000_000_000, tz=datetime.timezone.utc
                    ).strftime("%Y-%m-%d %H:%M:%S UTC")
                    if t_start > 0
                    else "Recent"
                )

                result.append({
                    "trace_id": trace_id,
                    "root_name": root["name"],
                    "start_time_iso": iso_time,
                    "duration_ms": total_duration,
                    "span_count": len(spans),
                    "class_name": class_name,
                    "confidence": confidence,
                    "status": root["status"],
                    "spans": spans_with_offset,
                })
                if len(result) >= limit:
                    break
            return result

    def get_trace(self, trace_id: str) -> Optional[dict[str, Any]]:
        traces = self.get_recent_traces(limit=self.capacity)
        return next((t for t in traces if t["trace_id"] == trace_id), None)


_GLOBAL_TRACE_STORE: Optional[InMemoryTraceStore] = None


def get_trace_store() -> InMemoryTraceStore:
    global _GLOBAL_TRACE_STORE
    if _GLOBAL_TRACE_STORE is None:
        _GLOBAL_TRACE_STORE = InMemoryTraceStore(capacity=100)
    return _GLOBAL_TRACE_STORE


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

    # Always register in-memory trace store for interactive UI and API inspection
    provider.add_span_processor(SimpleSpanProcessor(get_trace_store()))

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
