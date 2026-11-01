"""HTML and JavaScript UI for the interactive OpenTelemetry Trace Waterfall Dashboard."""

TRACE_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Vehicle Classifier | Distributed Tracing & Span Waterfall</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f17;
      --surface: #111827;
      --surface-elevated: #1f2937;
      --border: #374151;
      --text-main: #f9fafb;
      --text-muted: #9ca3af;
      --accent: #3b82f6;
      --accent-glow: rgba(59, 130, 246, 0.2);
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --purple: #8b5cf6;
      --cyan: #06b6d4;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text-main);
      font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 16px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      position: sticky;
      top: 0;
      z-index: 50;
    }
    .logo-group {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .logo-badge {
      background: linear-gradient(135deg, #3b82f6, #8b5cf6);
      width: 36px;
      height: 36px;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 18px;
      font-weight: 700;
    }
    h1 {
      font-size: 18px;
      font-weight: 700;
      letter-spacing: -0.02em;
    }
    .subtitle {
      font-size: 12px;
      color: var(--text-muted);
    }
    .header-actions {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .btn {
      background: var(--surface-elevated);
      color: var(--text-main);
      border: 1px solid var(--border);
      padding: 8px 14px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      text-decoration: none;
      transition: all 0.15s ease;
    }
    .btn:hover {
      background: var(--border);
      border-color: #4b5563;
    }
    .btn-primary {
      background: #2563eb;
      border-color: #3b82f6;
    }
    .btn-primary:hover {
      background: #1d4ed8;
    }
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 9999px;
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .pulse-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: currentColor;
      animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.85); }
    }
    main {
      flex: 1;
      display: grid;
      grid-template-columns: 360px 1fr;
      height: calc(100vh - 69px);
      overflow: hidden;
    }
    .sidebar {
      background: var(--surface);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .search-box {
      padding: 12px;
      border-bottom: 1px solid var(--border);
    }
    .search-box input {
      width: 100%;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 8px 12px;
      font-size: 13px;
      color: var(--text-main);
      font-family: inherit;
    }
    .search-box input:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 2px var(--accent-glow);
    }
    .trace-list {
      flex: 1;
      overflow-y: auto;
      list-style: none;
    }
    .trace-item {
      padding: 12px 14px;
      border-bottom: 1px solid rgba(55, 65, 81, 0.5);
      cursor: pointer;
      transition: background 0.15s ease;
    }
    .trace-item:hover {
      background: rgba(255, 255, 255, 0.03);
    }
    .trace-item.active {
      background: rgba(59, 130, 246, 0.12);
      border-left: 3px solid var(--accent);
    }
    .trace-item-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 6px;
    }
    .badge {
      font-size: 11px;
      font-weight: 600;
      padding: 2px 6px;
      border-radius: 4px;
      text-transform: uppercase;
      font-family: 'JetBrains Mono', monospace;
    }
    .badge-class {
      background: rgba(139, 92, 246, 0.2);
      color: #c4b5fd;
      border: 1px solid rgba(139, 92, 246, 0.4);
    }
    .badge-latency {
      background: rgba(59, 130, 246, 0.15);
      color: #93c5fd;
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
    }
    .trace-id {
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      color: var(--text-muted);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .trace-time {
      font-size: 11px;
      color: #6b7280;
      margin-top: 4px;
    }
    .waterfall-container {
      display: flex;
      flex-direction: column;
      overflow-y: auto;
      padding: 20px 24px;
      gap: 16px;
    }
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
    }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }
    .summary-item {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .summary-label {
      font-size: 11px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .summary-val {
      font-size: 16px;
      font-weight: 700;
      font-family: 'JetBrains Mono', monospace;
    }
    .timeline-title {
      font-size: 14px;
      font-weight: 600;
      margin-bottom: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .waterfall-ruler {
      position: relative;
      height: 20px;
      border-bottom: 1px dashed var(--border);
      margin-bottom: 12px;
      display: flex;
      justify-content: space-between;
      font-family: 'JetBrains Mono', monospace;
      font-size: 10px;
      color: var(--text-muted);
    }
    .span-row {
      display: flex;
      flex-direction: column;
      margin-bottom: 8px;
      padding: 8px 10px;
      border-radius: 6px;
      background: rgba(31, 41, 55, 0.4);
      cursor: pointer;
      border: 1px solid transparent;
      transition: all 0.15s ease;
    }
    .span-row:hover {
      background: rgba(31, 41, 55, 0.8);
      border-color: #4b5563;
    }
    .span-row.selected {
      border-color: var(--accent);
      background: rgba(59, 130, 246, 0.1);
    }
    .span-meta {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 12px;
      margin-bottom: 6px;
    }
    .span-name {
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .span-indent {
      color: #6b7280;
      font-family: monospace;
    }
    .span-duration {
      font-family: 'JetBrains Mono', monospace;
      color: var(--text-muted);
      font-size: 11px;
    }
    .bar-track {
      position: relative;
      height: 8px;
      background: rgba(255, 255, 255, 0.05);
      border-radius: 4px;
      overflow: hidden;
    }
    .bar-fill {
      position: absolute;
      top: 0;
      bottom: 0;
      border-radius: 4px;
      transition: width 0.3s ease;
    }
    .span-detail-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      font-size: 12px;
      font-family: 'JetBrains Mono', monospace;
    }
    .attr-table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
      font-size: 12px;
      font-family: 'JetBrains Mono', monospace;
    }
    .attr-table th, .attr-table td {
      text-align: left;
      padding: 6px 10px;
      border-bottom: 1px solid rgba(55, 65, 81, 0.5);
    }
    .attr-table th {
      color: var(--text-muted);
      font-size: 11px;
      text-transform: uppercase;
    }
    .empty-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 300px;
      color: var(--text-muted);
      gap: 12px;
    }
  </style>
</head>
<body>
  <header>
    <div class="logo-group">
      <div class="logo-badge">⚡</div>
      <div>
        <h1>Vehicle Classifier Distributed Tracing Explorer</h1>
        <div class="subtitle">OpenTelemetry W3C Distributed Context & Granular Span Timing</div>
      </div>
    </div>
    <div class="header-actions">
      <div class="status-pill">
        <span class="pulse-dot"></span>
        <span id="poll-status">LIVE STREAMING</span>
      </div>
      <button class="btn" id="refresh-btn" onclick="fetchTraces()">🔄 Refresh</button>
      <a class="btn btn-primary" href="http://20.249.161.211:3000/d/vehicle-classifier-overview" target="_blank">📊 Open Grafana Dashboard</a>
      <a class="btn" href="/docs" target="_blank">📖 Swagger Docs</a>
    </div>
  </header>

  <main>
    <div class="sidebar">
      <div class="search-box">
        <input type="text" id="filter-input" placeholder="Filter by Class, Trace ID, or Status..." oninput="renderTraceList()">
      </div>
      <ul class="trace-list" id="trace-list">
        <!-- Rendered dynamically -->
      </ul>
    </div>

    <div class="waterfall-container" id="waterfall-container">
      <div class="empty-state">
        <div>Loading trace telemetry from Azure AKS cluster...</div>
      </div>
    </div>
  </main>

  <script>
    let traces = [];
    let selectedTraceId = null;
    let selectedSpanId = null;
    let autoRefresh = true;

    const spanColors = {
      'predict_request': 'linear-gradient(90deg, #3b82f6, #60a5fa)',
      'read_upload_payload': 'linear-gradient(90deg, #06b6d4, #22d3ee)',
      'validate_and_decode_image': 'linear-gradient(90deg, #f59e0b, #fbbf24)',
      'convert_image_rgb': 'linear-gradient(90deg, #ec4899, #f472b6)',
      'model_inference': 'linear-gradient(90deg, #8b5cf6, #a78bfa)',
      'pytorch_forward_pass': 'linear-gradient(90deg, #10b981, #34d399)'
    };

    async function fetchTraces() {
      try {
        const resp = await fetch('/api/traces');
        if (!resp.ok) throw new Error('API request failed');
        const data = await resp.json();
        traces = data.traces || [];
        if (!selectedTraceId && traces.length > 0) {
          selectedTraceId = traces[0].trace_id;
        }
        renderTraceList();
        renderWaterfall();
      } catch (err) {
        console.error('Failed to fetch traces:', err);
      }
    }

    function renderTraceList() {
      const listEl = document.getElementById('trace-list');
      const filter = (document.getElementById('filter-input').value || '').toLowerCase().trim();
      
      const filtered = traces.filter(t => {
        if (!filter) return true;
        const cls = (t.class_name || '').toLowerCase();
        const id = (t.trace_id || '').toLowerCase();
        const status = (t.status || '').toLowerCase();
        return cls.includes(filter) || id.includes(filter) || status.includes(filter);
      });

      if (filtered.length === 0) {
        listEl.innerHTML = '<li style="padding: 24px; text-align: center; color: #6b7280; font-size: 13px;">No matching traces found</li>';
        return;
      }

      listEl.innerHTML = filtered.map(t => {
        const active = t.trace_id === selectedTraceId ? 'active' : '';
        const cls = t.class_name || 'Inference';
        const conf = t.confidence ? `(${(t.confidence * 100).toFixed(1)}%)` : '';
        return `
          <li class="trace-item ${active}" onclick="selectTrace('${t.trace_id}')">
            <div class="trace-item-header">
              <span class="badge badge-class">${cls} ${conf}</span>
              <span class="badge badge-latency">${t.duration_ms.toFixed(1)} ms</span>
            </div>
            <div class="trace-id">${t.trace_id}</div>
            <div class="trace-time">${t.start_time_iso} • ${t.span_count} spans</div>
          </li>
        `;
      }).join('');
    }

    function selectTrace(traceId) {
      selectedTraceId = traceId;
      selectedSpanId = null;
      renderTraceList();
      renderWaterfall();
    }

    function selectSpan(spanId) {
      selectedSpanId = spanId;
      renderWaterfall();
    }

    function renderWaterfall() {
      const container = document.getElementById('waterfall-container');
      const trace = traces.find(t => t.trace_id === selectedTraceId);

      if (!trace) {
        container.innerHTML = `
          <div class="empty-state">
            <div>Select a trace on the left panel to inspect the function call waterfall</div>
          </div>
        `;
        return;
      }

      const totalDuration = Math.max(0.1, trace.duration_ms);
      const spans = trace.spans || [];
      const selectedSpan = spans.find(s => s.span_id === selectedSpanId) || spans[0];

      // Build waterfall bars
      const waterfallRows = spans.map((s, idx) => {
        const offsetPct = Math.min(99, Math.max(0, (s.offset_ms / totalDuration) * 100));
        const widthPct = Math.min(100 - offsetPct, Math.max(1.5, (s.duration_ms / totalDuration) * 100));
        const color = spanColors[s.name] || 'linear-gradient(90deg, #6b7280, #9ca3af)';
        const isSelected = s.span_id === (selectedSpan ? selectedSpan.span_id : null);
        const indentLevel = s.parent_span_id ? (s.name === 'pytorch_forward_pass' ? '└── └── ' : '├── ') : '';

        return `
          <div class="span-row ${isSelected ? 'selected' : ''}" onclick="selectSpan('${s.span_id}')">
            <div class="span-meta">
              <span class="span-name">
                <span class="span-indent">${indentLevel}</span>
                ${s.name}
              </span>
              <span class="span-duration">${s.duration_ms.toFixed(2)} ms</span>
            </div>
            <div class="bar-track">
              <div class="bar-fill" style="left: ${offsetPct}%; width: ${widthPct}%; background: ${color};"></div>
            </div>
          </div>
        `;
      }).join('');

      // Attributes table
      const attrs = selectedSpan ? selectedSpan.attributes || {} : {};
      const attrRows = Object.entries(attrs).map(([k, v]) => `
        <tr>
          <td style="color: #93c5fd;">${k}</td>
          <td>${v}</td>
        </tr>
      `).join('');

      container.innerHTML = `
        <div class="card">
          <div class="summary-grid">
            <div class="summary-item">
              <div class="summary-label">Trace ID</div>
              <div class="summary-val" style="font-size: 13px; color: #60a5fa;">${trace.trace_id}</div>
            </div>
            <div class="summary-item">
              <div class="summary-label">Root Endpoint</div>
              <div class="summary-val">${trace.root_name}</div>
            </div>
            <div class="summary-item">
              <div class="summary-label">Total Execution Time</div>
              <div class="summary-val" style="color: #34d399;">${trace.duration_ms.toFixed(2)} ms</div>
            </div>
            <div class="summary-item">
              <div class="summary-label">Predicted Class</div>
              <div class="summary-val" style="color: #c4b5fd;">${trace.class_name || 'N/A'} ${trace.confidence ? '(' + (trace.confidence * 100).toFixed(1) + '%)' : ''}</div>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="timeline-title">
            <span>Granular Span Waterfall Execution Timeline</span>
            <span style="font-size: 12px; color: #9ca3af; font-family: 'JetBrains Mono', monospace;">Total: ${totalDuration.toFixed(1)} ms</span>
          </div>

          <div class="waterfall-ruler">
            <span>0 ms</span>
            <span>${(totalDuration * 0.25).toFixed(1)} ms</span>
            <span>${(totalDuration * 0.50).toFixed(1)} ms</span>
            <span>${(totalDuration * 0.75).toFixed(1)} ms</span>
            <span>${totalDuration.toFixed(1)} ms</span>
          </div>

          <div>${waterfallRows}</div>
        </div>

        ${selectedSpan ? `
          <div class="card">
            <div class="timeline-title">
              <span>Span Details: <code style="color: #34d399;">${selectedSpan.name}</code></span>
              <span class="badge badge-latency">${selectedSpan.duration_ms.toFixed(2)} ms</span>
            </div>
            <div class="span-detail-grid">
              <div><strong>Span ID:</strong> ${selectedSpan.span_id}</div>
              <div><strong>Parent Span ID:</strong> ${selectedSpan.parent_span_id || 'root (None)'}</div>
              <div><strong>Relative Offset:</strong> +${selectedSpan.offset_ms.toFixed(2)} ms</div>
              <div><strong>Span Status:</strong> <span style="color: #34d399;">${selectedSpan.status}</span></div>
            </div>
            
            <h4 style="font-size: 12px; text-transform: uppercase; color: #9ca3af; margin-top: 16px;">Span Attributes & Function Context</h4>
            <table class="attr-table">
              <thead>
                <tr>
                  <th>Attribute Name</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                ${attrRows || '<tr><td colspan="2" style="color: #6b7280;">No attributes attached to this span</td></tr>'}
              </tbody>
            </table>
          </div>
        ` : ''}
      `;
    }

    // Initial load and periodic polling
    fetchTraces();
    setInterval(() => {
      if (autoRefresh) fetchTraces();
    }, 2500);
  </script>
</body>
</html>
"""
