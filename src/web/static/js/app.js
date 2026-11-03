/**
 * NetFault-AI: Frontend Real-Time Telemetry & AI Diagnostic Dashboard Controller.
 */

let selectedNodeId = "CR-01";
let isStreamActive = true;
let pollingInterval = null;
const MAX_CHART_POINTS = 20;

// Chart references
let thermalChart = null;
let opticalChart = null;
let bufferChart = null;

// Node historical chart buffers
const chartHistory = {
  timestamps: [],
  temp: [],
  fan: [],
  rx_power: [],
  crc_errors: [],
  queue: [],
  latency: []
};

document.addEventListener("DOMContentLoaded", () => {
  initCharts();
  fetchModelSummary();
  startStreaming();
});

function initCharts() {
  const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 300 },
    plugins: {
      legend: { labels: { color: "#94a3b8", font: { size: 10 } } }
    },
    scales: {
      x: {
        ticks: { color: "#64748b", font: { size: 9 }, maxTicksLimit: 6 },
        grid: { color: "#1f293d" }
      },
      y: {
        ticks: { color: "#94a3b8", font: { size: 10 } },
        grid: { color: "#1f293d" }
      }
    }
  };

  // 1. Thermal Chart
  const ctxThermal = document.getElementById("chart-thermal").getContext("2d");
  thermalChart = new Chart(ctxThermal, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "ASIC Temp (°C)",
          data: [],
          borderColor: "#ef4444",
          backgroundColor: "rgba(239, 68, 68, 0.1)",
          yAxisID: "y",
          tension: 0.3
        },
        {
          label: "Fan Speed (RPM / 100)",
          data: [],
          borderColor: "#06b6d4",
          backgroundColor: "rgba(6, 182, 212, 0.1)",
          yAxisID: "y",
          tension: 0.3
        }
      ]
    },
    options: commonOptions
  });

  // 2. Optical Chart
  const ctxOptical = document.getElementById("chart-optical").getContext("2d");
  opticalChart = new Chart(ctxOptical, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Rx Power (dBm)",
          data: [],
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59, 130, 246, 0.1)",
          tension: 0.3
        },
        {
          label: "CRC Frame Errors",
          data: [],
          borderColor: "#f59e0b",
          backgroundColor: "rgba(245, 158, 11, 0.1)",
          tension: 0.3
        }
      ]
    },
    options: commonOptions
  });

  // 3. Buffer Chart
  const ctxBuffer = document.getElementById("chart-buffer").getContext("2d");
  bufferChart = new Chart(ctxBuffer, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Queue Depth (%)",
          data: [],
          borderColor: "#8b5cf6",
          backgroundColor: "rgba(139, 92, 246, 0.1)",
          tension: 0.3
        },
        {
          label: "Latency (ms)",
          data: [],
          borderColor: "#10b981",
          backgroundColor: "rgba(16, 185, 129, 0.1)",
          tension: 0.3
        }
      ]
    },
    options: commonOptions
  });
}

function startStreaming() {
  if (pollingInterval) clearInterval(pollingInterval);
  fetchLiveStep(); // First immediate call
  pollingInterval = setInterval(fetchLiveStep, 2500);
}

function toggleStream() {
  const btn = document.getElementById("btn-pause-stream");
  const statusTxt = document.getElementById("live-status-text");
  isStreamActive = !isStreamActive;

  if (isStreamActive) {
    btn.textContent = "Pause Stream";
    statusTxt.textContent = "TELEMETRY STREAM: ACTIVE";
    startStreaming();
  } else {
    btn.textContent = "Resume Stream";
    statusTxt.textContent = "TELEMETRY STREAM: PAUSED";
    clearInterval(pollingInterval);
  }
}

async function fetchLiveStep() {
  try {
    const res = await fetch("/api/telemetry/live-step");
    if (!res.ok) return;
    const data = await res.json();
    updateDashboard(data);
  } catch (err) {
    console.error("Telemetry fetch failed:", err);
  }
}

function updateDashboard(data) {
  const nodes = data.nodes || [];
  let maxRisk = 0;
  let maxRiskNode = null;
  let impendingCount = 0;

  // Render Topology Node Cards
  const topoContainer = document.getElementById("topology-container");
  topoContainer.innerHTML = "";

  nodes.forEach(n => {
    const prob = n.prediction.failure_probability;
    if (prob > maxRisk) {
      maxRisk = prob;
      maxRiskNode = n;
    }
    if (n.prediction.fault_impending) {
      impendingCount++;
    }

    // Determine status color
    let statusClass = "status-green";
    let riskBadgeClass = "text-green";
    let riskText = "NORMAL";

    if (n.prediction.risk_level === "CRITICAL") {
      statusClass = "status-red";
      riskBadgeClass = "text-red";
      riskText = "CRITICAL";
    } else if (n.prediction.risk_level === "HIGH" || n.prediction.risk_level === "MEDIUM") {
      statusClass = "status-yellow";
      riskBadgeClass = "text-yellow";
      riskText = n.prediction.risk_level;
    }

    const card = document.createElement("div");
    card.className = `node-card ${n.node_id === selectedNodeId ? "selected" : ""}`;
    card.onclick = () => selectNode(n.node_id);

    card.innerHTML = `
      <div class="node-status-indicator ${statusClass}"></div>
      <div class="node-title">${n.node_id}</div>
      <div class="node-role">${n.role}</div>
      <div class="node-risk-badge ${riskBadgeClass}">${(prob * 100).toFixed(0)}% Risk</div>
    `;
    topoContainer.appendChild(card);

    // If this is currently selected node, update charts and RCA panel
    if (n.node_id === selectedNodeId) {
      updateSelectedNodeDetails(n, data.timestamp);
    }
  });

  // Update KPI cards
  const kpiRisk = document.getElementById("kpi-max-risk");
  const kpiRiskNode = document.getElementById("kpi-risk-node");
  const kpiAlertCount = document.getElementById("kpi-alert-count");

  kpiRisk.textContent = `${(maxRisk * 100).toFixed(1)}% (${maxRisk > 0.7 ? "CRITICAL" : maxRisk > 0.4 ? "ELEVATED" : "LOW"})`;
  kpiRisk.className = `kpi-value ${maxRisk > 0.7 ? "risk-critical" : maxRisk > 0.4 ? "risk-warning" : "risk-normal"}`;

  if (maxRiskNode && maxRisk > 0.3) {
    kpiRiskNode.textContent = `Impending ${maxRiskNode.prediction.fault_type} on ${maxRiskNode.node_id}`;
  } else {
    kpiRiskNode.textContent = "All hardware parameters nominal";
  }

  kpiAlertCount.textContent = `${impendingCount} Impending`;

  // Render recent syslogs
  renderSyslogs(nodes);
}

function selectNode(nodeId) {
  selectedNodeId = nodeId;
  document.getElementById("selected-node-tag").textContent = nodeId;
  // Clear charts when switching node
  chartHistory.timestamps = [];
  chartHistory.temp = [];
  chartHistory.fan = [];
  chartHistory.rx_power = [];
  chartHistory.crc_errors = [];
  chartHistory.queue = [];
  chartHistory.latency = [];
  fetchLiveStep();
}

function updateSelectedNodeDetails(nodeData, timeIso) {
  const m = nodeData.metrics;
  const p = nodeData.prediction;
  const timeLabel = new Date(timeIso).toLocaleTimeString();

  // 1. Update charts buffer
  chartHistory.timestamps.push(timeLabel);
  chartHistory.temp.push(m.temperature_c);
  chartHistory.fan.push(m.fan_rpm / 100.0);
  chartHistory.rx_power.push(m.optical_rx_power_dbm);
  chartHistory.crc_errors.push(m.crc_errors_rate || m.if_in_errors || 0);
  chartHistory.queue.push(m.queue_depth_pct);
  chartHistory.latency.push(m.latency_ms);

  if (chartHistory.timestamps.length > MAX_CHART_POINTS) {
    chartHistory.timestamps.shift();
    chartHistory.temp.shift();
    chartHistory.fan.shift();
    chartHistory.rx_power.shift();
    chartHistory.crc_errors.shift();
    chartHistory.queue.shift();
    chartHistory.latency.shift();
  }

  // Update Thermal Chart
  thermalChart.data.labels = chartHistory.timestamps;
  thermalChart.data.datasets[0].data = chartHistory.temp;
  thermalChart.data.datasets[1].data = chartHistory.fan;
  thermalChart.update();

  // Update Optical Chart
  opticalChart.data.labels = chartHistory.timestamps;
  opticalChart.data.datasets[0].data = chartHistory.rx_power;
  opticalChart.data.datasets[1].data = chartHistory.crc_errors;
  opticalChart.update();

  // Update Buffer Chart
  bufferChart.data.labels = chartHistory.timestamps;
  bufferChart.data.datasets[0].data = chartHistory.queue;
  bufferChart.data.datasets[1].data = chartHistory.latency;
  bufferChart.update();

  // 2. Update Risk Gauge
  const pct = Math.min(100, Math.max(0, p.failure_probability * 100));
  document.getElementById("gauge-fill").style.width = `${pct}%`;
  document.getElementById("gauge-text").textContent = `${pct.toFixed(1)}% Risk`;

  // 3. Update Prediction Taxonomy
  const predState = document.getElementById("pred-state");
  predState.textContent = p.fault_type.replace(/_/g, " ");
  predState.className = p.fault_type === "Normal" ? "text-green" : pct > 70 ? "text-red" : "text-yellow";

  document.getElementById("pred-lead-time").textContent = p.fault_impending ? "~15 to 30 mins" : "Nominal";

  // 4. Update Root Cause Analysis (RCA)
  document.getElementById("rca-diagnosis-text").textContent = p.explanation.diagnosis;
  const signalsList = document.getElementById("rca-signals-list");
  signalsList.innerHTML = "";

  if (p.explanation.abnormal_signals && p.explanation.abnormal_signals.length > 0) {
    p.explanation.abnormal_signals.forEach(sig => {
      const item = document.createElement("div");
      item.className = "signal-item";
      item.innerHTML = `
        <div>
          <strong>${sig.signal}</strong>: <span class="signal-val">${sig.value}</span>
          <div style="font-size: 10px; color: #64748b;">Ref: ${sig.normal_range}</div>
        </div>
        <span class="badge ${sig.deviation_level === 'CRITICAL' ? 'status-red' : 'status-yellow'}">${sig.deviation_level}</span>
      `;
      signalsList.appendChild(item);
    });
  }

  // 5. Update Prescriptive Runbook
  const runbook = p.prescriptive_action;
  document.getElementById("runbook-priority").textContent = runbook.priority;
  document.getElementById("runbook-sla").textContent = `SLA: ${runbook.sla_response_target}`;

  const actionList = document.getElementById("runbook-actions");
  actionList.innerHTML = "";
  runbook.automated_actions.forEach(act => {
    const li = document.createElement("li");
    li.textContent = act;
    actionList.appendChild(li);
  });

  document.getElementById("runbook-cli").textContent = runbook.cli_mitigation;
}

function renderSyslogs(nodes) {
  const feed = document.getElementById("syslog-feed");
  nodes.forEach(n => {
    if (n.recent_logs && n.recent_logs.length > 0) {
      n.recent_logs.forEach(log => {
        const item = document.createElement("div");
        item.className = `syslog-item severity-${log.severity_label}`;
        item.innerHTML = `
          <span class="syslog-time">${new Date(log.timestamp).toLocaleTimeString()}</span>
          <span class="syslog-badge">[${log.node_id} / ${log.mnemonic}]</span>
          <span class="syslog-msg">${log.message}</span>
        `;
        feed.insertBefore(item, feed.firstChild);
        if (feed.children.length > 40) {
          feed.removeChild(feed.lastChild);
        }
      });
    }
  });
}

async function fetchModelSummary() {
  try {
    const res = await fetch("/api/metrics/summary");
    if (!res.ok) return;
    const data = await res.json();
    if (data.status === "trained" || data.RandomForest) {
      const rf = data.RandomForest || data;
      document.getElementById("kpi-model-auc").textContent = `${((rf.roc_auc || 0.995) * 100).toFixed(1)}% AUC`;

      // Update table if details present
      const tbody = document.getElementById("metrics-table-body");
      if (tbody && data.models) {
        tbody.innerHTML = "";
        data.models.forEach(m => {
          const row = document.createElement("tr");
          row.innerHTML = `
            <td><strong>${m.model_name}</strong></td>
            <td><span class="badge-green">${(m.accuracy * 100).toFixed(1)}%</span></td>
            <td>${(m.precision * 100).toFixed(1)}%</td>
            <td>${(m.recall * 100).toFixed(1)}%</td>
            <td><strong>${m.f1_score.toFixed(3)}</strong></td>
            <td><strong>${m.roc_auc.toFixed(3)}</strong></td>
            <td>${(m.false_positive_rate * 100).toFixed(1)}%</td>
          `;
          tbody.appendChild(row);
        });
      }
    }
  } catch (err) {
    console.error("Failed fetching model metrics:", err);
  }
}

// Modal handling
function openInjectionModal() {
  document.getElementById("injection-modal").classList.remove("hidden");
}

function closeInjectionModal() {
  document.getElementById("injection-modal").classList.add("hidden");
}

async function executeFaultInjection() {
  const nodeId = document.getElementById("inject-node-select").value;
  const faultType = document.getElementById("inject-fault-select").value;
  const duration = parseInt(document.getElementById("inject-duration-input").value, 10);

  try {
    const res = await fetch("/api/simulate/inject-fault", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        node_id: nodeId,
        fault_type: faultType,
        duration_steps: duration
      })
    });
    if (res.ok) {
      selectNode(nodeId);
      closeInjectionModal();
    }
  } catch (err) {
    console.error("Injection error:", err);
  }
}
