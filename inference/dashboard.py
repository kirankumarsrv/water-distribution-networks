"""Lightweight dashboard for EPANET network visualization and leak simulation."""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference.example_end_to_end import build_sample_df
from inference.real_time_detector import RealTimeLeakDetector
from integration.EPANET_Integration import EPANETIntegrator

INP_FILE = ROOT_DIR / "EPANETINPUTFILESFOR7NEWORKS" / "2_Extended Hanoi.inp"
PORT = 8000

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Water Network Leak Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f4f7fb; color: #24292f; }
    header { background: #0366d6; color: white; padding: 16px; }
    main { display: grid; grid-template-columns: 320px 1fr; gap: 16px; padding: 16px; }
    .panel { background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); padding: 16px; }
    .panel h2 { margin-top: 0; }
    label { display: block; margin: 12px 0 4px; font-weight: 600; }
    select, button { width: 100%; padding: 10px 12px; border-radius: 8px; border: 1px solid #cbd5e1; background: white; font-size: 14px; }
    button { cursor: pointer; background: #0366d6; color: white; border: none; margin-top: 12px; }
    button:hover { background: #024ea2; }
    #status { margin-top: 16px; white-space: pre-wrap; font-family: monospace; line-height: 1.4; }
    #network-svg { width: 100%; height: 650px; border-radius: 10px; border: 1px solid #d1d5db; background: #fff; }
    .node { cursor: pointer; }
    .node circle { fill: #0366d6; stroke: #fff; stroke-width: 1.8px; }
    .edge { stroke: #94a3b8; stroke-width: 2px; }
    .edge.selected { stroke: #dc2626; stroke-width: 3px; }
    .edge.predicted { stroke: #22c55e; stroke-width: 3px; }
    .node.selected circle { fill: #f97316; }
  </style>
</head>
<body>
  <header>
    <h1>Water Network Leak Dashboard</h1>
    <p>Visualize the Hanoi network, inject a leak or fault, and run the trained detector.</p>
  </header>
  <main>
    <div class="panel">
      <h2>Controls</h2>
      <p style="font-size: 13px; color: #475569; line-height: 1.5; margin-bottom: 12px;">
        Select one or more pipes, choose a fault type, and assign it before running the simulation. Mixed fault types are supported per pipe.
      </p>
      <label for="fault-type">Fault type for selected pipe(s)</label>
      <select id="fault-type">
        <option value="normal">Normal</option>
        <option value="leak">Leak</option>
        <option value="burst">Burst</option>
        <option value="blockage">Blockage</option>
      </select>
      <button id="assign-fault">Assign to selected pipe(s)</button>
      <label for="pipe">Pipe(s) to fault <small style="font-weight:normal;">(Ctrl/Cmd+click for multiple)</small></label>
      <select id="pipe" multiple size="10" style="min-height: 160px;"></select>
      <div id="assignments" style="margin: 12px 0; font-size: 14px; color: #334155; line-height: 1.4; background: #f8fafc; padding: 12px; border: 1px solid #e2e8f0; border-radius: 8px;">
        No pipe fault assignments yet.
      </div>
      <button id="simulate">Run simulation</button>
      <div id="status">Loading dashboard...</div>
      <div id="prediction-result" style="margin-top: 16px; white-space: pre-wrap; font-family: monospace; line-height: 1.4; background: #f8fafc; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0;"></div>
    </div>
    <div class="panel">
      <h2>Network view</h2>
      <svg id="network-svg" viewBox="0 0 1000 700"></svg>
    </div>
  </main>
  <script>
    const status = document.getElementById('status');
    const resultPanel = document.getElementById('prediction-result');
    const pipeSelect = document.getElementById('pipe');
    const faultTypeSelect = document.getElementById('fault-type');
    const assignFaultButton = document.getElementById('assign-fault');
    const assignmentsPanel = document.getElementById('assignments');
    const simulateButton = document.getElementById('simulate');
    const svg = document.getElementById('network-svg');
    let network = null;
    let selectedPipe = [];
    let prediction = null;
    let predictedZonePipes = [];
    let pipeFaultMap = {};

    function setStatus(text) {
      status.textContent = text;
    }

    function updateAssignmentsPanel() {
      const entries = Object.entries(pipeFaultMap);
      if (!entries.length) {
        assignmentsPanel.textContent = 'No pipe fault assignments yet.';
        return;
      }
      assignmentsPanel.innerHTML = entries
        .map(([pipe, type]) => `<div>Pipe ${pipe}: ${type}</div>`)
        .join('');
    }

    window.addEventListener('error', event => {
      setStatus('JavaScript error: ' + event.message);
      console.error('Dashboard error', event.error || event.message, event.filename, event.lineno, event.colno);
    });
    window.addEventListener('unhandledrejection', event => {
      setStatus('Unhandled promise rejection: ' + event.reason);
      console.error('Unhandled rejection', event.reason);
    });

    function renderPrediction() {
      if (!prediction) {
        resultPanel.textContent = 'Run a simulation to see detection and localization results.';
        return;
      }
      if (prediction.error) {
        resultPanel.textContent = 'Error: ' + prediction.error;
        return;
      }
      const det = prediction.result.detection || {};
      const loc = prediction.result.localization || {};
      const actualZones = Array.isArray(prediction.actual_zones) ? prediction.actual_zones : [prediction.actual_zone].filter(Boolean);
      const predictedZone = loc.zone_id !== undefined ? loc.zone_id : 'N/A';
      let text = `Fault assignments: ${prediction.faults?.length ? prediction.faults.map(f => `${f.pipe}:${f.scenario}`).join(', ') : 'none'}\n` +
        `Actual zones: ${actualZones.length ? actualZones.join(', ') : 'N/A'}\n` +
        `Detected label: ${det.label} (${det.label === 0 ? 'Normal' : 'Fault'})\n` +
        `Detection confidence: ${det.confidence?.toFixed?.(3) ?? 'N/A'}\n` +
        `Predicted zone: ${predictedZone}\n` +
        `Zone confidence: ${loc.zone_confidence?.toFixed?.(3) ?? 'N/A'}`;

      if (Array.isArray(loc.top_zones) && loc.top_zones.length > 0) {
        text += "\\nZone ranking:";
        loc.top_zones.slice(0, 5).forEach((zone, idx) => {
          text += `\n  ${idx + 1}. Zone ${zone.zone_id} (${(zone.probability * 100).toFixed(1)}%)`;
        });
      }

      resultPanel.textContent = text;
    }

    function requestNetwork() {
      setStatus('Loading network data...');
      return fetch('/network').then(r => {
        if (!r.ok) {
          throw new Error(`Network API error ${r.status}`);
        }
        return r.json();
      });
    }

    function populateControls(data) {
      network = data;
      setStatus(`Loaded ${data.pipes?.length ?? 0} pipes and ${data.nodes?.length ?? 0} nodes`);
      pipeSelect.innerHTML = '';
      data.pipes.forEach(pipe => {
        const option = document.createElement('option');
        option.value = pipe.id;
        option.textContent = `${pipe.id} (${pipe.source}→${pipe.target})`;
        pipeSelect.appendChild(option);
      });
      selectedPipe = [];
      pipeSelect.addEventListener('change', () => {
        selectedPipe = Array.from(pipeSelect.selectedOptions).map(opt => opt.value);
        drawNetwork();
      });
      assignFaultButton.addEventListener('click', () => {
        if (!selectedPipe || selectedPipe.length === 0) {
          setStatus('Select one or more pipes before assigning a fault type.');
          return;
        }
        const type = faultTypeSelect.value;
        selectedPipe.forEach(pipe => {
          pipeFaultMap[pipe] = type;
        });
        updateAssignmentsPanel();
        drawNetwork();
        setStatus(`Assigned ${type} to ${selectedPipe.length} pipe(s).`);
      });
      updateAssignmentsPanel();
      drawNetwork();
      renderPrediction();
    }

    function drawNetwork() {
      if (!network) return;
      svg.innerHTML = '';
      const width = 1000;
      const height = 700;
      const nodeById = {};
      network.nodes.forEach(node => {
        nodeById[node.id] = node;
      });
      network.edges = network.pipes.map(pipe => ({
        ...pipe,
        selected: Array.isArray(selectedPipe) ? selectedPipe.includes(pipe.id) : pipe.id === selectedPipe,
        predicted: predictedZonePipes.includes(pipe.id),
      }));
      network.edges.forEach(edge => {
        const source = nodeById[edge.source];
        const target = nodeById[edge.target];
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', source.x * width);
        line.setAttribute('y1', source.y * height);
        line.setAttribute('x2', target.x * width);
        line.setAttribute('y2', target.y * height);
        let cssClass = 'edge';
        if (edge.predicted) cssClass = 'edge predicted';
        const selected = Array.isArray(selectedPipe) ? selectedPipe.includes(edge.id) : edge.id === selectedPipe;
        if (selected) cssClass = 'edge selected';
        line.setAttribute('class', cssClass);
        line.dataset.pipe = edge.id;
        line.addEventListener('click', () => {
          selectedPipe = [edge.id];
          Array.from(pipeSelect.options).forEach(opt => {
            opt.selected = opt.value === edge.id;
          });
          drawNetwork();
        });
        svg.appendChild(line);
      });
      network.nodes.forEach(node => {
        const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        group.setAttribute('class', 'node');
        group.setAttribute('transform', `translate(${node.x * width}, ${node.y * height})`);
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('r', 12);
        group.appendChild(circle);
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', 16);
        label.setAttribute('y', 5);
        label.setAttribute('font-size', '12');
        label.textContent = node.id;
        group.appendChild(label);
        svg.appendChild(group);
      });
    }

    async function simulate() {
      const assignedFaults = Object.entries(pipeFaultMap).map(([pipe, scenario]) => ({ pipe, scenario }));
      const unassigned = selectedPipe.filter(pipe => !Object.prototype.hasOwnProperty.call(pipeFaultMap, pipe));
      const defaultType = faultTypeSelect.value;
      const unassignedFaults = unassigned.map(pipe => ({ pipe, scenario: defaultType }));
      const faults = [...assignedFaults, ...unassignedFaults];
      if (faults.length === 0 && defaultType !== 'normal') {
        setStatus('Please choose at least one pipe to fault.');
        return;
      }
      setStatus('Running simulation...');
      const body = { faults };
      try {
        const res = await fetch('/simulate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        const data = await res.json();
        if (data.error) {
          setStatus('Error: ' + data.error);
          prediction = { error: data.error };
          predictedZonePipes = [];
        } else {
          prediction = data;
          predictedZonePipes = Array.isArray(data.predicted_zone_pipes) ? data.predicted_zone_pipes : [];
          setStatus('Simulation completed.');
        }
      } catch (err) {
        setStatus('Request failed: ' + err.message);
        prediction = { error: err.message };
        predictedZonePipes = [];
      }
      drawNetwork();
      renderPrediction();
    }

    simulateButton.addEventListener('click', simulate);
    requestNetwork().then(populateControls).catch(err => setStatus('Failed to load network: ' + err.message));
  </script>
</body>
</html>
"""


def load_zone_definitions() -> dict[str, int]:
    zone_file = ROOT_DIR / "DATASETS" / "zone_definitions.json"
    if zone_file.exists():
        with zone_file.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def load_network_data() -> dict[str, Any]:
    integrator = EPANETIntegrator(str(INP_FILE))
    wn = integrator.wn

    nodes: list[dict[str, Any]] = []
    for name, node in wn.nodes():
        coords = getattr(node, "coordinates", None)
        x = float(coords[0]) if coords is not None and len(coords) > 0 else None
        y = float(coords[1]) if coords is not None and len(coords) > 1 else None
        nodes.append({"id": name, "type": getattr(node, "node_type", "node"), "x": x, "y": y})

    if any(node["x"] is None or node["y"] is None for node in nodes):
        count = len(nodes)
        for idx, node in enumerate(nodes):
            if node["x"] is None or node["y"] is None:
                angle = 2 * 3.141592653589793 * idx / max(1, count)
                node["x"] = 0.5 + 0.38 * float(__import__("math").cos(angle))
                node["y"] = 0.5 + 0.38 * float(__import__("math").sin(angle))

    xs = [node["x"] for node in nodes]
    ys = [node["y"] for node in nodes]
    if xs and ys:
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        range_x = max(max_x - min_x, 1e-6)
        range_y = max(max_y - min_y, 1e-6)
        for node in nodes:
            node["x"] = 0.05 + 0.9 * ((node["x"] - min_x) / range_x)
            node["y"] = 0.05 + 0.9 * ((node["y"] - min_y) / range_y)

    pipes: list[dict[str, Any]] = []
    for name, link in wn.links():
        if getattr(link, "link_type", "") != "Pipe":
            continue
        pipes.append({
            "id": name,
            "source": getattr(link, "start_node_name", ""),
            "target": getattr(link, "end_node_name", ""),
            "length": float(getattr(link, "length", 0.0) or 0.0),
            "diameter": float(getattr(link, "diameter", 0.0) or 0.0),
        })

    return {
        "nodes": nodes,
        "pipes": pipes,
        "zone_definitions": load_zone_definitions(),
    }


def _normalize_faults(faults_value: Any, default_scenario: str) -> list[dict[str, str]] | None:
    if faults_value is None:
        return None
    if isinstance(faults_value, list):
        normalized: list[dict[str, str]] = []
        for item in faults_value:
            if isinstance(item, dict):
                pipe_id = item.get('pipe') or item.get('pipe_id') or item.get('id')
                if pipe_id is None:
                    continue
                scenario = str(item.get('scenario', default_scenario))
                normalized.append({'pipe': str(pipe_id), 'scenario': scenario})
            else:
                normalized.append({'pipe': str(item), 'scenario': default_scenario})
        return normalized
    if isinstance(faults_value, str):
        return [{'pipe': faults_value, 'scenario': default_scenario}]
    return [{'pipe': str(faults_value), 'scenario': default_scenario}]


def simulate_leak(faults: Any, default_scenario: str) -> dict[str, Any]:
    if default_scenario not in {"normal", "leak", "burst", "blockage"}:
        raise ValueError("Invalid default scenario")

    target_faults = _normalize_faults(faults, default_scenario)
    if default_scenario != "normal" and not target_faults:
        raise ValueError("At least one pipe fault is required for a non-normal scenario")

    zone_definitions = load_zone_definitions()
    sample_df = build_sample_df(str(INP_FILE), default_scenario, target_pipe=target_faults)
    detector = RealTimeLeakDetector(models_dir=ROOT_DIR / "models")
    result = detector.infer(sample_df)
    zone_id = result.get("localization", {}).get("zone_id")
    predicted_zone_pipes = [pipe for pipe, zid in zone_definitions.items() if zid == zone_id] if zone_id is not None else []
    actual_zones = sorted({zone_definitions[pipe] for fault in (target_faults or []) for pipe in [fault['pipe']] if pipe in zone_definitions})

    return {
        "faults": target_faults,
        "actual_zones": actual_zones,
        "predicted_zone_pipes": predicted_zone_pipes,
        "result": result,
    }


class DashboardHandler(BaseHTTPRequestHandler):
    def _send_response(self, content: bytes, content_type: str = "text/html", status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/index"):
            self._send_response(HTML_PAGE.encode("utf-8"), "text/html")
            return
        if self.path.startswith("/network"):
            try:
                payload = json.dumps(load_network_data()).encode("utf-8")
                self._send_response(payload, "application/json")
            except Exception as exc:
                self._send_response(json.dumps({"error": str(exc)}).encode("utf-8"), "application/json", status=500)
            return
        self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        if self.path != "/simulate":
            self.send_error(404, "Not Found")
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8") if length else ""
        try:
            request = json.loads(body) if body else {}
            scenario = str(request.get("scenario", "leak"))
            faults = request.get("faults") if request.get("faults") is not None else None
            if faults is None:
                pipe_name = request.get("pipe") if request.get("pipe") is not None else request.get("pipes")
                faults = pipe_name
            result = simulate_leak(faults, scenario)
            payload = json.dumps(result).encode("utf-8")
            self._send_response(payload, "application/json")
        except Exception as exc:
            import traceback

            traceback.print_exc()
            payload = json.dumps({
                "error": str(exc),
                "type": exc.__class__.__name__,
            }).encode("utf-8")
            self._send_response(payload, "application/json", status=500)


def main() -> None:
    server_address = ("", PORT)
    httpd = HTTPServer(server_address, DashboardHandler)
    print(f"Dashboard running at http://localhost:{PORT}/")
    print("Use CTRL+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Stopping dashboard...")
        httpd.server_close()


if __name__ == "__main__":
    main()
