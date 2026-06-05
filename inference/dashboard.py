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
from isolation.valve_isolation import ValveIsolationManager
from restoration.supply_restoration import SupplyRestorationManager
from evaluation.evaluation_framework import PerformanceEvaluator, EvaluationEvent

INP_FILE = ROOT_DIR / "EPANETINPUTFILESFOR7NEWORKS" / "2_Extended Hanoi.inp"
PORT = 8000

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Water Network Fault Response Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f4f7fb; color: #24292f; }
    header { background: #0366d6; color: white; padding: 16px; }
    .tabs { display: flex; gap: 4px; padding: 12px 16px; background: #e1eef9; border-bottom: 2px solid #0366d6; }
    .tab-btn { padding: 8px 16px; cursor: pointer; background: transparent; border: none; font-weight: 500; color: #475569; border-radius: 4px 4px 0 0; }
    .tab-btn.active { background: white; color: #0366d6; border-bottom: 3px solid #0366d6; }
    .tab-btn:hover { background: rgba(3, 102, 214, 0.1); }
    main { display: grid; grid-template-columns: 380px 1fr; gap: 16px; padding: 16px; }
    .panel { background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); padding: 16px; }
    .panel h2 { margin-top: 0; font-size: 18px; }
    .panel h3 { margin: 16px 0 8px; font-size: 14px; color: #475569; }
    label { display: block; margin: 12px 0 4px; font-weight: 600; font-size: 13px; }
    select, textarea, button { width: 100%; padding: 10px 12px; border-radius: 8px; border: 1px solid #cbd5e1; background: white; font-size: 14px; font-family: inherit; }
    button { cursor: pointer; background: #0366d6; color: white; border: none; margin-top: 12px; }
    button:hover { background: #024ea2; }
    button.secondary { background: #6b7280; }
    button.secondary:hover { background: #4b5563; }
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    #status { margin-top: 12px; white-space: pre-wrap; font-family: monospace; line-height: 1.4; font-size: 12px; background: #f8fafc; padding: 8px; border-radius: 4px; }
    #network-svg { width: 100%; height: 650px; border-radius: 10px; border: 1px solid #d1d5db; background: #fff; }
    .node { cursor: pointer; }
    .node circle { fill: #0366d6; stroke: #fff; stroke-width: 1.8px; }
    .edge { stroke: #94a3b8; stroke-width: 2px; }
    .edge.selected { stroke: #dc2626; stroke-width: 3px; }
    .edge.predicted { stroke: #22c55e; stroke-width: 3px; }
    .edge.isolated { stroke: #b91c1c; stroke-width: 3px; stroke-dasharray: 6 3; }
    .edge.restored { stroke: #16a34a; stroke-width: 3px; }
    .node.selected circle { fill: #f97316; }
    .valve-label { font-size: 10px; fill: #0f172a; }     .customer-label { font-size: 10px; fill: #2563eb; font-weight: bold; }    .metric-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #e2e8f0; font-size: 13px; }
    .metric-row .label { font-weight: 500; }
    .metric-row .value { text-align: right; font-family: monospace; }
    .metric-row.target-met .value { color: #16a34a; font-weight: 600; }
    .metric-row.target-miss .value { color: #dc2626; font-weight: 600; }
    .card { background: #f8fafc; padding: 12px; margin: 8px 0; border-radius: 6px; border-left: 3px solid #0366d6; }
  </style>
</head>
<body>
  <header>
    <h1>Water Distribution Network - Fault Detection & Response System</h1>
    <p>Objectives 1-6: Detection, Localization, Isolation, Restoration, and Metrics</p>
  </header>
  
  <div class="tabs">
    <button class="tab-btn active" data-tab="detection">Detection & Localization</button>
    <button class="tab-btn" data-tab="isolation">Isolation (Obj 4)</button>
    <button class="tab-btn" data-tab="restoration">Restoration (Obj 5)</button>
    <button class="tab-btn" data-tab="metrics">Metrics (Obj 6)</button>
  </div>

  <main>
    <div class="panel">
      <!-- Detection Tab -->
      <div id="detection" class="tab-content active">
        <h2>Fault Detection & Localization</h2>
        <p style="font-size: 13px; color: #475569; line-height: 1.5; margin-bottom: 12px;">
          Select pipes, assign fault types, and run simulation for detection and localization.
        </p>
        <label for="fault-type">Fault type</label>
        <select id="fault-type">
          <option value="normal">Normal</option>
          <option value="leak">Leak</option>
          <option value="burst">Burst</option>
          <option value="blockage">Blockage</option>
        </select>
        <button id="assign-fault">Assign to selected pipe(s)</button>
        <label for="pipe">Select pipe(s) <small style="font-weight:normal;">(Ctrl+click for multiple)</small></label>
        <select id="pipe" multiple size="10" style="min-height: 160px;"></select>
        <div id="assignments" style="margin: 12px 0; font-size: 13px; color: #334155; line-height: 1.4; background: #f8fafc; padding: 12px; border: 1px solid #e2e8f0; border-radius: 8px; max-height: 120px; overflow-y: auto;">
          No assignments yet.
        </div>
        <button id="simulate">Run Simulation</button>
        <div id="status">Loading...</div>
        <div id="prediction-result" style="margin-top: 12px; white-space: pre-wrap; font-family: monospace; line-height: 1.4; font-size: 12px; background: #f8fafc; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0; max-height: 200px; overflow-y: auto;"></div>
      </div>

      <!-- Isolation Tab -->
      <div id="isolation" class="tab-content">
        <h2>Valve Isolation (Obj 4)</h2>
        <p style="font-size: 13px; color: #475569; line-height: 1.5; margin-bottom: 12px;">
          Run simulation first, then use isolation to compute valve closure set.
        </p>
        <label for="faulty-pipe">Faulty pipe ID</label>
        <input type="text" id="faulty-pipe" placeholder="e.g., 1" style="width: 100%; padding: 10px 12px; border-radius: 8px; border: 1px solid #cbd5e1;" />
        <button id="isolate" class="secondary">Compute Isolation</button>
        <div id="isolation-result" style="margin-top: 12px; white-space: pre-wrap; font-family: monospace; line-height: 1.4; font-size: 12px; background: #f8fafc; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0; max-height: 300px; overflow-y: auto;"></div>
      </div>

      <!-- Restoration Tab -->
      <div id="restoration" class="tab-content">
        <h2>Supply Restoration (Obj 5)</h2>
        <p style="font-size: 13px; color: #475569; line-height: 1.5; margin-bottom: 12px;">
          Compute alternative paths for isolated segment.
        </p>
        <label for="isolated-pipes">Isolated pipes (comma-separated)</label>
        <textarea id="isolated-pipes" placeholder="e.g., 1,2,3" style="height: 60px;"></textarea>
        <label for="isolated-nodes" style="margin-top: 12px;">Isolated nodes (comma-separated)</label>
        <textarea id="isolated-nodes" placeholder="e.g., 1,2,3" style="height: 60px;"></textarea>
        <label for="customer-map-file" style="margin-top: 12px;">Customer map (JSON)</label>
        <input type="file" id="customer-map-file" accept="application/json" />
        <button id="load-customer-map-button" class="secondary" style="margin-top:8px;">Load Customer Map File</button>
        <button id="use-customer-map-button" style="margin-top:8px;">Use Customer Map (from textarea)</button>
        <textarea id="customer-map-text" placeholder='{"5":120, "6":80}' style="height: 80px; margin-top:8px;"></textarea>
        <button id="restore" class="secondary">Compute Restoration</button>
        <div id="restoration-result" style="margin-top: 12px; white-space: pre-wrap; font-family: monospace; line-height: 1.4; font-size: 12px; background: #f8fafc; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0; max-height: 300px; overflow-y: auto;"></div>
      </div>

      <!-- Metrics Tab -->
      <div id="metrics" class="tab-content">
        <h2>Performance Metrics (Obj 6)</h2>
        <p style="font-size: 13px; color: #475569; line-height: 1.5; margin-bottom: 12px;">
          System performance against literature targets.
        </p>
        <button id="load-metrics" class="secondary">Load Metrics Report</button>
        <div id="metrics-result" style="margin-top: 12px;"></div>
      </div>
    </div>
    
    <div class="panel">
      <h2>Network Visualization</h2>
      <div style="margin-top:8px;margin-bottom:8px;">
        <label style="font-size:13px;color:#475569;">
          <input type="checkbox" id="overlay-toggle" checked style="margin-right:6px;" />
          Show restoration overlay (toggle restored pipes)
        </label>
      </div>
      <svg id="network-svg" viewBox="0 0 1000 700"></svg>
    </div>
  </main>
  <script>
    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const tabName = btn.getAttribute('data-tab');
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(tabName).classList.add('active');
      });
    });

    // State
    const status = document.getElementById('status');
    const resultPanel = document.getElementById('prediction-result');
    const pipeSelect = document.getElementById('pipe');
    const faultTypeSelect = document.getElementById('fault-type');
    const assignFaultButton = document.getElementById('assign-fault');
    const assignmentsPanel = document.getElementById('assignments');
    const simulateButton = document.getElementById('simulate');
    const isolateButton = document.getElementById('isolate');
    const restoreButton = document.getElementById('restore');
    const loadMetricsButton = document.getElementById('load-metrics');
    const svg = document.getElementById('network-svg');
    
    let network = null;
    let selectedPipe = [];
    let prediction = null;
    let predictedZonePipes = [];
    let pipeFaultMap = {};
    let lastIsolationResult = null;
    let lastRestorationResult = null;
    let isolatedPipes = [];
    let restorationPathPipes = [];
    let customerMap = null;

    function setStatus(text) {
      status.textContent = text;
    }

    function updateAssignmentsPanel() {
      const entries = Object.entries(pipeFaultMap);
      if (!entries.length) {
        assignmentsPanel.textContent = 'No assignments yet.';
        return;
      }
      assignmentsPanel.innerHTML = entries
        .map(([pipe, type]) => `<div>Pipe ${pipe}: ${type}</div>`)
        .join('');
    }

    window.addEventListener('error', event => {
      setStatus('JavaScript error: ' + event.message);
      console.error('Dashboard error', event.error || event.message);
    });

    function renderPrediction() {
      if (!prediction) {
        resultPanel.textContent = 'Run a simulation to see results.';
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
        if (!r.ok) throw new Error(`Network API error ${r.status}`);
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
        option.textContent = `${pipe.id} (${pipe.source}->${pipe.target})`;
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
      const width = 1000, height = 700;
      const nodeById = {};
      network.nodes.forEach(node => { nodeById[node.id] = node; });
      // decide overlay toggle state
      const showRestoration = document.getElementById('overlay-toggle') ? document.getElementById('overlay-toggle').checked : true;
      network.edges = network.pipes.map(pipe => ({
        ...pipe,
        selected: Array.isArray(selectedPipe) ? selectedPipe.includes(pipe.id) : pipe.id === selectedPipe,
        predicted: predictedZonePipes.includes(pipe.id),
        isolated: Array.isArray(isolatedPipes) ? isolatedPipes.includes(pipe.id) : false,
        restored: Array.isArray(restorationPathPipes) ? restorationPathPipes.includes(pipe.id) : false,
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
        // Priority: selected (user) > restored (if toggle on) > isolated (closed) > predicted (detection)
        const selected = Array.isArray(selectedPipe) ? selectedPipe.includes(edge.id) : edge.id === selectedPipe;
        if (selected) {
          cssClass = 'edge selected';
        } else if (showRestoration && edge.restored) {
          cssClass = 'edge restored';
        } else if (edge.isolated) {
          cssClass = 'edge isolated';
        } else if (edge.predicted) {
          cssClass = 'edge predicted';
        }
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
      // build valve->node mapping from isolation result
      const valveByNode = {};
      if (lastIsolationResult && Array.isArray(lastIsolationResult.valve_ids)) {
        lastIsolationResult.valve_ids.forEach(v => {
          let nodeId = null;
          if (v.startsWith('V_start_')) nodeId = v.slice('V_start_'.length);
          else if (v.startsWith('V_end_')) nodeId = v.slice('V_end_'.length);
          else if (v.startsWith('V_node_')) nodeId = v.slice('V_node_'.length);
          else {
            const parts = v.split('_'); nodeId = parts[parts.length - 1];
          }
          if (nodeId) {
            valveByNode[nodeId] = valveByNode[nodeId] || [];
            valveByNode[nodeId].push(v);
          }
        });
      }
      const valveActions = lastRestorationResult && lastRestorationResult.valve_changes ? lastRestorationResult.valve_changes : {};

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
        // render valve labels if present
        const valves = valveByNode[node.id] || [];
        if (valves.length) {
          const vtext = valves.map(v => {
            const action = valveActions[v];
            return action ? `${v}(${action})` : v;
          }).join(', ');
          const vl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
          vl.setAttribute('x', 16);
          vl.setAttribute('y', 20);
          vl.setAttribute('class', 'valve-label');
          vl.textContent = vtext;
          group.appendChild(vl);
        }
        // render customer count if customerMap is set
        if (customerMap && customerMap[node.id]) {
          const cl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
          cl.setAttribute('x', 16);
          cl.setAttribute('y', valves.length ? 33 : 20);
          cl.setAttribute('class', 'customer-label');
          cl.textContent = `Customers: ${customerMap[node.id]}`;
          group.appendChild(cl);
        }
        svg.appendChild(group);
      });
    }

    async function simulate() {
      const assignedFaults = Object.entries(pipeFaultMap).map(([pipe, scenario]) => ({ pipe, scenario }));
      const unassigned = selectedPipe.filter(pipe => !pipeFaultMap.hasOwnProperty(pipe));
      const defaultType = faultTypeSelect.value;
      const unassignedFaults = unassigned.map(pipe => ({ pipe, scenario: defaultType }));
      const faults = [...assignedFaults, ...unassignedFaults];
      if (faults.length === 0 && defaultType !== 'normal') {
        setStatus('Please choose at least one pipe to fault.');
        return;
      }
      setStatus('Running simulation...');
      try {
        const res = await fetch('/simulate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ faults }),
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

    async function computeIsolation() {
      const pipeId = document.getElementById('faulty-pipe').value;
      if (!pipeId) {
        alert('Enter faulty pipe ID');
        return;
      }
      const resultDiv = document.getElementById('isolation-result');
      resultDiv.textContent = 'Computing isolation...';
      try {
        const res = await fetch('/isolate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ pipe_id: pipeId }),
        });
        const data = await res.json();
        if (data.error) {
          resultDiv.textContent = 'Error: ' + data.error;
        } else {
          let text = `VALVE ISOLATION RESULT\n` +
            `======================\n\n` +
            `Valve IDs to close (${data.valve_ids.length}):\n  ${data.valve_ids.join(', ')}\n\n` +
            `Isolation segment pipes (${data.isolation_segment_pipes.length}):\n  ${data.isolation_segment_pipes.join(', ')}\n\n` +
            `Isolation segment nodes (${data.isolation_segment_nodes.length}):\n  ${data.isolation_segment_nodes.join(', ')}\n\n` +
            `Customers affected: ${data.customers_affected}\n` +
            `Feasible: ${data.feasible ? 'YES' : 'NO'}`;
          if (data.alternative_configs && data.alternative_configs.length > 0) {
            text += `\n\nAlternative configurations:`;
            data.alternative_configs.forEach((alt, i) => {
              text += `\n  ${i + 1}. ${alt.strategy}: ${alt.customers_affected} customers`;
            });
          }
          resultDiv.textContent = text;
          lastIsolationResult = data;
          // update isolated pipes state and auto-fill restoration inputs
          isolatedPipes = Array.isArray(data.isolation_segment_pipes) ? data.isolation_segment_pipes.slice() : [];
          restorationPathPipes = [];
          document.getElementById('isolated-pipes').value = data.isolation_segment_pipes?.join(', ') || '';
          document.getElementById('isolated-nodes').value = data.isolation_segment_nodes?.join(', ') || '';
          drawNetwork();
        }
      } catch (err) {
        resultDiv.textContent = 'Request failed: ' + err.message;
      }
    }

    async function computeRestoration() {
      const pipesText = document.getElementById('isolated-pipes').value;
      const nodesText = document.getElementById('isolated-nodes').value;
      let pipes = pipesText ? pipesText.split(',').map(s => s.trim()) : [];
      let nodes = nodesText ? nodesText.split(',').map(s => s.trim()) : [];
      if ((!pipes.length || !nodes.length) && lastIsolationResult) {
        pipes = pipes.length ? pipes : lastIsolationResult.isolation_segment_pipes || [];
        nodes = nodes.length ? nodes : lastIsolationResult.isolation_segment_nodes || [];
      }
      
      const resultDiv = document.getElementById('restoration-result');
      resultDiv.textContent = 'Computing restoration...';
      try {
        const res = await fetch('/restore', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ isolated_pipes: pipes, isolated_nodes: nodes, customer_map: customerMap }),
        });
        const data = await res.json();
        if (data.error) {
          resultDiv.textContent = 'Error: ' + data.error;
        } else {
          const alternativePaths = Array.isArray(data.alternative_paths) ? data.alternative_paths : [];
          const valveChanges = data.valve_changes || {};
          let valveChangeText = 'none';
          if (Object.keys(valveChanges).length > 0) {
            valveChangeText = Object.entries(valveChanges)
              .map(([k, v]) => `${k}=${v}`)
              .join(', ');
          }
          let text = `SUPPLY RESTORATION RESULT\n` +
            `=========================\n\n` +
            `Alternative paths: ${alternativePaths.length}\n` +
            `Valve changes: ${valveChangeText}\n` +
            `Restored customers: ${data.restored_customers ?? 'N/A'}\n` +
            `Feasible: ${data.feasible ? 'YES' : 'NO'}\n` +
            `Validation status: ${data.validation_status ?? 'N/A'}\n` +
            `Notes: ${data.notes ?? 'None'}`;
          if (alternativePaths.length > 0) {
            alternativePaths.forEach((path, i) => {
              text += `\n\nPath ${i + 1}:`;
              text += `\n  Source: ${path.source_node ?? 'N/A'}`;
              text += `\n  Target: ${path.target_node ?? 'N/A'}`;
              text += `\n  Nodes: ${Array.isArray(path.path_nodes) ? path.path_nodes.join(' -> ') : 'N/A'}`;
              text += `\n  Pipes: ${Array.isArray(path.path_pipes) ? path.path_pipes.join(', ') : 'N/A'}`;
              text += `\n  Length score: ${path.path_length?.toFixed ? path.path_length.toFixed(1) : path.path_length ?? 'N/A'}`;
              text += `\n  Valves to open: ${Array.isArray(path.valves_to_open) && path.valves_to_open.length ? path.valves_to_open.join(', ') : 'none'}`;
              text += `\n  Priority: ${path.priority?.toFixed ? path.priority.toFixed(2) : path.priority ?? 'N/A'}`;
            });
          } else {
            text += `\n\nNo alternative restoration paths were found.`;
          }
          resultDiv.textContent = text;
          // save restoration result for valve actions and mark restoration path pipes for visualization
          lastRestorationResult = data;
          restorationPathPipes = [];
          if (Array.isArray(alternativePaths) && alternativePaths.length > 0) {
            alternativePaths.forEach(p => {
              if (Array.isArray(p.path_pipes)) {
                p.path_pipes.forEach(pp => restorationPathPipes.push(pp));
              }
            });
            // dedupe
            restorationPathPipes = Array.from(new Set(restorationPathPipes));
          }
          drawNetwork();
        }
      } catch (err) {
        resultDiv.textContent = 'Request failed: ' + err.message;
      }
    }

    async function loadMetrics() {
      const resultDiv = document.getElementById('metrics-result');
      resultDiv.textContent = 'Loading metrics...';
      try {
        const res = await fetch('/metrics');
        const data = await res.json();
        if (data.error) {
          resultDiv.textContent = 'Error: ' + data.error;
        } else {
          const s = data.summary || {};
          let html = `<div class="card"><strong>SYSTEM METRICS</strong><br/>` +
            `Water Loss Reduction: ${(s.water_loss_reduction_percent || 0).toFixed(1)}% (target: >=85%)<br/>` +
            `End-to-End Latency: ${(s.end_to_end_latency_seconds || 0).toFixed(1)}s (target: <300s)<br/>` +
            `System Reliability: ${(s.system_reliability_percent || 0).toFixed(1)}% (target: >99%)</div>`;
          
          if (s.detection) {
            html += `<div class="card"><strong>DETECTION</strong><br/>` +
              `Accuracy: ${(s.detection.accuracy || 0).toFixed(3)} (target: >=0.90)<br/>` +
              `FPR: ${(s.detection.false_positive_rate || 0).toFixed(3)} (target: <0.05)<br/>` +
              `Latency: ${(s.detection.detection_latency_seconds || 0).toFixed(1)}s</div>`;
          }
          if (s.localization) {
            html += `<div class="card"><strong>LOCALIZATION</strong><br/>` +
              `Zone Accuracy: ${(s.localization.zone_accuracy || 0).toFixed(3)} (target: >=0.80)<br/>` +
              `Top-3 Accuracy: ${(s.localization.top_3_accuracy || 0).toFixed(3)}</div>`;
          }
          if (s.isolation) {
            html += `<div class="card"><strong>ISOLATION (OBJ 4)</strong><br/>` +
              `Response Time: ${(s.isolation.mean_response_time_seconds || 0).toFixed(1)}s (target: <120s)<br/>` +
              `Customers Affected: ${(s.isolation.mean_customers_affected || 0).toFixed(0)}</div>`;
          }
          if (s.restoration) {
            html += `<div class="card"><strong>RESTORATION (OBJ 5)</strong><br/>` +
              `Success Rate: ${Math.round((s.restoration.restoration_success_rate || 0) * 100)}% (target: >=60%)<br/>` +
              `Feasibility: ${Math.round((s.restoration.restoration_feasibility_rate || 0) * 100)}%</div>`;
          }
          resultDiv.innerHTML = html;
        }
      } catch (err) {
        resultDiv.textContent = 'Request failed: ' + err.message;
      }
    }

    simulateButton.addEventListener('click', simulate);
    isolateButton.addEventListener('click', computeIsolation);
    restoreButton.addEventListener('click', computeRestoration);
    // Customer map controls
    const customerMapFile = document.getElementById('customer-map-file');
    const customerMapText = document.getElementById('customer-map-text');
    const loadCustomerButton = document.getElementById('load-customer-map-button');
    const useCustomerButton = document.getElementById('use-customer-map-button');

    customerMapFile.addEventListener('change', (evt) => {
      const f = evt.target.files && evt.target.files[0];
      if (!f) return;
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const obj = JSON.parse(e.target.result);
          customerMap = obj;
          customerMapText.value = JSON.stringify(obj, null, 2);
          setStatus('Loaded customer map from file.');
        } catch (err) {
          setStatus('Invalid JSON in customer map file: ' + err.message);
        }
      };
      reader.readAsText(f);
    });

    loadCustomerButton.addEventListener('click', async () => {
      // try to fetch default dataset file from server
      try {
        const r = await fetch('/customer_map');
        if (!r.ok) throw new Error('No customer_map available');
        const obj = await r.json();
        customerMap = obj;
        customerMapText.value = JSON.stringify(obj, null, 2);
        setStatus('Loaded default customer_map from server.');
      } catch (err) {
        setStatus('Failed to load default customer_map: ' + err.message);
      }
    });

    useCustomerButton.addEventListener('click', () => {
      const txt = customerMapText.value;
      if (!txt) { setStatus('Customer map textarea empty'); return; }
      try {
        customerMap = JSON.parse(txt);
        setStatus('Customer map set from textarea.');
      } catch (err) {
        setStatus('Invalid JSON: ' + err.message);
      }
    });
    loadMetricsButton.addEventListener('click', loadMetrics);
    requestNetwork().then(populateControls).catch(err => setStatus('Failed to load network: ' + err.message));
    // auto-load default customer map
    fetch('/customer_map').then(r => r.json()).then(obj => {
      if (obj && Object.keys(obj).length) {
        customerMap = obj;
        customerMapText.value = JSON.stringify(obj, null, 2);
      }
    }).catch(() => {
      // silently skip if customer_map not available
    });
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
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
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
        if self.path.startswith("/customer_map"):
          try:
            cm_file = ROOT_DIR / "DATASETS" / "customer_map.json"
            if cm_file.exists():
              with cm_file.open('r', encoding='utf-8') as h:
                payload = h.read().encode('utf-8')
            else:
              payload = json.dumps({}).encode('utf-8')
            self._send_response(payload, "application/json")
          except Exception as exc:
            self._send_response(json.dumps({"error": str(exc)}).encode("utf-8"), "application/json", status=500)
          return
        if self.path.startswith("/metrics"):
            try:
                evaluator = PerformanceEvaluator()
                # Return empty report if no events (placeholder)
                report_data = {
                    "summary": {
                        "water_loss_reduction_percent": 0,
                        "end_to_end_latency_seconds": 0,
                        "system_reliability_percent": 99.5,
                        "detection": None,
                        "localization": None,
                        "isolation": None,
                        "restoration": None,
                    },
                    "num_events": 0,
                    "events": [],
                }
                payload = json.dumps(report_data).encode("utf-8")
                self._send_response(payload, "application/json")
            except Exception as exc:
                payload = json.dumps({"error": str(exc)}).encode("utf-8")
                self._send_response(payload, "application/json", status=500)
            return
        self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        if self.path == "/simulate":
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
            return

        if self.path == "/isolate":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else ""
            try:
                request = json.loads(body) if body else {}
                faulty_pipe_id = request.get("pipe_id")
                if not faulty_pipe_id:
                    raise ValueError("pipe_id required for isolation")
                
                manager = ValveIsolationManager(str(INP_FILE))
                result = manager.compute_isolation(faulty_pipe_id)
                payload = json.dumps(result.to_dict()).encode("utf-8")
                self._send_response(payload, "application/json")
            except Exception as exc:
                import traceback
                traceback.print_exc()
                payload = json.dumps({
                    "error": str(exc),
                    "type": exc.__class__.__name__,
                }).encode("utf-8")
                self._send_response(payload, "application/json", status=500)
            return

        if self.path == "/restore":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else ""
            try:
                request = json.loads(body) if body else {}
                isolated_pipes = request.get("isolated_pipes", [])
                isolated_nodes = request.get("isolated_nodes", [])
                customer_map = request.get("customer_map") if request.get("customer_map") is not None else None
                
                manager = SupplyRestorationManager(str(INP_FILE))
                result = manager.compute_restoration(isolated_pipes, isolated_nodes, customer_map=customer_map)
                payload = json.dumps(result.to_dict()).encode("utf-8")
                self._send_response(payload, "application/json")
            except Exception as exc:
                import traceback
                traceback.print_exc()
                payload = json.dumps({
                    "error": str(exc),
                    "type": exc.__class__.__name__,
                }).encode("utf-8")
                self._send_response(payload, "application/json", status=500)
            return

        self.send_error(404, "Not Found")


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
