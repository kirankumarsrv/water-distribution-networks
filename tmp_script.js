
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
      let text = `Fault assignments: ${prediction.faults?.length ? prediction.faults.map(f => `${f.pipe}:${f.scenario}`).join(', ') : 'none'}
` +
        `Actual zones: ${actualZones.length ? actualZones.join(', ') : 'N/A'}
` +
        `Detected label: ${det.label} (${det.label === 0 ? 'Normal' : 'Fault'})
` +
        `Detection confidence: ${det.confidence?.toFixed?.(3) ?? 'N/A'}
` +
        `Predicted zone: ${predictedZone}
` +
        `Zone confidence: ${loc.zone_confidence?.toFixed?.(3) ?? 'N/A'}`;
      if (Array.isArray(loc.top_zones) && loc.top_zones.length > 0) {
        text += "
Zone ranking:";
        loc.top_zones.slice(0, 5).forEach((zone, idx) => {
          text += `
  ${idx + 1}. Zone ${zone.zone_id} (${(zone.probability * 100).toFixed(1)}%)`;
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
          let text = `VALVE ISOLATION RESULT
` +
            `======================

` +
            `Valve IDs to close (${data.valve_ids.length}):
  ${data.valve_ids.join(', ')}

` +
            `Isolation segment pipes (${data.isolation_segment_pipes.length}):
  ${data.isolation_segment_pipes.join(', ')}

` +
            `Isolation segment nodes (${data.isolation_segment_nodes.length}):
  ${data.isolation_segment_nodes.join(', ')}

` +
            `Customers affected: ${data.customers_affected}
` +
            `Feasible: ${data.feasible ? 'YES' : 'NO'}`;
          if (data.alternative_configs && data.alternative_configs.length > 0) {
            text += `

Alternative configurations:`;
            data.alternative_configs.forEach((alt, i) => {
              text += `
  ${i + 1}. ${alt.strategy}: ${alt.customers_affected} customers`;
            });
          }
          resultDiv.textContent = text;
        }
      } catch (err) {
        resultDiv.textContent = 'Request failed: ' + err.message;
      }
    }

    async function computeRestoration() {
      const pipesText = document.getElementById('isolated-pipes').value;
      const nodesText = document.getElementById('isolated-nodes').value;
      const pipes = pipesText ? pipesText.split(',').map(s => s.trim()) : [];
      const nodes = nodesText ? nodesText.split(',').map(s => s.trim()) : [];
      
      const resultDiv = document.getElementById('restoration-result');
      resultDiv.textContent = 'Computing restoration...';
      try {
        const res = await fetch('/restore', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ isolated_pipes: pipes, isolated_nodes: nodes }),
        });
        const data = await res.json();
        if (data.error) {
          resultDiv.textContent = 'Error: ' + data.error;
        } else {
          let text = `SUPPLY RESTORATION RESULT
` +
            `=========================

` +
            `Alternative paths: ${data.alternative_paths.length}
` +
            `Valve changes: ${Object.entries(data.valve_changes).map(([k,v]) => `${k}=${v}`).join(', ')}
` +
            `Restored customers: ${data.restored_customers}
` +
            `Feasible: ${data.feasible ? 'YES' : 'NO'}
` +
            `Validation status: ${data.validation_status}
` +
            `Notes: ${data.notes}`;
          if (data.alternative_paths && data.alternative_paths.length > 0) {
            text += `

Paths:`;
            data.alternative_paths.forEach((path, i) => {
              text += `
  ${i + 1}. ${path.source_node} -> ${path.target_node} (${path.path_pipes.length} pipes)`;
            });
          }
          resultDiv.textContent = text;
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
    loadMetricsButton.addEventListener('click', loadMetrics);
    requestNetwork().then(populateControls).catch(err => setStatus('Failed to load network: ' + err.message));
  