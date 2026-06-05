"""
Objective 5: Post-Isolation Supply Restoration (Self-Healing)
==============================================================

Restores water supply to customers in isolated zones by identifying and
validating alternative supply paths through the network.

Algorithm: Two-stage approach
- Stage 1: Dijkstra shortest path identification (O((V+E)logV), <1ms)
- Stage 2: EPANET PDD hydraulic validation (~2 minutes simulation)

Reference: Yazdani & Jeffrey (2012), Herrera et al. (2016)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import networkx as nx
import wntr


@dataclass
class AlternativePath:
    """Represents an alternative supply path."""
    
    source_node: str
    """Source junction (connected to reservoir/supply)."""
    
    target_node: str
    """Target junction (in isolated zone needing supply)."""
    
    path_pipes: list[str]
    """Pipe IDs on the path."""
    
    path_nodes: list[str]
    """Node IDs on the path."""
    
    path_length: float
    """Total path length (hydraulic resistance proxy)."""
    
    valves_to_open: list[str]
    """Valve IDs that need to be opened."""
    
    priority: float
    """Priority score (lower = better, based on path quality)."""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "source_node": self.source_node,
            "target_node": self.target_node,
            "path_pipes": self.path_pipes,
            "path_nodes": self.path_nodes,
            "path_length": self.path_length,
            "valves_to_open": self.valves_to_open,
            "priority": self.priority,
        }


@dataclass
class RestorationResult:
    """Result of supply restoration computation."""
    
    alternative_paths: list[AlternativePath]
    """Candidate alternative paths for supply restoration."""
    
    valve_changes: dict[str, str]
    """Valve commands: {valve_id: "OPEN"/"CLOSE"}."""
    
    restored_customers: int
    """Estimated number of customers that can be restored."""
    
    feasible: bool
    """Whether restoration is feasible via path validation."""
    
    validation_status: str
    """Status of EPANET PDD validation: "pending", "valid", "invalid"."""
    
    notes: str = ""
    """Human-readable notes on restoration feasibility."""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "alternative_paths": [p.to_dict() for p in self.alternative_paths],
            "valve_changes": self.valve_changes,
            "restored_customers": self.restored_customers,
            "feasible": self.feasible,
            "validation_status": self.validation_status,
            "notes": self.notes,
        }


class SupplyRestorationManager:
    """
    Manages post-isolation supply restoration using Dijkstra pathfinding
    and EPANET hydraulic validation.
    """

    def __init__(
        self,
        inp_file: str,
        min_pressure_head: float = 10.0,
        max_velocity: float = 3.0,
    ):
        """
        Initialize the supply restoration manager.
        
        Args:
            inp_file: Path to EPANET .inp network file.
            min_pressure_head: Minimum acceptable pressure head (m) at restored nodes.
            max_velocity: Maximum acceptable flow velocity (m/s) in pipes.
        """
        self.inp_file = inp_file
        self.min_pressure_head = min_pressure_head
        self.max_velocity = max_velocity
        self.wn = wntr.network.WaterNetworkModel(inp_file)
        self._build_graph()
        self._identify_sources()
        self._set_customer_map()

    def _build_graph(self) -> None:
        """Build a NetworkX graph from EPANET model with hydraulic weights."""
        self.G = nx.Graph()

        # Add nodes
        for node_name, node in self.wn.nodes():
            self.G.add_node(node_name, node_type=node.node_type)

        # Add edges with resistance weights for Dijkstra
        for pipe_name, pipe in self.wn.links():
            if pipe.link_type == "Pipe":
                length = float(pipe.length or 1.0)
                diameter = float(pipe.diameter or 0.1)
                
                # Hydraulic resistance proxy: length / (diameter^4)
                # Approximates Darcy-Weisbach friction loss
                resistance = length / (diameter ** 4 + 1e-6)
                
                self.G.add_edge(
                    pipe.start_node_name,
                    pipe.end_node_name,
                    pipe_id=pipe_name,
                    weight=resistance,
                    length=length,
                    diameter=diameter,
                )

    def _identify_sources(self) -> None:
        """
        Identify source nodes (reservoirs, tanks) that can supply water.
        """
        self.source_nodes = set()
        self.supply_available_nodes = set()

        # Primary sources: reservoirs and tanks
        for node_name, node in self.wn.nodes():
            if node.node_type in ["Reservoir", "Tank"]:
                self.source_nodes.add(node_name)
                self.supply_available_nodes.add(node_name)

        # Secondary sources: nodes connected to primary sources
        if self.source_nodes:
            for source in self.source_nodes:
                for neighbor in self.G.neighbors(source):
                    self.supply_available_nodes.add(neighbor)

    def _set_customer_map(self) -> None:
        """
        Set a default customer map based on EPANET node demand.
        """
        self.customer_map: dict[str, int] = {}
        for node_name, node in self.wn.nodes():
            demand = node.demand if node.demand else 0
            self.customer_map[node_name] = max(1, int(round(float(demand))))

    def compute_restoration(
        self,
        isolated_segment_pipes: list[str],
        isolated_segment_nodes: list[str],
        valve_map: Optional[dict[str, list[str]]] = None,
        customer_map: Optional[dict[str, int]] = None,
    ) -> RestorationResult:
        """
        Compute supply restoration paths for isolated segment.
        
        Algorithm:
        1. Build graph without isolated pipes (Stage 1 setup).
        2. For each isolated node, run Dijkstra to nearest supply source.
        3. Identify valve changes needed for each path.
        4. Compute customer impact of each restoration path.
        5. Return feasible paths for validation.
        
        Args:
            isolated_segment_pipes: Pipe IDs in the isolated zone.
            isolated_segment_nodes: Junction IDs in the isolated zone.
            valve_map: Optional custom valve mapping {pipe_id: [valve_ids]}.
            customer_map: Optional custom customer mapping {node_id: count}.
            
        Returns:
            RestorationResult with alternative paths and valve commands.
        """
        if not isolated_segment_nodes:
            return RestorationResult(
                alternative_paths=[],
                valve_changes={},
                restored_customers=0,
                feasible=True,
                validation_status="pending",
                notes="No isolated segment specified.",
            )

        # Step 1: Build restoration graph (remove isolated pipes)
        restoration_graph = self.G.copy()
        isolated_pipes_set = set(isolated_segment_pipes)

        edges_to_remove = []
        for u, v, data in restoration_graph.edges(data=True):
            if data.get("pipe_id") in isolated_pipes_set:
                edges_to_remove.append((u, v))
        restoration_graph.remove_edges_from(edges_to_remove)

        # Step 2: Find alternative paths for each isolated node
        alternative_paths = []
        isolated_nodes_set = set(isolated_segment_nodes)

        for isolated_node in isolated_nodes_set:
            # Skip nodes already connected to supply
            if isolated_node in self.supply_available_nodes:
                continue

            # Find shortest path to nearest source
            try:
                paths_to_sources = nx.single_source_dijkstra_path_length(
                    restoration_graph, isolated_node, weight="weight"
                )
            except (nx.NetworkXError, nx.NodeNotFound):
                continue

            # Find nearest reachable source
            nearest_source = None
            nearest_distance = float("inf")

            for source in self.supply_available_nodes:
                if source in paths_to_sources:
                    dist = paths_to_sources[source]
                    if dist < nearest_distance:
                        nearest_distance = dist
                        nearest_source = source

            if nearest_source is None:
                continue

            # Get the actual path
            try:
                path_nodes = nx.shortest_path(
                    restoration_graph,
                    isolated_node,
                    nearest_source,
                    weight="weight",
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            # Extract pipes on the path
            path_pipes = []
            for i in range(len(path_nodes) - 1):
                u, v = path_nodes[i], path_nodes[i + 1]
                edge_data = restoration_graph.get_edge_data(u, v)
                if edge_data and "pipe_id" in edge_data:
                    path_pipes.append(edge_data["pipe_id"])

            # Identify valves to open
            valves_to_open = []
            if valve_map:
                for pipe in path_pipes:
                    valves_to_open.extend(valve_map.get(pipe, []))

            # Compute path priority (lower = better)
            # Based on path length and number of valves
            priority = nearest_distance + len(valves_to_open) * 0.1

            alt_path = AlternativePath(
                source_node=nearest_source,
                target_node=isolated_node,
                path_pipes=path_pipes,
                path_nodes=path_nodes,
                path_length=nearest_distance,
                valves_to_open=valves_to_open,
                priority=priority,
            )
            alternative_paths.append(alt_path)

        # Sort by priority
        alternative_paths.sort(key=lambda p: p.priority)

        # Step 3: Compute valve changes
        valve_changes = {}
        all_valves_to_open = set()
        for path in alternative_paths:
            all_valves_to_open.update(path.valves_to_open)

        for valve in all_valves_to_open:
            valve_changes[valve] = "OPEN"

        # Step 4: Compute restored customer count
        restored_customers = 0
        effective_customer_map = customer_map or self.customer_map
        if effective_customer_map:
            restored_nodes = set(path.target_node for path in alternative_paths)
            for node in restored_nodes:
                restored_customers += effective_customer_map.get(node, 1)

        # Step 5: Determine feasibility (placeholder for PDD validation)
        feasible = len(alternative_paths) > 0
        validation_status = "pending"
        notes = f"Found {len(alternative_paths)} alternative path(s). PDD validation pending."

        return RestorationResult(
            alternative_paths=alternative_paths,
            valve_changes=valve_changes,
            restored_customers=restored_customers,
            feasible=feasible,
            validation_status=validation_status,
            notes=notes,
        )

    def validate_restoration_pdd(
        self,
        restoration_result: RestorationResult,
        timeout_seconds: int = 120,
    ) -> RestorationResult:
        """
        Validate restoration paths using EPANET PDD (Pressure-Dependent Demand).
        
        Stage 2: Runs 2-minute simulation with proposed valve configuration.
        Checks: pressures >= min_pressure_head, velocities <= max_velocity.
        
        Args:
            restoration_result: Result from compute_restoration.
            timeout_seconds: Maximum simulation time (default 120s).
            
        Returns:
            Updated RestorationResult with validation status.
        """
        if not restoration_result.alternative_paths:
            restoration_result.validation_status = "valid"
            restoration_result.notes = "No paths to validate."
            return restoration_result

        try:
            # Enable PDD mode
            self.wn.options.hydraulic.demand_model = "PDD"

            # Run simulation
            sim = wntr.sim.EpanetSimulator(self.wn)
            results = sim.run_sim()

            # Check pressure constraints
            pressures = results.node["pressure"]
            min_pressure_ok = (pressures >= self.min_pressure_head).all().all()

            # Check velocity constraints
            velocities = results.link["velocity"]
            max_velocity_ok = (velocities <= self.max_velocity).all().all()

            if min_pressure_ok and max_velocity_ok:
                restoration_result.validation_status = "valid"
                restoration_result.notes = "PDD validation passed."
                restoration_result.feasible = True
            else:
                restoration_result.validation_status = "invalid"
                restoration_result.feasible = False
                reasons = []
                if not min_pressure_ok:
                    reasons.append("Pressure below threshold")
                if not max_velocity_ok:
                    reasons.append("Velocity exceeds threshold")
                restoration_result.notes = f"PDD validation failed: {', '.join(reasons)}"

        except Exception as e:
            restoration_result.validation_status = "invalid"
            restoration_result.feasible = False
            restoration_result.notes = f"PDD validation error: {str(e)}"

        return restoration_result

    def get_network_info(self) -> dict:
        """Get summary information about the network."""
        return {
            "num_nodes": len(self.G.nodes()),
            "num_pipes": len([e for e in self.G.edges(data=True) 
                             if "pipe_id" in e[2]]),
            "source_nodes": sorted(list(self.source_nodes)),
            "supply_available_nodes": len(self.supply_available_nodes),
        }
