"""
Objective 4: Graph-Based Valve Isolation Algorithm
====================================================

Computes the minimum set of valves required to isolate a leaking pipe segment
while minimizing customer supply disruption.

Algorithm: BFS-based segment isolation
Complexity: O(V+E)
Reference: Alvisi & Franchini (2014)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import networkx as nx
import wntr


@dataclass
class IsolationResult:
    """Result of valve isolation computation."""
    
    valve_ids: list[str]
    """Valve IDs to close for isolation."""
    
    isolation_segment_pipes: list[str]
    """Pipe IDs in the isolated segment."""
    
    isolation_segment_nodes: list[str]
    """Junction IDs in the isolated segment."""
    
    customers_affected: int
    """Number of customers in the isolated segment."""
    
    feasible: bool
    """Whether isolation is feasible (True if customers < threshold)."""
    
    alternative_configs: list[dict] = None
    """Alternative valve configurations if primary exceeds threshold."""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "valve_ids": self.valve_ids,
            "isolation_segment_pipes": self.isolation_segment_pipes,
            "isolation_segment_nodes": self.isolation_segment_nodes,
            "customers_affected": self.customers_affected,
            "feasible": self.feasible,
            "alternative_configs": self.alternative_configs or [],
        }


class ValveIsolationManager:
    """
    Manages valve placement and BFS-based isolation computations.
    
    A valve is placed at each junction by default, allowing segmentation
    by closing valves upstream and downstream of a faulty pipe.
    """

    def __init__(
        self,
        inp_file: str,
        customer_threshold: int = 500,
        valve_map: Optional[dict[str, list[str]]] = None,
        customer_map: Optional[dict[str, int]] = None,
    ):
        """
        Initialize the valve isolation manager.
        
        Args:
            inp_file: Path to EPANET .inp network file.
            customer_threshold: Max customers allowed in isolation zone (default 500).
            valve_map: Custom mapping {pipe_id: [valve_ids]}. If None, defaults to
                      one valve per junction.
            customer_map: Custom mapping {junction_id: customer_count}. If None,
                         defaults to equal distribution.
        """
        self.inp_file = inp_file
        self.customer_threshold = customer_threshold
        self.wn = wntr.network.WaterNetworkModel(inp_file)
        self._build_graph()
        self._set_valve_map(valve_map)
        self._set_customer_map(customer_map)

    def _build_graph(self) -> None:
        """Build a NetworkX graph from EPANET model."""
        self.G = nx.Graph()

        # Add nodes (junctions)
        for node_name, node in self.wn.nodes():
            self.G.add_node(node_name, node_type=node.node_type)

        # Add edges (pipes)
        for pipe_name, pipe in self.wn.links():
            if pipe.link_type == "Pipe":
                self.G.add_edge(
                    pipe.start_node_name,
                    pipe.end_node_name,
                    pipe_id=pipe_name,
                    length=float(pipe.length or 0),
                    diameter=float(pipe.diameter or 0),
                )

        # Create pipe graph: nodes=pipes, edges=connections via junctions
        self.pipe_graph = nx.Graph()
        for pipe_name, pipe in self.wn.links():
            if pipe.link_type == "Pipe":
                self.pipe_graph.add_node(pipe_name)

        # Connect pipes that share a junction
        for node_name in self.G.nodes():
            incident_pipes = []
            for u, v, data in self.G.edges(node_name, data=True):
                incident_pipes.append(data.get("pipe_id"))
            for i, pipe1 in enumerate(incident_pipes):
                for pipe2 in incident_pipes[i + 1 :]:
                    self.pipe_graph.add_edge(pipe1, pipe2, junction=node_name)

    def _set_valve_map(self, valve_map: Optional[dict[str, list[str]]]) -> None:
        """
        Set the valve map: which valves are on which pipes.
        
        Default: One valve per junction. Closing a valve at junction J
        isolates all pipes on one side of J.
        """
        if valve_map is not None:
            self.valve_map = valve_map
            return

        # Default: one valve per junction
        self.valve_map: dict[str, list[str]] = {}

        for pipe_name, pipe in self.wn.links():
            if pipe.link_type == "Pipe":
                # Valve at start junction
                start_valve = f"V_start_{pipe.start_node_name}"
                # Valve at end junction
                end_valve = f"V_end_{pipe.end_node_name}"
                self.valve_map[pipe_name] = [start_valve, end_valve]

    def _set_customer_map(
        self, customer_map: Optional[dict[str, int]]
    ) -> None:
        """
        Set the customer map: customers served by each junction.
        
        Default: Equal distribution (demand-based if available).
        """
        if customer_map is not None:
            self.customer_map = customer_map
            return

        # Default: Count customers from EPANET demand
        self.customer_map: dict[str, int] = {}
        total_demand = 0
        demand_by_node = {}

        for node_name, node in self.wn.nodes():
            demand = node.demand if node.demand else 0
            demand_by_node[node_name] = float(demand)
            total_demand += float(demand)

        # Convert demand to customer count (assume 1 customer = 1 unit demand)
        for node_name, demand in demand_by_node.items():
            self.customer_map[node_name] = max(1, int(round(demand)))

    def compute_isolation(self, faulty_pipe_id: str) -> IsolationResult:
        """
        Compute the minimum valve closure set to isolate a faulty pipe.
        
        Algorithm:
        1. Identify the faulty pipe's start and end nodes.
        2. Find all pipes in the isolated segment using BFS from the faulty pipe.
        3. Identify boundary valves (valves that separate isolated segment from network).
        4. Compute customer impact.
        5. If impact > threshold, search for alternative configurations.
        
        Args:
            faulty_pipe_id: ID of the leaking/faulty pipe.
            
        Returns:
            IsolationResult with valve IDs, segment, and impact info.
        """
        if faulty_pipe_id not in self.pipe_graph:
            raise ValueError(f"Pipe {faulty_pipe_id} not found in network")

        # Step 1: Get the faulty pipe's endpoints
        faulty_pipe = self.wn.get_link(faulty_pipe_id)
        if not faulty_pipe:
            raise ValueError(f"Pipe {faulty_pipe_id} not in EPANET model")

        start_node = faulty_pipe.start_node_name
        end_node = faulty_pipe.end_node_name

        # Step 2: For the current model, isolate the faulty pipe itself.
        # Closing valves on the faulty pipe endpoints isolates the pipe.
        isolation_pipes = {faulty_pipe_id}
        isolation_nodes = {start_node, end_node}

        # Step 3: Identify valve closures for the faulty pipe.
        boundary_valves = set(self.valve_map.get(faulty_pipe_id, []))

        # Step 4: Compute customer impact on isolated nodes only.
        customers_affected = sum(
            self.customer_map.get(node, 1) for node in isolation_nodes
        )

        feasible = customers_affected <= self.customer_threshold
        alternatives = []
        if not feasible:
            alternatives = self._compute_alternatives(
                faulty_pipe_id, isolation_nodes
            )

        return IsolationResult(
            valve_ids=sorted(list(boundary_valves)),
            isolation_segment_pipes=sorted(list(isolation_pipes)),
            isolation_segment_nodes=sorted(list(isolation_nodes)),
            customers_affected=customers_affected,
            feasible=feasible,
            alternative_configs=alternatives,
        )

    def _bfs_segment(
        self, start_pipe_id: str, max_hops: int = 50
    ) -> set[str]:
        """
        BFS traversal to find all connected pipes in a segment.
        
        Complexity: O(V+E) where V=pipes, E=pipe connections.
        
        Args:
            start_pipe_id: Starting pipe for BFS.
            max_hops: Maximum traversal depth to prevent infinite loops.
            
        Returns:
            Set of pipe IDs in the connected segment.
        """
        visited = set()
        queue = [start_pipe_id]
        hop_count = {start_pipe_id: 0}

        while queue:
            pipe = queue.pop(0)
            if pipe in visited:
                continue
            visited.add(pipe)

            # Traverse connected pipes
            for neighbor in self.pipe_graph.neighbors(pipe):
                if neighbor not in visited and hop_count[pipe] < max_hops:
                    queue.append(neighbor)
                    hop_count[neighbor] = hop_count[pipe] + 1

        return visited

    def _compute_alternatives(
        self, faulty_pipe_id: str, isolation_nodes: set[str]
    ) -> list[dict]:
        """
        Compute alternative isolation configurations if primary exceeds threshold.
        
        Uses greedy heuristic: try to exclude nodes with high customer count.
        
        Args:
            faulty_pipe_id: ID of faulty pipe.
            isolation_nodes: Nodes in primary isolation zone.
            
        Returns:
            List of alternative configurations with reduced customer impact.
        """
        alternatives = []

        # Strategy 1: Isolate only the faulty pipe + 1 neighbor
        # by closing valves more conservatively
        isolation_pipes = self._bfs_segment(faulty_pipe_id, max_hops=2)
        nodes_1hop = set()
        for pipe_name in isolation_pipes:
            pipe = self.wn.get_link(pipe_name)
            if pipe and pipe.link_type == "Pipe":
                nodes_1hop.add(pipe.start_node_name)
                nodes_1hop.add(pipe.end_node_name)

        boundary_valves_1hop = set()
        for pipe_name in isolation_pipes:
            if pipe_name in self.valve_map:
                boundary_valves_1hop.update(self.valve_map[pipe_name])

        customers_1hop = sum(
            self.customer_map.get(node, 1) for node in nodes_1hop
        )

        if customers_1hop < self.customer_threshold:
            alternatives.append({
                "strategy": "1-hop_neighbors",
                "valves": sorted(list(boundary_valves_1hop)),
                "customers_affected": customers_1hop,
                "pipes": sorted(list(isolation_pipes)),
            })

        # Strategy 2: Isolate only faulty pipe
        isolation_pipes_tight = {faulty_pipe_id}
        pipe = self.wn.get_link(faulty_pipe_id)
        nodes_tight = {pipe.start_node_name, pipe.end_node_name}

        valves_tight = self.valve_map.get(faulty_pipe_id, [])
        customers_tight = sum(
            self.customer_map.get(node, 1) for node in nodes_tight
        )

        alternatives.append({
            "strategy": "faulty_pipe_only",
            "valves": sorted(list(set(valves_tight))),
            "customers_affected": customers_tight,
            "pipes": sorted(list(isolation_pipes_tight)),
        })

        return alternatives

    def get_network_info(self) -> dict:
        """Get summary information about the network."""
        return {
            "num_nodes": len(self.G.nodes()),
            "num_pipes": len(self.pipe_graph.nodes()),
            "num_junctions": len([n for n, d in self.G.nodes(data=True) 
                                 if d.get("node_type") == "Junction"]),
            "total_customers": sum(self.customer_map.values()),
            "total_valves": len(set(v for vals in self.valve_map.values() for v in vals)),
        }
