"""
Net/connectivity data structures for efficient placement optimization.

Builds GPU-friendly tensor representations of net connectivity from
the Benchmark object for use in differentiable wirelength computation
and SA neighbor selection.
"""

import torch
import numpy as np
from macro_place.benchmark import Benchmark


class NetData:
    """Pre-computed net connectivity data for efficient placement optimization."""

    def __init__(self, benchmark: Benchmark, device: torch.device = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.num_hard = benchmark.num_hard_macros
        self.num_macros = benchmark.num_macros
        self.num_nets = benchmark.num_nets

        # Build edge list (pairs of connected hard macros with weights)
        self.edges, self.edge_weights = self._build_edges(benchmark)

        # Build adjacency lists for SA neighbor selection
        self.adjacency = self._build_adjacency()

        # Build pin-level net data for differentiable wirelength
        self.pin_data = self._build_pin_data(benchmark)

    def _build_edges(self, benchmark: Benchmark):
        """Build weighted edge list between hard macros from net connectivity."""
        edge_dict = {}

        for net_idx in range(benchmark.num_nets):
            nodes = benchmark.net_nodes[net_idx]
            # Filter to hard macros only
            hard_nodes = nodes[nodes < self.num_hard].tolist()
            if len(hard_nodes) < 2:
                continue

            weight = 1.0 / (len(hard_nodes) - 1)
            for i in range(len(hard_nodes)):
                for j in range(i + 1, len(hard_nodes)):
                    pair = (hard_nodes[i], hard_nodes[j])
                    edge_dict[pair] = edge_dict.get(pair, 0.0) + weight

        if not edge_dict:
            return (np.zeros((0, 2), dtype=np.int64),
                    np.zeros(0, dtype=np.float64))

        edges = np.array(list(edge_dict.keys()), dtype=np.int64)
        weights = np.array([edge_dict[tuple(e)] for e in edges], dtype=np.float64)
        return edges, weights

    def _build_adjacency(self):
        """Build adjacency lists from edge data."""
        adj = [[] for _ in range(self.num_hard)]
        for i, (u, v) in enumerate(self.edges):
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def _build_pin_data(self, benchmark: Benchmark):
        """
        Build padded tensors for differentiable pin-level HPWL computation.

        Returns a dict with:
          - net_pin_x_offsets: [num_nets, max_pins] pin x-offsets relative to owner center
          - net_pin_y_offsets: [num_nets, max_pins] pin y-offsets
          - net_pin_owners: [num_nets, max_pins] owner macro index (or -1 for padding)
          - net_pin_counts: [num_nets] actual number of pins in each net
          - owner_is_port: [num_nets, max_pins] bool, True if owner is an I/O port
        """
        if not benchmark.net_pin_nodes or len(benchmark.net_pin_nodes) == 0:
            return self._build_pin_data_from_net_nodes(benchmark)

        num_nets = benchmark.num_nets
        num_ports = benchmark.port_positions.shape[0]

        # Find max pins per net
        max_pins = max(len(npn) for npn in benchmark.net_pin_nodes) if num_nets > 0 else 0
        if max_pins == 0:
            return None

        pin_owners = torch.full((num_nets, max_pins), -1, dtype=torch.long)
        pin_x_off = torch.zeros(num_nets, max_pins, dtype=torch.float32)
        pin_y_off = torch.zeros(num_nets, max_pins, dtype=torch.float32)
        pin_counts = torch.zeros(num_nets, dtype=torch.long)
        is_port = torch.zeros(num_nets, max_pins, dtype=torch.bool)

        for net_idx in range(num_nets):
            npn = benchmark.net_pin_nodes[net_idx]  # [num_pins_in_net, 2]
            n_pins = npn.shape[0]
            pin_counts[net_idx] = n_pins

            for p in range(n_pins):
                owner_idx = npn[p, 0].item()
                pin_slot = npn[p, 1].item()

                pin_owners[net_idx, p] = owner_idx

                if owner_idx < benchmark.num_hard_macros:
                    # Hard macro — use pin offset
                    offsets = benchmark.macro_pin_offsets[owner_idx]
                    if offsets.shape[0] > pin_slot:
                        pin_x_off[net_idx, p] = offsets[pin_slot, 0]
                        pin_y_off[net_idx, p] = offsets[pin_slot, 1]
                elif owner_idx < benchmark.num_macros:
                    # Soft macro — pin at center (0, 0 offset)
                    pass
                else:
                    # I/O port — mark as port
                    is_port[net_idx, p] = True

        return {
            "pin_owners": pin_owners.to(self.device),
            "pin_x_offsets": pin_x_off.to(self.device),
            "pin_y_offsets": pin_y_off.to(self.device),
            "pin_counts": pin_counts.to(self.device),
            "is_port": is_port.to(self.device),
            "max_pins": max_pins,
            "port_positions": benchmark.port_positions.to(self.device),  # [num_ports, 2]
        }

    def _build_pin_data_from_net_nodes(self, benchmark: Benchmark):
        """Fallback: build pin data from net_nodes (macro-level, no pin offsets)."""
        num_nets = benchmark.num_nets
        if num_nets == 0:
            return None

        max_pins = max(len(nn) for nn in benchmark.net_nodes)
        pin_owners = torch.full((num_nets, max_pins), -1, dtype=torch.long)
        pin_x_off = torch.zeros(num_nets, max_pins, dtype=torch.float32)
        pin_y_off = torch.zeros(num_nets, max_pins, dtype=torch.float32)
        pin_counts = torch.zeros(num_nets, dtype=torch.long)
        is_port = torch.zeros(num_nets, max_pins, dtype=torch.bool)

        for net_idx in range(num_nets):
            nodes = benchmark.net_nodes[net_idx]
            n_pins = nodes.shape[0]
            pin_counts[net_idx] = n_pins
            for p in range(n_pins):
                owner = nodes[p].item()
                pin_owners[net_idx, p] = owner
                if owner >= benchmark.num_macros:
                    is_port[net_idx, p] = True

        return {
            "pin_owners": pin_owners.to(self.device),
            "pin_x_offsets": pin_x_off.to(self.device),
            "pin_y_offsets": pin_y_off.to(self.device),
            "pin_counts": pin_counts.to(self.device),
            "is_port": is_port.to(self.device),
            "max_pins": max_pins,
            "port_positions": benchmark.port_positions.to(self.device),
        }

    def fast_hpwl(self, positions: np.ndarray) -> float:
        """
        Compute approximate HPWL using edge list (fast, numpy).
        positions: [num_hard, 2] numpy array of hard macro positions.
        """
        if len(self.edges) == 0:
            return 0.0
        dx = np.abs(positions[self.edges[:, 0], 0] - positions[self.edges[:, 1], 0])
        dy = np.abs(positions[self.edges[:, 0], 1] - positions[self.edges[:, 1], 1])
        return float((self.edge_weights * (dx + dy)).sum())
