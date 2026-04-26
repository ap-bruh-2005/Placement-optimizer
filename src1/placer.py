"""
GPU-Accelerated Analytical Macro Placer

Multi-start analytical placement pipeline:
  1. Global placement via differentiable optimization (GPU Nesterov)
  2. Greedy minimum-displacement legalization
  3. SA refinement with net-aware moves
  4. Soft macro optimization via force-directed placement

Usage:
    uv run evaluate submissions/our_placer/placer.py
    uv run evaluate submissions/our_placer/placer.py --all
    uv run evaluate submissions/our_placer/placer.py -b ibm01
"""

import sys
import time
import numpy as np
import torch
from pathlib import Path

from macro_place.benchmark import Benchmark

# Add our_placer to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from net_builder import NetData
from legalizer import legalize
from global_placer import global_place
from refiner import sa_refine


def _load_plc(name):
    """Load the PlacementCost object for soft macro optimization."""
    from macro_place.loader import load_benchmark_from_dir, load_benchmark
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc
    ng45 = {"ariane133_ng45": "ariane133", "ariane136_ng45": "ariane136",
            "nvdla_ng45": "nvdla", "mempool_tile_ng45": "mempool_tile"}
    d = ng45.get(name)
    if d:
        base = Path("external/MacroPlacement/Flows/NanGate45") / d / "netlist" / "output_CT_Grouping"
        if (base / "netlist.pb.txt").exists():
            _, plc = load_benchmark(str(base / "netlist.pb.txt"), str(base / "initial.plc"))
            return plc
    return None


class AnalyticalPlacer:
    """
    Multi-start GPU-accelerated analytical macro placer.

    Strategy:
      1. Start from initial placement (always good density) + N random starts
      2. For each: run differentiable global optimization → legalize → fast SA screen
      3. Pick best by fast wirelength → do full SA refinement
      4. Optimize soft macros
    """

    def __init__(self,
                 n_random_starts=3,
                 global_iters=600,
                 sa_iters=20000,
                 seed=42):
        self.n_random_starts = n_random_starts
        self.global_iters = global_iters
        self.sa_iters = sa_iters
        self.seed = seed

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        n_hard = benchmark.num_hard_macros
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)
        sizes_np = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
        movable = benchmark.get_movable_mask()[:n_hard].numpy()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ─── Build net data ──────────────────────────────────────────
        net_data = NetData(benchmark, device=device)

        # ─── Candidate generation: initial + random starts ───────────
        candidates = []

        # Candidate 1: Start from initial placement (usually best for density)
        try:
            init_pos = benchmark.macro_positions[:n_hard].numpy().astype(np.float64)
            gp_init = global_place(
                benchmark, net_data,
                n_iters=self.global_iters,
                device=device,
                seed=self.seed,
                init_positions=init_pos
            )
            legal_init = legalize(gp_init, movable, sizes_np, canvas_w, canvas_h)
            cost_init = net_data.fast_hpwl(legal_init)
            candidates.append(("init", legal_init, cost_init))
        except Exception as e:
            pass

        # Candidate 2: Just legalize initial placement (no gradient optimization)
        try:
            init_pos = benchmark.macro_positions[:n_hard].numpy().astype(np.float64)
            legal_direct = legalize(init_pos, movable, sizes_np, canvas_w, canvas_h)
            cost_direct = net_data.fast_hpwl(legal_direct)
            candidates.append(("direct", legal_direct, cost_direct))
        except Exception:
            pass

        # Candidates 3+: Random starts
        for start_idx in range(self.n_random_starts):
            seed_i = self.seed + (start_idx + 1) * 137
            try:
                gp_pos = global_place(
                    benchmark, net_data,
                    n_iters=self.global_iters,
                    device=device,
                    seed=seed_i,
                    init_positions=None
                )
                legal_pos = legalize(gp_pos, movable, sizes_np, canvas_w, canvas_h)
                cost = net_data.fast_hpwl(legal_pos)
                candidates.append((f"random_{start_idx}", legal_pos, cost))
            except Exception:
                continue

        # ─── Pick best candidate ─────────────────────────────────────
        if not candidates:
            # Absolute fallback
            init_pos = benchmark.macro_positions[:n_hard].numpy().astype(np.float64)
            best_pos = legalize(init_pos, movable, sizes_np, canvas_w, canvas_h)
        else:
            candidates.sort(key=lambda c: c[2])
            best_pos = candidates[0][1]

        # ─── SA refinement on best candidate ─────────────────────────
        refined_pos = sa_refine(
            best_pos, movable, sizes_np, canvas_w, canvas_h,
            net_data, n_iters=self.sa_iters, seed=self.seed
        )

        # ─── Build full placement ────────────────────────────────────
        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = torch.tensor(refined_pos, dtype=torch.float32)

        # ─── Soft macro optimization ─────────────────────────────────
        try:
            plc = _load_plc(benchmark.name)
            if plc is not None:
                full_pos = self._optimize_soft_macros(full_pos, benchmark, plc)
        except Exception:
            pass

        return full_pos

    def _optimize_soft_macros(self, full_pos, benchmark, plc):
        """Optimize soft macro positions using force-directed placement."""
        from macro_place.objective import _set_placement

        _set_placement(plc, full_pos, benchmark)

        canvas_size = max(benchmark.canvas_width, benchmark.canvas_height)
        try:
            plc.optimize_stdcells(
                use_current_loc=False,
                move_stdcells=True,
                move_macros=False,
                log_scale_conns=False,
                use_sizes=False,
                io_factor=1.0,
                num_steps=[5, 5, 5],
                max_move_distance=[canvas_size / 50] * 3,
                attract_factor=[100, 1.0e-3, 1.0e-5],
                repel_factor=[0, 1.0e6, 1.0e7],
            )

            n_hard = benchmark.num_hard_macros
            for i, macro_idx in enumerate(benchmark.soft_macro_indices):
                node = plc.modules_w_pins[macro_idx]
                x, y = node.get_pos()
                full_pos[n_hard + i, 0] = x
                full_pos[n_hard + i, 1] = y
        except Exception:
            pass

        return full_pos
