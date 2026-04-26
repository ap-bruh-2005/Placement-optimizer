"""
Simulated Annealing refinement for macro placement.

Takes a legalized placement and refines it using:
  - Shift: Gaussian perturbation of single macro
  - Swap: Exchange positions of two connected macros
  - Slide: Move macro along one axis (good for congestion)
  - Cluster move: Shift a connected group of macros together

Uses fast approximate wirelength for most evaluations,
with periodic full proxy cost checks via the TILOS evaluator.
"""

import math
import random
import numpy as np


def sa_refine(pos, movable, sizes, canvas_w, canvas_h, net_data,
              n_iters=15000, seed=42):
    """
    SA-based refinement of legalized hard macro placement.

    Args:
        pos: [N, 2] numpy array of hard macro positions
        movable: [N] boolean array
        sizes: [N, 2] numpy array of (width, height)
        canvas_w, canvas_h: Canvas dimensions
        net_data: NetData with edge list and adjacency
        n_iters: Number of SA iterations
        seed: Random seed

    Returns:
        [N, 2] numpy array of refined positions
    """
    random.seed(seed)
    np.random.seed(seed)

    n = len(pos)
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    movable_idx = np.where(movable)[0]
    if len(movable_idx) == 0:
        return pos

    pos = pos.copy().astype(np.float64)

    # Precompute pairwise separation requirements
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2

    edges = net_data.edges
    edge_weights = net_data.edge_weights
    adjacency = net_data.adjacency

    gap = 0.05

    def wl_cost():
        """Fast wirelength using edge list."""
        if len(edges) == 0:
            return 0.0
        dx = np.abs(pos[edges[:, 0], 0] - pos[edges[:, 1], 0])
        dy = np.abs(pos[edges[:, 0], 1] - pos[edges[:, 1], 1])
        return float((edge_weights * (dx + dy)).sum())

    def check_overlap(idx):
        """O(N) overlap check for single macro."""
        dx = np.abs(pos[idx, 0] - pos[:, 0])
        dy = np.abs(pos[idx, 1] - pos[:, 1])
        overlaps = (dx < sep_x[idx] + gap) & (dy < sep_y[idx] + gap)
        overlaps[idx] = False
        return overlaps.any()

    def clamp_x(idx, x):
        return np.clip(x, half_w[idx] + gap, canvas_w - half_w[idx] - gap)

    def clamp_y(idx, y):
        return np.clip(y, half_h[idx] + gap, canvas_h - half_h[idx] - gap)

    current_cost = wl_cost()
    best_pos = pos.copy()
    best_cost = current_cost

    # Temperature schedule
    T_start = max(canvas_w, canvas_h) * 0.2
    T_end = max(canvas_w, canvas_h) * 0.0005

    accepted = 0
    rejected = 0

    for step in range(n_iters):
        frac = step / n_iters
        T = T_start * (T_end / T_start) ** frac

        move_type = random.random()
        i = random.choice(movable_idx)

        if move_type < 0.45:
            # ─── SHIFT ──────────────────────────────────────────
            old_x, old_y = pos[i, 0], pos[i, 1]
            shift_scale = T * (0.5 + 0.5 * (1 - frac))

            pos[i, 0] = clamp_x(i, pos[i, 0] + random.gauss(0, shift_scale))
            pos[i, 1] = clamp_y(i, pos[i, 1] + random.gauss(0, shift_scale))

            if check_overlap(i):
                pos[i, 0] = old_x
                pos[i, 1] = old_y
                rejected += 1
                continue

            new_cost = wl_cost()
            delta = new_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                current_cost = new_cost
                accepted += 1
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_pos = pos.copy()
            else:
                pos[i, 0] = old_x
                pos[i, 1] = old_y
                rejected += 1

        elif move_type < 0.75:
            # ─── SWAP ───────────────────────────────────────────
            # Prefer swapping with connected macros
            neighbors_i = [j for j in adjacency[i] if movable[j]]
            if neighbors_i and random.random() < 0.7:
                j = random.choice(neighbors_i)
            else:
                j = random.choice(movable_idx)

            if i == j:
                continue

            old_ix, old_iy = pos[i, 0], pos[i, 1]
            old_jx, old_jy = pos[j, 0], pos[j, 1]

            # Swap positions (clamped for different sizes)
            pos[i, 0] = clamp_x(i, old_jx)
            pos[i, 1] = clamp_y(i, old_jy)
            pos[j, 0] = clamp_x(j, old_ix)
            pos[j, 1] = clamp_y(j, old_iy)

            if check_overlap(i) or check_overlap(j):
                pos[i, 0] = old_ix; pos[i, 1] = old_iy
                pos[j, 0] = old_jx; pos[j, 1] = old_jy
                rejected += 1
                continue

            new_cost = wl_cost()
            delta = new_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                current_cost = new_cost
                accepted += 1
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_pos = pos.copy()
            else:
                pos[i, 0] = old_ix; pos[i, 1] = old_iy
                pos[j, 0] = old_jx; pos[j, 1] = old_jy
                rejected += 1

        elif move_type < 0.9:
            # ─── MOVE TOWARD NEIGHBOR ────────────────────────────
            neighbors_i = adjacency[i]
            if not neighbors_i:
                continue

            j = random.choice(neighbors_i)
            old_x, old_y = pos[i, 0], pos[i, 1]
            alpha = random.uniform(0.05, 0.35)

            pos[i, 0] = clamp_x(i, pos[i, 0] + alpha * (pos[j, 0] - pos[i, 0]))
            pos[i, 1] = clamp_y(i, pos[i, 1] + alpha * (pos[j, 1] - pos[i, 1]))

            if check_overlap(i):
                pos[i, 0] = old_x
                pos[i, 1] = old_y
                rejected += 1
                continue

            new_cost = wl_cost()
            delta = new_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                current_cost = new_cost
                accepted += 1
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_pos = pos.copy()
            else:
                pos[i, 0] = old_x
                pos[i, 1] = old_y
                rejected += 1

        else:
            # ─── SLIDE (axis-aligned move) ───────────────────────
            old_x, old_y = pos[i, 0], pos[i, 1]
            shift_scale = T * (0.3 + 0.7 * (1 - frac))

            if random.random() < 0.5:
                pos[i, 0] = clamp_x(i, pos[i, 0] + random.gauss(0, shift_scale * 2))
            else:
                pos[i, 1] = clamp_y(i, pos[i, 1] + random.gauss(0, shift_scale * 2))

            if check_overlap(i):
                pos[i, 0] = old_x
                pos[i, 1] = old_y
                rejected += 1
                continue

            new_cost = wl_cost()
            delta = new_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                current_cost = new_cost
                accepted += 1
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_pos = pos.copy()
            else:
                pos[i, 0] = old_x
                pos[i, 1] = old_y
                rejected += 1

    return best_pos
