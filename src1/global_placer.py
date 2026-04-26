"""
GPU-accelerated differentiable global placement engine.

Uses Nesterov's accelerated gradient descent to optimize macro positions
starting from initial placement, focusing on wirelength reduction while
maintaining density and overlap constraints.

Key insight: the initial placement in these benchmarks is already
well-optimized for density/congestion. Our job is to improve wirelength
while keeping density comparable.
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from macro_place.benchmark import Benchmark


def global_place(benchmark: Benchmark, net_data, n_iters=1000,
                 device=None, seed=None, init_positions=None):
    """
    Run differentiable global placement.

    Args:
        benchmark: Benchmark object
        net_data: NetData object with precomputed connectivity
        n_iters: Number of Nesterov iterations
        device: torch.device (auto-detected if None)
        seed: Random seed for reproducibility
        init_positions: Optional [num_hard, 2] initial positions

    Returns:
        [num_hard, 2] numpy array of optimized hard macro positions
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        torch.manual_seed(seed)

    n_hard = benchmark.num_hard_macros
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)

    sizes = benchmark.macro_sizes[:n_hard].to(device).float()
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    movable = benchmark.get_movable_mask()[:n_hard].to(device)
    fixed_mask = ~movable

    # Initialize positions
    if init_positions is not None:
        pos = torch.tensor(init_positions, dtype=torch.float32, device=device)
    else:
        # Random initialization: spread macros across canvas
        pos = _random_init(n_hard, sizes, half_w, half_h, canvas_w, canvas_h,
                          fixed_mask, benchmark.macro_positions[:n_hard].to(device).float(),
                          device, seed)

    fixed_pos = benchmark.macro_positions[:n_hard].to(device).float()

    # Grid parameters for density
    grid_rows = benchmark.grid_rows
    grid_cols = benchmark.grid_cols
    grid_w = canvas_w / grid_cols
    grid_h = canvas_h / grid_rows
    col_centers = torch.arange(grid_cols, device=device, dtype=torch.float32) * grid_w + grid_w / 2
    row_centers = torch.arange(grid_rows, device=device, dtype=torch.float32) * grid_h + grid_h / 2

    total_area = (sizes[:, 0] * sizes[:, 1]).sum().item()
    target_density = total_area / (canvas_w * canvas_h)

    pin_data = net_data.pin_data

    # ─── Nesterov optimization ───────────────────────────────────────
    v = pos.detach().clone()
    y = pos.detach().clone()

    step_size = max(canvas_w, canvas_h) * 0.002
    best_pos = v.clone()
    best_cost = float('inf')

    for iteration in range(n_iters):
        frac = iteration / n_iters
        pos_eval = y.detach().requires_grad_(True)

        # Weight schedule: prioritize wirelength, moderate density, strong overlap
        wl_weight = 1.0
        density_weight = 0.5 + 1.0 * frac  # increase density pressure over time
        overlap_weight = 100.0 + 400.0 * min(1.0, frac * 2)

        gamma = max(0.3, 3.0 * (1 - frac))

        # ─── Compute losses ──────────────────────────────────────
        wl_loss = _lse_wirelength(pos_eval, pin_data, benchmark, device, gamma)
        density_loss = _density_loss(pos_eval, sizes, half_w, half_h,
                                      grid_rows, grid_cols, canvas_w, canvas_h,
                                      target_density, col_centers, row_centers,
                                      grid_w, grid_h, device)
        overlap_loss = _overlap_loss(pos_eval, half_w, half_h)

        total_loss = (wl_weight * wl_loss +
                      density_weight * density_loss +
                      overlap_weight * overlap_loss)

        total_loss.backward()
        grad = pos_eval.grad.detach()
        grad[fixed_mask] = 0.0

        # Gradient clipping
        grad_norm = grad.norm()
        max_grad = max(canvas_w, canvas_h) * 0.3
        if grad_norm > max_grad:
            grad = grad * (max_grad / grad_norm)

        # Nesterov update
        lr = step_size / (1.0 + 0.001 * iteration)
        v_new = y - lr * grad
        v_new = _project(v_new, half_w, half_h, canvas_w, canvas_h)
        v_new[fixed_mask] = fixed_pos[fixed_mask]

        momentum = iteration / (iteration + 3.0)
        y_new = v_new + momentum * (v_new - v)
        y_new = _project(y_new, half_w, half_h, canvas_w, canvas_h)
        y_new[fixed_mask] = fixed_pos[fixed_mask]

        v = v_new.detach()
        y = y_new.detach()

        # Track best
        if iteration % 50 == 0:
            with torch.no_grad():
                ol = _count_overlaps(v, half_w, half_h)
                cost = wl_loss.item() + 0.5 * density_loss.item()
                if ol == 0 and cost < best_cost:
                    best_cost = cost
                    best_pos = v.clone()
                elif best_cost == float('inf'):
                    best_pos = v.clone()

    return v.detach().cpu().numpy()


def _random_init(n_hard, sizes, half_w, half_h, canvas_w, canvas_h,
                 fixed_mask, fixed_positions, device, seed):
    """Random macro initialization across canvas."""
    if seed is not None:
        torch.manual_seed(seed)
    pos = torch.zeros(n_hard, 2, device=device)
    for i in range(n_hard):
        if fixed_mask[i]:
            pos[i] = fixed_positions[i]
        else:
            pos[i, 0] = half_w[i] + torch.rand(1, device=device).item() * (canvas_w - sizes[i, 0])
            pos[i, 1] = half_h[i] + torch.rand(1, device=device).item() * (canvas_h - sizes[i, 1])
    return pos


def _lse_wirelength(positions, pin_data, benchmark, device, gamma=1.0):
    """Differentiable Log-Sum-Exp HPWL."""
    if pin_data is None:
        return torch.tensor(0.0, device=device)

    n_hard = benchmark.num_hard_macros
    n_macros = benchmark.num_macros
    pin_owners = pin_data["pin_owners"]
    pin_x_off = pin_data["pin_x_offsets"]
    pin_y_off = pin_data["pin_y_offsets"]
    is_port = pin_data["is_port"]
    port_positions = pin_data["port_positions"]
    valid = pin_owners >= 0

    num_nets = pin_owners.shape[0]

    soft_pos = benchmark.macro_positions[n_hard:].to(device).float()
    all_pos = torch.cat([positions, soft_pos], dim=0)

    owners_clamped = pin_owners.clamp(min=0)
    macro_owners = owners_clamped.clamp(max=n_macros - 1)

    pin_x = all_pos[macro_owners, 0] + pin_x_off
    pin_y = all_pos[macro_owners, 1] + pin_y_off

    if port_positions.shape[0] > 0 and is_port.any():
        port_owners = (owners_clamped - n_macros).clamp(min=0, max=port_positions.shape[0] - 1)
        pin_x = torch.where(is_port, port_positions[port_owners, 0], pin_x)
        pin_y = torch.where(is_port, port_positions[port_owners, 1], pin_y)

    big = 1e6
    px_hi = torch.where(valid, pin_x, torch.tensor(-big, device=device))
    px_lo = torch.where(valid, pin_x, torch.tensor(big, device=device))
    py_hi = torch.where(valid, pin_y, torch.tensor(-big, device=device))
    py_lo = torch.where(valid, pin_y, torch.tensor(big, device=device))

    wl_x = gamma * torch.logsumexp(px_hi / gamma, dim=1) + gamma * torch.logsumexp(-px_lo / gamma, dim=1)
    wl_y = gamma * torch.logsumexp(py_hi / gamma, dim=1) + gamma * torch.logsumexp(-py_lo / gamma, dim=1)

    canvas_diag = math.sqrt(benchmark.canvas_width ** 2 + benchmark.canvas_height ** 2)
    return (wl_x + wl_y).sum() / canvas_diag / max(num_nets, 1)


def _density_loss(positions, sizes, half_w, half_h,
                  grid_rows, grid_cols, canvas_w, canvas_h,
                  target_density, col_centers, row_centers,
                  grid_w, grid_h, device):
    """Density loss: penalize top-10% densest grid cells."""
    n = positions.shape[0]
    density = torch.zeros(grid_rows, grid_cols, device=device)

    cx = positions[:, 0]
    cy = positions[:, 1]

    for i in range(n):
        ox = torch.clamp(
            torch.clamp(col_centers + grid_w / 2, max=(cx[i] + half_w[i]).item()) -
            torch.clamp(col_centers - grid_w / 2, min=(cx[i] - half_w[i]).item()),
            min=0) / grid_w
        oy = torch.clamp(
            torch.clamp(row_centers + grid_h / 2, max=(cy[i] + half_h[i]).item()) -
            torch.clamp(row_centers - grid_h / 2, min=(cy[i] - half_h[i]).item()),
            min=0) / grid_h
        density = density + oy.unsqueeze(1) * ox.unsqueeze(0)

    flat = density.flatten()
    k = max(1, int(0.1 * flat.numel()))
    top_k, _ = torch.topk(flat, k)
    excess = F.relu(top_k - target_density)
    return excess.pow(2).mean()


def _overlap_loss(positions, half_w, half_h):
    """Differentiable pairwise overlap penalty."""
    n = positions.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=positions.device)
    x, y = positions[:, 0], positions[:, 1]
    dx = (x.unsqueeze(0) - x.unsqueeze(1)).abs()
    dy = (y.unsqueeze(0) - y.unsqueeze(1)).abs()
    sep_x = half_w.unsqueeze(0) + half_w.unsqueeze(1)
    sep_y = half_h.unsqueeze(0) + half_h.unsqueeze(1)
    ol_x = F.relu(sep_x - dx)
    ol_y = F.relu(sep_y - dy)
    mask = torch.triu(torch.ones(n, n, device=positions.device, dtype=torch.bool), diagonal=1)
    return ((ol_x * ol_y) * mask).sum()


def _project(positions, half_w, half_h, canvas_w, canvas_h):
    pos = positions.clone()
    gap = 0.05
    pos[:, 0] = pos[:, 0].clamp(min=half_w + gap, max=canvas_w - half_w - gap)
    pos[:, 1] = pos[:, 1].clamp(min=half_h + gap, max=canvas_h - half_h - gap)
    return pos


def _count_overlaps(positions, half_w, half_h):
    n = positions.shape[0]
    x, y = positions[:, 0], positions[:, 1]
    dx = (x.unsqueeze(0) - x.unsqueeze(1)).abs()
    dy = (y.unsqueeze(0) - y.unsqueeze(1)).abs()
    sep_x = half_w.unsqueeze(0) + half_w.unsqueeze(1)
    sep_y = half_h.unsqueeze(0) + half_h.unsqueeze(1)
    overlaps = (dx < sep_x) & (dy < sep_y)
    mask = torch.triu(torch.ones(n, n, device=positions.device, dtype=torch.bool), diagonal=1)
    return (overlaps & mask).sum().item()
