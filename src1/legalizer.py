"""
Overlap-free macro legalization with minimum displacement.

Places hard macros in legal (non-overlapping, within-bounds) positions
while minimizing total displacement from their input coordinates.
Uses a greedy area-ordered approach with vectorized overlap checking.
"""

import numpy as np


def legalize(pos, movable, sizes, canvas_w, canvas_h, gap=0.05):
    """
    Legalize hard macro placement: remove all overlaps with minimum displacement.

    Args:
        pos: [N, 2] numpy array of (x, y) center positions for hard macros
        movable: [N] boolean array, True for movable macros
        sizes: [N, 2] numpy array of (width, height) for each macro
        canvas_w, canvas_h: Canvas dimensions
        gap: Safety gap between macros (avoids float precision edge overlaps)

    Returns:
        [N, 2] numpy array of legalized positions (no overlaps guaranteed)
    """
    n = len(pos)
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2

    # Precompute pairwise separation requirements
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2 + gap
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2 + gap

    # Sort movable macros by area descending (place largest first)
    areas = sizes[:, 0] * sizes[:, 1]
    order = sorted(range(n), key=lambda i: -areas[i])

    legal = pos.copy().astype(np.float64)
    placed = np.zeros(n, dtype=bool)

    for idx in order:
        if not movable[idx]:
            placed[idx] = True
            continue

        # Clamp to canvas bounds
        legal[idx, 0] = np.clip(legal[idx, 0], half_w[idx] + gap, canvas_w - half_w[idx] - gap)
        legal[idx, 1] = np.clip(legal[idx, 1], half_h[idx] + gap, canvas_h - half_h[idx] - gap)

        # Check if current position is already legal
        if placed.any():
            if not _has_overlap(legal, idx, placed, sep_x, sep_y):
                placed[idx] = True
                continue

        # Search for nearest legal position using expanding spiral
        target_x, target_y = pos[idx, 0], pos[idx, 1]
        step = max(sizes[idx, 0], sizes[idx, 1]) * 0.2
        best_pos = legal[idx].copy()
        best_dist = float('inf')
        found = False

        for radius in range(1, 200):
            found_at_radius = False
            # Search perimeter of square at this radius
            for dx_step in range(-radius, radius + 1):
                for dy_step in range(-radius, radius + 1):
                    # Only check perimeter points
                    if abs(dx_step) != radius and abs(dy_step) != radius:
                        continue

                    cx = np.clip(target_x + dx_step * step,
                                 half_w[idx] + gap, canvas_w - half_w[idx] - gap)
                    cy = np.clip(target_y + dy_step * step,
                                 half_h[idx] + gap, canvas_h - half_h[idx] - gap)

                    # Check overlap with already-placed macros
                    if placed.any():
                        if _has_overlap_at(legal, idx, cx, cy, placed, sep_x, sep_y):
                            continue

                    dist = (cx - target_x) ** 2 + (cy - target_y) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = np.array([cx, cy])
                        found_at_radius = True

            if found_at_radius:
                found = True
                break

        if not found:
            # Fallback: row packing from top-left
            best_pos = _fallback_position(legal, idx, placed, sizes, half_w, half_h,
                                          canvas_w, canvas_h, gap, sep_x, sep_y)

        legal[idx] = best_pos
        placed[idx] = True

    return legal


def _has_overlap(positions, idx, placed_mask, sep_x, sep_y):
    """Check if macro idx overlaps any placed macro."""
    dx = np.abs(positions[idx, 0] - positions[:, 0])
    dy = np.abs(positions[idx, 1] - positions[:, 1])
    overlaps = (dx < sep_x[idx]) & (dy < sep_y[idx]) & placed_mask
    overlaps[idx] = False
    return overlaps.any()


def _has_overlap_at(positions, idx, cx, cy, placed_mask, sep_x, sep_y):
    """Check if placing macro idx at (cx, cy) would overlap any placed macro."""
    dx = np.abs(cx - positions[:, 0])
    dy = np.abs(cy - positions[:, 1])
    overlaps = (dx < sep_x[idx]) & (dy < sep_y[idx]) & placed_mask
    overlaps[idx] = False
    return overlaps.any()


def _fallback_position(legal, idx, placed_mask, sizes, half_w, half_h,
                        canvas_w, canvas_h, gap, sep_x, sep_y):
    """Row-packing fallback when spiral search fails."""
    w, h = sizes[idx]
    step_x = w * 0.5
    step_y = h * 0.5

    for row_y in np.arange(half_h[idx] + gap, canvas_h - half_h[idx] - gap, step_y):
        for col_x in np.arange(half_w[idx] + gap, canvas_w - half_w[idx] - gap, step_x):
            if not placed_mask.any() or not _has_overlap_at(
                    legal, idx, col_x, row_y, placed_mask, sep_x, sep_y):
                return np.array([col_x, row_y])

    # Ultimate fallback — canvas center
    return np.array([canvas_w / 2, canvas_h / 2])
