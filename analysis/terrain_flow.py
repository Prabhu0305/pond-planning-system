"""
terrain_flow.py

Day 3 goal: given the DEM grid (from dem_builder.py), compute:
  1. Fill depressions  -- remove fake pits caused by interpolation noise
  2. Flow direction (D8) -- for every cell, which neighbor is "downhill"
  3. Flow accumulation -- for every cell, how many upstream cells drain into it

VIVA POINT: "D8" means each cell can flow into exactly one of its 8
surrounding neighbors -- whichever one is the steepest way down. This is
the standard, most widely used method in hydrology for computing flow
direction on a grid.
"""

import heapq
import numpy as np


# The 8 neighbor directions: (row offset, col offset, distance).
# Diagonal neighbors are farther away (sqrt(2) times the cell size), so we
# divide by that distance when comparing slopes -- otherwise diagonal
# neighbors would be unfairly favored just for being "closer" in index terms.
NEIGHBORS = [
    (-1, -1, np.sqrt(2)), (-1, 0, 1), (-1, 1, np.sqrt(2)),
    (0, -1, 1),                       (0, 1, 1),
    (1, -1, np.sqrt(2)),  (1, 0, 1),  (1, 1, np.sqrt(2)),
]


def fill_depressions(Z):
    """
    Removes local pits from the DEM so that every cell has a continuous
    downhill path to the edge of the map.

    This uses the "priority-flood" method: think of water rising from the
    outer edge of the map inward. Any tiny pit gets raised to match the
    lowest point on its "rim" -- exactly what would happen if you flooded
    the landscape from the boundary in real life.
    """
    rows, cols = Z.shape
    filled = Z.copy()
    visited = np.zeros_like(Z, dtype=bool)
    heap = []

    # Start from every boundary cell (the edge of the map)
    for c in range(cols):
        for r in (0, rows - 1):
            heapq.heappush(heap, (filled[r, c], r, c))
            visited[r, c] = True
    for r in range(rows):
        for c in (0, cols - 1):
            if not visited[r, c]:
                heapq.heappush(heap, (filled[r, c], r, c))
                visited[r, c] = True

    while heap:
        elev, r, c = heapq.heappop(heap)
        for dr, dc, _dist in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                visited[nr, nc] = True
                # If the neighbor is lower than the current flood level,
                # raise it -- this is what "fills" the depression.
                if filled[nr, nc] < elev:
                    filled[nr, nc] = elev
                heapq.heappush(heap, (filled[nr, nc], nr, nc))

    return filled


def compute_flow_direction(Z):
    """
    For every cell, finds which of its 8 neighbors is the steepest way
    downhill. Returns a grid of direction indices (0-7, matching NEIGHBORS),
    or -1 if the cell has no downhill neighbor (edge of map).
    """
    rows, cols = Z.shape
    direction = np.full((rows, cols), -1, dtype=int)

    for r in range(rows):
        for c in range(cols):
            best_slope = 0.0
            best_idx = -1
            for idx, (dr, dc, dist) in enumerate(NEIGHBORS):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    slope = (Z[r, c] - Z[nr, nc]) / dist
                    if slope > best_slope:
                        best_slope = slope
                        best_idx = idx
            direction[r, c] = best_idx

    return direction


def compute_flow_accumulation(Z, direction):
    """
    For every cell, counts how many upstream cells eventually drain
    through it (including itself). High accumulation = a natural drainage
    path (a place where a lot of water collects).

    Method: process cells from HIGHEST elevation to LOWEST. Since water
    only flows downhill, by the time we reach a cell, every cell that
    drains INTO it has already been processed and added its contribution.
    """
    rows, cols = Z.shape
    accumulation = np.ones((rows, cols), dtype=float)  # every cell counts itself

    order = np.argsort(-Z.ravel())  # cell indices, highest elevation first

    for flat_idx in order:
        r, c = divmod(flat_idx, cols)
        d = direction[r, c]
        if d == -1:
            continue  # edge cell, water leaves the map here
        dr, dc, _dist = NEIGHBORS[d]
        nr, nc = r + dr, c + dc
        accumulation[nr, nc] += accumulation[r, c]

    return accumulation


def find_pour_point(Z, accumulation, low_elevation_percentile=25, edge_margin_fraction=0.08):
    """
    Picks the pond site: the cell with the highest flow accumulation among
    cells that are also relatively low-lying (bottom 25% of elevation).

    VIVA POINT -- edge effect: cells right at the boundary of the mapped
    area often show artificially high flow accumulation, simply because
    that's where water exits the study area, not because of a genuine
    natural basin. We exclude a margin around the edges (default 8% of the
    grid) so the chosen site is a real interior low point, not a map-edge
    artifact.
    """
    rows, cols = Z.shape
    row_margin = max(int(rows * edge_margin_fraction), 1)
    col_margin = max(int(cols * edge_margin_fraction), 1)

    interior_mask = np.zeros_like(Z, dtype=bool)
    interior_mask[row_margin:rows - row_margin, col_margin:cols - col_margin] = True

    threshold = np.percentile(Z[interior_mask], low_elevation_percentile)
    candidate_mask = interior_mask & (Z <= threshold)

    masked_accum = np.where(candidate_mask, accumulation, -1)
    flat_idx = np.argmax(masked_accum)
    row, col = divmod(flat_idx, cols)
    return row, col


def delineate_catchment(direction, pour_row, pour_col):
    """
    Finds every cell that eventually drains into the pour point (the pond
    site). This IS the catchment area.

    Method: start at the pour point and walk UPSTREAM -- for each cell,
    check its neighbors; if a neighbor's flow direction points at this
    cell, that neighbor is upstream, so add it and keep walking outward
    from there. This naturally traces the whole watershed boundary.
    """
    rows, cols = direction.shape
    catchment = np.zeros((rows, cols), dtype=bool)
    catchment[pour_row, pour_col] = True
    stack = [(pour_row, pour_col)]

    while stack:
        r, c = stack.pop()
        for idx, (dr, dc, _dist) in enumerate(NEIGHBORS):
            nr, nc = r - dr, c - dc  # candidate upstream neighbor
            if 0 <= nr < rows and 0 <= nc < cols and not catchment[nr, nc]:
                if direction[nr, nc] == idx:  # does it flow INTO (r, c)?
                    catchment[nr, nc] = True
                    stack.append((nr, nc))

    return catchment


if __name__ == "__main__":
    from kml_parser import parse_contour_kml
    from dem_builder import build_dem

    points, _ = parse_contour_kml("sample_data/contours_1m.kml")
    Z, grid_lons, grid_lats = build_dem(points, resolution_m=10)

    print("Filling depressions...")
    Z_filled = fill_depressions(Z)
    print(f"Cells raised by fill: {(Z_filled > Z).sum()} out of {Z.size}")

    print("Computing flow direction...")
    direction = compute_flow_direction(Z_filled)

    print("Computing flow accumulation...")
    accumulation = compute_flow_accumulation(Z_filled, direction)
    print(f"Max accumulation (cells draining to one point): {accumulation.max():.0f}")

    print("Finding pour point (pond site)...")
    pour_row, pour_col = find_pour_point(Z_filled, accumulation)
    pond_lat = grid_lats[pour_row]
    pond_lon = grid_lons[pour_col]
    print(f"Pond site: row={pour_row}, col={pour_col}")
    print(f"Pond site lat/lon: {pond_lat:.6f}, {pond_lon:.6f}")
    print(f"Pond site elevation: {Z_filled[pour_row, pour_col]:.2f} m")

    print("Delineating catchment...")
    catchment = delineate_catchment(direction, pour_row, pour_col)
    print(f"Catchment cell count: {catchment.sum()} out of {Z.size} total cells")