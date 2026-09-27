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
from scipy import ndimage


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


def latlon_to_grid_index(lat, lon, grid_lats, grid_lons):
    """
    Converts a real-world (lat, lon) -- e.g. a point the admin clicked on
    the map -- into the nearest row/col on our DEM grid.

    VIVA POINT: our grid has fixed spacing (grid_lats, grid_lons are evenly
    spaced arrays), so "nearest point" is just finding the closest value in
    each array -- no complex geometry needed.
    """
    row = int(np.argmin(np.abs(grid_lats - lat)))
    col = int(np.argmin(np.abs(grid_lons - lon)))
    return row, col


def estimate_pond_depth_and_storage(catchment_area_m2, pond_area_fraction=0.02,
                                     min_depth_m=1.5, max_depth_m=4.0):
    """
    Provisional pond sizing, used before rainfall/runoff data is available.

    ASSUMPTION (documented, not hidden): the pond's own footprint is
    estimated as a small fraction of its catchment area -- 2% is a common
    rough planning guideline for small farm/check-dam ponds relative to the
    area draining into them. Depth is set to a typical small-pond range
    (1.5m-4m), scaled mildly by catchment size (a bigger catchment can
    reasonably support a somewhat deeper pond, within sensible bounds).

    This is intentionally a simple, explainable placeholder -- once runoff
    volume is available (Day 3), storage capacity should be cross-checked
    against it rather than relying on this estimate alone.
    """
    pond_area_m2 = catchment_area_m2 * pond_area_fraction

    # scale depth mildly with catchment size, clamped to a sensible range
    size_factor = min(catchment_area_m2 / 50000, 1.0)  # saturates at 50,000 m2
    recommended_depth_m = min_depth_m + size_factor * (max_depth_m - min_depth_m)

    storage_capacity_m3 = pond_area_m2 * recommended_depth_m

    return {
        "pond_area_m2": round(pond_area_m2, 2),
        "recommended_depth_m": round(recommended_depth_m, 2),
        "storage_capacity_m3": round(storage_capacity_m3, 2),
    }


def identify_main_channel(accumulation, percentile=97):
    """
    Identifies the single dominant river/drainage channel: the largest
    CONNECTED group of cells with very high flow accumulation.

    VIVA POINT: this is different from just "all high-accumulation cells"
    -- a threshold alone catches every minor tributary too. Taking the
    LARGEST CONNECTED COMPONENT isolates the one genuinely dominant
    channel, which is what we actually want to keep pond sites away from.
    """
    threshold = np.percentile(accumulation, percentile)
    high_accum_mask = accumulation >= threshold

    labeled, num_features = ndimage.label(high_accum_mask, structure=np.ones((3, 3)))
    if num_features == 0:
        return np.zeros_like(accumulation, dtype=bool)

    sizes = ndimage.sum(high_accum_mask, labeled, range(1, num_features + 1))
    main_channel_label = np.argmax(sizes) + 1
    return labeled == main_channel_label


def extract_catchment_polygon(catchment_mask, grid_lons, grid_lats, max_vertices=300):
    """
    Extracts the outer boundary polygon of the delineated catchment mask
    as a list of [lat, lon] coordinates suitable for Leaflet vector overlays.
    Uses contourpy for high-speed, zero-copy boundary contouring.
    """
    try:
        import contourpy
        c = contourpy.contour_generator(z=catchment_mask.astype(float))
        lines = c.lines(0.5)
        if not lines:
            return []

        # If multiple contours exist (islands/disjoint areas), take the largest
        longest_line = max(lines, key=lambda l: len(l))
        if len(longest_line) < 3:
            return []

        # Downsample if too dense, to keep JSON payload lightweight (<20 KB)
        step = max(1, len(longest_line) // max_vertices)
        sampled = longest_line[::step]
        if not np.array_equal(sampled[0], sampled[-1]):
            sampled = np.vstack([sampled, sampled[0]])  # close polygon loop

        poly_lons = np.interp(sampled[:, 0], np.arange(len(grid_lons)), grid_lons)
        poly_lats = np.interp(sampled[:, 1], np.arange(len(grid_lats)), grid_lats)

        return [[round(float(lat), 6), round(float(lon), 6)] for lat, lon in zip(poly_lats, poly_lons)]
    except Exception as e:
        return []


def compute_water_yield(catchment_area_m2, annual_rainfall_mm=1000.0, runoff_coeff=0.30):
    """
    Computes expected harvestable water volume based on standard watershed hydrological equation:
        V_runoff = A_catchment * (P_annual / 1000) * C_runoff
    where:
        A_catchment: Contributing drainage area in m2
        P_annual: Annual precipitation in mm (default 1000 mm for Central India)
        C_runoff: Runoff coefficient (default 0.30 for agricultural/clay loam terrain)
    """
    rainfall_m = annual_rainfall_mm / 1000.0
    runoff_volume_m3 = catchment_area_m2 * rainfall_m * runoff_coeff
    volume_liters = runoff_volume_m3 * 1000.0

    # Rule of thumb for protective/supplementary irrigation:
    # ~500 m3 per hectare per watering event for kharif crops during dry spells
    irrigation_hectares = runoff_volume_m3 / 500.0

    return {
        "annual_rainfall_mm": float(annual_rainfall_mm),
        "runoff_coefficient": float(runoff_coeff),
        "runoff_volume_m3": round(float(runoff_volume_m3), 2),
        "runoff_volume_lakh_liters": round(float(volume_liters / 100000.0), 2),
        "runoff_volume_million_liters": round(float(volume_liters / 1000000.0), 3),
        "irrigation_hectares_supported": round(float(irrigation_hectares), 2),
    }


def find_top_n_pond_sites(Z, accumulation, direction, n=4, low_elevation_percentile=25,
                           edge_margin_fraction=0.08, min_catchment_cells=5,
                           min_distance_from_channel_cells=20, bounds=None,
                           grid_lons=None, grid_lats=None):
    """
    Finds up to N distinct candidate pond sites, ranked by flow accumulation
    (highest first), each with its OWN, NON-OVERLAPPING catchment, and each
    kept at a safe minimum distance from the identified main river channel.

    If `bounds` is provided (e.g. user selected land area on the map), candidate
    sites are strictly constrained to fall WITHIN that land area while tracking
    the upstream flow draining into it.
    """
    rows, cols = Z.shape
    row_margin = max(int(rows * edge_margin_fraction), 1)
    col_margin = max(int(cols * edge_margin_fraction), 1)

    interior_mask = np.zeros_like(Z, dtype=bool)
    interior_mask[row_margin:rows - row_margin, col_margin:cols - col_margin] = True

    # Distance (in cells) from every cell to the nearest main-channel cell
    main_channel_mask = identify_main_channel(accumulation)
    if main_channel_mask.any():
        distance_from_channel = ndimage.distance_transform_edt(~main_channel_mask)
    else:
        distance_from_channel = np.full_like(Z, np.inf)

    safe_from_channel_mask = distance_from_channel >= min_distance_from_channel_cells

    # Base low elevation threshold
    threshold = np.percentile(Z[interior_mask], low_elevation_percentile)

    # Apply spatial bounding box constraint if specified by user
    if bounds and grid_lons is not None and grid_lats is not None:
        min_lat = bounds.get("min_lat", bounds.get("south", -np.inf))
        max_lat = bounds.get("max_lat", bounds.get("north", np.inf))
        min_lon = bounds.get("min_lon", bounds.get("west", -np.inf))
        max_lon = bounds.get("max_lon", bounds.get("east", np.inf))

        lon_in_bounds = (grid_lons >= min_lon) & (grid_lons <= max_lon)
        lat_in_bounds = (grid_lats >= min_lat) & (grid_lats <= max_lat)
        bounds_2d = np.outer(lat_in_bounds, lon_in_bounds)

        if bounds_2d.any():
            # Prefer low elevation & channel safe inside bounds
            cand = bounds_2d & safe_from_channel_mask & (Z <= threshold)
            if not cand.any():
                cand = bounds_2d & safe_from_channel_mask
            if not cand.any():
                cand = bounds_2d
            candidate_mask = cand
        else:
            candidate_mask = interior_mask & (Z <= threshold) & safe_from_channel_mask
    else:
        candidate_mask = interior_mask & (Z <= threshold) & safe_from_channel_mask

    candidate_rows, candidate_cols = np.where(candidate_mask)
    if len(candidate_rows) == 0:
        return []

    candidate_accum = accumulation[candidate_rows, candidate_cols]
    order = np.argsort(-candidate_accum)

    used_mask = np.zeros_like(Z, dtype=bool)
    sites = []

    for idx in order:
        if len(sites) >= n:
            break

        row, col = candidate_rows[idx], candidate_cols[idx]
        if used_mask[row, col]:
            continue

        catchment = delineate_catchment(direction, row, col)

        if catchment.sum() < min_catchment_cells:
            continue

        sites.append({
            "row": int(row),
            "col": int(col),
            "elevation": float(Z[row, col]),
            "accumulation": float(accumulation[row, col]),
            "catchment_mask": catchment,
            "catchment_cell_count": int(catchment.sum()),
            "distance_from_main_channel_m": round(float(distance_from_channel[row, col]) * 10, 1),
        })
        used_mask |= catchment

    return sites



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