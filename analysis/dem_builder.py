"""
dem_builder.py

Day 2 goal: take the point cloud from kml_parser.py (scattered lon/lat/elevation
points pulled from contour lines) and interpolate it onto a regular grid.
That grid IS the DEM (Digital Elevation Model) that later steps (slope, flow
direction, flow accumulation, catchment) will run on.

VIVA POINT: contour lines only tell us elevation ALONG those specific lines.
Everywhere else (between contour lines) is unknown until we interpolate.
Interpolation estimates those in-between values from the nearest known points.
"""

import numpy as np
from scipy.interpolate import griddata
from kml_parser import parse_contour_kml


def build_dem(points, resolution_m=10):
    """
    points: list of (lon, lat, elevation) — from kml_parser.parse_contour_kml
    resolution_m: target grid cell size in meters (10m = one elevation
                  value every 10 meters on the ground)

    Returns:
        Z: 2D numpy array of elevation values (the DEM itself)
        grid_lons: 1D array of longitude values for each grid column
        grid_lats: 1D array of latitude values for each grid row
    """
    points_arr = np.array(points)
    lons = points_arr[:, 0]
    lats = points_arr[:, 1]
    elevs = points_arr[:, 2]

    min_lon, max_lon = lons.min(), lons.max()
    min_lat, max_lat = lats.min(), lats.max()

    # Convert degrees to meters so we can decide a sensible grid size.
    # 1 degree of latitude is always ~111,320 meters. 1 degree of longitude
    # is smaller depending how far from the equator you are (that's why the
    # cos(latitude) term is here) — this is a standard, well-known conversion,
    # not something specific to our data.
    mean_lat = (min_lat + max_lat) / 2
    meters_per_deg_lat = 111320
    meters_per_deg_lon = 111320 * np.cos(np.radians(mean_lat))

    width_m = (max_lon - min_lon) * meters_per_deg_lon
    height_m = (max_lat - min_lat) * meters_per_deg_lat

    n_cols = max(int(width_m / resolution_m), 2)
    n_rows = max(int(height_m / resolution_m), 2)

    grid_lons = np.linspace(min_lon, max_lon, n_cols)
    grid_lats = np.linspace(min_lat, max_lat, n_rows)
    grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lons, grid_lats)

    # "linear" interpolation works well INSIDE the area covered by known
    # points, but leaves gaps (NaN) at the very edges of the grid where
    # there's no surrounding data. We fill those edge gaps using "nearest"
    # as a fallback, so the final DEM has no holes in it.
    Z_linear = griddata(
        (lons, lats), elevs, (grid_lon_mesh, grid_lat_mesh), method="linear"
    )
    Z_nearest = griddata(
        (lons, lats), elevs, (grid_lon_mesh, grid_lat_mesh), method="nearest"
    )
    Z = np.where(np.isnan(Z_linear), Z_nearest, Z_linear)

    return Z, grid_lons, grid_lats


if __name__ == "__main__":
    points, line_count = parse_contour_kml("sample_data/contours_1m.kml")
    Z, grid_lons, grid_lats = build_dem(points, resolution_m=10)

    print(f"DEM grid shape: {Z.shape[0]} rows x {Z.shape[1]} cols")
    print(f"Elevation range in DEM: {np.nanmin(Z):.2f} to {np.nanmax(Z):.2f}")
    print(f"NaN cells remaining (should be 0): {np.isnan(Z).sum()}")

    # Save an image so you can SEE the terrain surface we just built.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 7))
    plt.imshow(
        Z, origin="lower", cmap="terrain",
        extent=[grid_lons.min(), grid_lons.max(), grid_lats.min(), grid_lats.max()],
    )
    plt.colorbar(label="Elevation (m)")
    plt.title("Interpolated DEM from contour KML")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig("sample_data/dem_preview.png", dpi=150)
    print("Saved preview image to sample_data/dem_preview.png")