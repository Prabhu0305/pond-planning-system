import json
import os
import sys
import tempfile
import zipfile

from flask import Flask, request, jsonify

# The analysis/*.py files were written to be run standalone (e.g.
# "python analysis/dem_builder.py"), and they import each other using
# plain names like "from kml_parser import ...". To reuse them here in
# app.py without rewriting them, we add the analysis/ folder itself to
# Python's search path, so those same plain imports keep working.
ANALYSIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis")
sys.path.insert(0, ANALYSIS_DIR)

from kml_parser import parse_contour_kml           # noqa: E402
from dem_builder import build_dem                  # noqa: E402
from terrain_flow import (                          # noqa: E402
    fill_depressions,
    compute_flow_direction,
    compute_flow_accumulation,
    find_top_n_pond_sites,
    estimate_pond_depth_and_storage,
    extract_catchment_polygon,
    compute_water_yield,
)

import numpy as np

app = Flask(__name__)

# In-memory terrain cache to power instant sub-second land-area queries (<100ms)
# without re-parsing 160,000 points or repeating expensive SciPy grid interpolations.
TERRAIN_CACHE = {}


def extract_kml_from_kmz(kmz_path):
    """
    A .kmz file is just a .zip archive that contains a .kml file inside
    (plus, sometimes, images/icons). This finds the .kml inside and
    extracts it to a temporary file, returning that file's path.
    """
    with zipfile.ZipFile(kmz_path, "r") as archive:
        kml_names = [n for n in archive.namelist() if n.lower().endswith(".kml")]
        if not kml_names:
            raise ValueError("This .kmz file does not contain any .kml file inside it.")

        # A KMZ conventionally has a main "doc.kml" -- prefer that if present,
        # otherwise just take the first .kml found.
        chosen_name = next((n for n in kml_names if os.path.basename(n).lower() == "doc.kml"), kml_names[0])
        extracted_bytes = archive.read(chosen_name)

    with tempfile.NamedTemporaryFile(suffix=".kml", delete=False) as tmp:
        tmp.write(extracted_bytes)
        return tmp.name


def estimate_cell_area_m2(grid_lons, grid_lats):
    """
    Works out the real-world area (in square meters) that ONE grid cell
    covers, based on the actual spacing of our grid -- needed to convert
    a "number of cells" into a real catchment area.
    """
    mean_lat = (grid_lats.min() + grid_lats.max()) / 2
    meters_per_deg_lat = 111320
    meters_per_deg_lon = 111320 * np.cos(np.radians(mean_lat))

    cell_width_deg = grid_lons[1] - grid_lons[0]
    cell_height_deg = grid_lats[1] - grid_lats[0]

    cell_width_m = cell_width_deg * meters_per_deg_lon
    cell_height_m = cell_height_deg * meters_per_deg_lat

    return abs(cell_width_m * cell_height_m)


def format_site_results(raw_sites, grid_lons, grid_lats, cell_area_m2,
                        rainfall_mm=1000.0, runoff_coeff=0.30):
    """
    Formats raw sited cells into structured JSON with:
      - Sited coordinates (lat, lon, elevation)
      - Catchment metrics (cells, area in m2 and hectares)
      - Physical pond sizing (area, depth, storage capacity m3)
      - Expected harvestable water volume (V = A * P * C)
      - Catchment vector boundary polygon for Leaflet map overlay
    """
    pond_sites = []
    for rank, site in enumerate(raw_sites, start=1):
        catchment_area_m2 = site["catchment_cell_count"] * cell_area_m2
        sizing = estimate_pond_depth_and_storage(catchment_area_m2)
        water_yield = compute_water_yield(
            catchment_area_m2, annual_rainfall_mm=rainfall_mm, runoff_coeff=runoff_coeff
        )

        # Extract vector polygon coordinates for Leaflet overlay
        polygon_coords = extract_catchment_polygon(
            site["catchment_mask"], grid_lons, grid_lats
        )

        pond_sites.append({
            "rank": rank,
            "latitude": float(grid_lats[site["row"]]),
            "longitude": float(grid_lons[site["col"]]),
            "elevation_m": round(float(site["elevation"]), 2),
            "distance_from_main_channel_m": site["distance_from_main_channel_m"],
            "catchment": {
                "cell_count": site["catchment_cell_count"],
                "cell_area_m2": round(float(cell_area_m2), 2),
                "catchment_area_m2": round(float(catchment_area_m2), 2),
                "catchment_area_ha": round(float(catchment_area_m2) / 10000.0, 2),
            },
            "pond_sizing": sizing,
            "water_yield": water_yield,
            "catchment_polygon": polygon_coords,
        })
    return pond_sites


def run_full_analysis(kml_file_path, resolution_m=10, num_sites=4,
                      bounds=None, rainfall_mm=1000.0, runoff_coeff=0.30):
    """
    Runs the complete pipeline on a given KML file path and returns a
    plain Python dict, ready to be converted to JSON.

    Returns up to `num_sites` distinct, ranked candidate pond locations,
    each with its own non-overlapping catchment and vector polygon overlay.
    """
    points, line_count = parse_contour_kml(kml_file_path)

    if len(points) < 10:
        raise ValueError(
            "Not enough contour points found in this file to build a DEM. "
            "Check that the KML contains contour LineStrings with elevation "
            "in each Placemark's name."
        )

    Z, grid_lons, grid_lats = build_dem(points, resolution_m=resolution_m)

    Z_filled = fill_depressions(Z)
    direction = compute_flow_direction(Z_filled)
    accumulation = compute_flow_accumulation(Z_filled, direction)

    cell_area_m2 = estimate_cell_area_m2(grid_lons, grid_lats)

    # Save to terrain cache for ultra-fast subsequent land-area selections (<100ms)
    TERRAIN_CACHE["latest"] = {
        "Z_filled": Z_filled,
        "direction": direction,
        "accumulation": accumulation,
        "grid_lons": grid_lons,
        "grid_lats": grid_lats,
        "cell_area_m2": cell_area_m2,
        "line_count": line_count,
        "Z": Z,
        "resolution_m": resolution_m,
    }

    raw_sites = find_top_n_pond_sites(
        Z_filled, accumulation, direction,
        n=num_sites, bounds=bounds,
        grid_lons=grid_lons, grid_lats=grid_lats
    )

    pond_sites = format_site_results(
        raw_sites, grid_lons, grid_lats, cell_area_m2,
        rainfall_mm=rainfall_mm, runoff_coeff=runoff_coeff
    )

    map_bounds = {
        "north": float(grid_lats.max()),
        "south": float(grid_lats.min()),
        "east": float(grid_lons.max()),
        "west": float(grid_lons.min()),
    }

    return {
        "input_summary": {
            "contour_lines_parsed": line_count,
            "elevation_range_m": {
                "min": float(np.nanmin(Z)),
                "max": float(np.nanmax(Z)),
            },
            "dem_grid_shape": {"rows": int(Z.shape[0]), "cols": int(Z.shape[1])},
            "dem_resolution_m": resolution_m,
            "map_bounds": map_bounds,
        },
        "selected_bounds": bounds,
        "pond_sites": pond_sites,
        "notes": (
            f"Returned {len(pond_sites)} distinct candidate pond "
            "sites, ranked by flow accumulation. Catchment boundary polygons "
            "and expected harvestable water yields are computed dynamically. "
            "All results are derived automatically from the uploaded contour data."
        ),
    }


@app.route("/analyzeContour", methods=["POST"])
def analyze_contour():
    if "contour_map" in request.files:
        uploaded_file = request.files["contour_map"]
    elif "file" in request.files:
        uploaded_file = request.files["file"]
    else:
        return jsonify({
            "error": "No file uploaded. Send it as form field 'contour_map' "
                     "(or 'file' for backward compatibility)."
        }), 400

    if uploaded_file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    filename_lower = uploaded_file.filename.lower()
    if not (filename_lower.endswith(".kml") or filename_lower.endswith(".kmz")):
        return jsonify({"error": "Only .kml or .kmz files are supported."}), 400

    # Optional bounds / rainfall parameters passed in form data
    bounds = None
    if "bounds" in request.form:
        try:
            bounds = json.loads(request.form["bounds"])
        except Exception:
            pass

    rainfall_mm = float(request.form.get("rainfall_mm", 1000.0))
    runoff_coeff = float(request.form.get("runoff_coeff", 0.30))

    is_kmz = filename_lower.endswith(".kmz")
    upload_suffix = ".kmz" if is_kmz else ".kml"

    with tempfile.NamedTemporaryFile(suffix=upload_suffix, delete=False) as tmp:
        uploaded_file.save(tmp.name)
        tmp_path = tmp.name

    kml_path = tmp_path
    extracted_kml_path = None

    try:
        if is_kmz:
            extracted_kml_path = extract_kml_from_kmz(tmp_path)
            kml_path = extracted_kml_path

        result = run_full_analysis(
            kml_path, bounds=bounds, rainfall_mm=rainfall_mm, runoff_coeff=runoff_coeff
        )
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if extracted_kml_path and os.path.exists(extracted_kml_path):
            os.remove(extracted_kml_path)


@app.route("/analyzeArea", methods=["POST"])
def analyze_area():
    """
    Sub-second endpoint (<100ms) for analyzing a selected land area bounding box
    directly on precomputed cached terrain flow matrices.
    Accepts JSON:
      {
        "bounds": { "north": float, "south": float, "east": float, "west": float },
        "rainfall_mm": float (optional, default 1000),
        "runoff_coeff": float (optional, default 0.30),
        "num_sites": int (optional, default 4)
      }
    """
    cache = TERRAIN_CACHE.get("latest")
    if not cache:
        return jsonify({
            "error": "No contour map analyzed yet. Please upload a .kml/.kmz file first."
        }), 400

    data = request.get_json(silent=True) or {}
    bounds = data.get("bounds")
    rainfall_mm = float(data.get("rainfall_mm", 1000.0))
    runoff_coeff = float(data.get("runoff_coeff", 0.30))
    num_sites = int(data.get("num_sites", 4))

    try:
        raw_sites = find_top_n_pond_sites(
            cache["Z_filled"],
            cache["accumulation"],
            cache["direction"],
            n=num_sites,
            bounds=bounds,
            grid_lons=cache["grid_lons"],
            grid_lats=cache["grid_lats"],
        )

        pond_sites = format_site_results(
            raw_sites,
            cache["grid_lons"],
            cache["grid_lats"],
            cache["cell_area_m2"],
            rainfall_mm=rainfall_mm,
            runoff_coeff=runoff_coeff,
        )

        return jsonify({
            "pond_sites": pond_sites,
            "selected_bounds": bounds,
            "cached": True,
            "sites_found": len(pond_sites),
        }), 200
    except Exception as e:
        return jsonify({"error": f"Area analysis failed: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for cluster load balancers (Sys 1) and watchdogs."""
    return jsonify({
        "status": "healthy",
        "service": "pond-planning-phase3",
        "cache_ready": "latest" in TERRAIN_CACHE,
    }), 200


@app.route("/planner", methods=["GET"])
def planner():
    template_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "templates", "planner.html"
    )
    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "Pond Catchment Analysis API (Phase 3) is running.",
        "endpoints": {
            "POST /analyzeContour": "Upload .kml or .kmz contour file for complete terrain analysis",
            "POST /analyzeArea": "Instant analysis (<100ms) for selected land area bounds on cached terrain",
            "GET /planner": "Interactive GIS Web Interface with land area selection & catchment overlays",
            "GET /health": "Cluster health status probe",
        }
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=False)