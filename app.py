"""
app.py

Day 4 goal: wrap the full pipeline (parse KML -> build DEM -> fill
depressions -> flow direction -> flow accumulation -> pond site ->
catchment) into a single Flask API endpoint.

Endpoint: POST /analyzeContour
  - Accepts an uploaded .kml file (multipart/form-data, field name "file")
  - Returns catchment analysis results as JSON

VIVA POINT: this endpoint does NOT hardcode anything about the sample
village. Every result (pond location, catchment area, elevation stats) is
computed fresh from whatever KML file is uploaded -- feed it a different
village's contour file and it derives different results, which is exactly
what the assignment requires ("no hardcoding, must generalize").
"""

import os
import sys
import tempfile

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
    find_pour_point,
    delineate_catchment,
)

import numpy as np

app = Flask(__name__)


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


def run_full_analysis(kml_file_path, resolution_m=10):
    """
    Runs the complete pipeline on a given KML file path and returns a
    plain Python dict, ready to be converted to JSON.
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

    pour_row, pour_col = find_pour_point(Z_filled, accumulation)
    catchment_mask = delineate_catchment(direction, pour_row, pour_col)

    cell_area_m2 = estimate_cell_area_m2(grid_lons, grid_lats)
    catchment_cell_count = int(catchment_mask.sum())
    catchment_area_m2 = catchment_cell_count * cell_area_m2

    pond_lat = float(grid_lats[pour_row])
    pond_lon = float(grid_lons[pour_col])
    pond_elevation = float(Z_filled[pour_row, pour_col])

    return {
        "input_summary": {
            "contour_lines_parsed": line_count,
            "elevation_range_m": {
                "min": float(np.nanmin(Z)),
                "max": float(np.nanmax(Z)),
            },
            "dem_grid_shape": {"rows": int(Z.shape[0]), "cols": int(Z.shape[1])},
            "dem_resolution_m": resolution_m,
        },
        "pond_site": {
            "latitude": pond_lat,
            "longitude": pond_lon,
            "elevation_m": pond_elevation,
        },
        "catchment": {
            "cell_count": catchment_cell_count,
            "cell_area_m2": round(cell_area_m2, 2),
            "catchment_area_m2": round(catchment_area_m2, 2),
        },
        "notes": (
            "Pond site and catchment are derived automatically from the "
            "uploaded contour data. Nothing here is hardcoded to a "
            "specific village."
        ),
    }


@app.route("/analyzeContour", methods=["POST"])
def analyze_contour():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send it as form field 'file'."}), 400

    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    if not uploaded_file.filename.lower().endswith(".kml"):
        return jsonify({"error": "Only .kml files are supported right now."}), 400

    # Save the uploaded file to a temporary path so our existing
    # file-based parser can read it.
    with tempfile.NamedTemporaryFile(suffix=".kml", delete=False) as tmp:
        uploaded_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = run_full_analysis(tmp_path)
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # Anything unexpected -- we don't want the server to just crash
        # with a raw traceback for the person calling the API.
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500
    finally:
        os.remove(tmp_path)


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "Pond Catchment Analysis API is running.",
        "endpoint": "POST /analyzeContour with a .kml file as form field 'file'",
    })


if __name__ == "__main__":
    # host="0.0.0.0" makes it reachable from outside your own machine --
    # required later when this runs on the professor's server.
    app.run(host="0.0.0.0", port=5000, debug=True)