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
)

import numpy as np

app = Flask(__name__)


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


def run_full_analysis(kml_file_path, resolution_m=10, num_sites=4):
    """
    Runs the complete pipeline on a given KML file path and returns a
    plain Python dict, ready to be converted to JSON.

    Returns up to `num_sites` distinct, ranked candidate pond locations,
    each with its own non-overlapping catchment -- not just one "the"
    answer. This matters in practice: the top-ranked site might turn out
    to be unsuitable for reasons contour data alone can't capture (rocky
    ground, existing farmland in active use, access constraints), so
    having genuine alternatives ready means a field visit can substitute
    the next-best option instead of starting over.
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

    raw_sites = find_top_n_pond_sites(Z_filled, accumulation, direction, n=num_sites)

    pond_sites = []
    for rank, site in enumerate(raw_sites, start=1):
        catchment_area_m2 = site["catchment_cell_count"] * cell_area_m2
        sizing = estimate_pond_depth_and_storage(catchment_area_m2)

        pond_sites.append({
            "rank": rank,
            "latitude": float(grid_lats[site["row"]]),
            "longitude": float(grid_lons[site["col"]]),
            "elevation_m": site["elevation"],
            "distance_from_main_channel_m": site["distance_from_main_channel_m"],
            "catchment": {
                "cell_count": site["catchment_cell_count"],
                "cell_area_m2": round(cell_area_m2, 2),
                "catchment_area_m2": round(catchment_area_m2, 2),
            },
            "pond_sizing": sizing,
        })

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
        "pond_sites": pond_sites,
        "notes": (
            f"Returned {len(pond_sites)} distinct, non-overlapping candidate pond "
            "sites, ranked by flow accumulation (highest first = most natural water "
            "collection). Multiple sites are provided because terrain data alone "
            "cannot capture ground conditions like rock composition or existing land "
            "use -- if the top-ranked site proves unsuitable on inspection, the next "
            "ranked site is a genuine, independent alternative. All results are "
            "derived automatically from the uploaded contour data; nothing is "
            "hardcoded to a specific village."
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

    is_kmz = filename_lower.endswith(".kmz")

    # Save the uploaded file to a temporary path so our existing
    # file-based parser can read it.
    upload_suffix = ".kmz" if is_kmz else ".kml"
    with tempfile.NamedTemporaryFile(suffix=upload_suffix, delete=False) as tmp:
        uploaded_file.save(tmp.name)
        tmp_path = tmp.name

    kml_path = tmp_path
    extracted_kml_path = None

    try:
        if is_kmz:
            # A KMZ file is just a KML file zipped up (usually alongside
            # images/icons). We extract the .kml that's inside it and
            # point our existing parser at that instead.
            extracted_kml_path = extract_kml_from_kmz(tmp_path)
            kml_path = extracted_kml_path

        result = run_full_analysis(kml_path)
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # Anything unexpected -- we don't want the server to just crash
        # with a raw traceback for the person calling the API.
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500
    finally:
        os.remove(tmp_path)
        if extracted_kml_path and os.path.exists(extracted_kml_path):
            os.remove(extracted_kml_path)


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
        "message": "Pond Catchment Analysis API is running.",
        "endpoint": "POST /analyzeContour with a .kml or .kmz file as form field 'contour_map'",
        "frontend": "GET /planner for the interactive tool",
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=False)