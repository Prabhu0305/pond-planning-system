"""
kml_parser.py

Day 1 goal: read a contour KML file and extract every contour line as a
list of (longitude, latitude, elevation) points.

VIVA POINT: A KML contour file stores elevation as the *name* of each line
(e.g. a Placemark named "277.0" means every point on that line is at 277m
elevation), and the actual shape of the line as a LineString of
lon,lat coordinate pairs. We are not given elevation per-point directly —
we assign the line's elevation to every point on that line.
"""

from lxml import etree


def parse_contour_kml(file_path):
    """
    Reads a KML file and returns a list of points, where each point is
    (longitude, latitude, elevation).

    This works regardless of which village/contour map is given — it does
    not assume any specific coordinates, only the KML structure.
    """
    tree = etree.parse(file_path)
    root = tree.getroot()

    # KML files use an XML namespace. We need to handle both namespaced
    # and non-namespaced tags because different KML generators format
    # this slightly differently (we saw this in the sample file).
    nsmap = {"kml": "http://www.opengis.net/kml/2.2"}

    points = []
    line_count = 0

    # Find every Placemark in the document, regardless of namespace
    placemarks = root.findall(".//kml:Placemark", nsmap)
    if not placemarks:
        # fallback: some KML exports omit the namespace on certain tags
        placemarks = root.findall(".//Placemark")

    for placemark in placemarks:
        # Get elevation from the Placemark's <name> (or <n> in this file's
        # non-standard export)
        name_elem = placemark.find("kml:name", nsmap)
        if name_elem is None:
            name_elem = placemark.find("name")
        if name_elem is None:
            name_elem = placemark.find("n")  # this sample file uses <n>

        if name_elem is None or name_elem.text is None:
            continue  # skip placemarks with no elevation label

        try:
            elevation = float(name_elem.text.strip())
        except ValueError:
            continue  # skip anything whose name isn't a number

        # Get the coordinates of the line
        coords_elem = placemark.find(".//kml:coordinates", nsmap)
        if coords_elem is None:
            coords_elem = placemark.find(".//coordinates")

        if coords_elem is None or coords_elem.text is None:
            continue

        raw_coords = coords_elem.text.strip().split()
        line_points = []
        for coord in raw_coords:
            parts = coord.split(",")
            lon = float(parts[0])
            lat = float(parts[1])
            line_points.append((lon, lat, elevation))

        points.extend(line_points)
        line_count += 1

    return points, line_count


if __name__ == "__main__":
    # Quick manual test — run this file directly to check parsing works
    file_path = "sample_data/contours_1m.kml"
    points, line_count = parse_contour_kml(file_path)

    elevations = [p[2] for p in points]

    print(f"Contour lines parsed: {line_count}")
    print(f"Total points extracted: {len(points)}")
    print(f"Elevation range: {min(elevations)} to {max(elevations)}")
    print(f"Sample point: {points[0]}")