---
description: Guardrails and best practices for parsing and validating KML/KMZ contour maps
trigger: model_decision
---

# KML/KMZ Contour Processing Guardrails

When working with topographic and hydrological KML/KMZ datasets in this repository:

1. **Distinguish LineStrings from Label Points**:
   - GIS tools often export separate folders for contour lines (`LineString`) and text annotations (`Point`).
   - Do NOT count or ingest `Point` placemarks as contour lines when building Digital Elevation Models (DEMs).
   - Only parse geometry nodes matching `.//kml:LineString` or `.//LineString`.

2. **Deduplication Invariant**:
   - Check for duplicate coordinate vertices and duplicate placemarks before passing points to interpolation grids.
   - Do not commit identical test files (e.g., `contours2_1m.kml`) if an existing sample dataset already exists.

3. **Workspace Path Resolution**:
   - The primary assignment folder is located at `Universal/CSD_Assignment-1/pond-planning`.
