# floorplan-ingestion

Normalizes floor plans (PNG/JPEG/BMP/TIFF/WebP, PDF, DXF) into a uniform `IngestionResult`.

```python
import logging
from ingestion import IngestionService, IngestionConfig

logging.basicConfig(level=logging.INFO)
service = IngestionService(IngestionConfig(target_dpi=300, max_dimension=2048))

result = service.process("plan.pdf")
if result.input_type.value == "raster":
    print(result.image.shape, result.metadata.dpi, result.metadata.scale_factor)
else:
    print(result.metadata.units, result.metadata.bounds, len(result.vectors))
```

## Development
```
pip install -e ".[dev]"
pytest && ruff check . && mypy src
```

## Behavior notes
- Format is detected from file content first, extension second. DWG is rejected with a hint to export DXF.
- Dewarping applies to photos/scans only (never PDF renders), and only to skewed quads;
  axis-aligned drawing frames are ignored.
- PDF DPI is lowered automatically to stay under `max_render_pixels`; `metadata.dpi` reports the
  resolution of the *returned* image relative to the physical page.
- DXF: LINE, LWPOLYLINE, POLYLINE, ARC, CIRCLE, ELLIPSE, SPLINE are flattened to polylines;
  block INSERTs are exploded (depth-limited); hidden layers skipped; unsupported types are
  counted in `metadata.skipped_entities`.
- Only one PDF page and the first TIFF frame are processed (a warning is recorded for PDFs).
- Untrusted files: run ingestion in a sandboxed worker with OS-level memory/time limits;
  the built-in limits (file size, render pixels, entity count) are a first line of defense only.

---

# Module 2: perception (`perception` package)

Extracts a wall mask, door/window/opening symbols and text from the normalized
image produced by ingestion.

```python
from ingestion import IngestionService
from perception import PerceptionService, PerceptionConfig, OcrMode

ingested = IngestionService().process("plan.png")
service = PerceptionService(PerceptionConfig(
    ocr_mode=OcrMode.AUTO,                 # OFF | AUTO | REQUIRED
    symbol_model_path="best_floorplan_yolo.pt",   # optional YOLO weights
))
result = service.extract_from_ingestion(ingested)
# result.wall_mask, .opening_mask, .doors, .windows, .annotations, .warnings
```

Optional dependencies: `pip install -e ".[ocr]"` (EasyOCR), `pip install -e ".[detection]"` (Ultralytics).
End-to-end demo: `python examples/run_pipeline.py plan.pdf --out out` (writes overlay/mask images).

Custom models: pass objects implementing `TextDetector.detect(image)` or
`SymbolDetector.detect(image, wall_mask)` via `PerceptionService(text_detector=..., symbol_detector=...)`.

All coordinates are in the pixel space of the image passed in. For a PDF/large image that
ingestion downscaled, divide by `ingested.metadata.scale_factor` to return to the source pixels.

---

# Module 3: topology (`topology` package)

Turns the wall mask (plus detected doors/windows) into a planar wall graph, closed rooms,
and openings positioned along their host walls.

```python
from topology import TopologyEngine, TopologyConfig

graph = TopologyEngine().build_from_perception(perception_result)
# or: TopologyEngine().build_topology(wall_mask, doors=[...], windows=[...])

print(len(graph.rooms), graph.thickness_px, graph.rotation_deg, graph.warnings)
for edge in graph.edges:
    for o in edge.openings:      # o.opening_type, o.t1, o.t2, o.width_px, o.center_px
        ...
json_ready = graph.to_dict()
```

Coordinates are in the pixel space of the mask. Check `graph.warnings`,
`graph.dangling_vertices` and `graph.unattached_openings` before trusting a result.
Requires `scipy`, `shapely>=2`, `scikit-image` (installed with the package).

---

# Module 4: scale & dimensionality (`scale` package)

Resolves the real-world scale (metres per pixel) and converts the pixel wall graph into a
metric plan with parametric elevations, ready for 3D construction.

```python
from scale import ScaleEngine, ManualAnchor

engine = ScaleEngine()
plan = engine.build_from_perception(
    graph, perception_result,
    manual_anchors=[ManualAnchor((120, 80), (940, 80), 8.20)],   # optional, overrides all
    image_dpi=ingested.metadata.dpi,                              # enables "1:100" notation
)
print(plan.scale.meters_per_pixel, plan.scale.source, plan.scale.level, plan.scale.warnings)
for wall in plan.walls:   # metres, Y up, origin at the plan's min corner
    wall.start, wall.end, wall.thickness_m, wall.height_m, wall.kind, wall.openings
for room in plan.rooms:   # CCW rings; room.area_m2, room.interior_area_m2, room.area_sqft
    ...
payload = plan.to_dict()  # JSON-ready, "units": "meters"
```

## How the scale is chosen

| Tier | Evidence | Notes |
|---|---|---|
| 1 | `ManualAnchor` (two points + known length) | Ground truth; automated evidence only produces contradiction warnings. Multiple anchors use the median. |
| 2 | OCR dimensions matched to wall spans and `A x B` room labels, plus `1:N` notation when the DPI is known | Pooled and reduced by **consensus**: the scale most distinct texts agree on wins, so misread numbers and wrong associations lose the vote. |
| 3 | Median detected door width vs. a standard door (0.85 m) | Last resort, about +/-10-15 %; reported as such. |

Accuracy mechanisms in tier 2:
- **Unit-aware parser** (`mm`, `cm`, `m`, `ft-in`, fractions, decimal commas, OCR O/0 and l/1 fixes). Bare numbers
  (`3600`, `360`) keep every plausible unit as a weighted alternative; areas (`12 m2`) are rejected.
- **Span association**: each text is matched to collinear runs of wall edges that are parallel, nearby and centred on
  the text, plus the whole plan extent. Each span is tried as centreline, clear-interior (`L - t`) and outside-face
  (`L + t`) length.
- **Convention coherence**: drawings follow one dimensioning convention per sheet, so a consensus where every text
  agrees on one convention scores higher. Tight agreement beats loose agreement (Gaussian kernel).
- **Cross-checks**: single-text evidence that contradicts the door prior is rejected as a misread; implausible plan
  sizes, wall thicknesses and door widths raise warnings and lower confidence.

`ScaleResult` reports `confidence` (0-1 heuristic), `level` (high / medium / low), a heuristic
`relative_uncertainty`, `support` (agreeing texts) and the evidence considered. **Treat `LOW` and door-prior results
as "ask the user for an anchor".** `ScaleResolutionError` is raised when there is no evidence at all.

## Elevation defaults (`ScaleConfig`)
Ceiling 2.70 m; door sill 0 / head 2.10 m; window sill 0.90 / head 2.10 m (1.20 m tall); cased openings 0 / 2.10 m;
wall thickness clamped to 0.10-0.35 m (clamped walls are flagged and summarised in `plan.warnings`).
Walls are classified `exterior` (bounds one room), `interior` (shared) or `unknown`.

Vector input: `ScaleEngine.scale_from_drawing_units(metadata.units)` returns the exact drawing-unit-to-metre factor.

---

# Learned accuracy upgrades (`perception`)

Deep learning is used where the problem is *semantic*; geometry stays classical where it must be *exact*.

**Learned wall segmentation** (optional): give `PerceptionConfig(wall_model_path="walls.onnx")` an ONNX network
(for example a U-Net fine-tuned on CubiCasa5k). Inference is tiled with smooth blending and runs on
`onnxruntime` (`pip install -e ".[learned]"`). `wall_fusion` selects how it is combined:
`GUIDED` (default) keeps pixel-exact classical strokes only where the network says "wall", which removes dimension
lines, hatching and furniture without blurring edges; `LEARNED` uses the network mask as-is; `CLASSICAL` ignores it.
An empty model output falls back to classical segmentation with a warning.
I/O contract: input `float32 [1, C, H, W]`, C in {1, 3}, 0 = ink, 1 = paper, normalisation baked into the graph;
output wall probability (or logits with `wall_model_logits=True`), or K class scores with `wall_model_channel`.
**No weights are shipped**: you must train/export one and validate it on your own plans.

**Tiled OCR**: `PerceptionConfig(ocr_tile_px=1024)` reads small dimension text on large sheets by running OCR on
overlapping tiles (the overlap must exceed your longest text string). Better text means better scale.

## Evaluating accuracy
Hand-label 10-20 real plans (wall centrelines, openings, one known length) and track wall F1, opening recall, room
count and scale error against any change. Without this, tuning is guesswork.
