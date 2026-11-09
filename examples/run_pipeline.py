"""Run ingestion -> perception -> topology on a file and save debug images.

Usage:
    python examples/run_pipeline.py plan.pdf [--model best.pt] [--wall-model walls.onnx]
        [--ocr] [--ocr-tile 1024] [--anchor x1,y1,x2,y2,meters] [--out out_dir]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from ingestion import IngestionError, IngestionService
from perception import OcrMode, PerceptionConfig, PerceptionError, PerceptionService
from scale import ManualAnchor, ScaleEngine, ScaleError
from topology import TopologyEngine, TopologyError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--model", help="YOLO symbol weights (.pt)")
    parser.add_argument("--ocr", action="store_true", help="require EasyOCR")
    parser.add_argument("--wall-model", help="ONNX wall-segmentation network (.onnx)")
    parser.add_argument("--ocr-tile", type=int, default=0, help="OCR tile size in px (0 = off)")
    parser.add_argument(
        "--anchor",
        action="append",
        default=[],
        help="manual scale anchor 'x1,y1,x2,y2,meters' in processed-image pixels (repeatable)",
    )
    parser.add_argument("--heuristic-doors", action="store_true", help="experimental")
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    try:
        ingested = IngestionService().process(args.file)
        if ingested.input_type.value != "raster":
            print("DXF input is vector data; perception applies to raster images only.")
            print(f"Primitives: {ingested.metadata.entity_count}, units: {ingested.metadata.units}")
            return 0

        config = PerceptionConfig(
            ocr_mode=OcrMode.REQUIRED if args.ocr else OcrMode.AUTO,
            symbol_model_path=args.model,
            wall_model_path=args.wall_model,
            ocr_tile_px=args.ocr_tile,
            heuristic_symbols=args.heuristic_doors and args.model is None,
        )
        result = PerceptionService(config).extract_from_ingestion(ingested)
        graph = TopologyEngine().build_from_perception(result)
        anchors = [_parse_anchor(a) for a in args.anchor]
        plan = ScaleEngine().build_from_perception(
            graph, result, anchors, image_dpi=ingested.metadata.dpi
        )
    except FileNotFoundError:
        print(f"File not found: {args.file}")
        return 1
    except (IngestionError, PerceptionError, TopologyError, ScaleError, ValueError) as exc:
        print(f"Failed: {exc}")
        return 1

    overlay = cv2.cvtColor(result.cleaned_image, cv2.COLOR_GRAY2BGR)
    overlay[result.wall_mask > 0] = (0, 0, 255)
    overlay[result.opening_mask > 0] = (0, 200, 0)
    for symbol in result.doors + result.windows + result.openings:
        x1, y1, x2, y2 = symbol.box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 120, 0), 2)

    for room in graph.rooms:
        ring = np.array(room.polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [ring], True, (0, 160, 255), 3)
    for edge in graph.edges:
        a, b = graph.vertices[edge.u], graph.vertices[edge.v]
        cv2.line(overlay, (int(a.x), int(a.y)), (int(b.x), int(b.y)), (255, 0, 255), 1)
    (args.out / "graph.json").write_text(json.dumps(graph.to_dict(), indent=2))
    (args.out / "plan_metric.json").write_text(json.dumps(plan.to_dict(), indent=2))

    cv2.imwrite(str(args.out / "cleaned.png"), result.cleaned_image)
    cv2.imwrite(str(args.out / "wall_mask.png"), result.wall_mask)
    cv2.imwrite(str(args.out / "overlay.png"), overlay)

    coverage = 100.0 * np.count_nonzero(result.wall_mask) / result.wall_mask.size
    print(
        f"Wall coverage: {coverage:.2f}%  | doors: {len(result.doors)}  "
        f"| windows: {len(result.windows)}  | text items: {len(result.annotations)}"
    )
    openings = sum(len(e.openings) for e in graph.edges)
    print(
        f"Graph: {len(graph.vertices)} vertices, {len(graph.edges)} edges, "
        f"{len(graph.rooms)} rooms, {openings} openings, "
        f"wall thickness {graph.thickness_px:.1f}px, tilt {graph.rotation_deg:.2f} deg"
    )
    scale = plan.scale
    print(
        f"Scale: {scale.meters_per_pixel:.5f} m/px from {scale.source.value} "
        f"(confidence {scale.confidence:.2f} {scale.level.value}, "
        f"+/-{scale.relative_uncertainty:.1%}, support {scale.support})"
    )
    floor = sum(r.area_m2 for r in plan.rooms)
    print(f"Metric plan: {len(plan.walls)} walls, {len(plan.rooms)} rooms, {floor:.1f} m2 total")
    for warning in result.warnings + graph.warnings + scale.warnings + plan.warnings:
        print("WARNING:", warning)
    print(f"Saved debug images to {args.out.resolve()}")
    return 0


def _parse_anchor(spec: str) -> ManualAnchor:
    try:
        x1, y1, x2, y2, meters = (float(v) for v in spec.split(","))
    except ValueError as exc:
        raise ValueError(f"--anchor expects 'x1,y1,x2,y2,meters', got {spec!r}") from exc
    return ManualAnchor((x1, y1), (x2, y2), meters)


if __name__ == "__main__":
    raise SystemExit(main())
