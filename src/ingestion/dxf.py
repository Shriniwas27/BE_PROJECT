"""DXF (CAD) parsing into flattened 2-D polylines."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ezdxf
from ezdxf import units
from ezdxf.path import make_path

from .config import IngestionConfig
from .errors import CorruptFileError, ResourceLimitError
from .models import (
    Bounds,
    IngestionMetadata,
    IngestionResult,
    InputType,
    VectorPrimitive,
)

logger = logging.getLogger(__name__)

_PATH_ENTITIES = frozenset({"LINE", "LWPOLYLINE", "POLYLINE", "ARC", "CIRCLE", "ELLIPSE", "SPLINE"})
_MIN_CURVE_SEGMENTS = 8


@dataclass
class _State:
    doc: Any
    tolerance: float
    primitives: list[VectorPrimitive] = field(default_factory=list)
    skipped: Counter = field(default_factory=Counter)  # type: ignore[type-arg]
    hidden_cache: dict = field(default_factory=dict)  # type: ignore[type-arg]


class DxfParser:
    """Extracts modelspace geometry, exploding blocks and flattening curves."""

    def __init__(self, config: IngestionConfig) -> None:
        self._cfg = config

    def parse(self, path: Path) -> IngestionResult:
        doc = self._read(path)
        unit_code = doc.units
        state = _State(doc=doc, tolerance=self._tolerance(unit_code))

        self._walk(doc.modelspace(), state, depth=0)

        warnings: list[str] = []
        if unit_code == 0:
            warnings.append("DXF declares no drawing units; real-world scale is unknown.")
        if state.skipped:
            logger.info("Skipped DXF entities: %s", dict(state.skipped))

        metadata = IngestionMetadata(
            source_format="dxf",
            units=None if unit_code == 0 else units.decode(unit_code),
            bounds=self._bounds(state.primitives),
            entity_count=len(state.primitives),
            skipped_entities=dict(state.skipped),
            warnings=warnings,
        )
        return IngestionResult(InputType.VECTOR, metadata, vectors=state.primitives)

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _read(path: Path) -> Any:
        try:
            return ezdxf.readfile(str(path))
        except (ezdxf.DXFError, OSError, UnicodeDecodeError, ValueError) as exc:
            raise CorruptFileError(f"Invalid or corrupt DXF file {path.name}: {exc}") from exc

    def _tolerance(self, unit_code: int) -> float:
        """Convert the millimetre curve tolerance into drawing units."""
        if unit_code == 0:
            return self._cfg.curve_tolerance_mm
        try:
            to_mm = units.conversion_factor(
                units.InsertUnits(unit_code), units.InsertUnits.Millimeters
            )
        except (KeyError, ValueError, ZeroDivisionError):
            return self._cfg.curve_tolerance_mm
        return self._cfg.curve_tolerance_mm / to_mm if to_mm else self._cfg.curve_tolerance_mm

    def _walk(self, entities: Iterable[Any], state: _State, depth: int) -> None:
        for entity in entities:
            etype = entity.dxftype()
            if etype == "INSERT":
                self._explode_insert(entity, state, depth)
            elif etype in _PATH_ENTITIES:
                self._extract(entity, etype, state)
            else:
                state.skipped[etype] += 1

    def _explode_insert(self, insert: Any, state: _State, depth: int) -> None:
        if depth >= self._cfg.max_block_depth:
            state.skipped["INSERT(max_depth)"] += 1
            return
        try:
            children = list(insert.virtual_entities())
        except (ezdxf.DXFError, ValueError, TypeError, ArithmeticError) as exc:
            logger.debug("Could not explode INSERT: %s", exc)
            state.skipped["INSERT(failed)"] += 1
            return
        self._walk(children, state, depth + 1)

    def _extract(self, entity: Any, etype: str, state: _State) -> None:
        layer_name = entity.dxf.layer
        if self._cfg.skip_hidden_layers and self._is_hidden(state, layer_name):
            state.skipped["(hidden layer)"] += 1
            return
        try:
            path = make_path(entity)
            for sub in path.sub_paths():
                vertices = list(sub.flattening(state.tolerance, segments=_MIN_CURVE_SEGMENTS))
                points = tuple((float(v.x), float(v.y)) for v in vertices)
                if len(points) < 2:
                    continue
                self._append(state, layer_name, etype, points, sub.is_closed)
        except (TypeError, ValueError, ezdxf.DXFError, ArithmeticError) as exc:
            logger.debug("Failed to convert %s: %s", etype, exc)
            state.skipped[f"{etype}(failed)"] += 1

    def _append(
        self,
        state: _State,
        layer: str,
        etype: str,
        points: tuple[tuple[float, float], ...],
        closed: bool,
    ) -> None:
        if len(state.primitives) >= self._cfg.max_dxf_entities:
            raise ResourceLimitError(f"DXF exceeds {self._cfg.max_dxf_entities} primitives.")
        if closed and points[0] != points[-1]:
            points = (*points, points[0])
        state.primitives.append(VectorPrimitive(layer.upper(), etype, points, closed))

    @staticmethod
    def _is_hidden(state: _State, layer_name: str) -> bool:
        cache = state.hidden_cache
        if layer_name not in cache:
            try:
                layer = state.doc.layers.get(layer_name)
                cache[layer_name] = layer.is_off() or layer.is_frozen()
            except (KeyError, ezdxf.DXFError):
                cache[layer_name] = False
        return bool(cache[layer_name])

    @staticmethod
    def _bounds(primitives: list[VectorPrimitive]) -> Bounds | None:
        xs = [x for p in primitives for x, _ in p.points]
        ys = [y for p in primitives for _, y in p.points]
        if not xs:
            return None
        return Bounds(min(xs), min(ys), max(xs), max(ys))
