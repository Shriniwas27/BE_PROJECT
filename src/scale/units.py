"""Length units and conversions."""

from __future__ import annotations

METERS_PER_UNIT: dict[str, float] = {
    "mm": 0.001,
    "cm": 0.01,
    "m": 1.0,
    "ft": 0.3048,
    "in": 0.0254,
}

# Unit names as produced by ``ezdxf.units.decode`` (Module 1's ``metadata.units``).
_DRAWING_UNIT_METERS: dict[str, float] = {
    "inches": 0.0254,
    "feet": 0.3048,
    "yards": 0.9144,
    "miles": 1609.344,
    "millimeters": 0.001,
    "centimeters": 0.01,
    "decimeters": 0.1,
    "meters": 1.0,
    "dekameters": 10.0,
    "hectometers": 100.0,
    "kilometers": 1000.0,
    "microinches": 2.54e-8,
    "mils": 2.54e-5,
}

SQUARE_FEET_PER_SQUARE_METER = 10.763910416709722


def drawing_units_to_meters(name: str | None) -> float | None:
    """Metres per drawing unit for a DXF unit name; ``None`` if unknown/unitless."""
    if not name:
        return None
    return _DRAWING_UNIT_METERS.get(name.strip().lower())
