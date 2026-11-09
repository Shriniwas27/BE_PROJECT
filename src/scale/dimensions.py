"""Parse OCR text into lengths, room sizes, areas and drawing-scale ratios.

Accepted linear forms (case-insensitive)::

    3.60m   3,60 m   3600   3600mm   360cm   12 ft
    12'     12'-6"   12' 6 1/2"   6"   12'6

Bare numbers carry no unit, so :func:`parse_text` returns every *plausible*
interpretation with a prior weight (``3600`` is most likely millimetres,
``360`` could be centimetres or millimetres, ``3.6`` is metres). Downstream
consensus keeps whichever interpretation agrees with the rest of the plan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .units import METERS_PER_UNIT

DEFAULT_PLAUSIBLE_RANGE_M = (0.3, 60.0)

_EXPLICIT_WEIGHT = 1.0
_BARE_METER_WEIGHT = 0.9
_BARE_MILLIMETER_LARGE_WEIGHT = 0.9  # integers >= 1000
_BARE_AMBIGUOUS_WEIGHT = 0.5  # integers 100..999 (cm or mm)
_BARE_UNLIKELY_WEIGHT = 0.25
_BARE_DECIMAL_LARGE_WEIGHT = 0.6  # decimals >= 100 (cm or mm)

_UNIT_ALIASES = {
    "feet": "ft",
    "foot": "ft",
    "inch": "in",
    "inches": "in",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "mtr": "m",
    "mtrs": "m",
}
_UNIT_PATTERN = r"mm|cm|m|ft|feet|foot|in|inch|inches|meters?|metres?|mtrs?"

_QUOTES = str.maketrans(
    {
        "\u2019": "'",
        "\u2018": "'",
        "\u2032": "'",
        "\u00b4": "'",
        "`": "'",
        "\u201d": '"',
        "\u201c": '"',
        "\u2033": '"',
        "\u00d7": "x",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)

_AREA = re.compile(
    r"m\u00b2|m2\b|\u33a1|ft\u00b2|ft2\b|sq\.?\s*(?:m|ft|feet|meters?)?\b|sqm|sqft|\bsf\b",
    re.IGNORECASE,
)
_RATIO = re.compile(r"^(?:scale\s*[:=]?\s*)?1\s*:\s*(\d{1,5})$", re.IGNORECASE)
_RATIO_SLASH = re.compile(r"^scale\s*[:=]?\s*1\s*/\s*(\d{1,5})$", re.IGNORECASE)
_SIZE_SPLIT = re.compile(r"^(?P<a>.+?)\s*[x*]\s*(?P<b>.+)$", re.IGNORECASE)
_FEET_INCH = re.compile(
    r"""^(\d+(?:\.\d+)?)\s*'\s*(?:-\s*)?"""
    r"""(?:(\d+(?:\.\d+)?)(?:\s*[- ]\s*(\d+)\s*/\s*(\d+))?\s*"?)?$"""
)
_INCH_ONLY = re.compile(r"""^(?:(\d+(?:\.\d+)?)\s*)?(?:(\d+)\s*/\s*(\d+)\s*)?"$""")
_NUM_UNIT = re.compile(rf"^(?P<num>\d[\d.,\s]*?)\s*(?P<unit>{_UNIT_PATTERN})?$", re.IGNORECASE)
_THOUSANDS_COMMA = re.compile(r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$")
_THOUSANDS_SPACE = re.compile(r"^\d{1,3}(?: \d{3})+(?:[.,]\d+)?$")
_PLAIN_NUMBER = re.compile(r"^\d+(?:[.,]\d+)?$")


class TextKind(str, Enum):
    LINEAR = "linear"  # one length: "3.60m"
    ROOM_SIZE = "room_size"  # two lengths: "3.5 x 4.0"
    AREA = "area"  # "12.5 m2" - never usable as a length
    SCALE_RATIO = "scale_ratio"  # "1:100"
    OTHER = "other"


@dataclass(frozen=True)
class LengthReading:
    """One interpretation of a measurement.

    Attributes:
        meters: The length in metres under this interpretation.
        weight: Prior (0-1] that this interpretation is the intended one;
            1.0 when the text carried an explicit unit.
        unit: Interpreted unit label (``mm``, ``cm``, ``m``, ``ft``, ``in``, ``ft-in``).
    """

    meters: float
    weight: float
    unit: str


@dataclass(frozen=True)
class ParsedText:
    """Result of :func:`parse_text`.

    ``lengths`` holds one tuple of alternatives per measurement ("slot"):
    one slot for LINEAR, two for ROOM_SIZE, none otherwise. ``ratio`` is the
    N of a ``1:N`` drawing scale.
    """

    raw: str
    kind: TextKind
    lengths: tuple[tuple[LengthReading, ...], ...] = ()
    ratio: float | None = None


def parse_text(
    text: str, plausible_range_m: tuple[float, float] = DEFAULT_PLAUSIBLE_RANGE_M
) -> ParsedText:
    """Interpret one OCR string. Never raises; unrecognised text is ``OTHER``."""
    raw = text
    stripped = text.strip()
    if not stripped:
        return ParsedText(raw, TextKind.OTHER)

    cleaned = _normalise(stripped)

    ratio = _parse_ratio(cleaned)
    if ratio is not None:
        return ParsedText(raw, TextKind.SCALE_RATIO, ratio=ratio)
    if _AREA.search(stripped):
        return ParsedText(raw, TextKind.AREA)

    size = _SIZE_SPLIT.match(cleaned)
    if size is not None:
        slots = _parse_pair(size.group("a"), size.group("b"), plausible_range_m)
        if slots is not None:
            return ParsedText(raw, TextKind.ROOM_SIZE, lengths=slots)

    readings = _parse_length(cleaned, plausible_range_m)
    if readings:
        return ParsedText(raw, TextKind.LINEAR, lengths=(readings,))
    return ParsedText(raw, TextKind.OTHER)


# -- normalisation ----------------------------------------------------------


def _normalise(text: str) -> str:
    out = text.translate(_QUOTES).replace("''", '"')
    out = re.sub(r"\s+", " ", out).strip()
    # Common OCR confusions between letters and digits, applied only next to digits.
    out = re.sub(r"(?<=\d)O(?=[\d.,\s'\"-]|$)", "0", out)
    out = re.sub(r"(?<![A-Za-z])O(?=\d)", "0", out)
    out = re.sub(r"(?<=\d)[lI|](?=[\d.,\s'\"-]|$)", "1", out)
    out = re.sub(r"(?<![A-Za-z])[lI|](?=\d)", "1", out)
    return out


def _parse_ratio(text: str) -> float | None:
    match = _RATIO.match(text) or _RATIO_SLASH.match(text)
    if match is None:
        return None
    value = int(match.group(1))
    return float(value) if value >= 1 else None


# -- lengths ----------------------------------------------------------------


def _parse_pair(
    a: str, b: str, window: tuple[float, float]
) -> tuple[tuple[LengthReading, ...], tuple[LengthReading, ...]] | None:
    first, second = _parse_length(a, window), _parse_length(b, window)
    # "3.5 x 4.0 m": a unit on one side applies to the other.
    if first and second:
        if _is_bare(first) and not _is_bare(second):
            first = _parse_length(a, window, hint_unit=_single_unit(second)) or first
        elif _is_bare(second) and not _is_bare(first):
            second = _parse_length(b, window, hint_unit=_single_unit(first)) or second
    if not first or not second:
        return None
    return first, second


def _is_bare(readings: tuple[LengthReading, ...]) -> bool:
    return any(r.weight < _EXPLICIT_WEIGHT for r in readings)


def _single_unit(readings: tuple[LengthReading, ...]) -> str | None:
    units = {r.unit for r in readings}
    only = next(iter(units)) if len(units) == 1 else None
    return only if only in METERS_PER_UNIT else None


def _parse_length(
    token: str, window: tuple[float, float], hint_unit: str | None = None
) -> tuple[LengthReading, ...]:
    token = token.strip()
    if not token:
        return ()

    feet_inch = _FEET_INCH.match(token)
    if feet_inch is not None:
        feet = float(feet_inch.group(1))
        inches = float(feet_inch.group(2) or 0.0)
        if feet_inch.group(3) and feet_inch.group(4):
            denominator = int(feet_inch.group(4))
            if denominator == 0:
                return ()
            inches += int(feet_inch.group(3)) / denominator
        if inches > 12.0:
            return ()
        return _explicit((feet * 12.0 + inches) * METERS_PER_UNIT["in"], "ft-in", window)

    inch_only = _INCH_ONLY.match(token)
    if inch_only is not None and (inch_only.group(1) or inch_only.group(2)):
        inches = float(inch_only.group(1) or 0.0)
        if inch_only.group(2) and inch_only.group(3):
            denominator = int(inch_only.group(3))
            if denominator == 0:
                return ()
            inches += int(inch_only.group(2)) / denominator
        return _explicit(inches * METERS_PER_UNIT["in"], "in", window)

    match = _NUM_UNIT.match(token)
    if match is None:
        return ()
    values = _numeric_values(match.group("num").strip())
    unit = match.group("unit")
    unit = _UNIT_ALIASES.get(unit.lower(), unit.lower()) if unit else hint_unit
    readings: dict[float, LengthReading] = {}
    for value in values:
        candidates = (
            [(value * METERS_PER_UNIT[unit], _EXPLICIT_WEIGHT, unit)]
            if unit
            else _bare_interpretations(
                value, "." in match.group("num") or "," in match.group("num")
            )
        )
        for meters, weight, label in candidates:
            if window[0] <= meters <= window[1]:
                key = round(meters, 6)
                if key not in readings or readings[key].weight < weight:
                    readings[key] = LengthReading(meters, weight, label)
    return tuple(sorted(readings.values(), key=lambda r: -r.weight))


def _explicit(meters: float, unit: str, window: tuple[float, float]) -> tuple[LengthReading, ...]:
    if window[0] <= meters <= window[1]:
        return (LengthReading(meters, _EXPLICIT_WEIGHT, unit),)
    return ()


def _numeric_values(number: str) -> tuple[float, ...]:
    """All plausible numeric readings (thousands separator vs decimal comma)."""
    if _THOUSANDS_COMMA.match(number):
        values = [float(number.replace(",", ""))]
        if re.match(r"^\d+,\d{3}$", number):  # "3,600": thousands or decimal
            values.append(float(number.replace(",", ".")))
        return tuple(values)
    if _THOUSANDS_SPACE.match(number):
        return (float(number.replace(" ", "").replace(",", ".")),)
    if _PLAIN_NUMBER.match(number):
        return (float(number.replace(",", ".")),)
    return ()


def _bare_interpretations(value: float, has_separator: bool) -> list[tuple[float, float, str]]:
    """Plausible (meters, weight, unit) readings of a unitless number."""
    mm, cm, m = METERS_PER_UNIT["mm"], METERS_PER_UNIT["cm"], METERS_PER_UNIT["m"]
    if has_separator and value != int(value):
        if value < 100:
            return [(value * m, _BARE_METER_WEIGHT, "m")]
        return [
            (value * cm, _BARE_DECIMAL_LARGE_WEIGHT, "cm"),
            (value * mm, _BARE_DECIMAL_LARGE_WEIGHT, "mm"),
        ]
    if value < 100:
        return [(value * m, _BARE_METER_WEIGHT, "m")]
    if value < 1000:
        return [
            (value * cm, _BARE_AMBIGUOUS_WEIGHT, "cm"),
            (value * mm, _BARE_AMBIGUOUS_WEIGHT, "mm"),
        ]
    return [
        (value * mm, _BARE_MILLIMETER_LARGE_WEIGHT, "mm"),
        (value * cm, _BARE_UNLIKELY_WEIGHT, "cm"),
    ]
