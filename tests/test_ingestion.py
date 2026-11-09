from __future__ import annotations

from pathlib import Path

import cv2
import ezdxf
import numpy as np
import pymupdf
import pytest

from ingestion import (
    CorruptFileError,
    IngestionConfig,
    IngestionService,
    InputType,
    InvalidPageError,
    ResourceLimitError,
    UnsupportedFormatError,
)
from ingestion.raster import order_corners

# -- fixtures -----------------------------------------------------------------


def _plan(width: int = 600, height: int = 400) -> np.ndarray:
    img = np.full((height, width, 3), 255, np.uint8)
    cv2.rectangle(img, (50, 50), (width - 50, height - 50), (0, 0, 0), 4)
    cv2.line(img, (width // 2, 50), (width // 2, height - 50), (0, 0, 0), 4)
    return img


@pytest.fixture()
def service() -> IngestionService:
    return IngestionService()


@pytest.fixture()
def png(tmp_path: Path) -> Path:
    path = tmp_path / "plan.png"
    cv2.imwrite(str(path), _plan())
    return path


@pytest.fixture()
def pdf(tmp_path: Path) -> Path:
    path = tmp_path / "plan.pdf"
    doc = pymupdf.open()
    for _ in range(2):
        page = doc.new_page(width=595, height=842)
        page.draw_rect(pymupdf.Rect(50, 50, 545, 792), width=2)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture()
def dxf(tmp_path: Path) -> Path:
    path = tmp_path / "plan.dxf"
    doc = ezdxf.new(setup=True)
    doc.units = ezdxf.units.MM
    doc.layers.add("WALLS")
    doc.layers.add("HIDDEN", dxfattribs={"flags": 1})  # frozen
    msp = doc.modelspace()
    msp.add_line((0, 0), (1000, 0), dxfattribs={"layer": "walls"})
    msp.add_lwpolyline(
        [(0, 0), (1000, 0), (1000, 800), (0, 800)], close=True, dxfattribs={"layer": "WALLS"}
    )
    msp.add_arc((500, 400), 200, 0, 90, dxfattribs={"layer": "WALLS"})
    msp.add_circle((200, 200), 50, dxfattribs={"layer": "WALLS"})
    msp.add_line((0, 0), (5, 5), dxfattribs={"layer": "HIDDEN"})
    msp.add_text("label")
    block = doc.blocks.new("DOOR")
    block.add_line((0, 0), (900, 0))
    block.add_arc((0, 0), 900, 0, 90)
    msp.add_blockref("DOOR", (100, 100), dxfattribs={"layer": "WALLS"})
    doc.saveas(str(path))
    return path


# -- raster -------------------------------------------------------------------


def test_png_is_normalized(service: IngestionService, png: Path) -> None:
    result = service.process(png)
    assert result.input_type is InputType.RASTER
    assert result.image is not None and result.image.dtype == np.uint8
    assert result.image.ndim == 2
    assert result.original_dimensions == (600, 400)
    assert result.metadata.scale_factor == 1.0
    assert not result.metadata.dewarped


def test_inverted_blueprint_is_flipped_to_dark_on_light(
    service: IngestionService, tmp_path: Path
) -> None:
    path = tmp_path / "blue.png"
    cv2.imwrite(str(path), 255 - _plan())
    image = service.process(path).image
    assert image is not None and np.median(image) > 200


def test_large_image_is_downscaled_with_scale_recorded(tmp_path: Path) -> None:
    path = tmp_path / "big.png"
    cv2.imwrite(str(path), _plan(4000, 2000))
    result = IngestionService(IngestionConfig(max_dimension=1000)).process(path)
    assert result.image is not None and max(result.image.shape) == 1000
    assert result.metadata.scale_factor == pytest.approx(0.25)


def test_alpha_png_is_composited_on_white(service: IngestionService, tmp_path: Path) -> None:
    rgba = np.zeros((200, 300, 4), np.uint8)  # fully transparent
    path = tmp_path / "alpha.png"
    cv2.imwrite(str(path), rgba)
    image = service.process(path).image
    assert image is not None and np.median(image) > 200


def test_16bit_image_is_supported(service: IngestionService, tmp_path: Path) -> None:
    path = tmp_path / "deep.png"
    cv2.imwrite(str(path), (_plan()[:, :, 0].astype(np.uint16)) * 257)
    result = service.process(path)
    assert result.image is not None and result.image.dtype == np.uint8


def test_photo_of_skewed_paper_is_dewarped(service: IngestionService, tmp_path: Path) -> None:
    photo = np.full((800, 1000, 3), 40, np.uint8)
    quad = np.array([[150, 120], [850, 180], [800, 700], [100, 640]], np.int32)
    cv2.fillConvexPoly(photo, quad, (240, 240, 240))
    path = tmp_path / "photo.jpg"
    cv2.imwrite(str(path), photo)
    result = service.process(path)
    assert result.metadata.dewarped
    assert result.image is not None and np.median(result.image) > 200


def test_clean_scan_with_page_border_is_not_dewarped(
    service: IngestionService, tmp_path: Path
) -> None:
    img = _plan()
    cv2.rectangle(img, (0, 0), (599, 399), (0, 0, 0), 3)  # frame hugging the image edge
    path = tmp_path / "framed.png"
    cv2.imwrite(str(path), img)
    assert not service.process(path).metadata.dewarped


def test_order_corners() -> None:
    pts = np.array([[10, 90], [90, 10], [10, 10], [90, 90]])
    ordered = order_corners(pts)
    assert ordered.tolist() == [[10, 10], [90, 10], [90, 90], [10, 90]]


def test_unicode_path(service: IngestionService, tmp_path: Path) -> None:
    path = tmp_path / "план_é.png"
    cv2.imwrite(str(tmp_path / "t.png"), _plan())
    (tmp_path / "t.png").rename(path)
    assert service.process(path).image is not None


# -- pdf ----------------------------------------------------------------------


def test_pdf_renders_with_dpi_and_page_info(service: IngestionService, pdf: Path) -> None:
    result = service.process(pdf)
    meta = result.metadata
    assert result.image is not None and max(result.image.shape) <= 2048
    assert meta.page_count == 2 and meta.page_index == 0
    assert meta.dpi == pytest.approx(300 * meta.scale_factor)
    assert any("only page 0" in w for w in meta.warnings)
    assert not meta.dewarped


def test_pdf_render_size_is_capped(pdf: Path) -> None:
    cfg = IngestionConfig(max_render_pixels=1_000_000, max_dimension=4000)
    result = IngestionService(cfg).process(pdf)
    assert result.image is not None
    assert result.image.size <= 1_000_000 * 1.01
    assert result.metadata.dpi is not None and result.metadata.dpi < 300


def test_pdf_page_selection_and_invalid_page(pdf: Path) -> None:
    assert IngestionService(IngestionConfig(pdf_page=1)).process(pdf).metadata.page_index == 1
    with pytest.raises(InvalidPageError):
        IngestionService(IngestionConfig(pdf_page=5)).process(pdf)


def test_corrupt_pdf(service: IngestionService, tmp_path: Path) -> None:
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"%PDF-1.7\nthis is not a real pdf")
    with pytest.raises(CorruptFileError):
        service.process(path)


# -- dxf ----------------------------------------------------------------------


def test_dxf_extracts_geometry_units_and_bounds(service: IngestionService, dxf: Path) -> None:
    result = service.process(dxf)
    assert result.input_type is InputType.VECTOR and result.image is None
    assert result.vectors is not None
    types = {v.entity_type for v in result.vectors}
    assert {"LINE", "LWPOLYLINE", "ARC", "CIRCLE"} <= types
    meta = result.metadata
    assert meta.units == "mm"
    assert meta.entity_count == len(result.vectors)
    assert meta.bounds is not None and meta.bounds.width >= 1000
    assert meta.skipped_entities.get("TEXT") == 1


def test_dxf_layers_uppercased_closed_polylines_and_blocks(
    service: IngestionService, dxf: Path
) -> None:
    vectors = service.process(dxf).vectors
    assert vectors is not None
    assert all(v.layer == v.layer.upper() for v in vectors)
    poly = next(v for v in vectors if v.entity_type == "LWPOLYLINE")
    assert poly.closed and poly.points[0] == poly.points[-1]
    # DOOR block: line starting at insert point (100, 100) exploded into modelspace
    assert any(v.points[0] == (100.0, 100.0) and v.points[-1] == (1000.0, 100.0) for v in vectors)


def test_dxf_hidden_layers_skipped_unless_disabled(dxf: Path) -> None:
    default = IngestionService().process(dxf)
    assert default.vectors is not None
    assert all(v.layer != "HIDDEN" for v in default.vectors)
    shown = IngestionService(IngestionConfig(skip_hidden_layers=False)).process(dxf)
    assert shown.vectors is not None and any(v.layer == "HIDDEN" for v in shown.vectors)


def test_dxf_curves_are_flattened_within_tolerance(service: IngestionService, dxf: Path) -> None:
    vectors = service.process(dxf).vectors
    assert vectors is not None
    circle = next(v for v in vectors if v.entity_type == "CIRCLE")
    assert len(circle.points) > 8
    radii = [np.hypot(x - 200, y - 200) for x, y in circle.points]
    assert all(abs(r - 50) <= 1.0 for r in radii)


def test_dxf_entity_cap(dxf: Path) -> None:
    with pytest.raises(ResourceLimitError):
        IngestionService(IngestionConfig(max_dxf_entities=2)).process(dxf)


def test_corrupt_dxf(service: IngestionService, tmp_path: Path) -> None:
    path = tmp_path / "bad.dxf"
    path.write_text("not a dxf file at all")
    with pytest.raises(CorruptFileError):
        service.process(path)


# -- validation ---------------------------------------------------------------


def test_missing_file(service: IngestionService, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        service.process(tmp_path / "nope.png")


def test_empty_file(service: IngestionService, tmp_path: Path) -> None:
    path = tmp_path / "empty.png"
    path.touch()
    with pytest.raises(CorruptFileError):
        service.process(path)


def test_unsupported_extension(service: IngestionService, tmp_path: Path) -> None:
    path = tmp_path / "plan.xyz"
    path.write_bytes(b"hello world")
    with pytest.raises(UnsupportedFormatError):
        service.process(path)


def test_dwg_gives_helpful_error(service: IngestionService, tmp_path: Path) -> None:
    path = tmp_path / "plan.dwg"
    path.write_bytes(b"AC1032" + b"\x00" * 64)
    with pytest.raises(UnsupportedFormatError, match="DXF"):
        service.process(path)


def test_content_trumps_extension(service: IngestionService, png: Path, tmp_path: Path) -> None:
    disguised = tmp_path / "actually_png.pdf"
    disguised.write_bytes(png.read_bytes())
    assert service.process(disguised).metadata.source_format == "png"


def test_extension_with_garbage_content_is_rejected(
    service: IngestionService, tmp_path: Path
) -> None:
    path = tmp_path / "fake.png"
    path.write_bytes(b"definitely not an image")
    with pytest.raises(CorruptFileError):
        service.process(path)


def test_file_size_limit(png: Path) -> None:
    with pytest.raises(ResourceLimitError):
        IngestionService(IngestionConfig(max_file_bytes=10)).process(png)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_dpi": 0},
        {"pdf_page": -1},
        {"polarity_threshold": 300},
        {"dewarp_min_area_ratio": 0.99, "dewarp_max_area_ratio": 0.5},
    ],
)
def test_invalid_config_rejected(kwargs: dict) -> None:  # type: ignore[type-arg]
    with pytest.raises(ValueError):
        IngestionConfig(**kwargs)
