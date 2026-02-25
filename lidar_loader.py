"""LAZ/LAS loading helpers for LiDAR View."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import laspy
import numpy as np


@dataclass
class LoadedTile:
    path: str
    points: np.ndarray
    crs: Optional[str]
    classification: Optional[np.ndarray]


def read_laz_precise(path: str) -> np.ndarray:
    """Read LAS/LAZ coordinates in float64 with header scale/offset precision."""
    las = laspy.read(path)
    scales = np.asarray(las.header.scales, dtype=np.float64)
    offsets = np.asarray(las.header.offsets, dtype=np.float64)
    ints = np.column_stack((las.X, las.Y, las.Z)).astype(np.float64)
    return ints * scales + offsets


def infer_crs_from_points(points: np.ndarray) -> Optional[str]:
    if points.size == 0:
        return None

    x = points[:, 0]
    y = points[:, 1]
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))

    if xmin >= -180 and xmax <= 180 and ymin >= -90 and ymax <= 90:
        return "EPSG:4326"

    if xmin >= -100000 and xmax <= 1400000 and ymin >= 5900000 and ymax <= 7300000:
        return "EPSG:2154"

    if xmin >= -20037508 and xmax <= 20037508 and ymin >= -20037508 and ymax <= 20037508:
        return "EPSG:3857"

    return None


def read_crs_string(las: laspy.LasData) -> Optional[str]:
    try:
        parsed = las.header.parse_crs()
        if parsed is not None:
            return parsed.to_string()
    except Exception:
        return None
    return None


def load_laz_tile(path: str) -> LoadedTile:
    las = laspy.read(path)

    # Convert integer LAS storage to real-world coordinates using header metadata.
    scales = np.asarray(las.header.scales, dtype=np.float64)
    offsets = np.asarray(las.header.offsets, dtype=np.float64)
    ints = np.column_stack((las.X, las.Y, las.Z)).astype(np.float64)
    points = ints * scales + offsets

    crs = read_crs_string(las)
    if crs is None:
        crs = infer_crs_from_points(points)

    classification = None
    try:
        # Keep raw LAS classes (uint8) so UI filters/type coloring stay lightweight.
        if hasattr(las, "classification"):
            classification = np.asarray(las.classification, dtype=np.uint8)
            if len(classification) != len(points):
                classification = None
    except Exception:
        classification = None

    return LoadedTile(
        path=str(Path(path).resolve()),
        points=points,
        crs=crs,
        classification=classification,
    )


def load_laz(path: str, max_points: Optional[int] = None):
    """Backward-compatible helper used by older scripts/tests."""
    tile = load_laz_tile(path)
    points = tile.points.copy()

    mask = np.isfinite(points).all(axis=1)
    points = points[mask]

    colors = None
    if max_points and len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[idx]

    center = points.mean(axis=0) if len(points) else np.zeros(3, dtype=np.float64)
    points = (points - center).astype(np.float32)

    return points, colors

