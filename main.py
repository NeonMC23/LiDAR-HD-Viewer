"""LiDAR View main application.

This module contains the Qt main window, rendering orchestration, asynchronous
loading/build threads, color pipelines, and UI settings.
"""
import math
import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph.opengl as gl
from PyQt6.QtCore import QThread, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QFontMetrics, QVector3D
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from lidar_loader import load_laz_tile


POINT_SIZE = 1.5
SAT_TILE_SIZE = 256
SURFACE_GRID_MIN = 220
SURFACE_GRID_MAX = 700
SAT_DEFAULT_URL = (
    "https://services.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

LIDAR_CLASS_LABELS = {
    0: "Created / Never classified",
    1: "Unclassified",
    2: "Ground",
    3: "Low vegetation",
    4: "Medium vegetation",
    5: "High vegetation",
    6: "Building",
    7: "Low point (noise)",
    8: "Model key-point",
    9: "Water",
    10: "Rail",
    11: "Road surface",
    12: "Overlap",
    13: "Wire guard",
    14: "Wire conductor",
    15: "Transmission tower",
    16: "Wire-structure connector",
    17: "Bridge deck",
    18: "High noise",
}

LIDAR_CLASS_COLORS = {
    0: (0.65, 0.65, 0.65, 1.0),
    1: (0.80, 0.80, 0.80, 1.0),
    2: (0.55, 0.35, 0.22, 1.0),
    3: (0.55, 0.78, 0.28, 1.0),
    4: (0.30, 0.68, 0.22, 1.0),
    5: (0.16, 0.50, 0.18, 1.0),
    6: (0.82, 0.22, 0.18, 1.0),
    7: (1.00, 0.20, 1.00, 1.0),
    8: (1.00, 0.82, 0.18, 1.0),
    9: (0.20, 0.48, 0.95, 1.0),
    10: (0.40, 0.40, 0.40, 1.0),
    11: (0.25, 0.25, 0.25, 1.0),
    12: (0.84, 0.84, 0.84, 1.0),
    13: (0.86, 0.42, 0.16, 1.0),
    14: (0.96, 0.56, 0.22, 1.0),
    15: (0.50, 0.50, 0.88, 1.0),
    16: (0.56, 0.56, 0.96, 1.0),
    17: (0.58, 0.38, 0.24, 1.0),
    18: (1.00, 0.00, 0.00, 1.0),
}

THEMES = {
    "midnight": {
        "bg": "#0b1220",
        "fg": "#e2e8f0",
        "muted": "#cbd5e1",
        "panel": "#1e293b",
        "panel_hover": "#334155",
        "panel_disabled": "#111827",
        "border": "#334155",
        "accent": "#14b8a6",
        "accent_soft": "#0f766e",
        "slider_sub": "#0891b2",
        "slider_handle": "#67e8f9",
        "popup_bg": "rgba(15, 23, 42, 236)",
    },
    "dawn": {
        "bg": "#f6f8fb",
        "fg": "#10233a",
        "muted": "#2d455f",
        "panel": "#ffffff",
        "panel_hover": "#e8edf4",
        "panel_disabled": "#dfe7f1",
        "border": "#b8c4d2",
        "accent": "#118ab2",
        "accent_soft": "#0f6b89",
        "slider_sub": "#3ba9cf",
        "slider_handle": "#0f6b89",
        "popup_bg": "rgba(255, 255, 255, 244)",
    },
    "volcanic": {
        "bg": "#1a1412",
        "fg": "#f7e8df",
        "muted": "#d6bfb1",
        "panel": "#2b1f1a",
        "panel_hover": "#3a2a24",
        "panel_disabled": "#201813",
        "border": "#634338",
        "accent": "#e76f51",
        "accent_soft": "#b84a35",
        "slider_sub": "#f4a261",
        "slider_handle": "#ffd7b2",
        "popup_bg": "rgba(34, 24, 19, 238)",
    },
    "forest": {
        "bg": "#0f1b14",
        "fg": "#e6f3e8",
        "muted": "#bfd5c4",
        "panel": "#1a2d21",
        "panel_hover": "#24402f",
        "panel_disabled": "#132219",
        "border": "#355844",
        "accent": "#4caf50",
        "accent_soft": "#2e7d32",
        "slider_sub": "#66bb6a",
        "slider_handle": "#c8e6c9",
        "popup_bg": "rgba(18, 34, 25, 238)",
    },
    "blueprint": {
        "bg": "#0a1b2b",
        "fg": "#e5f3ff",
        "muted": "#b7d2e7",
        "panel": "#14314a",
        "panel_hover": "#1d4668",
        "panel_disabled": "#10263a",
        "border": "#2b5d85",
        "accent": "#3fb6ff",
        "accent_soft": "#1883c7",
        "slider_sub": "#4fc3f7",
        "slider_handle": "#d7f1ff",
        "popup_bg": "rgba(14, 39, 61, 238)",
    },
    "sandstone": {
        "bg": "#f5efe4",
        "fg": "#3b2f23",
        "muted": "#6c5a47",
        "panel": "#fffaf1",
        "panel_hover": "#efe4d3",
        "panel_disabled": "#e8dbc8",
        "border": "#c7b091",
        "accent": "#c0843d",
        "accent_soft": "#9f6428",
        "slider_sub": "#d9a15f",
        "slider_handle": "#8b5a2b",
        "popup_bg": "rgba(255, 248, 235, 245)",
    },
}

I18N = {
    "en": {
        "window_title": "LiDAR Viewer V3",
        "load_laz": "Load .laz",
        "add_laz": "Add .laz",
        "multi_off": "Multi mode: OFF",
        "multi_on": "Multi mode: ON",
        "reset_data": "Reset data",
        "refresh_data": "Refresh data",
        "remove_tile": "Remove tile",
        "filters": "Filters",
        "point_class_filters": "Point class filters",
        "select_all": "Select all",
        "view": "View",
        "surface": "Surface",
        "color": "Color",
        "sat_zoom": "Sat zoom",
        "quality": "Quality",
        "settings": "Settings",
        "open_settings": "Open settings",
        "view_points": "Points",
        "view_surface": "Opaque surface",
        "surface_auto": "Auto (recommended)",
        "surface_manual": "Manual",
        "color_elevation": "Elevation",
        "color_satellite": "Satellite",
        "color_type": "Type",
        "status_ready": "Ready.",
        "processing": "Processing...",
        "reading_laz": "Reading LAZ files...",
        "preparing_scene": "Preparing scene...",
        "refreshing_data": "Refreshing scene...",
        "generating_surface": "Generating surface...",
        "sat_coloring": "Satellite coloring z{zoom}...",
        "multi_active": "Multi mode active: add one or more .laz files.",
        "single_active": "Single mode active: each load replaces scene.",
        "select_tile": "Select a loaded tile",
        "settings_title": "Application settings",
        "language": "Language",
        "theme": "Theme",
        "ui_scale": "UI scale (%)",
        "popup_width": "Popup width (%)",
        "show_tile_centers": "Show tile outlines in 3D",
        "compact_toolbar": "Compact toolbar",
        "lang_english": "English",
        "lang_french": "French",
        "theme_midnight": "Midnight",
        "theme_dawn": "Dawn",
        "theme_volcanic": "Volcanic",
        "theme_forest": "Forest",
        "theme_blueprint": "Blueprint",
        "theme_sandstone": "Sandstone",
        "loading_in_progress": "Loading already in progress...",
        "open_file": "Open a .laz file",
        "add_files": "Add .laz files",
        "loading_files": "Loading {count} file(s)...",
        "no_tiles_loaded": "No tiles loaded.",
        "data_reset": "Data reset.",
        "enable_multi_remove": "Enable multi mode to remove a specific tile.",
        "no_tile_selected": "No tile selected.",
        "selected_tile": "Selected tile: {name}",
        "scene_build_error": "Scene build error: {error}",
        "no_active_points": "No active points to display (check filters).",
        "copy_report": "Copy report",
        "copy_report_tip": "Copy diagnostics (errors, warnings, loaded tiles) to clipboard",
        "report_copied": "Diagnostics report copied to clipboard.",
    },
    "fr": {
        "window_title": "LiDAR Viewer V3",
        "load_laz": "Charger .laz",
        "add_laz": "Ajouter .laz",
        "multi_off": "Mode multi : OFF",
        "multi_on": "Mode multi : ON",
        "reset_data": "R?initialiser les donn?es",
        "refresh_data": "Rafra?chir l'affichage",
        "remove_tile": "Supprimer la tuile",
        "filters": "Filtres",
        "point_class_filters": "Filtres des classes de points",
        "select_all": "Tout s?lectionner",
        "view": "Vue",
        "surface": "Surface",
        "color": "Couleur",
        "sat_zoom": "Zoom sat",
        "quality": "Qualit?",
        "settings": "Param?tres",
        "open_settings": "Ouvrir les param?tres",
        "view_points": "Points",
        "view_surface": "Surface opaque",
        "surface_auto": "Auto (recommand?)",
        "surface_manual": "Manuel",
        "color_elevation": "Altim?trie",
        "color_satellite": "Satellite",
        "color_type": "Type",
        "status_ready": "Pr?t.",
        "processing": "Traitement...",
        "reading_laz": "Lecture des fichiers LAZ...",
        "preparing_scene": "Pr?paration de la sc?ne...",
        "refreshing_data": "Rafra?chissement de la sc?ne...",
        "generating_surface": "G?n?ration de la surface...",
        "sat_coloring": "Colorisation satellite z{zoom}...",
        "multi_active": "Mode multi actif : ajoute un ou plusieurs fichiers .laz.",
        "single_active": "Mode simple actif : chaque chargement remplace la sc?ne.",
        "select_tile": "S?lectionner une tuile charg?e",
        "settings_title": "Param?tres de l'application",
        "language": "Langue",
        "theme": "Th?me",
        "ui_scale": "?chelle UI (%)",
        "popup_width": "Largeur popups (%)",
        "show_tile_centers": "Afficher les contours de tuiles en 3D",
        "compact_toolbar": "Barre compacte",
        "lang_english": "Anglais",
        "lang_french": "Fran?ais",
        "theme_midnight": "Minuit",
        "theme_dawn": "Aurore",
        "theme_volcanic": "Volcanique",
        "theme_forest": "For?t",
        "theme_blueprint": "Plan bleu",
        "theme_sandstone": "Gr?s",
        "loading_in_progress": "Un chargement est d?j? en cours...",
        "open_file": "Ouvrir un fichier .laz",
        "add_files": "Ajouter des fichiers .laz",
        "loading_files": "Chargement de {count} fichier(s)...",
        "no_tiles_loaded": "Aucune tuile charg?e.",
        "data_reset": "Donn?es r?initialis?es.",
        "enable_multi_remove": "Active le mode multi pour supprimer une tuile sp?cifique.",
        "no_tile_selected": "Aucune tuile s?lectionn?e.",
        "selected_tile": "Tuile s?lectionn?e : {name}",
        "scene_build_error": "Erreur de pr?paration de sc?ne : {error}",
        "no_active_points": "Aucun point actif ? afficher (v?rifie les filtres).",
        "copy_report": "Copier rapport",
        "copy_report_tip": "Copier le diagnostic (erreurs, avertissements, tuiles) dans le presse-papiers",
        "report_copied": "Rapport de diagnostic copi? dans le presse-papiers.",
    },
}


class LoaderThread(QThread):
    loaded = pyqtSignal(object, object)
    failed = pyqtSignal(str)

    def __init__(self, paths):
        super().__init__()
        self.paths = paths

    def run(self):
        loaded_tiles = []
        warnings = []
        cpu = os.cpu_count() or 4
        max_workers = max(1, min(len(self.paths), 8, max(4, cpu - 1)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(load_laz_tile, path): path for path in self.paths}
            for fut in as_completed(futures):
                src_path = futures[fut]
                try:
                    tile = fut.result()
                except Exception as exc:
                    warnings.append(f"{Path(src_path).name}: {exc}")
                    continue

                if tile.points.size == 0:
                    warnings.append(f"{Path(src_path).name}: empty file")
                    continue

                # Keep tile points in float32 to reduce RAM pressure in multi-tile sessions.
                pts = np.asarray(tile.points, dtype=np.float32)
                if not pts.flags["C_CONTIGUOUS"]:
                    pts = np.ascontiguousarray(pts, dtype=np.float32)
                loaded_tiles.append(
                    {
                        "path": tile.path,
                        "points": pts,
                        "crs": tile.crs,
                        "classification": tile.classification,
                    }
                )

        if not loaded_tiles:
            if warnings:
                self.failed.emit("No file loaded. " + " | ".join(warnings[:2]))
            else:
                self.failed.emit("No file loaded.")
            return

        self.loaded.emit(loaded_tiles, warnings)

class SatelliteColorizer:
    GLOBAL_TILE_CACHE = {}

    def __init__(self, url_template=SAT_DEFAULT_URL):
        self.url_template = url_template
        self.tile_cache = SatelliteColorizer.GLOBAL_TILE_CACHE
        self.transformer_cache = {}

    def clear_cache(self):
        self.tile_cache.clear()

    def set_url_template(self, url_template):
        if url_template != self.url_template:
            self.url_template = url_template
            self.clear_cache()

    def colorize(self, points_world_xy, src_crs, zoom):
        lon, lat = self._to_lon_lat(points_world_xy, src_crs)

        valid = np.isfinite(lon) & np.isfinite(lat) & (lat > -85.05112878) & (lat < 85.05112878)
        rgba = np.zeros((len(points_world_xy), 4), dtype=np.float32)
        rgba[:, 0:3] = 0.72
        rgba[:, 3] = 1.0

        if not np.any(valid):
            return rgba, "No projectable points for satellite tiles."

        n = (2**zoom) * SAT_TILE_SIZE
        lon_v = lon[valid]
        lat_v = lat[valid]

        px = (lon_v + 180.0) / 360.0 * n
        lat_rad = np.radians(lat_v)
        py = (
            (1.0 - np.log(np.tan(lat_rad) + (1.0 / np.cos(lat_rad))) / math.pi)
            / 2.0
            * n
        )

        tile_x = np.floor(px / SAT_TILE_SIZE).astype(np.int64)
        tile_y = np.floor(py / SAT_TILE_SIZE).astype(np.int64)
        pix_x = np.clip((px - tile_x * SAT_TILE_SIZE).astype(np.int64), 0, SAT_TILE_SIZE - 1)
        pix_y = np.clip((py - tile_y * SAT_TILE_SIZE).astype(np.int64), 0, SAT_TILE_SIZE - 1)

        max_tile = 2**zoom
        in_bounds = (tile_x >= 0) & (tile_x < max_tile) & (tile_y >= 0) & (tile_y < max_tile)

        valid_idx = np.flatnonzero(valid)
        mapped_idx = valid_idx[in_bounds]
        tile_x = tile_x[in_bounds]
        tile_y = tile_y[in_bounds]
        pix_x = pix_x[in_bounds]
        pix_y = pix_y[in_bounds]

        if len(mapped_idx) == 0:
            return rgba, "Satellite projection outside selected zoom coverage."

        keys = np.stack((tile_x, tile_y), axis=1)
        uniq_keys, inv = np.unique(keys, axis=0, return_inverse=True)

        failed = 0
        for k_i, (tx, ty) in enumerate(uniq_keys):
            tile = self._fetch_tile(zoom, int(tx), int(ty))
            match = inv == k_i
            idx = mapped_idx[match]
            if tile is None:
                failed += 1
                continue
            rgb = tile[pix_y[match], pix_x[match]].astype(np.float32) / 255.0
            rgba[idx, 0:3] = rgb

        msg = None
        if failed:
            msg = f"{failed} satellite tile(s) unavailable."
        return rgba, msg

    def _fetch_tile(self, z, x, y):
        key = (z, x, y)
        if key in self.tile_cache:
            return self.tile_cache[key]

        url = self.url_template.format(z=z, x=x, y=y)
        try:
            with urllib.request.urlopen(url, timeout=8) as resp:
                data = resp.read()
        except (urllib.error.URLError, TimeoutError, ValueError):
            self.tile_cache[key] = None
            return None

        image = QImage.fromData(data)
        if image.isNull():
            self.tile_cache[key] = None
            return None

        image = image.convertToFormat(QImage.Format.Format_RGB888)
        width = image.width()
        height = image.height()

        if width <= 0 or height <= 0:
            self.tile_cache[key] = None
            return None

        ptr = image.bits()
        ptr.setsize(width * height * 3)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 3)).copy()
        self.tile_cache[key] = arr
        return arr

    def _to_lon_lat(self, points_world_xy, src_crs):
        x = points_world_xy[:, 0].astype(np.float64, copy=False)
        y = points_world_xy[:, 1].astype(np.float64, copy=False)

        if not src_crs:
            raise RuntimeError("Missing CRS: cannot project to satellite.")

        src = src_crs.lower()
        if "4326" in src or "wgs 84" in src:
            return x, y

        if "3857" in src:
            lon = x / 6378137.0 * 180.0 / math.pi
            lat = (2.0 * np.arctan(np.exp(y / 6378137.0)) - math.pi / 2.0) * 180.0 / math.pi
            return lon, lat

        try:
            from pyproj import CRS, Transformer
        except Exception as exc:
            raise RuntimeError(f"pyproj required for CRS {src_crs}: {exc}") from exc

        if src_crs not in self.transformer_cache:
            src_obj = CRS.from_user_input(src_crs)
            dst_obj = CRS.from_epsg(4326)
            self.transformer_cache[src_crs] = Transformer.from_crs(src_obj, dst_obj, always_xy=True)

        transformer = self.transformer_cache[src_crs]
        lon, lat = transformer.transform(x, y)
        return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


class SatelliteColorThread(QThread):
    result = pyqtSignal(int, object, str)
    failed = pyqtSignal(int, str)

    def __init__(self, request_id, points_world_xy, src_crs, zoom, url_template):
        super().__init__()
        self.request_id = request_id
        self.points_world_xy = points_world_xy
        self.src_crs = src_crs
        self.zoom = zoom
        self.url_template = url_template

    def run(self):
        try:
            colorizer = SatelliteColorizer(url_template=self.url_template)
            rgba, msg = colorizer.colorize(self.points_world_xy, self.src_crs, self.zoom)
            self.result.emit(self.request_id, rgba.astype(np.float32), msg or "")
        except Exception as exc:
            self.failed.emit(self.request_id, str(exc))


class SceneBuildThread(QThread):
    built = pyqtSignal(int, object, object, object, object, object, object, object)
    failed = pyqtSignal(int, str)

    def __init__(self, request_id, tiles_snapshot):
        super().__init__()
        self.request_id = request_id
        self.tiles_snapshot = tiles_snapshot

    def run(self):
        try:
            if not self.tiles_snapshot:
                self.built.emit(self.request_id, None, None, None, None, None, None, None)
                return

            tile_info = {}
            total_points = 0
            sum_xyz = np.zeros(3, dtype=np.float64)
            for tile in self.tiles_snapshot:
                pts = tile["points"]
                cls = tile.get("classification")
                if cls is None or len(cls) != len(pts):
                    cls = np.ones(len(pts), dtype=np.uint8)
                else:
                    cls = np.asarray(cls, dtype=np.uint8)

                n = len(pts)
                if n == 0:
                    continue
                total_points += n
                sum_xyz += np.sum(pts, axis=0, dtype=np.float64)
                tile_info[tile["path"]] = {
                    "start": 0,
                    "end": 0,
                    "center_world": pts.mean(axis=0).astype(np.float64),
                    "min_world": np.min(pts, axis=0).astype(np.float64),
                    "max_world": np.max(pts, axis=0).astype(np.float64),
                    "classification": cls,
                }
            if total_points == 0:
                self.built.emit(self.request_id, None, None, None, None, None, None, None)
                return

            world_center = sum_xyz / float(total_points)
            pts_centered = np.empty((total_points, 3), dtype=np.float32)
            classes_full = np.empty(total_points, dtype=np.uint8)
            offset = 0
            for tile in self.tiles_snapshot:
                pts = tile["points"]
                n = len(pts)
                if n == 0:
                    continue
                cls = tile_info[tile["path"]]["classification"]
                tile_info[tile["path"]]["start"] = offset
                tile_info[tile["path"]]["end"] = offset + n
                pts_centered[offset : offset + n] = (pts - world_center).astype(np.float32, copy=False)
                classes_full[offset : offset + n] = cls
                offset += n

            z = pts_centered[:, 2]
            den = float(np.ptp(z))
            if den < 1e-12:
                z_norm = np.zeros_like(z, dtype=np.float32)
            else:
                z_norm = ((z - z.min()) / den).astype(np.float32)

            crs_values = sorted({t.get("crs") for t in self.tiles_snapshot if t.get("crs")})
            active_crs = crs_values[0] if len(crs_values) == 1 else None

            self.built.emit(
                self.request_id,
                pts_centered,
                z_norm,
                active_crs,
                world_center,
                classes_full,
                tile_info,
                (float(np.min(z)), float(np.max(z))),
            )
        except Exception as exc:
            self.failed.emit(self.request_id, str(exc))


class SurfaceBuildThread(QThread):
    built = pyqtSignal(int, object, str)
    failed = pyqtSignal(int, str)

    def __init__(
        self,
        request_id,
        tiles_snapshot,
        world_center,
        global_zmin,
        global_zmax,
        color_mode,
        active_crs,
        sat_zoom,
        surface_mode,
        surface_value,
        sat_url_template,
        active_classes,
        class_lut,
    ):
        super().__init__()
        self.request_id = request_id
        self.tiles_snapshot = tiles_snapshot
        self.world_center = world_center
        self.global_zmin = global_zmin
        self.global_zmax = global_zmax
        self.color_mode = color_mode
        self.active_crs = active_crs
        self.sat_zoom = sat_zoom
        self.surface_mode = surface_mode
        self.surface_value = surface_value
        self.sat_url_template = sat_url_template
        self.active_classes = set(active_classes)
        self.class_lut = np.asarray(class_lut, dtype=np.float32)

    def run(self):
        try:
            payload = []
            notes = []
            sat = SatelliteColorizer(url_template=self.sat_url_template)

            for tile_path, tile in self.tiles_snapshot.items():
                tile_world = tile["points"]
                if tile_world is None or len(tile_world) < 3:
                    continue
                cls = tile.get("classification")
                if cls is None or len(cls) != len(tile_world):
                    cls = np.ones(len(tile_world), dtype=np.uint8)
                else:
                    cls = np.asarray(cls, dtype=np.uint8)

                if self.active_classes:
                    m = np.isin(cls, np.fromiter(self.active_classes, dtype=np.uint8))
                    if not np.any(m):
                        continue
                    tile_world = tile_world[m]
                    cls = cls[m]
                    if len(tile_world) < 3:
                        continue

                p = (tile_world - self.world_center).astype(np.float64, copy=False)
                x = p[:, 0]
                y = p[:, 1]
                z = p[:, 2]

                xmin, xmax = float(np.min(x)), float(np.max(x))
                ymin, ymax = float(np.min(y)), float(np.max(y))
                if xmax - xmin < 1e-6 or ymax - ymin < 1e-6:
                    continue

                n_points = len(p)
                if self.surface_mode == "Manual":
                    nx = int(self.surface_value)
                else:
                    nx = int(np.sqrt(max(n_points, 1)) * 0.75)
                    nx = max(SURFACE_GRID_MIN, min(SURFACE_GRID_MAX, nx))
                ny = nx

                xi = np.clip(((x - xmin) / (xmax - xmin) * (nx - 1)).astype(np.int32), 0, nx - 1)
                yi = np.clip(((y - ymin) / (ymax - ymin) * (ny - 1)).astype(np.int32), 0, ny - 1)
                flat = xi * ny + yi

                z_sum = np.bincount(flat, weights=z, minlength=nx * ny)
                z_cnt = np.bincount(flat, minlength=nx * ny).astype(np.float64)
                z_grid = np.divide(
                    z_sum, z_cnt, out=np.full(nx * ny, np.nan), where=z_cnt > 0
                ).reshape((nx, ny))
                if np.isnan(z_grid).all():
                    continue

                z_grid = MainWindow._fill_nan_surface_static(z_grid)
                xs = np.linspace(xmin, xmax, nx, dtype=np.float32)
                ys = np.linspace(ymin, ymax, ny, dtype=np.float32)
                zf = z_grid.astype(np.float32)

                if self.color_mode == "Satellite" and self.active_crs is not None:
                    gx, gy = np.meshgrid(xs.astype(np.float64), ys.astype(np.float64), indexing="ij")
                    world_xy = np.c_[
                        gx.ravel() + float(self.world_center[0]),
                        gy.ravel() + float(self.world_center[1]),
                    ]
                    sat_rgba, sat_msg = sat.colorize(world_xy, self.active_crs, self.sat_zoom)
                    colors_vertices = sat_rgba.reshape(-1, 4).astype(np.float32)
                    if sat_msg:
                        notes.append(sat_msg)
                elif self.color_mode == "Type":
                    cls_rgb = self.class_lut[cls][:, :3]
                    c_sum_r = np.bincount(flat, weights=cls_rgb[:, 0], minlength=nx * ny)
                    c_sum_g = np.bincount(flat, weights=cls_rgb[:, 1], minlength=nx * ny)
                    c_sum_b = np.bincount(flat, weights=cls_rgb[:, 2], minlength=nx * ny)
                    den_c = np.maximum(z_cnt, 1.0)
                    rgb = np.stack(
                        [
                            (c_sum_r / den_c).reshape((nx, ny)),
                            (c_sum_g / den_c).reshape((nx, ny)),
                            (c_sum_b / den_c).reshape((nx, ny)),
                        ],
                        axis=-1,
                    ).astype(np.float32)
                    if np.any(z_cnt == 0):
                        valid = z_cnt.reshape((nx, ny)) > 0
                        mean_rgb = np.array(
                            [
                                float(np.mean(cls_rgb[:, 0])),
                                float(np.mean(cls_rgb[:, 1])),
                                float(np.mean(cls_rgb[:, 2])),
                            ],
                            dtype=np.float32,
                        )
                        rgb[~valid] = mean_rgb
                    colors = np.empty((nx, ny, 4), dtype=np.float32)
                    colors[..., :3] = rgb
                    colors[..., 3] = 1.0
                    colors_vertices = colors.reshape(-1, 4).astype(np.float32)
                else:
                    den = max(float(self.global_zmax - self.global_zmin), 1e-12)
                    zn = ((z_grid - float(self.global_zmin)) / den).astype(np.float32)
                    colors = np.empty((nx, ny, 4), dtype=np.float32)
                    colors[..., 0] = zn
                    colors[..., 1] = 0.25
                    colors[..., 2] = 1.0 - zn
                    colors[..., 3] = 1.0
                    colors_vertices = colors.reshape(-1, 4).astype(np.float32)

                payload.append((tile_path, xs, ys, zf, colors_vertices))

            note = "; ".join(sorted(set(notes))) if notes else ""
            self.built.emit(self.request_id, payload, note)
        except Exception as exc:
            self.failed.emit(self.request_id, str(exc))


class SettingsDialog(QDialog):
    def __init__(self, parent, settings):
        super().__init__(parent)
        self.setModal(True)
        self.setMinimumWidth(460)
        self.setSizeGripEnabled(True)
        self._parent = parent

        self.language_combo = QComboBox()
        self.language_combo.addItem(parent._t("lang_english"), "en")
        self.language_combo.addItem(parent._t("lang_french"), "fr")
        self.language_combo.setCurrentIndex(max(0, self.language_combo.findData(settings["language"])))

        self.theme_combo = QComboBox()
        for theme_id in THEMES.keys():
            self.theme_combo.addItem(parent._theme_display_name(theme_id), theme_id)
        self.theme_combo.setCurrentIndex(max(0, self.theme_combo.findData(settings["theme"])))

        self.scale_spin = QSpinBox()
        self.scale_spin.setRange(85, 150)
        self.scale_spin.setSingleStep(5)
        self.scale_spin.setValue(int(settings["ui_scale"]))

        self.popup_width_spin = QSpinBox()
        self.popup_width_spin.setRange(15, 60)
        self.popup_width_spin.setSingleStep(5)
        self.popup_width_spin.setValue(int(settings["popup_width_percent"]))

        self.compact_check = QCheckBox(parent._t("compact_toolbar"))
        self.compact_check.setChecked(bool(settings["compact_toolbar"]))

        self.tile_centers_check = QCheckBox(parent._t("show_tile_centers"))
        self.tile_centers_check.setChecked(bool(settings["show_tile_centers"]))

        visual_group = QGroupBox(parent._t("settings"))
        visual_form = QFormLayout()
        visual_form.addRow(parent._t("language"), self.language_combo)
        visual_form.addRow(parent._t("theme"), self.theme_combo)
        visual_form.addRow(parent._t("ui_scale"), self.scale_spin)
        visual_form.addRow(parent._t("popup_width"), self.popup_width_spin)
        visual_form.addRow(self.compact_check)
        visual_form.addRow(self.tile_centers_check)
        visual_group.setLayout(visual_form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        root = QVBoxLayout()
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)
        root.addWidget(visual_group)
        root.addWidget(buttons)
        self.setLayout(root)
        self._refresh_texts()

    def _refresh_texts(self):
        self.setWindowTitle(self._parent._t("settings_title"))

    def values(self):
        return {
            "language": self.language_combo.currentData(),
            "theme": self.theme_combo.currentData(),
            "ui_scale": int(self.scale_spin.value()),
            "popup_width_percent": int(self.popup_width_spin.value()),
            "compact_toolbar": bool(self.compact_check.isChecked()),
            "show_tile_centers": bool(self.tile_centers_check.isChecked()),
        }


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui_settings = {
            "language": "en",
            "theme": "midnight",
            "ui_scale": 100,
            "popup_width_percent": 34,
            "compact_toolbar": False,
            "show_tile_centers": True,
        }
        self.status_key = "status_ready"
        self.status_kwargs = {}

        self.setWindowTitle(self._t("window_title"))
        self.resize(1500, 920)

        self.view = gl.GLViewWidget()
        self.view.opts["center"] = QVector3D(0, 0, 0)
        self.view.opts["distance"] = 80
        self.view.opts["azimuth"] = 45
        self.view.opts["elevation"] = 20
        self.view.mousePressEvent = self._on_view_mouse_press

        self.scatter = None
        self.tile_outline_items = {}
        self.tile_fill_items = {}
        self.surface_items = {}
        self.points_centered = None
        self.world_center = None
        self.z_norm_full = None
        self.class_full = None
        self.tile_info = {}
        self.global_zrange = (0.0, 1.0)

        self.tiles = {}
        self.class_checkboxes = {}
        self.active_classes = set()
        self.multi_mode = False
        self.thread = None
        self.pending_replace_all = False
        self.active_crs = None
        self.last_color_note = ""
        self.sat_thread = None
        self.sat_request_counter = 0
        self.sat_active_request = None
        self.sat_pending_request = None
        self.sat_payload = {}
        self.sat_color_cache = {}
        self.sat_note_cache = {}
        self.scene_thread = None
        self.scene_request_counter = 0
        self.scene_active_request = None
        self.surface_thread = None
        self.surface_request_counter = 0
        self.surface_active_request = None
        self.surface_pending_rebuild = False
        self.selected_tile_path = None
        self.last_load_warnings = []
        self.last_error_message = ""

        self.sat_colorizer = SatelliteColorizer()
        self._base_font_size = QApplication.font().pointSizeF()
        if self._base_font_size <= 0:
            self._base_font_size = 10.0

        self.btn_load = QPushButton()
        self.btn_load.setMinimumHeight(28)
        self.btn_load.clicked.connect(self.load_from_main_button)
        self.btn_load.setToolTip("Load one file in single mode, or add files in multi mode")

        self.btn_multi = QPushButton(self._t("multi_off"))
        self.btn_multi.setCheckable(True)
        self.btn_multi.setMinimumHeight(28)
        self.btn_multi.toggled.connect(self.toggle_multi_mode)
        self.btn_multi.setToolTip("OFF: replace scene. ON: add tiles.")

        self.btn_clear = QPushButton(self._t("reset_data"))
        self.btn_clear.setMinimumHeight(28)
        self.btn_clear.clicked.connect(self.clear_loaded_data)
        self.btn_clear.setToolTip("Clear all loaded tiles")

        self.btn_reset = QPushButton(self._t("refresh_data"))
        self.btn_reset.setMinimumHeight(28)
        self.btn_reset.clicked.connect(self.refresh_data)
        self.btn_reset.setToolTip("Rebuild display from loaded tiles")

        self.tile_select_combo = QComboBox()
        self.tile_select_combo.setMinimumHeight(28)
        self.tile_select_combo.setMinimumWidth(220)
        self.tile_select_combo.setToolTip(self._t("select_tile"))
        self.tile_select_combo.currentIndexChanged.connect(self.on_tile_combo_changed)

        self.btn_remove_tile = QPushButton(self._t("remove_tile"))
        self.btn_remove_tile.setMinimumHeight(28)
        self.btn_remove_tile.clicked.connect(self.remove_selected_tile)
        self.btn_remove_tile.setToolTip("Remove selected tile (multi mode)")

        self.btn_filters = QPushButton(self._t("filters"))
        self.btn_filters.setCheckable(True)
        self.btn_filters.setMinimumHeight(28)
        self.btn_filters.toggled.connect(self.toggle_filter_popup)
        self.btn_filters.setToolTip("Show/hide class filters popup")

        self.btn_copy_report = QPushButton(self._t("copy_report"))
        self.btn_copy_report.setMinimumHeight(28)
        self.btn_copy_report.clicked.connect(self.copy_diagnostics_report)
        self.btn_copy_report.setToolTip(self._t("copy_report_tip"))

        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItems(
            [self._t("color_elevation"), self._t("color_satellite"), self._t("color_type")]
        )
        self.color_mode_combo.setMinimumHeight(28)
        self.color_mode_combo.currentTextChanged.connect(self.on_color_mode_changed)
        self.color_mode_combo.setToolTip("Choose point/surface coloring mode")

        self.sat_zoom_combo = QComboBox()
        self.sat_zoom_combo.addItems(["16", "17", "18", "19"])
        self.sat_zoom_combo.setCurrentText("19")
        self.sat_zoom_combo.setMinimumHeight(28)
        self.sat_zoom_combo.currentTextChanged.connect(self.on_sat_zoom_changed)
        self.sat_zoom_combo.setToolTip("16 = wider area, 19 = finer detail (more tiles to load)")

        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems([self._t("view_points"), self._t("view_surface")])
        self.view_mode_combo.setMinimumHeight(28)
        self.view_mode_combo.currentTextChanged.connect(self.on_view_mode_changed)
        self.view_mode_combo.setToolTip("Points: point cloud. Opaque surface: continuous terrain")
        self.view_mode_combo.setToolTip(
            "Points: point cloud. Opaque surface: continuous terrain. Shift+Left click selects tile in 3D."
        )

        self.surface_precision_mode = QComboBox()
        self.surface_precision_mode.addItems([self._t("surface_auto"), self._t("surface_manual")])
        self.surface_precision_mode.setMinimumHeight(28)
        self.surface_precision_mode.currentTextChanged.connect(self.on_surface_precision_mode_changed)

        self.surface_precision_spin = QSpinBox()
        self.surface_precision_spin.setRange(120, 1400)
        self.surface_precision_spin.setSingleStep(20)
        self.surface_precision_spin.setValue(SURFACE_GRID_MIN)
        self.surface_precision_spin.setMinimumHeight(28)
        self.surface_precision_spin.setEnabled(False)
        self.surface_precision_spin.valueChanged.connect(self.on_surface_precision_changed)
        self.surface_precision_spin.setToolTip("Surface grid resolution (higher = more detail)")

        self.status = QLabel(self._t("status_ready"))
        self.status.setMinimumHeight(28)
        self._status_full_text = self._t("status_ready")
        self.status.setWordWrap(False)

        self.quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.quality_slider.setMinimum(1)
        self.quality_slider.setMaximum(5000)
        self.quality_slider.setValue(300)
        self.quality_slider.setMinimumWidth(180)
        self.quality_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.quality_slider.setSingleStep(1)
        self.quality_slider.setPageStep(50)
        self.quality_slider.valueChanged.connect(self._schedule_lod_update)
        self.quality_slider.setToolTip("Quality LOD: quantity of displayed points")

        self.status.setMinimumWidth(0)
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

        self.actions_layout = QHBoxLayout()
        self.actions_layout.setSpacing(10)
        self.actions_layout.addWidget(self.btn_load)
        self.actions_layout.addWidget(self.btn_multi)
        self.actions_layout.addWidget(self.btn_clear)
        self.actions_layout.addWidget(self.btn_reset)
        self.actions_layout.addWidget(self.tile_select_combo)
        self.actions_layout.addWidget(self.btn_remove_tile)
        self.actions_layout.addWidget(self.btn_filters)
        self.actions_layout.addStretch()
        self.actions_layout.addWidget(self.btn_copy_report)
        self.btn_settings = QPushButton(self._t("settings"))
        self.btn_settings.setMinimumHeight(28)
        self.btn_settings.clicked.connect(self.open_settings_dialog)
        self.actions_layout.addWidget(self.btn_settings)

        self.lbl_view = QLabel(self._t("view"))
        self.lbl_surface = QLabel(self._t("surface"))
        self.lbl_color = QLabel(self._t("color"))
        self.lbl_sat_zoom = QLabel(self._t("sat_zoom"))
        self.lbl_quality = QLabel(self._t("quality"))
        self.viz_layout = QHBoxLayout()
        self.viz_layout.setSpacing(10)
        self.viz_layout.addWidget(self.lbl_view)
        self.viz_layout.addWidget(self.view_mode_combo)
        self.viz_layout.addWidget(self.lbl_surface)
        self.viz_layout.addWidget(self.surface_precision_mode)
        self.viz_layout.addWidget(self.surface_precision_spin)
        self.viz_layout.addWidget(self.lbl_color)
        self.viz_layout.addWidget(self.color_mode_combo)
        self.viz_layout.addWidget(self.lbl_sat_zoom)
        self.viz_layout.addWidget(self.sat_zoom_combo)
        self.viz_layout.addWidget(self.lbl_quality)
        self.viz_layout.addWidget(self.quality_slider, 1)
        self.viz_layout.addWidget(self.status, 2)

        self.top_layout = QGridLayout()
        self.top_layout.setContentsMargins(10, 6, 10, 6)
        self.top_layout.setHorizontalSpacing(10)
        self.top_layout.setVerticalSpacing(6)
        self.top_layout.addLayout(self.actions_layout, 0, 0)
        self.top_layout.addLayout(self.viz_layout, 1, 0)

        layout = QVBoxLayout()
        layout.addLayout(self.top_layout)
        layout.addWidget(self.view)

        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)

        self._apply_app_style()
        self._refresh_load_button_text()
        self._update_sat_controls_state()
        self._update_surface_precision_controls()
        self._refresh_tile_selector()
        self._build_busy_popup()
        self._build_filter_popup()

        self.lod_timer = QTimer(self)
        self.lod_timer.setSingleShot(True)
        self.lod_timer.timeout.connect(self._do_lod_update)
        self._lod_force_recolor = False

    def _t(self, key, **kwargs):
        lang = self.ui_settings.get("language", "en")
        table = I18N.get(lang, I18N["en"])
        text = table.get(key, I18N["en"].get(key, key))
        if kwargs:
            try:
                return text.format(**kwargs)
            except Exception:
                return text
        return text

    def _theme(self):
        return THEMES.get(self.ui_settings.get("theme", "midnight"), THEMES["midnight"])

    @staticmethod
    def _hex_to_rgb01(value):
        s = str(value).strip().lstrip("#")
        if len(s) != 6:
            return (1.0, 1.0, 1.0)
        try:
            r = int(s[0:2], 16) / 255.0
            g = int(s[2:4], 16) / 255.0
            b = int(s[4:6], 16) / 255.0
            return (r, g, b)
        except Exception:
            return (1.0, 1.0, 1.0)

    def _theme_display_name(self, theme_id):
        key = f"theme_{theme_id}"
        label = self._t(key)
        if label == key:
            return theme_id.replace("_", " ").title()
        return label

    def _apply_ui_scale(self):
        app = QApplication.instance()
        if app is None:
            return
        scale = max(85, min(150, int(self.ui_settings.get("ui_scale", 100)))) / 100.0
        font = app.font()
        font.setPointSizeF(self._base_font_size * scale)
        app.setFont(font)

        compact = bool(self.ui_settings.get("compact_toolbar", False))
        h = int(round((24 if compact else 30) * scale))
        for widget in (
            self.btn_load,
            self.btn_multi,
            self.btn_clear,
            self.btn_reset,
            self.tile_select_combo,
            self.btn_remove_tile,
            self.btn_filters,
            self.btn_copy_report,
            self.btn_settings,
            self.color_mode_combo,
            self.sat_zoom_combo,
            self.view_mode_combo,
            self.surface_precision_mode,
            self.surface_precision_spin,
        ):
            widget.setFixedHeight(h)
        self.status.setFixedHeight(h)

        if hasattr(self, "btn_filter_all"):
            self.btn_filter_all.setFixedHeight(max(20, int(round((22 if compact else 24) * scale))))
        if hasattr(self, "busy_bar"):
            self.busy_bar.setFixedHeight(max(8, int(round(10 * scale))))
        self.quality_slider.setFixedHeight(max(20, int(round(24 * scale))))

        # Compact mode also tightens spacing/margins, not only control height.
        spacing = 6 if compact else 10
        self.actions_layout.setSpacing(spacing)
        self.viz_layout.setSpacing(spacing)
        self.top_layout.setHorizontalSpacing(spacing)
        self.top_layout.setVerticalSpacing(4 if compact else 6)
        m = 6 if compact else 10
        self.top_layout.setContentsMargins(m, 4 if compact else 6, m, 4 if compact else 6)

    def _apply_app_style(self):
        t = self._theme()
        self.setStyleSheet(
            f"""
            QMainWindow, QWidget {{
                background: {t['bg']};
                color: {t['fg']};
            }}
            QPushButton, QComboBox, QSpinBox {{
                background: {t['panel']};
                color: {t['fg']};
                border: 1px solid {t['border']};
                border-radius: 8px;
                padding: 5px 10px;
                font-weight: 600;
            }}
            QPushButton:hover, QComboBox:hover, QSpinBox:hover {{
                background: {t['panel_hover']};
            }}
            QPushButton:checked {{
                background: {t['accent_soft']};
                border: 1px solid {t['accent']};
            }}
            QPushButton:disabled, QComboBox:disabled, QSpinBox:disabled {{
                color: {t['muted']};
                background: {t['panel_disabled']};
            }}
            QComboBox QAbstractItemView {{
                background: {t['panel_disabled']};
                color: {t['fg']};
                selection-background-color: {t['accent_soft']};
            }}
            QLabel {{
                color: {t['muted']};
            }}
            QSlider::groove:horizontal {{
                height: 7px;
                background: {t['panel_disabled']};
                border-radius: 4px;
            }}
            QSlider::sub-page:horizontal {{
                background: {t['slider_sub']};
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background: {t['slider_handle']};
                border: 1px solid {t['slider_sub']};
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }}
            QProgressBar {{
                border: 1px solid {t['border']};
                border-radius: 5px;
                background: {t['panel_disabled']};
            }}
            QProgressBar::chunk {{
                background: {t['accent']};
                border-radius: 5px;
            }}
            QFrame#BusyPopup, QFrame#FilterPopup {{
                background: {t['popup_bg']};
                border: 1px solid {t['border']};
                border-radius: 12px;
            }}
            QToolTip {{
                background: {t['panel_disabled']};
                color: {t['fg']};
                border: 1px solid {t['border']};
            }}
            """
        )
        self._apply_ui_scale()

    def _refresh_load_button_text(self):
        if self.multi_mode:
            self.btn_load.setText(self._t("add_laz"))
        else:
            self.btn_load.setText(self._t("load_laz"))

    def _is_surface_view(self):
        return self.view_mode_combo.currentIndex() == 1

    def _is_satellite_mode(self):
        return self.color_mode_combo.currentIndex() == 1

    def _is_type_mode(self):
        return self.color_mode_combo.currentIndex() == 2

    def _is_manual_surface(self):
        return self.surface_precision_mode.currentIndex() == 1

    def _update_sat_controls_state(self):
        self.sat_zoom_combo.setEnabled(self._is_satellite_mode())

    def _build_busy_popup(self):
        self.busy_popup = QFrame(self)
        self.busy_popup.setObjectName("BusyPopup")
        self.busy_popup.setVisible(False)

        lay = QVBoxLayout()
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)

        self.busy_title = QLabel(self._t("processing"))
        self.busy_title.setWordWrap(True)
        self.busy_title.setStyleSheet("font-weight: 700;")
        self.busy_bar = QProgressBar()
        self.busy_bar.setRange(0, 0)
        self.busy_bar.setFixedHeight(10)
        self.busy_bar.setTextVisible(False)

        lay.addWidget(self.busy_title)
        lay.addWidget(self.busy_bar)
        self.busy_popup.setLayout(lay)
        shadow = QGraphicsDropShadowEffect(self.busy_popup)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 10)
        self.busy_popup.setGraphicsEffect(shadow)
        self._place_busy_popup()

    def _build_filter_popup(self):
        self.filter_popup = QFrame(self)
        self.filter_popup.setObjectName("FilterPopup")
        self.filter_popup.setVisible(False)

        lay = QVBoxLayout()
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)
        self.filter_title = QLabel(self._t("point_class_filters"))
        self.filter_title.setStyleSheet("font-weight: 700;")
        lay.addWidget(self.filter_title)

        self.btn_filter_all = QPushButton(self._t("select_all"))
        self.btn_filter_all.setFixedHeight(24)
        self.btn_filter_all.clicked.connect(self.select_all_classes)
        lay.addWidget(self.btn_filter_all)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        self.filter_items_layout = QVBoxLayout()
        self.filter_items_layout.setSpacing(4)
        self.filter_items_layout.addStretch()
        content.setLayout(self.filter_items_layout)
        scroll.setWidget(content)
        lay.addWidget(scroll, 1)
        lay.addStretch()
        self.filter_popup.setLayout(lay)
        shadow = QGraphicsDropShadowEffect(self.filter_popup)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 10)
        self.filter_popup.setGraphicsEffect(shadow)
        self._place_filter_popup()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._place_busy_popup()
        self._place_filter_popup()
        self._apply_status_text()

    def _place_busy_popup(self):
        if not hasattr(self, "busy_popup"):
            return
        ratio = max(15, min(60, int(self.ui_settings.get("popup_width_percent", 34)))) / 100.0
        width = min(560, max(260, int(self.width() * ratio)))
        height = 64
        x = 20
        y = self.height() - height - 24
        self.busy_popup.setGeometry(x, y, width, height)

    def _place_filter_popup(self):
        if not hasattr(self, "filter_popup"):
            return
        ratio = max(15, min(60, int(self.ui_settings.get("popup_width_percent", 34)))) / 100.0
        width = min(560, max(220, int(self.width() * ratio)))
        height = min(420, max(180, int(self.height() * 0.45)))
        x = self.width() - width - 20
        y = self.height() - height - 24
        self.filter_popup.setGeometry(x, y, width, height)

    def toggle_filter_popup(self, visible):
        if hasattr(self, "filter_popup"):
            self.filter_popup.setVisible(visible)

    def _set_status(self, text, remember=False):
        if not remember:
            self.status_key = None
            self.status_kwargs = {}
        self._status_full_text = str(text)
        self.status.setToolTip(self._status_full_text)
        self._apply_status_text()

    def _set_status_key(self, key, **kwargs):
        self.status_key = key
        self.status_kwargs = kwargs
        self._set_status(self._t(key, **kwargs), remember=True)

    def _remember_error(self, message):
        self.last_error_message = str(message or "").strip()

    def _apply_status_text(self):
        if not hasattr(self, "status"):
            return
        full_text = getattr(self, "_status_full_text", "")
        width = max(40, self.status.width() - 8)
        fm: QFontMetrics = self.status.fontMetrics()
        self.status.setText(fm.elidedText(full_text, Qt.TextElideMode.ElideRight, width))

    def _refresh_ui_texts(self):
        self.setWindowTitle(self._t("window_title"))
        self.btn_multi.setText(self._t("multi_on") if self.multi_mode else self._t("multi_off"))
        self.btn_clear.setText(self._t("reset_data"))
        self.btn_reset.setText(self._t("refresh_data"))
        self.btn_remove_tile.setText(self._t("remove_tile"))
        self.btn_filters.setText(self._t("filters"))
        self.btn_copy_report.setText(self._t("copy_report"))
        self.btn_copy_report.setToolTip(self._t("copy_report_tip"))
        self.btn_settings.setText(self._t("settings"))
        self.lbl_view.setText(self._t("view"))
        self.lbl_surface.setText(self._t("surface"))
        self.lbl_color.setText(self._t("color"))
        self.lbl_sat_zoom.setText(self._t("sat_zoom"))
        self.lbl_quality.setText(self._t("quality"))
        self.filter_title.setText(self._t("point_class_filters"))
        self.btn_filter_all.setText(self._t("select_all"))

        current_view_idx = self.view_mode_combo.currentIndex()
        current_color_idx = self.color_mode_combo.currentIndex()
        current_surface_idx = self.surface_precision_mode.currentIndex()

        self.view_mode_combo.blockSignals(True)
        self.view_mode_combo.clear()
        self.view_mode_combo.addItems([self._t("view_points"), self._t("view_surface")])
        self.view_mode_combo.setCurrentIndex(1 if current_view_idx == 1 else 0)
        self.view_mode_combo.blockSignals(False)

        self.color_mode_combo.blockSignals(True)
        self.color_mode_combo.clear()
        self.color_mode_combo.addItems(
            [self._t("color_elevation"), self._t("color_satellite"), self._t("color_type")]
        )
        self.color_mode_combo.setCurrentIndex(max(0, min(2, current_color_idx)))
        self.color_mode_combo.blockSignals(False)

        self.surface_precision_mode.blockSignals(True)
        self.surface_precision_mode.clear()
        self.surface_precision_mode.addItems([self._t("surface_auto"), self._t("surface_manual")])
        self.surface_precision_mode.setCurrentIndex(1 if current_surface_idx == 1 else 0)
        self.surface_precision_mode.blockSignals(False)

        self._refresh_load_button_text()
        if self.status_key:
            self._set_status(self._t(self.status_key, **self.status_kwargs))
        else:
            self._apply_status_text()

    def open_settings_dialog(self):
        dlg = SettingsDialog(self, self.ui_settings.copy())
        if dlg.exec():
            self.ui_settings.update(dlg.values())
            self._apply_app_style()
            self._refresh_ui_texts()
            self._place_busy_popup()
            self._place_filter_popup()
            self._update_tile_centers_overlay()

    def copy_diagnostics_report(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            f"LiDAR View diagnostics - {now}",
            f"Status: {getattr(self, '_status_full_text', '')}",
            f"Last error: {self.last_error_message or '-'}",
            f"Warnings count: {len(self.last_load_warnings)}",
        ]
        if self.last_load_warnings:
            lines.append("Warnings:")
            for w in self.last_load_warnings:
                lines.append(f"- {w}")

        lines.append(f"Loaded tiles: {len(self.tiles)}")
        if self.tiles:
            lines.append("Tiles:")
            for path in sorted(self.tiles.keys()):
                t = self.tiles[path]
                n = len(t["points"]) if t.get("points") is not None else 0
                crs = t.get("crs") or "Unknown CRS"
                lines.append(f"- {Path(path).name} | points={n:,} | crs={crs}")

        lines.append(f"View mode: {self.view_mode_combo.currentText()}")
        lines.append(f"Color mode: {self.color_mode_combo.currentText()}")
        lines.append(f"Multi mode: {'ON' if self.multi_mode else 'OFF'}")

        QApplication.clipboard().setText("\n".join(lines))
        self._set_status_key("report_copied")

    def toggle_multi_mode(self, enabled):
        self.multi_mode = enabled
        self.btn_multi.setText(self._t("multi_on") if enabled else self._t("multi_off"))
        self._refresh_load_button_text()
        self._refresh_tile_selector()
        if enabled:
            self._set_status_key("multi_active")
        else:
            self._set_status_key("single_active")

    def _refresh_tile_selector(self):
        current = self.selected_tile_path or self.tile_select_combo.currentData()
        self.tile_select_combo.blockSignals(True)
        self.tile_select_combo.clear()
        for path in sorted(self.tiles.keys()):
            self.tile_select_combo.addItem(Path(path).name, path)
        self.tile_select_combo.blockSignals(False)

        if current is not None:
            idx = self.tile_select_combo.findData(current)
            if idx >= 0:
                self.tile_select_combo.setCurrentIndex(idx)
                self.selected_tile_path = current
            elif self.tile_select_combo.count() > 0:
                self.selected_tile_path = self.tile_select_combo.itemData(0)
            else:
                self.selected_tile_path = None
        elif self.tile_select_combo.count() > 0:
            self.selected_tile_path = self.tile_select_combo.itemData(0)
        else:
            self.selected_tile_path = None

        has_tiles = len(self.tiles) > 0
        self.tile_select_combo.setEnabled(self.multi_mode and has_tiles)
        self.btn_remove_tile.setEnabled(self.multi_mode and has_tiles)
        self._update_tile_centers_overlay()

    def on_tile_combo_changed(self, index):
        if index < 0:
            self.selected_tile_path = None
        else:
            self.selected_tile_path = self.tile_select_combo.itemData(index)
        self._update_tile_centers_overlay()

    def _on_view_mouse_press(self, event):
        if (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.ShiftModifier
            and len(self.tiles) > 0
        ):
            picked = self._pick_tile_from_screen(event.position().x(), event.position().y())
            if picked is not None:
                self.selected_tile_path = picked
                idx = self.tile_select_combo.findData(picked)
                if idx >= 0:
                    self.tile_select_combo.setCurrentIndex(idx)
                self._update_tile_centers_overlay()
                self._set_status(self._t("selected_tile", name=Path(picked).name))
                return
        gl.GLViewWidget.mousePressEvent(self.view, event)

    def _pick_tile_from_screen(self, sx, sy):
        if not self.tile_info or self.world_center is None:
            return None

        az = math.radians(float(self.view.opts.get("azimuth", 45.0)))
        el = math.radians(float(self.view.opts.get("elevation", 20.0)))
        dist = float(self.view.opts.get("distance", 80.0))
        c = self.view.opts.get("center", QVector3D(0, 0, 0))
        center = np.array([float(c.x()), float(c.y()), float(c.z())], dtype=np.float64)

        cam = np.array(
            [
                center[0] + dist * math.cos(el) * math.cos(az),
                center[1] + dist * math.cos(el) * math.sin(az),
                center[2] + dist * math.sin(el),
            ],
            dtype=np.float64,
        )
        forward = center - cam
        fn = np.linalg.norm(forward)
        if fn < 1e-12:
            return None
        forward /= fn
        up_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, up_world)
        rn = np.linalg.norm(right)
        if rn < 1e-12:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            right /= rn
        up = np.cross(right, forward)

        w = max(1, self.view.width())
        h = max(1, self.view.height())
        aspect = w / h
        fov = math.radians(float(self.view.opts.get("fov", 60.0)))
        tan_half = math.tan(fov / 2.0)

        best = None
        best_d2 = float("inf")
        for path, info in self.tile_info.items():
            p = (info["center_world"] - self.world_center).astype(np.float64)
            rel = p - cam
            vx = float(np.dot(rel, right))
            vy = float(np.dot(rel, up))
            vz = float(np.dot(rel, forward))
            if vz <= 1e-6:
                continue
            ndc_x = (vx / (vz * tan_half * aspect))
            ndc_y = (vy / (vz * tan_half))
            px = (ndc_x + 1.0) * 0.5 * w
            py = (1.0 - ndc_y) * 0.5 * h
            d2 = (px - sx) ** 2 + (py - sy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = path
        return best

    def _update_tile_centers_overlay(self):
        if (
            not self.ui_settings.get("show_tile_centers", True)
            or self.points_centered is None
            or self.world_center is None
            or not self.tile_info
        ):
            for item in self.tile_outline_items.values():
                self.view.removeItem(item)
            self.tile_outline_items.clear()
            for item in self.tile_fill_items.values():
                self.view.removeItem(item)
            self.tile_fill_items.clear()
            return

        active_paths = set()
        accent_rgb = self._hex_to_rgb01(self._theme().get("accent", "#14b8a6"))
        muted_rgb = self._hex_to_rgb01(self._theme().get("muted", "#cbd5e1"))
        for path, info in sorted(self.tile_info.items()):
            active_paths.add(path)
            mn = info.get("min_world")
            mx = info.get("max_world")
            if mn is None or mx is None:
                continue
            x0 = float(mn[0] - self.world_center[0])
            x1 = float(mx[0] - self.world_center[0])
            y0 = float(mn[1] - self.world_center[1])
            y1 = float(mx[1] - self.world_center[1])
            span = max(abs(x1 - x0), abs(y1 - y0), 1.0)
            lift = max(0.20, 0.006 * span)
            zc = float(mx[2] - self.world_center[2] + lift)
            line_segments = np.array(
                [
                    [x0, y0, zc], [x1, y0, zc],
                    [x1, y0, zc], [x1, y1, zc],
                    [x1, y1, zc], [x0, y1, zc],
                    [x0, y1, zc], [x0, y0, zc],
                ],
                dtype=np.float32,
            )
            fill_vertices = np.array(
                [
                    [x0, y0, zc - 0.01],
                    [x1, y0, zc - 0.01],
                    [x1, y1, zc - 0.01],
                    [x0, y1, zc - 0.01],
                ],
                dtype=np.float32,
            )
            fill_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
            if path == self.selected_tile_path:
                color = np.array([accent_rgb[0], accent_rgb[1], accent_rgb[2], 1.0], dtype=np.float32)
                fill_alpha = 0.16
                width = 3.2
            else:
                color = np.array([muted_rgb[0], muted_rgb[1], muted_rgb[2], 0.42], dtype=np.float32)
                fill_alpha = 0.05
                width = 1.0

            prev_line = self.tile_outline_items.get(path)
            if prev_line is not None:
                self.view.removeItem(prev_line)
            prev_fill = self.tile_fill_items.get(path)
            if prev_fill is not None:
                self.view.removeItem(prev_fill)

            face_colors = np.array(
                [
                    [color[0], color[1], color[2], fill_alpha],
                    [color[0], color[1], color[2], fill_alpha],
                ],
                dtype=np.float32,
            )
            fill_item = gl.GLMeshItem(
                vertexes=fill_vertices,
                faces=fill_faces,
                faceColors=face_colors,
                smooth=False,
                drawEdges=False,
                drawFaces=True,
            )
            fill_item.setGLOptions("translucent")
            self.view.addItem(fill_item)
            self.tile_fill_items[path] = fill_item

            line_item = gl.GLLinePlotItem(
                pos=line_segments,
                color=color,
                width=width,
                antialias=True,
                mode="lines",
            )
            line_item.setGLOptions("translucent")
            self.view.addItem(line_item)
            self.tile_outline_items[path] = line_item

        for stale_path in list(self.tile_outline_items.keys()):
            if stale_path not in active_paths:
                self.view.removeItem(self.tile_outline_items[stale_path])
                del self.tile_outline_items[stale_path]
        for stale_path in list(self.tile_fill_items.keys()):
            if stale_path not in active_paths:
                self.view.removeItem(self.tile_fill_items[stale_path])
                del self.tile_fill_items[stale_path]

    def remove_selected_tile(self):
        if not self.multi_mode:
            self._set_status_key("enable_multi_remove")
            return
        path = self.tile_select_combo.currentData()
        if path is None or path not in self.tiles:
            self._set_status_key("no_tile_selected")
            return
        del self.tiles[path]
        if self.selected_tile_path == path:
            self.selected_tile_path = None
        self._refresh_tile_selector()
        self._start_scene_build()

    def _refresh_class_filters(self):
        while self.filter_items_layout.count():
            item = self.filter_items_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.class_checkboxes.clear()

        if self.class_full is None or len(self.class_full) == 0:
            self.active_classes = set()
            return

        classes, counts = np.unique(self.class_full, return_counts=True)
        if not self.active_classes:
            self.active_classes = set(int(c) for c in classes)
        else:
            self.active_classes = {c for c in self.active_classes if c in set(int(x) for x in classes)}
            if not self.active_classes:
                self.active_classes = set(int(c) for c in classes)

        for cid, cnt in zip(classes, counts):
            cid = int(cid)
            label = LIDAR_CLASS_LABELS.get(cid, f"Class {cid}")
            cb = QCheckBox(f"{cid} - {label} ({int(cnt):,})")
            cb.setChecked(cid in self.active_classes)
            cb.toggled.connect(lambda checked, c=cid: self._on_class_toggled(c, checked))
            self.filter_items_layout.addWidget(cb)
            self.class_checkboxes[cid] = cb
        self.filter_items_layout.addStretch()

    def _on_class_toggled(self, class_id, checked):
        if checked:
            self.active_classes.add(int(class_id))
        else:
            self.active_classes.discard(int(class_id))
        self.update_lod(force_recolor=True)

    def select_all_classes(self):
        for cid, cb in self.class_checkboxes.items():
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.blockSignals(False)
            self.active_classes.add(int(cid))
        self.update_lod(force_recolor=True)

    def _active_mask(self):
        if self.class_full is None or len(self.class_full) == 0:
            return None
        if not self.active_classes:
            return np.zeros(len(self.class_full), dtype=bool)
        cls = self.class_full
        return np.isin(cls, np.fromiter(self.active_classes, dtype=np.uint8))

    def _build_type_colors(self, idx):
        if self.class_full is None:
            out = np.zeros((len(idx), 4), dtype=np.float32)
            out[:, :3] = 0.8
            out[:, 3] = 1.0
            return out
        cls = self.class_full[idx]
        out = np.zeros((len(idx), 4), dtype=np.float32)
        for cid in np.unique(cls):
            color = LIDAR_CLASS_COLORS.get(int(cid), (0.8, 0.8, 0.8, 1.0))
            out[cls == cid] = color
        return out

    def on_color_mode_changed(self, _mode):
        self._update_sat_controls_state()
        self.update_lod(force_recolor=True)

    def on_sat_zoom_changed(self, _zoom):
        self.update_lod(force_recolor=True)

    def on_view_mode_changed(self, _mode):
        self._update_sat_controls_state()
        if not self._is_surface_view():
            # Invalidate any in-flight surface build so late thread results are ignored.
            self.surface_active_request = None
            self.surface_pending_rebuild = False
            self._set_busy(False)
            self._clear_surface_items()
        self.update_lod(force_recolor=True)

    def on_surface_precision_mode_changed(self, _mode):
        self._update_surface_precision_controls()
        if self._is_surface_view():
            self.update_lod(force_recolor=True)

    def on_surface_precision_changed(self, _value):
        if (
            self._is_manual_surface()
            and self._is_surface_view()
        ):
            self.update_lod(force_recolor=True)

    def _update_surface_precision_controls(self):
        self.surface_precision_spin.setEnabled(self._is_manual_surface())

    def _schedule_lod_update(self):
        if self._is_surface_view():
            return
        self.lod_timer.start(70)

    def _do_lod_update(self):
        self.update_lod(force_recolor=self._lod_force_recolor)
        self._lod_force_recolor = False

    def load_from_main_button(self):
        if self.thread is not None and self.thread.isRunning():
            self._set_status_key("loading_in_progress")
            return

        if self.multi_mode:
            paths, _ = QFileDialog.getOpenFileNames(self, self._t("add_files"), "", "LAZ (*.laz)")
            if not paths:
                return
            self._start_loading(paths, replace_all=False)
            return

        path, _ = QFileDialog.getOpenFileName(self, self._t("open_file"), "", "LAZ (*.laz)")
        if not path:
            return
        self._start_loading([path], replace_all=True)

    def _start_loading(self, paths, replace_all):
        self.pending_replace_all = replace_all
        self.btn_load.setEnabled(False)
        self._set_status(self._t("loading_files", count=len(paths)))
        self._set_busy(True, self._t("reading_laz"))

        self.thread = LoaderThread(paths)
        self.thread.loaded.connect(self.on_tiles_loaded)
        self.thread.failed.connect(self.on_load_failed)
        self.thread.start()

    def on_tiles_loaded(self, loaded_tiles, warnings):
        self._remember_error("")
        if self.pending_replace_all:
            self.tiles.clear()

        for tile in loaded_tiles:
            self.tiles[tile["path"]] = {
                "points": tile["points"],
                "crs": tile["crs"],
                "classification": tile.get("classification"),
            }

        self.last_load_warnings = warnings
        self._refresh_tile_selector()
        self._start_scene_build()

        self.btn_load.setEnabled(True)
        self.pending_replace_all = False
        self.thread = None

    def on_load_failed(self, message):
        self._remember_error(message)
        self._set_status(message)
        self.btn_load.setEnabled(True)
        self.pending_replace_all = False
        self.thread = None
        self._set_busy(False)

    def _start_scene_build(self):
        self._refresh_tile_selector()
        self.scene_request_counter += 1
        request_id = self.scene_request_counter
        self.scene_active_request = request_id
        tiles_snapshot = []
        for path, tile in self.tiles.items():
            tiles_snapshot.append(
                {
                    "path": path,
                    "points": tile["points"],
                    "crs": tile.get("crs"),
                    "classification": tile.get("classification"),
                }
            )
        self._set_busy(True, self._t("preparing_scene"))

        self.scene_thread = SceneBuildThread(request_id, tiles_snapshot)
        self.scene_thread.built.connect(self._on_scene_built)
        self.scene_thread.failed.connect(self._on_scene_build_failed)
        self.scene_thread.start()

    def _on_scene_built(
        self,
        request_id,
        pts_centered,
        z_norm,
        active_crs,
        world_center,
        class_full,
        tile_info,
        global_zrange,
    ):
        if request_id != self.scene_active_request:
            return

        if pts_centered is None:
            self.points_centered = None
            self.world_center = None
            self.z_norm_full = None
            self.class_full = None
            self.tile_info = {}
            self.global_zrange = (0.0, 1.0)
            self.active_crs = None
            self.sat_color_cache.clear()
            self.sat_note_cache.clear()
            self.sat_pending_request = None
            self.sat_payload.clear()
            self.sat_active_request = None
            if self.scatter is not None:
                self.view.removeItem(self.scatter)
                self.scatter = None
            self._clear_surface_items()
            self._refresh_tile_selector()
            self._refresh_class_filters()
            self._set_status_key("no_tiles_loaded")
            self._set_busy(False)
            return

        self.points_centered = pts_centered
        self.world_center = world_center
        self.z_norm_full = z_norm
        self.class_full = class_full
        self.tile_info = tile_info or {}
        self.global_zrange = global_zrange or (float(np.min(pts_centered[:, 2])), float(np.max(pts_centered[:, 2])))
        self.active_crs = active_crs
        self.sat_color_cache.clear()
        self.sat_note_cache.clear()
        self.sat_pending_request = None
        self.sat_payload.clear()
        self.sat_active_request = None
        self._refresh_tile_selector()
        self._refresh_class_filters()
        self._update_tile_centers_overlay()
        if not self._is_manual_surface():
            rec = self._recommended_surface_grid(len(self.points_centered))
            self.surface_precision_spin.blockSignals(True)
            self.surface_precision_spin.setValue(rec)
            self.surface_precision_spin.blockSignals(False)

        if self.scatter is None:
            self.scatter = gl.GLScatterPlotItem()
            self.view.addItem(self.scatter)
        self._clear_surface_items()

        count = len(self.tiles)
        msg = f"{count} tile(s) loaded."
        crs_values = sorted({t.get("crs") for t in self.tiles.values() if t.get("crs")})
        if len(crs_values) > 1:
            msg += " Warning: different CRS detected."
        elif len(crs_values) == 1:
            msg += f" CRS: {crs_values[0]}."
        else:
            msg += " CRS not detected."
        if self.last_load_warnings:
            msg += f" ({len(self.last_load_warnings)} warning(s))"
        self._set_status(msg)

        self._set_busy(False)
        self.update_lod(force_recolor=True)

    def _on_scene_build_failed(self, request_id, error):
        if request_id != self.scene_active_request:
            return
        self._remember_error(error)
        self._set_busy(False)
        self._set_status(self._t("scene_build_error", error=error))

    def _build_altitude_colors(self, idx):
        z_norm = self.z_norm_full[idx]
        return np.c_[
            z_norm,
            0.3 * np.ones_like(z_norm, dtype=np.float32),
            1.0 - z_norm,
            np.ones_like(z_norm, dtype=np.float32),
        ].astype(np.float32)

    def clear_loaded_data(self):
        self.tiles.clear()
        self.points_centered = None
        self.world_center = None
        self.z_norm_full = None
        self.class_full = None
        self.tile_info = {}
        self.global_zrange = (0.0, 1.0)
        self.active_crs = None
        self.scene_active_request = None
        self.sat_color_cache.clear()
        self.sat_note_cache.clear()
        self.sat_pending_request = None
        self.sat_payload.clear()
        self.sat_active_request = None
        self.active_classes.clear()
        self.lod_timer.stop()
        self.surface_active_request = None
        self.surface_pending_rebuild = False
        self._set_busy(False)
        self._refresh_tile_selector()
        self._refresh_class_filters()
        if self.scatter is not None:
            self.view.removeItem(self.scatter)
            self.scatter = None
        self._clear_surface_items()
        self._set_status_key("data_reset")

    def update_lod(self, force_recolor=False):
        if self.points_centered is None or self.z_norm_full is None or self.scatter is None:
            return

        active_mask = self._active_mask()
        if active_mask is None:
            active_idx = np.arange(len(self.points_centered), dtype=np.int64)
        else:
            active_idx = np.flatnonzero(active_mask)

        total = len(active_idx)
        if total == 0:
            self.scatter.setData(
                pos=np.empty((0, 3), dtype=np.float32),
                color=np.empty((0, 4), dtype=np.float32),
            )
            self._clear_surface_items()
            self._set_status_key("no_active_points")
            return

        slider_min = self.quality_slider.minimum()
        slider_max = self.quality_slider.maximum()
        ratio = (self.quality_slider.value() - slider_min) / (slider_max - slider_min)

        gamma = 2.5
        density = ratio**gamma
        n_points = int(1 + density * (total - 1))

        local_idx = np.linspace(0, total - 1, n_points, dtype=int)
        idx = active_idx[local_idx]
        pts = self.points_centered[idx]

        if self._is_surface_view():
            self._request_surface_rebuild()
            if self.scatter is not None:
                self.scatter.setData(
                    pos=np.empty((0, 3), dtype=np.float32),
                    color=np.empty((0, 4), dtype=np.float32),
                )
            mode_label = (
                f"Satellite surface z{self.sat_zoom_combo.currentText()}"
                if self._is_satellite_mode()
                else ("Type surface" if self._is_type_mode() else "Elevation surface")
            )
            note = self.last_color_note or "building..."
            base = f"{len(self.tiles)} tile(s) | {mode_label} | {total:,} source points | {note}"
            self._set_status(base)
            return

        mode = self.color_mode_combo.currentIndex()
        mode_label = "Elevation"
        note = ""
        if mode == 1:
            try:
                zoom = int(self.sat_zoom_combo.currentText())
                colors = self._build_altitude_colors(idx)

                known, cached_colors = self._lookup_satellite_cache(zoom, idx)
                if np.any(known):
                    colors[known] = cached_colors[known]

                missing_idx = idx[~known]
                if len(missing_idx) > 0:
                    self._request_satellite_colors(missing_idx, zoom)
                    note = f"coloring in progress... {len(missing_idx):,} pts remaining"
                else:
                    note = self.sat_note_cache.get(zoom, "")

                mode_label = f"Satellite z{self.sat_zoom_combo.currentText()}"
            except Exception as exc:
                colors = self._build_altitude_colors(idx)
                mode_label = "Elevation (fallback)"
                note = f"Satellite unavailable: {exc}"
                self.last_color_note = note
        elif mode == 2:
            colors = self._build_type_colors(idx)
            mode_label = "Type"
        else:
            colors = self._build_altitude_colors(idx)
            self.last_color_note = ""

        self.scatter.setData(pos=pts, size=POINT_SIZE, color=colors)
        self._clear_surface_items()
        self._update_tile_centers_overlay()
        base = f"{len(self.tiles)} tile(s) | {mode_label} | {n_points:,} / {total:,} points displayed"
        if note:
            base += f" | {note}"
        self._set_status(base)

    def refresh_data(self):
        if self.thread is not None and self.thread.isRunning():
            self._set_status_key("loading_in_progress")
            return
        if len(self.tiles) == 0:
            self._set_status_key("no_tiles_loaded")
            return

        # Refresh rebuilds scene data/caches without dropping loaded tiles.
        self._set_status(self._t("refreshing_data"))
        self.sat_color_cache.clear()
        self.sat_note_cache.clear()
        self.sat_pending_request = None
        self.sat_payload.clear()
        self.sat_active_request = None
        self.surface_active_request = None
        self.surface_pending_rebuild = False
        self._clear_surface_items()
        if self.scatter is not None:
            self.scatter.setData(
                pos=np.empty((0, 3), dtype=np.float32),
                color=np.empty((0, 4), dtype=np.float32),
            )
        self._start_scene_build()

    def reset_camera(self):
        self.view.setCameraPosition(distance=80, azimuth=45, elevation=20)

    def _set_busy(self, busy, title=None):
        if not hasattr(self, "busy_popup"):
            return
        self.busy_title.setText(title or self._t("processing"))
        self.busy_popup.setVisible(busy)

    def _clear_surface_items(self):
        if not hasattr(self, "surface_items"):
            self.surface_items = {}
            return
        for item in self.surface_items.values():
            self.view.removeItem(item)
        self.surface_items.clear()

    def _request_surface_rebuild(self):
        if self.points_centered is None or self.world_center is None or len(self.tiles) == 0:
            return
        self.surface_pending_rebuild = True
        if self.surface_thread is not None and self.surface_thread.isRunning():
            return
        self._launch_surface_rebuild()

    def _launch_surface_rebuild(self):
        if not self.surface_pending_rebuild:
            return
        self.surface_pending_rebuild = False

        # Each build request has a monotonically increasing id. Late thread
        # results are ignored to prevent stale surfaces from overriding newer UI state.
        self.surface_request_counter += 1
        rid = self.surface_request_counter
        self.surface_active_request = rid

        zmin, zmax = self.global_zrange

        tiles_snapshot = {
            k: {"points": v["points"], "classification": v.get("classification")} for k, v in self.tiles.items()
        }
        class_lut = np.zeros((256, 4), dtype=np.float32)
        class_lut[:, :] = (0.8, 0.8, 0.8, 1.0)
        for cid, col in LIDAR_CLASS_COLORS.items():
            class_lut[int(cid)] = col
        self.surface_thread = SurfaceBuildThread(
            request_id=rid,
            tiles_snapshot=tiles_snapshot,
            world_center=self.world_center.copy(),
            global_zmin=zmin,
            global_zmax=zmax,
            color_mode=["Elevation", "Satellite", "Type"][self.color_mode_combo.currentIndex()],
            active_crs=self.active_crs,
            sat_zoom=int(self.sat_zoom_combo.currentText()),
            surface_mode=["Auto (recommended)", "Manual"][self.surface_precision_mode.currentIndex()],
            surface_value=int(self.surface_precision_spin.value()),
            sat_url_template=self.sat_colorizer.url_template,
            active_classes=list(self.active_classes),
            class_lut=class_lut,
        )
        self.surface_thread.built.connect(self._on_surface_built)
        self.surface_thread.failed.connect(self._on_surface_failed)
        self._set_busy(True, self._t("generating_surface"))
        self.surface_thread.start()

    def _on_surface_built(self, request_id, payload, note):
        if request_id != self.surface_active_request:
            return
        # Surface generation is asynchronous; ignore payload if user switched
        # back to point mode while the thread was still running.
        if not self._is_surface_view():
            return

        new_items = {}
        for tile_path, xs, ys, zf, colors_vertices in payload:
            try:
                item = gl.GLSurfacePlotItem(
                    x=xs, y=ys, z=zf, colors=colors_vertices, shader="shaded", smooth=False
                )
                new_items[tile_path] = item
            except Exception:
                continue

        if new_items:
            self._clear_surface_items()
            for item in new_items.values():
                self.view.addItem(item)
            self.surface_items = new_items
            self.last_color_note = note or ""
            self._update_tile_centers_overlay()
            if self._is_surface_view():
                mode_label = (
                    f"Satellite surface z{self.sat_zoom_combo.currentText()}"
                    if self._is_satellite_mode()
                    else ("Type surface" if self._is_type_mode() else "Elevation surface")
                )
                info = self.last_color_note or "ready"
                self._set_status(f"{len(self.tiles)} tile(s) | {mode_label} | {info}")
        else:
            self.last_color_note = "surface unavailable"

        self._set_busy(False)
        if self.surface_pending_rebuild and self._is_surface_view():
            self._launch_surface_rebuild()

    def _on_surface_failed(self, request_id, error):
        if request_id != self.surface_active_request:
            return
        self._remember_error(error)
        self._set_busy(False)
        self.last_color_note = f"surface error: {error}"
        if self._is_surface_view():
            self._set_status(self.last_color_note)
        if self.surface_pending_rebuild and self._is_surface_view():
            self._launch_surface_rebuild()

    def _ensure_sat_cache(self, zoom):
        cache = self.sat_color_cache.get(zoom)
        if cache is None:
            cache = {
                "indices": np.empty(0, dtype=np.int64),
                "colors": np.empty((0, 4), dtype=np.float32),
            }
            self.sat_color_cache[zoom] = cache
        return cache

    def _lookup_satellite_cache(self, zoom, idx):
        cache = self._ensure_sat_cache(zoom)
        known = np.zeros(len(idx), dtype=bool)
        colors = np.empty((len(idx), 4), dtype=np.float32)
        colors[:, 0:3] = np.nan
        colors[:, 3] = 1.0
        if len(idx) == 0 or len(cache["indices"]) == 0:
            return known, colors

        pos = np.searchsorted(cache["indices"], idx)
        valid = (pos >= 0) & (pos < len(cache["indices"]))
        valid_idx = np.flatnonzero(valid)
        if len(valid_idx) == 0:
            return known, colors
        match = cache["indices"][pos[valid_idx]] == idx[valid_idx]
        if np.any(match):
            good = valid_idx[match]
            known[good] = True
            colors[good] = cache["colors"][pos[good]]
        return known, colors

    def _store_satellite_cache(self, zoom, idx, colors):
        if idx is None or len(idx) == 0:
            return
        cache = self._ensure_sat_cache(zoom)
        idx = np.asarray(idx, dtype=np.int64)
        colors = np.asarray(colors, dtype=np.float32)
        if len(idx) != len(colors):
            return

        all_idx = np.concatenate([cache["indices"], idx])
        all_colors = np.concatenate([cache["colors"], colors], axis=0)
        order = np.argsort(all_idx, kind="mergesort")
        all_idx = all_idx[order]
        all_colors = all_colors[order]

        if len(all_idx) > 1:
            rev_idx = all_idx[::-1]
            _uniq, first_rev = np.unique(rev_idx, return_index=True)
            keep = (len(all_idx) - 1 - first_rev)
            keep.sort()
            all_idx = all_idx[keep]
            all_colors = all_colors[keep]

        cache["indices"] = all_idx
        cache["colors"] = all_colors

    def _request_satellite_colors(self, idx, zoom):
        if self.active_crs is None:
            raise RuntimeError("CRS not unique or missing: satellite coloring disabled.")

        if idx is None or len(idx) == 0:
            return

        unique_idx = np.unique(idx)
        known, _colors = self._lookup_satellite_cache(zoom, unique_idx)
        unresolved = unique_idx[~known]
        if len(unresolved) == 0:
            return

        self.sat_pending_request = (unresolved.copy(), zoom)

        if self.sat_thread is not None and self.sat_thread.isRunning():
            return
        self._launch_next_satellite_request()

    def _launch_next_satellite_request(self):
        if self.sat_pending_request is None:
            return

        idx, zoom = self.sat_pending_request
        self.sat_pending_request = None

        self.sat_request_counter += 1
        request_id = self.sat_request_counter
        self.sat_active_request = request_id
        self.sat_payload[request_id] = (idx, zoom)

        world_xy = (
            self.points_centered[idx, :2].astype(np.float64, copy=False)
            + self.world_center[:2].astype(np.float64, copy=False)
        )
        self.sat_thread = SatelliteColorThread(
            request_id=request_id,
            points_world_xy=world_xy,
            src_crs=self.active_crs,
            zoom=zoom,
            url_template=self.sat_colorizer.url_template,
        )
        self.sat_thread.result.connect(self._on_satellite_result)
        self.sat_thread.failed.connect(self._on_satellite_failed)
        self._set_busy(True, self._t("sat_coloring", zoom=zoom))
        self.sat_thread.start()

    def _on_satellite_result(self, request_id, colors, sat_msg):
        payload = self.sat_payload.pop(request_id, None)
        if payload:
            idx, zoom = payload
            self._store_satellite_cache(zoom, idx, colors)
            if sat_msg:
                self.sat_note_cache[zoom] = sat_msg

        if self.sat_pending_request is None:
            self._set_busy(False)
        if not self._is_surface_view():
            self._schedule_lod_update()
        self._launch_next_satellite_request()

    def _on_satellite_failed(self, request_id, error):
        payload = self.sat_payload.pop(request_id, None)
        self._remember_error(error)
        if payload:
            _idx, zoom = payload
            self.sat_note_cache[zoom] = f"tiles unavailable: {error}"

        if self.sat_pending_request is None:
            self._set_busy(False)
        if not self._is_surface_view():
            self._schedule_lod_update()
        self._launch_next_satellite_request()

    def _surface_grid_size(self, n_points):
        if self._is_manual_surface():
            return int(self.surface_precision_spin.value())
        size = self._recommended_surface_grid(n_points)
        self.surface_precision_spin.blockSignals(True)
        self.surface_precision_spin.setValue(size)
        self.surface_precision_spin.blockSignals(False)
        return size

    @staticmethod
    def _recommended_surface_grid(n_points):
        size = int(np.sqrt(max(n_points, 1)) * 0.75)
        return max(SURFACE_GRID_MIN, min(SURFACE_GRID_MAX, size))

    @staticmethod
    def _fill_nan_surface_static(z_grid):
        filled = z_grid.copy()
        mask = ~np.isfinite(filled)
        if not np.any(mask):
            return filled

        ny, nx = filled.shape
        for y in range(ny):
            row = filled[y]
            valid = np.isfinite(row)
            if np.any(valid):
                first = int(np.argmax(valid))
                last = int(nx - 1 - np.argmax(valid[::-1]))
                row[:first] = row[first]
                row[last + 1 :] = row[last]
                filled[y] = row
        for x in range(nx):
            col = filled[:, x]
            valid = np.isfinite(col)
            if np.any(valid):
                first = int(np.argmax(valid))
                last = int(ny - 1 - np.argmax(valid[::-1]))
                col[:first] = col[first]
                col[last + 1 :] = col[last]
                filled[:, x] = col

        mask = ~np.isfinite(filled)
        for _ in range(nx + ny):
            if not np.any(mask):
                break
            acc = np.zeros_like(filled, dtype=np.float64)
            cnt = np.zeros_like(filled, dtype=np.float64)

            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                y0 = max(0, -dy)
                y1 = min(ny, ny - dy)
                x0 = max(0, -dx)
                x1 = min(nx, nx - dx)

                src = filled[y0:y1, x0:x1]
                dst_y0 = y0 + dy
                dst_y1 = y1 + dy
                dst_x0 = x0 + dx
                dst_x1 = x1 + dx
                valid = np.isfinite(src)
                acc[dst_y0:dst_y1, dst_x0:dst_x1] += np.where(valid, src, 0.0)
                cnt[dst_y0:dst_y1, dst_x0:dst_x1] += valid

            to_fill = mask & (cnt > 0)
            filled[to_fill] = acc[to_fill] / cnt[to_fill]
            mask = ~np.isfinite(filled)

        if np.any(mask):
            fallback = float(np.nanmean(filled))
            if not np.isfinite(fallback):
                fallback = 0.0
            filled[mask] = fallback
        return filled

    def _fill_nan_surface(self, z_grid):
        return MainWindow._fill_nan_surface_static(z_grid)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
