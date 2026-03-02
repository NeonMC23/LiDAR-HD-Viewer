# LiDAR View

LiDAR View is a desktop viewer for `.laz` / `.las` point clouds (LiDAR tiles), with multi-tile loading, LOD rendering, satellite color projection, and opaque surface rendering.

## Features

- Load single or multiple `.laz` tiles
- Single mode (replace scene) and multi mode (append tiles)
- Precise coordinate handling (scale/offset-based decoding)
- Camera controls for 3D exploration
- LOD quality slider for performance vs detail
- Color modes:
  - `Elevation`
  - `Satellite` (tile-based web imagery projection)
  - `Type` (LAS classification colors)
- View modes:
  - `Points`
  - `Opaque surface` (terrain-like rendering)
- Surface precision:
  - `Auto (recommended)`
  - `Manual` (custom grid resolution)
- Class filter popup (bottom-right), with live filtering by LAS classification
- Settings menu (themes, language FR/EN, UI scale, popup width, compact toolbar, tile overlays)
- Non-blocking processing with background threads and bottom progress popup

## Project Structure

- `main.py`: Main application UI, rendering logic, async processing, satellite coloring, surface generation
- `lidar_loader.py`: LAZ loading helpers, precision decoding, CRS inference
- `gl_viewer.py`: Legacy OpenGL fallback viewer widget
- `test_laz.py`: Small local test script (if used)
- `requirements.txt`: Python dependencies
- `install_and_verify.ps1`: Automated install + environment verification (Windows PowerShell)

## Requirements

- Python 3.10+ (3.11/3.12 recommended)
- Windows/macOS/Linux with OpenGL support
- Internet access for Satellite mode tile fetch

## Quick Start (Windows PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -File .\install_and_verify.ps1
.\.venv\Scripts\Activate.ps1
python .\main.py
```

## Manual Installation

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Run.

```powershell
python .\main.py
```

## Controls

### Top Bar

- `Load .laz (replace)`: load one file and replace current scene
- `Multi mode: OFF/ON`: switch between replace/append behavior
- `Reset data`: clear loaded data
- `Refresh data`: rebuild rendering/cache state from currently loaded tiles
- `View`: `Points` or `Opaque surface`
- `Surface`: `Auto (recommended)` or `Manual`
- `Surface precision` (when Manual): custom grid size
- `Color`: `Elevation`, `Satellite`, or `Type`
- `Filters`: open/close the class filter popup
- `Sat zoom`: imagery zoom level (16..20)
- `Quality`: LOD density for displayed points
- `Settings`: open full UI settings dialog

### 3D Navigation

- Left click + drag: rotate
- Mouse wheel: zoom
- `Shift + Left click`: select nearest loaded tile directly in 3D

### Settings Menu

- `Language`: English / French
- `Theme`: Midnight, Dawn, Volcanic, Forest, Blueprint, Sandstone
- `UI scale`: global interface scaling
- `Popup width`: responsive width ratio for busy/filter popups (15% to 60%)
- `Compact toolbar`: denser top controls
- `Show tile outlines in 3D`: toggle tile outline + translucent tile area highlight

### Tile Selection Overlay

- All tiles are displayed with subtle outlines.
- The selected tile is emphasized with a stronger accent border and area highlight.
- Selection is available from the tile combo box or by `Shift + Left click` in 3D view.

## Satellite Coloring Notes

- Works best when all loaded tiles share one valid CRS.
- If CRS is missing, loader attempts inference (`EPSG:4326`, `EPSG:3857`, `EPSG:2154`).
- For projected CRSs requiring conversion, `pyproj` is used.
- Tile imagery source default:
  - Esri World Imagery (`World_Imagery/MapServer/tile/{z}/{y}/{x}`)

## Performance Recommendations

- Use lower `Quality` when loading very large areas.
- Keep `Sat zoom` moderate (17-19) for better responsiveness.
- Prefer `Surface = Auto (recommended)` unless you need strict manual control.
- In multi-tile workflows, append progressively and validate alignment before adding many tiles.
- If the machine is constrained, enable `Compact toolbar` and lower `Quality`.
- Tile points are internally kept in `float32` for lower RAM usage in large multi-tile projects.

## Troubleshooting

### App starts but Satellite mode falls back

- Check that tiles have a consistent CRS.
- Ensure `pyproj` is installed.
- Ensure internet access is available.

### OpenGL/surface errors or unstable rendering

- Update GPU drivers.
- Lower surface precision or switch to `Points`.
- Try a smaller loaded area first.

### Slow loading or temporary freezes

- Loading and scene prep are threaded, but very large datasets can still cause short UI stalls.
- Reduce `Quality` or process tiles in smaller batches.

## Development Notes

- Main logic is in `main.py`.
- Loading utilities should stay centralized in `lidar_loader.py`.
- Keep user-facing messages concise and actionable.
- Install dependencies with `install_and_verify.ps1` before opening issues.
- Keep formatting consistent with `.editorconfig`.

## Repository Readiness

- Standard project files included: `.gitignore`, `.editorconfig`, `requirements.txt`, `LICENSE`.
- The codebase compiles with:
  - `python -m py_compile main.py lidar_loader.py gl_viewer.py`
- Contribution workflow is documented in `CONTRIBUTING.md`.

## License

MIT. See `LICENSE`.
