param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path "requirements.txt")) {
    throw "requirements.txt not found in current directory."
}

Write-Host "[1/6] Checking Python..." -ForegroundColor Cyan
& $PythonExe --version

Write-Host "[2/6] Creating virtual environment (.venv)..." -ForegroundColor Cyan
if (!(Test-Path ".venv")) {
    & $PythonExe -m venv .venv
}

$venvPython = ".\.venv\Scripts\python.exe"
if (!(Test-Path $venvPython)) {
    throw "Virtual environment python not found at $venvPython"
}

Write-Host "[3/6] Upgrading pip..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip

Write-Host "[4/6] Installing dependencies from requirements.txt..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade -r requirements.txt

Write-Host "[5/6] Verifying runtime imports and OpenGL stack..." -ForegroundColor Cyan
$verifyScript = @"
import importlib
import sys

modules = [
    "numpy",
    "laspy",
    "PyQt6",
    "pyqtgraph",
    "OpenGL",
    "pyproj",
]

failed = []
for m in modules:
    try:
        importlib.import_module(m)
    except Exception as e:
        failed.append((m, str(e)))

if failed:
    print("Import verification failed:")
    for name, err in failed:
        print(f" - {name}: {err}")
    sys.exit(1)

print("All imports OK.")

try:
    import pyqtgraph.opengl as _gl  # noqa: F401
    print("pyqtgraph.opengl OK.")
except Exception as e:
    print(f"pyqtgraph.opengl check failed: {e}")
    sys.exit(1)

print("Environment verification completed successfully.")
"@

& $venvPython -c $verifyScript

Write-Host "[6/6] Compiling project files..." -ForegroundColor Cyan
& $venvPython -m py_compile main.py lidar_loader.py gl_viewer.py

Write-Host "Done. Activate venv and run app:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python .\main.py"
