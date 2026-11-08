"""
Environment bootstrap for Windows Python DLL loading.
Dynamically detects Python and Anaconda Library directories to ensure
that C/C++ native DLLs (Scipy, Scikit-learn, XGBoost) load smoothly across all Windows environments.
"""
import os
import sys

# Silence loky and OpenMP warnings on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"

_DLL_HANDLES = []

def setup_environment():
    """Dynamically register DLL directories based on active Python environment."""
    if sys.platform == "win32":
        prefixes = [
            getattr(sys, "base_prefix", sys.prefix),
            sys.prefix,
            os.environ.get("CONDA_PREFIX", ""),
        ]
        
        candidates = []
        for p in prefixes:
            if p and os.path.isdir(p):
                candidates.extend([
                    os.path.join(p, "Library", "bin"),
                    os.path.join(p, "Library", "mingw-w64", "bin"),
                    os.path.join(p, "Library", "usr", "bin"),
                    os.path.join(p, "bin"),
                    os.path.join(p, "Scripts"),
                ])

        for path in set(candidates):
            if os.path.isdir(path):
                try:
                    handle = os.add_dll_directory(path)
                    _DLL_HANDLES.append(handle)
                except Exception:
                    pass

setup_environment()
