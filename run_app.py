"""
Launcher script to run the Streamlit Supplier Risk Intelligence Dashboard.
Sets up the local profile environment so Streamlit launches cleanly on all platforms.
"""
import os
import sys
from pathlib import Path

# Ensure UTF-8 stdout on Windows console to prevent charmap UnicodeEncodeErrors
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ["PYTHONIOENCODING"] = "utf-8"

ROOT = Path(__file__).resolve().parent

# Ensure Windows Anaconda DLLs and local profile are set
import env_setup

os.environ["USERPROFILE"] = str(ROOT)
os.environ["HOMEPATH"] = str(ROOT)[2:]
os.environ["HOMEDRIVE"] = str(ROOT)[:2]

if __name__ == "__main__":
    from streamlit.web import cli as stcli
    sys.argv = [
        "streamlit",
        "run",
        str(ROOT / "app.py"),
        "--server.headless=true",
        "--browser.gatherUsageStats=false"
    ] + sys.argv[1:]
    sys.exit(stcli.main())
