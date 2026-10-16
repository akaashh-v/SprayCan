import sys
from pathlib import Path

# Add backend directory to sys.path so tests can import app, config, etc.
_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
