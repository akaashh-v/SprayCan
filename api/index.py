import sys
from pathlib import Path

# Add project root and backend directory to sys.path so app and modules are discoverable
project_root = Path(__file__).resolve().parent.parent
backend_dir = project_root / "backend"
for p in [str(backend_dir), str(project_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.app import app

# Vercel serverless WSGI entry point
# Exports 'app' for @vercel/python
