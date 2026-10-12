import sys
from pathlib import Path

# Add project root directory to sys.path so app and modules are discoverable
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app import app

# Vercel serverless WSGI entry point
# Exports 'app' for @vercel/python
