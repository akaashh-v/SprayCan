"""Root entrypoint shim pointing to backend/app.py.

For unified application startup:
    python run.py
or
    python backend/app.py
"""

import sys
from pathlib import Path

# Add backend directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Re-export Flask application and utilities
from backend.app import app, ensure_demo_samples, engine, logger

if __name__ == "__main__":
    ensure_demo_samples()
    logger.info("Starting Flask Server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
