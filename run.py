"""Unified Launcher for Precision Aerosol Spray Can Sound Detector.

Starts the Flask backend server which automatically serves the decoupled frontend
(HTML, CSS stylesheets, JavaScript, and assets) from the frontend/ directory.

Usage:
    python run.py
    python run.py --port 5000 --host 127.0.0.1
"""

import argparse
import sys
from pathlib import Path

# Add backend directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

for p in [str(BACKEND_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app import app, ensure_demo_samples, logger, start_esp32_receiver_service


def main():
    parser = argparse.ArgumentParser(description="Start Aerosol Spray Can Detector Application")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind server (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    print("=" * 70)
    print("  [+] PRECISION AEROSOL SPRAY CAN SOUND DETECTOR")
    print("=" * 70)
    print(f"  * Project Root:  {PROJECT_ROOT}")
    print(f"  * Backend Path:  {BACKEND_DIR}")
    print(f"  * Frontend Path: {FRONTEND_DIR}")
    print(f"  * Web Dashboard: http://{args.host}:{args.port}")
    print("=" * 70)

    # Ensure synthetic and authentic sample fixtures exist in backend/data/demo_samples
    ensure_demo_samples()

    # Pre-flight check: ensure ports are not already occupied by another running instance
    import socket
    for check_port in [5001, args.port]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", check_port)) == 0:
                print(f"\n[ERROR] Port {check_port} is already in use by another running process!")
                print("To prevent conflicts, please terminate any existing 'python run.py' instances first.\n")
                sys.exit(1)

    # Launch dedicated ESP32 hardware receiver on port 5001 (daemon thread)
    start_esp32_receiver_service()

    logger.info("Starting Web Application on http://%s:%d ...", args.host, args.port)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
