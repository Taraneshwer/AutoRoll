"""
AutoRoll Worker Command Test Script.
Usage: python scripts/run_worker.py [--video path_to_video] [--device auto|cpu|cuda]
"""
import sys
from pathlib import Path
BACKEND_ROOT = str(Path(__file__).resolve().parents[2])
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

import sys
from pathlib import Path


from app.workers.main import main

if __name__ == "__main__":
    main()
