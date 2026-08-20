"""
AutoRoll Worker Command Test Script.
Usage: python scripts/run_worker.py [--video path_to_video] [--device auto|cpu|cuda]
"""

from worker.main import main

if __name__ == "__main__":
    main()
