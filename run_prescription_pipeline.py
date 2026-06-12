#!/usr/bin/env python3
"""Entry point: python run_prescription_pipeline.py page_image/"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prescription_pipeline.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
