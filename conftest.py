import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

for _p in (ROOT, ROOT / "src"):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)
