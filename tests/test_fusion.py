import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fusion

if __name__ == "__main__":
    fusion._selfcheck()
