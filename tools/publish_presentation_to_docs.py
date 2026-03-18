from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "presentation"
DST = ROOT / "docs" / "presentation"


def main() -> None:
    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST)
    print(f"published {SRC} -> {DST}")


if __name__ == "__main__":
    main()
