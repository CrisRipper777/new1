"""Single-run entry point with a fixed metadata-defined NC label universe."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.tasks.nc as nc_task


def _declared_nc_classes(data):
    return list(range(int(data.num_classes)))


nc_task._resolve_nc_eval_labels = _declared_nc_classes

from src.main import main  # noqa: E402


if __name__ == "__main__":
    main()
