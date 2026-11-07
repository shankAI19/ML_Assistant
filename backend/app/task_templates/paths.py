"""Use the sandbox workspace by default, or the standalone runner's data folder."""

import os
from pathlib import Path

WORK = Path(os.environ.get("ML_ASSISTANT_WORKDIR", "/work"))
OUT = WORK / "out"
