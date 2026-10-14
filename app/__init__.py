"""SOP-guided insurance claims agent."""
import os
import re
from pathlib import Path


def _load_dotenv() -> None:
    """Minimal .env loader (KEY=VALUE lines). Real environment variables win."""
    path = Path(__file__).resolve().parent.parent / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = re.split(r"\s+#", v, 1)[0]          # allow `KEY=value   # comment`
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()
