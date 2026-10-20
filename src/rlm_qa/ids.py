from __future__ import annotations

from datetime import datetime
from uuid import uuid4


def new_id(prefix: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d")
    return f"{prefix}_{stamp}_{uuid4().hex[:12]}"
