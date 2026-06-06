from __future__ import annotations

import os
import sys
from typing import Any


def _stdout_is_tty() -> bool:
    force_tty = os.environ.get("TQDM_FORCE_TTY", "")
    if str(force_tty).strip() in ("1", "true", "yes"):
        return True
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _prefer_rich() -> bool:
    if os.environ.get("TQDM_DISABLE_RICH", "") in ("1", "true", "yes"):
        return False
    if os.environ.get("TQDM_PLAIN", "") in ("1", "true", "yes"):
        return False
    return _stdout_is_tty()


def tqdm_auto(*args: Any, **kwargs: Any):
    """Create a tqdm progress iterator appropriate for the current output."""
    if _prefer_rich():
        from tqdm.rich import tqdm as _rtqdm

        return _rtqdm(*args, **kwargs)

    from tqdm import tqdm as _ptqdm

    if "bar_format" not in kwargs:
        kwargs["bar_format"] = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"

    env_mininterval = float(os.environ.get("TQDM_MININTERVAL", "0.5"))
    kwargs.setdefault("mininterval", env_mininterval)
    kwargs.setdefault("ascii", True)
    kwargs.setdefault("smoothing", 0.0)
    kwargs.setdefault("disable", False)
    return _ptqdm(*args, **kwargs)
