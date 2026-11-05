from __future__ import annotations

import asyncio
import functools
from typing import Any, Callable


async def run_blocking(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Executa uma função bloqueante num thread pool (compatível com Python 3.8)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))


def truncate(text: Any, limit: int = 12000) -> str:
    if text is None:
        return ""
    value = str(text)
    if len(value) <= limit:
        return value
    return value[:limit] + "\n... [saída truncada em {} caracteres]".format(limit)


def human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return "{:.1f} {}".format(num_bytes, unit)
        num_bytes /= 1024.0
    return "{:.1f} PB".format(num_bytes)
