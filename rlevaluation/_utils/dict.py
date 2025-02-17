from typing import Any, TypeVar

D = TypeVar('D', bound=dict[str, Any])

def subset(d: D, keys: set[str] | list[str]) -> D:
    out: Any = { k: v for k, v in d.items() if k in keys }
    return out
