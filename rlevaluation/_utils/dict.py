from typing import Any, Dict, Set, List, TypeVar

D = TypeVar('D', bound=Dict[str, Any])

def subset(d: D, keys: Set[str] | List[str]) -> D:
    out: Any = { k: v for k, v in d.items() if k in keys }
    return out
