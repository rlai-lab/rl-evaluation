from typing import Any, Iterable, List as _List
from typing import TypeVar

T = TypeVar('T')

class List(_List[T]):
    def __init__(self, it: Iterable[Any] | None = None):
        ...
