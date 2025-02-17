from collections.abc import Callable
from typing import Any, TypeVar
from rlevaluation._utils.logging import logger

_has_warned = False
T = TypeVar('T', bound=Callable[..., Any])

def try2jit(f: T) -> T:
    try:
        from numba import njit
        return njit(f, cache=True, nogil=True, fastmath=True)
    except Exception as e:
        global _has_warned
        if not _has_warned:
            _has_warned = True
            logger.warning('Could not jit compile --- expect slow performance', e)

        return f

def try2jit_no_cache(f: T) -> T:
    try:
        from numba import njit
        return njit(f, nogil=True, fastmath=True)
    except Exception as e:
        global _has_warned
        if not _has_warned:
            _has_warned = True
            logger.warning('Could not jit compile --- expect slow performance', e)

        return f

def try2pjit(f: T) -> T:
    try:
        from numba import njit
        return njit(f, cache=True, nogil=True, fastmath=True, parallel=True)
    except Exception as e:
        global _has_warned
        if not _has_warned:
            _has_warned = True
            logger.warning('Could not jit compile --- expect slow performance', e)

        return f
