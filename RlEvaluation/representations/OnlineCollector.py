import numpy as np

class OnlineCollector:
    def __init__(self, max_size: int, replacement_rate: float, full_by: int | None = None):
        self._max_size = max_size
        self._p = replacement_rate
        self._full_by = full_by

        self._ramp = self._p

        if full_by is not None:
            assert max_size <= full_by

            p = max_size / full_by
            self._ramp = max(self._p, p)


        self._init = False
        self._store = np.empty(0)
