from typing import Any, Dict, List, Sequence, Tuple

class TerminalTable:
    def __init__(self):
        self._d: Dict[Tuple[int, int], Any] = {}
        self._row_headers: List[str] = []

        self.m_row: int = 0
        self.m_col: int = 0

    def add_row(self, row: Sequence[Any]):
        for i, x in enumerate(row):
            self._d[(self.m_row, i)] = x

        self.m_row += 1
        self.m_col = max(self.m_col, len(row))

    def add_col(self, col: Sequence[Any]):
        for i, x in enumerate(col):
            self._d[(i, self.m_col)] = x

        self.m_col += 1
        self.m_row = max(self.m_row, len(col))

    def add_row_headers(self, headers: Sequence[str]):
        self._row_headers = list(headers)
        self.m_row = max(len(headers), self.m_row)

    def _col_widths(self):
        widths = [0] * (self.m_col + 1)
        for i in range(self.m_row + 1):
            for j in range(self.m_col + 1):
                item = self._d.get((i, j), '')
                widths[j] = max(widths[j], len(item))

        has_headers = len(self._row_headers) > 0
        if has_headers:
            m = max(map(len, self._row_headers))
            widths = widths + [m]

        return widths

    def show(self):
        has_headers = len(self._row_headers) > 0
        widths = self._col_widths()

        out = ''
        for i in range(self.m_row):
            if has_headers:
                h = self._row_headers[i]
                out += right_pad(h, widths[-1] + 1)

            for j in range(self.m_col):
                item = self._d.get((i, j), '')
                out += left_pad(item, widths[j] + 2)

            out += '\n'

        print(out)


def right_pad(s: str, l: int):
    needed = l - len(s)
    if needed <= 0:
        return s

    return s + ' ' * needed

def left_pad(s: str, l: int):
    needed = l - len(s)
    if needed <= 0:
        return s

    return (' ' * needed) + s
