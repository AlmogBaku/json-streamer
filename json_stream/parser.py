from abc import ABC, abstractmethod
from enum import Enum
from typing import Generator, Optional, Tuple, Dict, List


class ParseState(Enum):
    UNKNOWN = -1
    PARTIAL = 0
    COMPLETE = 1


class Parser(ABC):
    @staticmethod
    @abstractmethod
    def opening_symbols() -> List[chr]:
        raise NotImplementedError

    def closing_symbols(self) -> List[chr]:
        return [self._opposite_symbol(s) for s in self.opening_symbols()]

    @abstractmethod
    def raw_decode(self, s: str) -> Tuple[Dict, int]:
        raise NotImplementedError

    @staticmethod
    def _opposite_symbol(symbol: chr):
        """
        Returns the opposite symbol for a given symbol
        :param symbol: The symbol to find the opposite for
        :type symbol: chr
        :return: The opposite symbol
        """

        symbol_map = {'{': '}', '[': ']', '"': '"', '}': '{', ']': '['}
        return symbol_map.get(symbol, None)

    def __init__(self):
        self.buffer = []
        self.symbol_stack = []
        self.escaping = False

    def _process_part(self, part: str) -> Optional[Tuple[ParseState, dict]]:
        if part is None or part == '':
            return
        self.buffer.append(part)

        for c in part:
            self.escaping = c == '\\' and not self.escaping
            if not self.escaping:
                if self.symbol_stack and c in self.closing_symbols() and c == self._opposite_symbol(
                        self.symbol_stack[-1]):
                    self.symbol_stack.pop()
                elif c in self.opening_symbols():
                    self.symbol_stack.append(c)

        if not self.symbol_stack:
            try:
                obj, _ = self.raw_decode("".join(self.buffer))
                return ParseState.COMPLETE, obj
            except Exception:
                pass

        new_buffer = self.buffer + [self._opposite_symbol(s) for s in reversed(self.symbol_stack)]
        new_buffer = "".join(new_buffer)

        try:
            obj, _ = self.raw_decode(new_buffer)
            return ParseState.PARTIAL, obj
        except Exception:
            pass

    def parse_part(self, part: str) -> Generator[Tuple[ParseState, dict], None, None]:
        if part is None or part == '':
            return
        _obj = self._process_part(part)
        if _obj is not None:
            if _obj[0] == ParseState.PARTIAL and len(_obj[1]) == 0:
                return
            yield _obj

    def __call__(self, stream: Optional[Generator[chr, None, None]] = None) \
            -> Generator[Tuple[ParseState, dict], Optional[str], None]:
        """
        Parses a stream of partial JSON objects and yields partial objects as they are constructed
        When a complete object is constructed, it is returned

        :param stream: The stream to parse
        :type stream: Generator[str, None, None]
        :return: A generator that yields partial objects as they are constructed. When a complete object is constructed,
        it is returned.
        """

        if stream is not None:
            for part in stream:
                yield from self.parse_part(part)
        else:
            while True:
                try:
                    part = yield
                    yield from self.parse_part(part)
                except StopIteration:
                    return
