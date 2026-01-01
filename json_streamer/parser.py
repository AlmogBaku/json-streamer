from abc import ABC, abstractmethod
from enum import Enum
from typing import Generator, Optional, Tuple, List, Any
import re
import json

_SYMBOL_MAP = {'{': '}', '[': ']', '"': '"', '}': '{', ']': '['}


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
    def raw_decode(self, s: str) -> Tuple[Any, int]:
        raise NotImplementedError

    @staticmethod
    def _opposite_symbol(symbol: chr):
        """
        Returns the opposite symbol for a given symbol
        :param symbol: The symbol to find the opposite for
        :type symbol: chr
        :return: The opposite symbol
        """

        return _SYMBOL_MAP.get(symbol, None)

    # Compile regex pattern once for performance
    _STRUCTURAL_CHARS = re.compile(r'["\[\]{}]')

    def __init__(self):
        self.buffer = ""
        self.symbol_stack = []
        self.in_string = False

    def _is_escaped(self, text: str, pos: int) -> bool:
        """Check if character at position is escaped by counting preceding backslashes"""
        count = 0
        i = pos - 1
        while i >= 0 and text[i] == '\\':
            count += 1
            i -= 1
        return count % 2 == 1

    def _process_part(self, part: str) -> Optional[Tuple[ParseState, dict]]:
        if not part:
            return

        self.buffer += part
        buffer_text = self.buffer
        start_pos = len(buffer_text) - len(part)

        # Use regex to find only structural characters in the new chunk
        # This skips all content between structural chars (10-100x faster)
        for match in self._STRUCTURAL_CHARS.finditer(part):
            pos = start_pos + match.start()
            char = match.group()

            # Handle quotes - check for escape
            if char == '"':
                if self._is_escaped(buffer_text, pos):
                    continue  # Skip escaped quotes

                if self.symbol_stack and self.symbol_stack[-1] == '"':
                    self.symbol_stack.pop()
                    self.in_string = False
                else:
                    self.symbol_stack.append('"')
                    self.in_string = True

            # Only process structural chars outside strings
            elif not self.in_string:
                if char in '{[':
                    self.symbol_stack.append(char)
                elif char in '}]' and self.symbol_stack:
                    if char == self._opposite_symbol(self.symbol_stack[-1]):
                        self.symbol_stack.pop()

        # Try to parse complete JSON
        if not self.symbol_stack:
            try:
                obj, _ = self.raw_decode(buffer_text)
                return ParseState.COMPLETE, obj
            except json.JSONDecodeError:
                pass

        # Complete partial JSON by appending closing symbols
        completed = buffer_text + "".join(
            self._opposite_symbol(s) for s in reversed(self.symbol_stack)
        )
        try:
            obj, _ = self.raw_decode(completed)
            return ParseState.PARTIAL, obj
        except json.JSONDecodeError:
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
