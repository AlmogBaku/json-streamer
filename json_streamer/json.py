import json
from typing import List, Dict, Tuple, Optional, Generator

from .parser import Parser, ParseState


class JsonParser(Parser):
    def __init__(self):
        super().__init__()
        self._decoder = json.JSONDecoder()

    @staticmethod
    def opening_symbols() -> List[chr]:
        return ['{', '[', '"']

    def raw_decode(self, s: str) -> Tuple[Dict, int]:
        return self._decoder.raw_decode(s)


def loads(s: Optional[Generator[chr, None, None]] = None) -> Generator[Tuple[ParseState, dict], Optional[str], None]:
    return JsonParser()(s)
