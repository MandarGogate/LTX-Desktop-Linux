from __future__ import annotations

from collections import deque
from datetime import datetime
import sys
import threading
from typing import TextIO

_MAX_LOG_LINES = 2000
_lines: deque[str] = deque(maxlen=_MAX_LOG_LINES)
_lock = threading.RLock()
_installed = False


def _format_timestamp() -> str:
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S,") + f"{now.microsecond // 1000:03d}"


def _append_line(level: str, source: str, message: str) -> None:
    text = message.rstrip("\r")
    if not text:
        return
    line = f"{_format_timestamp()} - {level} - [{source}] {text}"
    with _lock:
        _lines.append(line)


def get_recent_lines(limit: int = 200) -> list[str]:
    safe_limit = max(1, min(limit, _MAX_LOG_LINES))
    with _lock:
        return list(_lines)[-safe_limit:]


class _LineBufferingTee:
    def __init__(self, stream: TextIO, *, level: str, source: str) -> None:
        self._stream = stream
        self._level = level
        self._source = source
        self._pending = ""

    def write(self, data: str) -> int:
        written = self._stream.write(data)
        self._pending += data

        while True:
            newline_index = self._pending.find("\n")
            if newline_index == -1:
                break
            line = self._pending[:newline_index]
            self._pending = self._pending[newline_index + 1 :]
            _append_line(self._level, self._source, line)

        return written

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._stream, "isatty", lambda: False)())

    def fileno(self) -> int:
        return int(self._stream.fileno())

    @property
    def encoding(self) -> str | None:
        return getattr(self._stream, "encoding", None)

    @property
    def errors(self) -> str | None:
        return getattr(self._stream, "errors", None)

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


def install_web_log_capture() -> None:
    global _installed
    if _installed:
        return

    sys.stdout = _LineBufferingTee(sys.stdout, level="INFO", source="Backend")
    sys.stderr = _LineBufferingTee(sys.stderr, level="ERROR", source="Backend")
    _installed = True
