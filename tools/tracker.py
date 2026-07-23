from contextlib import contextmanager
from typing     import Any, Mapping, Optional

import numpy as np


class Tracker:
    def __init__(self, writer=None):
        self.writer  = writer
        self._step   = 0
        self._scopes = []

    @property
    def active(self) -> bool:
        return self.writer is not None

    @property
    def current_step(self) -> int:
        return self._step

    def set_step(self, step: int) -> None:
        self._step = int(step)

    def advance(self, n: int = 1) -> int:
        self._step += n
        return self._step

    @contextmanager
    def scope(self, name: str):
        self._scopes.append(str(name))
        try:
            yield self
        finally:
            self._scopes.pop()

    def log_scalar(self, tag: str, value, step: Optional[int] = None) -> None:
        self._emit("add_scalar", tag, float(value), step)

    def log_metrics(self, prefix: str, values: Mapping[str, Any], step: Optional[int] = None) -> None:
        for key, value in values.items():
            try:
                self._emit("add_scalar", f"{prefix}/{key}", float(value), step)
            except (TypeError, ValueError):
                continue

    def log_histogram(self, tag: str, values, step: Optional[int] = None, bins="auto") -> None:
        flat = np.asarray(values).ravel().astype(np.float32)
        try:
            self._emit("add_histogram", tag, flat, step, bins=bins)
        except (ValueError, RuntimeError):
            pass

    def _tag(self, tag: str) -> str:
        return "/".join([*self._scopes, str(tag)])

    def _resolve(self, step: Optional[int]) -> int:
        return self._step if step is None else int(step)

    def _emit(self, method: str, tag: str, payload: Any, step: Optional[int], **kwargs: Any) -> None:
        if self.writer is None:
            return
        getattr(self.writer, method)(self._tag(tag), payload, self._resolve(step), **kwargs)

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


class NullTracker(Tracker):
    def __init__(self):
        super().__init__(writer=None)
