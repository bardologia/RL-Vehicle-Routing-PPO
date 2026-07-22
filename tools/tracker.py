from typing import Any, Mapping, Optional

import numpy as np


class Tracker:
    def __init__(self, writer=None):
        self.writer = writer

    @property
    def active(self) -> bool:
        return self.writer is not None

    def log_scalar(self, tag: str, value, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, float(value), step)

    def log_metrics(self, prefix: str, values: Mapping[str, Any], step: int) -> None:
        if self.writer is None:
            return
        for key, value in values.items():
            try:
                self.writer.add_scalar(f"{prefix}/{key}", float(value), step)
            except (TypeError, ValueError):
                continue

    def log_histogram(self, tag: str, values, step: int, bins="auto") -> None:
        if self.writer is None:
            return
        flat = np.asarray(values).ravel().astype(np.float32)
        try:
            self.writer.add_histogram(tag, flat, step, bins=bins)
        except (ValueError, RuntimeError):
            pass

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


class NullTracker(Tracker):
    def __init__(self):
        super().__init__(writer=None)
