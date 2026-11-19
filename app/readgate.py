"""Shared cap on concurrent chunk reads, with priority for resharding.

Diagnose (checkpoint diff) and reshard jobs both read the source volume in
chunks.  A single :class:`ReadGate` limits how many chunk reads may be in
flight at once across BOTH activities.  When a reshard reader is waiting,
freed slots go to reshards first; diagnose readers only take a slot while
no reshard is queued.  Slots are always released — including when the read
they protected failed.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator, Literal

ReadKind = Literal["reshard", "diagnose"]


class ReadGate:
    """Counting gate with reshard-first wakeup.  Thread-safe."""

    def __init__(self, max_slots: int):
        if max_slots < 1:
            raise ValueError("max_slots must be >= 1")
        self._max = max_slots
        self._available = max_slots
        self._cond = threading.Condition()
        self._waiting_reshards = 0
        self._waiting_diagnoses = 0

    @property
    def max_slots(self) -> int:
        return self._max

    @property
    def available_slots(self) -> int:
        with self._cond:
            return self._available

    @property
    def waiting_reshards(self) -> int:
        with self._cond:
            return self._waiting_reshards

    @property
    def waiting_diagnoses(self) -> int:
        with self._cond:
            return self._waiting_diagnoses

    @contextmanager
    def slot(self, kind: ReadKind) -> Iterator[None]:
        """Hold one read slot for the duration of one chunk read.

        The slot is released on exit no matter how the block ends, so a
        failed read never leaks capacity.
        """
        self._acquire(kind)
        try:
            yield
        finally:
            self._release()

    def _acquire(self, kind: ReadKind) -> None:
        with self._cond:
            if kind == "reshard":
                self._waiting_reshards += 1
            else:
                self._waiting_diagnoses += 1
            try:
                while True:
                    # A queued reshard gets first pick of freed slots:
                    # diagnose readers yield while any reshard is waiting.
                    if self._available > 0 and (
                        kind == "reshard" or self._waiting_reshards == 0
                    ):
                        self._available -= 1
                        return
                    self._cond.wait()
            finally:
                if kind == "reshard":
                    self._waiting_reshards -= 1
                else:
                    self._waiting_diagnoses -= 1

    def _release(self) -> None:
        with self._cond:
            self._available += 1
            self._cond.notify_all()
