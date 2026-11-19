"""Checkpoint diff: locate parameters that turned NaN/Inf between two checkpoints.

Both checkpoints live on read-only volumes and use the same manifest format
as reshard sources.  The four segment layouts (old/new x weights/momentum)
may all shard the tensor differently; every tensor is re-aligned by GLOBAL
index in chunk-sized windows before the element-wise comparison, so split
points never leak into the result.

Only finite -> non-finite transitions count: a parameter already NaN/Inf in
the old checkpoint is not a *new* anomaly.  New anomalies are classified as
weights-only, momentum-only, or both; each category reports its total count
plus the smallest few global indices.

The scan is all-or-nothing: any read failure raises and no partial counts
escape to the caller.
"""

from __future__ import annotations

import array
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .manifest import FLOAT32_BYTES, Segment, load_manifest
from .plans import plan_copy_ops
from .readgate import ReadGate

MAX_EXAMPLES = 5  # per category: the smallest global indices


class DiagnoseError(RuntimeError):
    """Base class: the comparison could not be completed."""


class LengthMismatchError(DiagnoseError):
    """The two checkpoints do not cover the same number of elements."""


class DiagnoseReadError(DiagnoseError):
    """A chunk read failed (IO error / short read) in the middle of the scan."""


@dataclass
class Category:
    count: int = 0
    first_indices: list[int] = field(default_factory=list)

    def record(self, index: int) -> None:
        self.count += 1
        # Windows are scanned in ascending order, so the list stays sorted.
        if len(self.first_indices) < MAX_EXAMPLES:
            self.first_indices.append(index)


@dataclass
class DiagReport:
    step_old: int
    step_new: int
    total_length: int
    weights_only: Category
    momentum_only: Category
    both: Category


def _read_window(
    source_dir: Path,
    segments: tuple[Segment, ...],
    start: int,
    end: int,
    gate: ReadGate,
    chunk_bytes: int,
) -> bytes:
    """Materialise global indices ``[start, end)`` of one tensor into a buffer.

    ``plan_copy_ops`` re-aligns whatever shard layout the checkpoint uses to
    the requested global window; every chunk read holds one gate slot, which
    is released even if the read fails.
    """
    buf = bytearray((end - start) * FLOAT32_BYTES)
    for op in plan_copy_ops(segments, start, end):
        remaining = op.length * FLOAT32_BYTES
        offset = op.src_offset * FLOAT32_BYTES
        dst = op.dst_offset * FLOAT32_BYTES
        try:
            fh = open(source_dir / op.path, "rb", buffering=0)
        except OSError as exc:
            raise DiagnoseReadError(f"cannot open {op.path!r}: {exc}") from exc
        with fh:
            while remaining > 0:
                try:
                    with gate.slot("diagnose"):
                        data = os.pread(fh.fileno(), min(chunk_bytes, remaining), offset)
                except OSError as exc:
                    raise DiagnoseReadError(
                        f"read failed on {op.path!r} at byte {offset}: {exc}"
                    ) from exc
                if not data:
                    raise DiagnoseReadError(
                        f"short read on {op.path!r} at byte {offset}: "
                        "checkpoint changed underneath the scan?"
                    )
                buf[dst : dst + len(data)] = data
                offset += len(data)
                dst += len(data)
                remaining -= len(data)
    return bytes(buf)


def _floats(data: bytes) -> array.array:
    vals = array.array("f", data)
    if sys.byteorder == "big":
        vals.byteswap()  # checkpoint files are little-endian
    return vals


def _scan_window(
    report: DiagReport,
    w_old: bytes,
    w_new: bytes,
    m_old: bytes,
    m_new: bytes,
    base: int,
) -> None:
    if w_old == w_new and m_old == m_new:
        return  # byte-identical window: nothing changed here
    wo, wn = _floats(w_old), _floats(w_new)
    mo, mn = _floats(m_old), _floats(m_new)
    for i in range(len(wo)):
        w_bad = not math.isfinite(wn[i]) and math.isfinite(wo[i])
        m_bad = not math.isfinite(mn[i]) and math.isfinite(mo[i])
        if w_bad and m_bad:
            report.both.record(base + i)
        elif w_bad:
            report.weights_only.record(base + i)
        elif m_bad:
            report.momentum_only.record(base + i)


def compare_checkpoints(
    old_dir: Path,
    new_dir: Path,
    gate: ReadGate,
    chunk_bytes: int,
) -> DiagReport:
    """Scan two equal-length checkpoints and classify every new NaN/Inf.

    Raises ManifestError (invalid source), LengthMismatchError (different
    total_length) or DiagnoseReadError (IO failure); on any of these the
    caller gets no partial result.
    """
    old = load_manifest(old_dir)  # re-validates gaps/overlaps/sizes
    new = load_manifest(new_dir)
    if old.total_length != new.total_length:
        raise LengthMismatchError(
            f"checkpoints differ in length: old total_length={old.total_length}, "
            f"new total_length={new.total_length}"
        )
    total = old.total_length
    report = DiagReport(
        step_old=old.step,
        step_new=new.step,
        total_length=total,
        weights_only=Category(),
        momentum_only=Category(),
        both=Category(),
    )
    window = max(1, chunk_bytes // FLOAT32_BYTES)  # window size in elements
    for start in range(0, total, window):
        end = min(start + window, total)
        w_old = _read_window(old_dir, old.weights, start, end, gate, chunk_bytes)
        w_new = _read_window(new_dir, new.weights, start, end, gate, chunk_bytes)
        m_old = _read_window(old_dir, old.momentum, start, end, gate, chunk_bytes)
        m_new = _read_window(new_dir, new.momentum, start, end, gate, chunk_bytes)
        _scan_window(report, w_old, w_new, m_old, m_new, start)
    return report
