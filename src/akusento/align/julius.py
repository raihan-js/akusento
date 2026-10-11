"""Join jsut-lab forced alignments (Julius, per-phoneme 100ns times) with our
mora labels: mora audio segments = union of member phoneme intervals.

r9y9/jsut-lab labels are OpenJTalk-generated like ours, so accent agreement
validates plumbing only — linguistic truth comes from the fluent check.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from akusento.labels.accent import _parse_row

_LAB_RE = re.compile(r"^(\S+)\s+(\S+)\s+(.+)$")


def _to_sec(tok: str) -> float:
    v = float(tok)
    return v / 1e7 if "." not in tok else v  # jsut-lab: 100ns ints; JVS: seconds


def parse_lab(path: str | Path) -> list[dict[str, Any]]:
    """Parse one .lab file -> [{start, end, phon, a, fid}]. Pauses kept (a None)."""
    out: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = _LAB_RE.match(line.strip())
        if not m:
            continue
        start, end = _to_sec(m.group(1)), _to_sec(m.group(2))
        p3, a, fid = _parse_row(m.group(3))
        out.append({"start": start, "end": end, "phon": p3, "a": a, "fid": fid})
    return out


def mora_segments(lab_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group phoneme rows into mora segments by A2 runs (pauses dropped).

    Returns [{start, end, a2, a1}]; one entry per mora, in order.
    """
    content = [r for r in lab_rows if r["a"] is not None]
    segs: list[dict[str, Any]] = []
    for r in content:
        a = r["a"]
        assert isinstance(a, tuple)
        if segs and segs[-1]["a2"] == a[1]:
            segs[-1]["end"] = r["end"]
        else:
            segs.append({"start": r["start"], "end": r["end"], "a2": a[1], "a1": a[0]})
    return segs
