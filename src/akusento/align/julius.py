"""Join jsut-lab forced alignments (Julius, per-phoneme 100ns times) with our
mora labels: mora audio segments = union of member phoneme intervals.

r9y9/jsut-lab labels are OpenJTalk-generated like ours, so accent agreement
validates plumbing only — linguistic truth comes from the fluent check.
"""

from __future__ import annotations

import re
from pathlib import Path

from akusento.labels.accent import _parse_row

_LAB_RE = re.compile(r"^(\d+)\s+(\d+)\s+(.+)$")


def parse_lab(path: str | Path) -> list[dict]:
    """Parse one .lab file -> [{start, end, phon, a}]. Pauses kept (a None)."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = _LAB_RE.match(line.strip())
        if not m:
            continue
        start, end = int(m.group(1)) / 1e7, int(m.group(2)) / 1e7
        p3, a, _ = _parse_row(m.group(3))
        out.append({"start": start, "end": end, "phon": p3, "a": a})
    return out


def mora_segments(lab_rows: list[dict]) -> list[dict]:
    """Group phoneme rows into mora segments by A2 runs (pauses dropped).

    Returns [{start, end, a2, a1}]; one entry per mora, in order.
    """
    content = [r for r in lab_rows if r["a"] is not None]
    segs = []
    for r in content:
        if segs and segs[-1]["a2"] == r["a"][1]:
            segs[-1]["end"] = r["end"]
        else:
            segs.append({"start": r["start"], "end": r["end"], "a2": r["a"][1], "a1": r["a"][0]})
    return segs
