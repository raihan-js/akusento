"""Alignment join tests (synthetic .lab rows)."""

from akusento.align.julius import mora_segments


def _row(s, e, phon, a):
    return {"start": s, "end": e, "phon": phon, "a": a}


def test_groups_by_a2_and_drops_pauses():
    rows = [
        _row(0.0, 0.1, "sil", None),
        _row(0.1, 0.2, "k", (0, 1, 2)),
        _row(0.2, 0.3, "a", (0, 1, 2)),
        _row(0.3, 0.4, "N", (1, 2, 1)),
        _row(0.4, 0.5, "pau", None),
        _row(0.5, 0.6, "t", (0, 1, 1)),
        _row(0.6, 0.7, "a", (0, 1, 1)),
    ]
    segs = mora_segments(rows)
    assert [(round(s["start"], 1), round(s["end"], 1)) for s in segs] == [
        (0.1, 0.3), (0.3, 0.4), (0.5, 0.7)]
    assert [s["a2"] for s in segs] == [1, 2, 1]
