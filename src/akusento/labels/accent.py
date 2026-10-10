"""Mora-level accent labels from OpenJTalk full-context labels.

Method (Kurihara, as in piper-plus/jpreprocess — verified field semantics
against docs.rs/haqumei-jlabel + piper-plus source, 2026-10-10):
- A1 = relative accent position, A2 = mora position forward in the phrase,
  A3 = backward. `]` (pitch fall / nucleus end) at the mora with a1==0 whose
  next mora has a2 == a2+1; `#` (phrase boundary) when a2==a3 and next a2==1.
- H/L per mora (Tokyo): 1-mora phrase -> [H]; else mora1 L, morae 2..nucleus
  H, after nucleus L; heiban (no fall) -> mora1 L, rest H.
- Mora TEXT comes from run_frontend pron (katakana); counts must match the
  label morae per phrase (asserted). OpenJTalk's dictionary choices
  (heiban vs odaka in isolation, Tokyo-only) are NOT verified here — the
  200-utterance fluent check (docs) measures label truth separately.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pyopenjtalk

_A_RE = re.compile(r"/A:([^/]+)/")
_F_RE = re.compile(r"/F:([^/]+)/")
# p-fields never contain ^ - + = @, so negated classes (the old greedy
# (.*) version silently misparsed p3 whenever later fields had dashes).
_PHON_RE = re.compile(r"^([^^]+)\^([^^-]+)-([^-+]+)\+([^+=]+)=([^@]+)@")

_PAUSES = {"sil", "pau", "xx", "silB", "silE"}


def _parse_row(label: str) -> tuple[str, tuple[int, int, int] | None, str]:
    """(current phoneme p3, (a1,a2,a3) or None for pauses, F-field phrase id)."""
    m = _PHON_RE.match(label)
    p3 = m.group(3) if m else ""
    f = _F_RE.search(label)
    fid = f.group(1) if f else ""
    if p3 in _PAUSES or fid.startswith("xx_xx#"):
        return p3, None, ""
    a = _A_RE.search(label)
    if not a:
        return p3, None, ""
    parts = a.group(1).split("+")
    if len(parts) != 3 or any(not p.lstrip("-").isdigit() for p in parts):
        return p3, None, ""
    return p3, (int(parts[0]), int(parts[1]), int(parts[2])), fid


@dataclass
class Mora:
    text: str  # katakana mora, e.g. "キャ"
    high: bool  # True = H, False = L


@dataclass
class AccentPhrase:
    morae: list[Mora] = field(default_factory=list)
    nucleus: int | None = None  # 1-indexed mora position of the fall, None = heiban

    @property
    def pattern(self) -> str:
        return "".join("H" if m.high else "L" for m in self.morae)


def _mora_positions(labels: list[str]) -> list[tuple[int, int, int]]:
    """Per-phoneme (a1, a2, a3); pauses dropped."""
    return [a for _, a in (_parse_row(lab) for lab in labels) if a is not None]


_KATAKANA_SMALL_Y = set("ャュョァィゥェォ")
_SOKUON = "ッ"
_HATSUON = "ン"
_CHOON = "ー"


def split_morae(pron: str) -> list[str]:
    """Split a katakana pronunciation into morae.

    きゃ/しゅ/ちょ (consonant + small y-row) = 1 mora; ッ/ン/ー = 1 mora each.
    """
    out: list[str] = []
    i = 0
    while i < len(pron):
        ch = pron[i]
        nxt = pron[i + 1] if i + 1 < len(pron) else ""
        if nxt in _KATAKANA_SMALL_Y and ch not in _KATAKANA_SMALL_Y:
            out.append(ch + nxt)
            i += 2
        else:
            out.append(ch)
            i += 1
    return out



def _njd_phrases(text: str) -> list[list[str]]:
    """Chunk the text into accent phrases via NJD chain_flag.

    A word with chain_flag == 1 attaches to the previous word (same phrase);
    otherwise it starts a new phrase. Punctuation always ends a phrase
    (labels emit pause rows there). Returns per-phrase katakana mora lists.
    """
    _PUNCT = set("、。？！「」『』…―")
    phrases: list[list[str]] = []
    cur: list[str] = []
    for node in pyopenjtalk.run_frontend(text):
        if node.get("string") in _PUNCT or node.get("pos") == "記号":
            if cur:
                phrases.append(cur)
                cur = []
            continue
        pron = "".join(ch for ch in node.get("pron", "") if "\u30a1" <= ch <= "\u30f6" or ch == "ー")
        if not pron:
            continue
        if node.get("chain_flag") != 1 and cur:
            phrases.append(cur)
            cur = []
        cur.extend(split_morae(pron))
    if cur:
        phrases.append(cur)
    return phrases


def extract(text: str) -> list[AccentPhrase]:
    """Full pipeline: text -> accent phrases with H/L morae.

    Boundaries combine three label-space signals (any fires): F-field change,
    A2 reset (prev>1 to 1), pause flush. NJD chain_flag chunking provides mora
    TEXT; all three counts (F1, NJD, A2-groups) must agree per phrase, else
    ValueError (never silently misaligned).
    """
    labels = pyopenjtalk.extract_fullcontext(text)
    rows = [_parse_row(lab) for lab in labels]
    # Segment label rows. A boundary falls before a row iff any fires:
    # F-field change, A2 reset (prev>1 to 1), or a pause row between
    # (pauses only sit at real boundaries; they are dropped, never morae).
    lab_phrases: list[tuple[str, list[tuple[int, int, int]]]] = []
    cur: list[tuple[int, int, int]] = []
    cur_fid = ""
    for _, a, fid in rows:
        if a is None:
            if cur:
                lab_phrases.append((cur_fid, cur))
                cur = []
                cur_fid = ""
            continue
        if cur and (fid != cur_fid or (a[1] == 1 and cur[-1][1] > 1)):
            lab_phrases.append((cur_fid, cur))
            cur = []
            cur_fid = ""
        if not cur:
            cur_fid = fid
        cur.append(a)
    if cur:
        lab_phrases.append((cur_fid, cur))
    njd = _njd_phrases(text)
    if len(lab_phrases) != len(njd):
        raise ValueError(f"phrase count mismatch: njd={len(njd)} labels={len(lab_phrases)} in {text!r}")
    result: list[AccentPhrase] = []
    for (fid, lrows), pmorae in zip(lab_phrases, njd):
        f1 = int(fid.split("#")[0].split("_")[0])
        # Morae: group phonemes by A2 within the phrase.
        mora_groups: list[list[tuple[int, int, int]]] = []
        for a in lrows:
            if mora_groups and mora_groups[-1][0][1] == a[1]:
                mora_groups[-1].append(a)
            else:
                mora_groups.append([a])
        if not (len(mora_groups) == len(pmorae) == f1):
            raise ValueError(
                f"phrase mora mismatch: njd={len(pmorae)} labels={len(mora_groups)} F1={f1} in {text!r}")
        n = len(mora_groups)
        # Fall detection: mora with a1==0 whose NEXT mora exists (Kurihara ]).
        a1s = [g[0][0] for g in mora_groups]
        nucleus: int | None = None
        for i, a1 in enumerate(a1s):
            if a1 == 0 and i + 1 < n:
                nucleus = i + 1  # 1-indexed
        ap = AccentPhrase(nucleus=nucleus)
        for i, tm in enumerate(pmorae, start=1):
            if n == 1:
                high = True
            elif nucleus == 1:
                high = (i == 1)
            elif nucleus is None:
                high = (i > 1)
            else:
                high = (1 < i <= nucleus)
            ap.morae.append(Mora(text=tm, high=high))
        result.append(ap)
    return result
