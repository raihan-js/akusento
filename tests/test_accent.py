"""Mora parser + accent assignment tests (structural; linguistic truth is
measured by the 200-utterance fluent check, not asserted here)."""

from akusento.labels.accent import extract, split_morae


def test_split_morae():
    assert split_morae("キャット") == ["キャ", "ッ", "ト"]
    assert split_morae("シンブン") == ["シ", "ン", "ブ", "ン"]
    assert split_morae("コーヒー") == ["コ", "ー", "ヒ", "ー"]
    assert split_morae("アメ") == ["ア", "メ"]


def test_atamadaka_ame_ga():
    ph = extract("雨が。")
    assert len(ph) == 1
    assert ph[0].pattern == "HLL", ph[0].pattern
    assert ph[0].nucleus == 1
    assert [m.text for m in ph[0].morae] == ["ア", "メ", "ガ"]


def test_heiban_ame_ga():
    ph = extract("飴が。")
    assert len(ph) == 1
    assert ph[0].pattern == "LHH", ph[0].pattern
    assert ph[0].nucleus is None


def test_minimal_pair_differs():
    assert extract("雨が。")[0].pattern != extract("飴が。")[0].pattern
    assert extract("箸が。")[0].pattern != extract("橋が。")[0].pattern


def test_heiban_sakura():
    ph = extract("桜。")
    assert ph[0].pattern == "LHH", ph[0].pattern


def test_pron_text_attached():
    ph = extract("日本語。")
    assert all(m.text for p in ph for m in p.morae)
    assert len(ph[0].morae) == 4
