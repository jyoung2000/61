"""Auto name-consistency: unify rare proper-noun spellings to the dominant one,
without collapsing genuinely distinct names. Pure logic (stdlib only)."""

from backend.services.name_consistency import (
    unify_proper_noun_variants, build_consistency_map,
)


def _seg(text):
    return {"text": text, "start": 0.0, "end": 1.0, "speaker": "Speaker 1"}


def test_unifies_close_variant_to_dominant():
    # "Doria" ×4 dominates the one-off "Dorian" misspelling.
    segs = [_seg("Doria arrived"), _seg("Mr Doria"), _seg("Doria spoke again"),
            _seg("hello Doria"), _seg("it was Dorian back then")]
    out, n = unify_proper_noun_variants(segs)
    blob = " ".join(s["text"] for s in out)
    assert "Dorian" not in blob
    assert blob.count("Doria") >= 5
    assert n == 1


def test_unifies_three_way_variants():
    segs = ([_seg("Lilina is here")] * 5
            + [_seg("where is Lilyna")]
            + [_seg("Liliana waved")])
    out, n = unify_proper_noun_variants(segs)
    blob = " ".join(s["text"] for s in out)
    assert "Lilyna" not in blob and "Liliana" not in blob
    assert blob.count("Lilina") >= 7


def test_preserves_codominant_distinct_names():
    # Leo and Leon are BOTH frequent → not merged (could be two people).
    segs = [_seg("Leo here")] * 4 + [_seg("Leon here")] * 3
    out, n = unify_proper_noun_variants(segs)
    blob = " ".join(s["text"] for s in out)
    assert "Leo here" in blob and "Leon here" in blob
    assert n == 0


def test_no_merge_across_different_initials():
    # "Catur" vs "Kator" share no first letter — never clustered.
    segs = [_seg("Catur reports")] * 3 + [_seg("Kator reports")]
    assert build_consistency_map(segs) == {}


def test_noop_without_recurrence():
    segs = [_seg("A lone Name appears"), _seg("nothing here")]
    out, n = unify_proper_noun_variants(segs)
    assert n == 0
    assert [s["text"] for s in out] == ["A lone Name appears", "nothing here"]


def test_stopwords_not_clustered():
    # Sentence-initial function words must never be treated as names.
    segs = [_seg("There it goes")] * 4 + [_seg("Their plan failed")] * 3
    assert build_consistency_map(segs) == {}


def test_handles_pydantic_like_objects():
    class _S:
        def __init__(self, text):
            self.text = text
    segs = [_S("Doria ran")] * 4 + [_S("then Dorian fell")]
    out, n = unify_proper_noun_variants(segs)
    assert n == 1
    assert all("Dorian" not in s.text for s in out)
