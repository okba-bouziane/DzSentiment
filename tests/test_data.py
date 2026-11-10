from collections import Counter

from dzsent import data as D


def rows(n_per_label=50):
    return [{"text": f"text {l} {i}", "label": l} for l in (0, 1, 2) for i in range(n_per_label)]


def test_clean_collapses_whitespace_and_normalises():
    assert D.clean("  salam \n\t  khoya  ") == "salam khoya"
    # decomposed "é" becomes the composed form
    assert D.clean("café") == "café"


def test_script_detection():
    assert D.script_of("واش راك خويا") == "arabic"
    assert D.script_of("wach rak khoya") == "latin"
    assert D.script_of("rani mli7 الحمد لله") == "mixed"
    assert D.script_of("👍👍 !!") == "other"


def test_split_is_deterministic_and_stratified():
    a = D.stratified_split(rows(), seed=42)
    b = D.stratified_split(rows(), seed=42)
    assert [r["text"] for r in a["test"]] == [r["text"] for r in b["test"]]
    assert sum(len(v) for v in a.values()) == 150
    assert Counter(r["label"] for r in a["test"]) == {0: 5, 1: 5, 2: 5}
    c = D.stratified_split(rows(), seed=1)
    assert [r["text"] for r in a["test"]] != [r["text"] for r in c["test"]]


def test_split_has_no_overlap():
    s = D.stratified_split(rows(), seed=42)
    sets = [{r["text"] for r in s[k]} for k in D.SPLITS]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])


def test_dedupe_drops_duplicates_and_conflicting_labels():
    rs = [
        {"text": "Bravo!", "label": 2},
        {"text": "bravo! ", "label": 2},   # same text after case folding
        {"text": "normal", "label": 1},
        {"text": "Normal", "label": 0},    # same text, other label: dropped entirely
        {"text": "   ", "label": 1},        # empty after cleaning
    ]
    kept, dups, conflicts = D.dedupe(rs)
    assert [r["text"] for r in kept] == ["Bravo!"]
    assert conflicts == 1
    assert dups == 3


def test_cross_split_duplicates_are_removed_from_eval_splits():
    splits = {
        "train": [{"text": "zwin bezzaf", "label": 2}],
        "validation": [{"text": "Zwin  bezzaf", "label": 2}, {"text": "khayeb", "label": 0}],
        "test": [{"text": "khayeb", "label": 0}, {"text": "ok", "label": 1}],
    }
    dropped = D.remove_cross_split_duplicates(splits)
    assert dropped == {"validation": 1, "test": 1}
    assert [r["text"] for r in splits["test"]] == ["ok"]
