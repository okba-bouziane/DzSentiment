import pytest

from dzsent.metrics import score


def make(lang, labels, script="arabic"):
    return [{"id": f"{lang}-{i}", "label": l, "lang": lang, "script": script} for i, l in enumerate(labels)]


def test_perfect_predictions():
    rows = make("darija", [0, 1, 2, 2]) + make("french", [0, 1, 2])
    m = score(rows, {r["id"]: r["label"] for r in rows})
    assert m["overall"]["macro_f1"] == 1.0
    assert m["mean_lang_macro_f1"] == 1.0
    assert m["by_lang"]["darija"]["confusion"] == [[1, 0, 0], [0, 1, 0], [0, 0, 2]]
    assert set(m["darija_by_script"]) == {"arabic"}


def test_languages_weigh_the_same_in_the_mean():
    rows = make("darija", [0, 1, 2] * 100) + make("english", [0, 1, 2])
    preds = {r["id"]: r["label"] for r in rows}
    for r in rows:
        if r["lang"] == "english":
            preds[r["id"]] = 0  # all wrong except negatives
    m = score(rows, preds)
    assert m["by_lang"]["darija"]["macro_f1"] == 1.0
    assert m["by_lang"]["english"]["macro_f1"] == pytest.approx(1 / 6)  # F1 0.5 on negative, 0 on others
    assert m["mean_lang_macro_f1"] == pytest.approx((1 + 1 / 6) / 2)
    assert m["overall"]["macro_f1"] > m["mean_lang_macro_f1"]  # the big language hides the small one


def test_missing_prediction_is_an_error():
    rows = make("darija", [0, 1])
    with pytest.raises(ValueError):
        score(rows, {rows[0]["id"]: 0})
