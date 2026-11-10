from dzsent.llm import build_prompt, parse_labels, sample_ids


def test_prompt_numbers_every_text_and_keeps_unicode():
    p = build_prompt(["ماشي مليح", 'il a dit "non"'])
    assert '1. "ماشي مليح"' in p
    assert '2. "il a dit \\"non\\""' in p
    assert "with 2 labels" in p


def test_parse_valid_reply_with_surrounding_text():
    reply = 'Sure:\n{"labels": ["Positive", "neutral", "negative"]}'
    assert parse_labels(reply, 3) == [2, 1, 0]


def test_parse_rejects_wrong_count_unknown_label_and_garbage():
    assert parse_labels('{"labels": ["positive"]}', 2) is None
    assert parse_labels('{"labels": ["happy", "neutral"]}', 2) is None
    assert parse_labels("no json here", 1) is None
    assert parse_labels('{"labels": "positive"}', 1) is None
    assert parse_labels("", 1) is None


def test_sample_is_fixed_and_per_language():
    rows = [{"id": f"{l}-{i:03d}", "lang": l} for l in ("darija", "french") for i in range(300)]
    a, b = sample_ids(rows, 50), sample_ids(rows, 50)
    assert a == b and len(a) == 100
    assert sum(i.startswith("darija") for i in a) == 50
