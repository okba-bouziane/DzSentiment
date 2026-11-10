import os

os.environ["DZSENT_NO_AUTOLOAD"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import MAX_TEXTS, create_app  # noqa: E402
from dzsent.classifier import Prediction  # noqa: E402


class FakeClassifier:
    def __init__(self):
        self.calls = []

    def predict(self, texts):
        self.calls.append(texts)
        label = lambda t: "negative" if "khayeb" in t else "positive" if "zwin" in t else "neutral"  # noqa: E731
        return [Prediction(t, label(t), {"negative": 0.1, "neutral": 0.2, "positive": 0.7}) for t in texts]


def client():
    fake = FakeClassifier()
    return TestClient(create_app(fake, "fake")), fake


def test_classify_returns_one_result_per_comment_and_counts():
    c, fake = client()
    r = c.post("/api/classify", json={"texts": ["service khayeb", " zwin bezzaf ", "", "win rah el colis?"]})
    assert r.status_code == 200
    body = r.json()
    assert [x["label"] for x in body["results"]] == ["negative", "positive", "neutral"]
    assert body["counts"] == {"negative": 1, "neutral": 1, "positive": 1}
    assert fake.calls == [["service khayeb", "zwin bezzaf", "win rah el colis?"]]  # trimmed, empty dropped


def test_limits():
    c, fake = client()
    assert c.post("/api/classify", json={"texts": ["  "]}).status_code == 422
    assert c.post("/api/classify", json={"texts": ["a"] * (MAX_TEXTS + 1)}).status_code == 422
    assert c.post("/api/classify", json={"texts": ["x" * 1001]}).status_code == 422
    assert fake.calls == []


def test_page_and_health():
    c, _ = client()
    assert c.get("/health").json() == {"ok": True, "model": "fake"}
    page = c.get("/")
    assert page.status_code == 200 and "<textarea" in page.text
