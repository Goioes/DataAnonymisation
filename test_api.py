"""API checks with a stub detector (no model download). Run: uv run python test_api.py (or pytest)"""
from fastapi.testclient import TestClient

import api
from detectors import DETECTORS, Entity

client = TestClient(api.app)


class Stub:
    def detect(self, text):
        return [Entity("Bob", "PERSON", 5, 8)]


api.get_detector = lambda name: Stub()


def test_models():
    assert client.get("/models").json() == list(DETECTORS)


def test_anonymize_defaults_to_gliner():
    r = client.post("/anonymize", json={"text": "Call Bob now"})
    assert r.status_code == 200
    assert r.json() == {
        "anonymized_text": "Call <PERSON> now",
        "entities": [{"text": "Bob", "label": "PERSON", "start": 5, "end": 8}],
    }


def test_validation():
    for body in [
        {"text": ""},
        {"text": "x" * (api.MAX_CHARS + 1)},
        {"text": "Call Bob", "model": "nope"},
    ]:
        assert client.post("/anonymize", json=body).status_code == 422, body


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
