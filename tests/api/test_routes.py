"""Integration tests for the API routes, via a FastAPI `TestClient` wired to
an injected `AppState` (tiny DINOv2 model, synthetic gallery, no real
checkpoint/network access)."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_identify_returns_ranked_matches(client: TestClient, sample_image_bytes: bytes) -> None:
    response = client.post(
        "/identify", files={"file": ("sample.png", sample_image_bytes, "image/png")}
    )

    assert response.status_code == 200
    matches = response.json()["matches"]
    assert len(matches) == 5
    assert all({"label", "similarity"} == set(match.keys()) for match in matches)


def test_identify_respects_top_k(client: TestClient, sample_image_bytes: bytes) -> None:
    response = client.post(
        "/identify",
        files={"file": ("sample.png", sample_image_bytes, "image/png")},
        params={"top_k": 2},
    )

    assert response.status_code == 200
    assert len(response.json()["matches"]) == 2


def test_identify_rejects_non_image_upload(client: TestClient) -> None:
    response = client.post(
        "/identify", files={"file": ("not_an_image.txt", b"hello world", "text/plain")}
    )

    assert response.status_code == 400
    assert "not a decodable image" in response.json()["detail"]


def test_identify_without_gallery_returns_400(
    client_without_gallery: TestClient, sample_image_bytes: bytes
) -> None:
    response = client_without_gallery.post(
        "/identify", files={"file": ("sample.png", sample_image_bytes, "image/png")}
    )

    assert response.status_code == 400
    assert "No gallery configured" in response.json()["detail"]


def test_enroll_adds_samples_and_updates_gallery_size(
    client_without_gallery: TestClient, sample_image_bytes: bytes
) -> None:
    response = client_without_gallery.post(
        "/enroll",
        data={"writer_id": "shehroz"},
        files=[
            ("files", ("me1.png", sample_image_bytes, "image/png")),
            ("files", ("me2.png", sample_image_bytes, "image/png")),
        ],
    )

    assert response.status_code == 200
    body = response.json()
    assert body == {"writer_id": "shehroz", "samples_added": 2, "gallery_size": 2}


def test_enroll_then_identify_finds_enrolled_writer(
    client_without_gallery: TestClient, sample_image_bytes: bytes
) -> None:
    client_without_gallery.post(
        "/enroll",
        data={"writer_id": "shehroz"},
        files=[("files", ("me1.png", sample_image_bytes, "image/png"))],
    )

    response = client_without_gallery.post(
        "/identify", files={"file": ("me2.png", sample_image_bytes, "image/png")}
    )

    assert response.status_code == 200
    matches = response.json()["matches"]
    assert matches[0]["label"] == "shehroz"
