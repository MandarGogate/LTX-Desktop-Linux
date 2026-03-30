"""Tests for web-mode routes (/web/*)."""

from __future__ import annotations

from pathlib import Path


def test_web_app_info(client) -> None:  # type: ignore[no-untyped-def]
    resp = client.get("/web/app-info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "web"
    assert "version" in data
    assert "modelsPath" in data
    assert "outputsPath" in data


def test_web_gpu_info(client) -> None:  # type: ignore[no-untyped-def]
    resp = client.get("/web/gpu-info")
    assert resp.status_code == 200
    data = resp.json()
    assert "available" in data


def test_web_vram_profile(client) -> None:  # type: ignore[no-untyped-def]
    resp = client.get("/web/vram-profile")
    assert resp.status_code == 200
    data = resp.json()
    assert "tier" in data
    assert "vram_total_gb" in data
    assert "offload_strategy" in data
    assert "available_resolutions" in data
    assert "gguf_recommended" in data


def test_web_gguf_models(client) -> None:  # type: ignore[no-untyped-def]
    resp = client.get("/web/gguf-models")
    assert resp.status_code == 200
    data = resp.json()
    assert "available_models" in data


def test_web_file_read_nonexistent(client) -> None:  # type: ignore[no-untyped-def]
    resp = client.post(
        "/web/file/read",
        json={"path": "/nonexistent/file.txt"},
    )
    assert resp.status_code == 200
    data = resp.json()
    # Should return empty data for nonexistent/disallowed paths
    assert data["data"] == ""


def test_web_file_save_disallowed_path(client) -> None:  # type: ignore[no-untyped-def]
    resp = client.post(
        "/web/file/save",
        json={"path": "/etc/shadow", "content": "hack"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
