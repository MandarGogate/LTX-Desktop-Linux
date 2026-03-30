"""Tests for runtime policy decision helper."""

from __future__ import annotations

from runtime_config.runtime_policy import MIN_LOCAL_VRAM_GB, decide_force_api_generations


def test_darwin_always_forces_api() -> None:
    assert decide_force_api_generations(system="Darwin", cuda_available=True, vram_gb=24) is True
    assert decide_force_api_generations(system="Darwin", cuda_available=False, vram_gb=None) is True


def test_windows_without_cuda_forces_api() -> None:
    assert decide_force_api_generations(system="Windows", cuda_available=False, vram_gb=24) is True


def test_windows_with_very_low_vram_forces_api() -> None:
    """GPUs below the minimum (8GB) threshold should be forced to API."""
    assert decide_force_api_generations(system="Windows", cuda_available=True, vram_gb=6) is True


def test_windows_with_unknown_vram_forces_api() -> None:
    assert decide_force_api_generations(system="Windows", cuda_available=True, vram_gb=None) is True


def test_windows_with_8gb_allows_local_mode() -> None:
    """8GB GPUs should be able to generate locally with low-VRAM pipeline."""
    assert decide_force_api_generations(system="Windows", cuda_available=True, vram_gb=8) is False


def test_windows_with_12gb_allows_local_mode() -> None:
    assert decide_force_api_generations(system="Windows", cuda_available=True, vram_gb=12) is False


def test_windows_with_24gb_allows_local_mode() -> None:
    """RTX 4090 (24GB) should definitely work locally."""
    assert decide_force_api_generations(system="Windows", cuda_available=True, vram_gb=24) is False


def test_windows_with_required_vram_allows_local_mode() -> None:
    assert decide_force_api_generations(system="Windows", cuda_available=True, vram_gb=31) is False


def test_linux_without_cuda_forces_api() -> None:
    assert decide_force_api_generations(system="Linux", cuda_available=False, vram_gb=24) is True


def test_linux_with_very_low_vram_forces_api() -> None:
    assert decide_force_api_generations(system="Linux", cuda_available=True, vram_gb=6) is True


def test_linux_with_unknown_vram_forces_api() -> None:
    assert decide_force_api_generations(system="Linux", cuda_available=True, vram_gb=None) is True


def test_linux_with_8gb_allows_local_mode() -> None:
    assert decide_force_api_generations(system="Linux", cuda_available=True, vram_gb=8) is False


def test_linux_with_12gb_allows_local_mode() -> None:
    assert decide_force_api_generations(system="Linux", cuda_available=True, vram_gb=12) is False


def test_linux_with_24gb_allows_local_mode() -> None:
    assert decide_force_api_generations(system="Linux", cuda_available=True, vram_gb=24) is False


def test_linux_with_required_vram_allows_local_mode() -> None:
    assert decide_force_api_generations(system="Linux", cuda_available=True, vram_gb=31) is False


def test_other_systems_fail_closed() -> None:
    assert decide_force_api_generations(system="FreeBSD", cuda_available=True, vram_gb=48) is True


def test_min_vram_threshold_constant() -> None:
    """Verify the exported constant matches expected value."""
    assert MIN_LOCAL_VRAM_GB == 8
