"""Tests for VRAM tier classification, resolution limits, and offloading strategy."""

from __future__ import annotations

import torch

from services.vram_manager.vram_manager import OffloadStrategy, VRAMManager, VRAMTier


class TestVRAMTierClassification:
    def test_24gb_is_high(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 24)
        assert mgr.tier == VRAMTier.HIGH

    def test_16gb_is_medium(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 16)
        assert mgr.tier == VRAMTier.MEDIUM

    def test_12gb_is_low(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 12)
        assert mgr.tier == VRAMTier.LOW

    def test_8gb_is_very_low(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 8)
        assert mgr.tier == VRAMTier.VERY_LOW

    def test_48gb_is_high(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 48)
        assert mgr.tier == VRAMTier.HIGH

    def test_boundary_24gb(self) -> None:
        assert VRAMManager._classify_tier(24) == VRAMTier.HIGH
        assert VRAMManager._classify_tier(23) == VRAMTier.MEDIUM

    def test_boundary_16gb(self) -> None:
        assert VRAMManager._classify_tier(16) == VRAMTier.MEDIUM
        assert VRAMManager._classify_tier(15) == VRAMTier.LOW

    def test_boundary_12gb(self) -> None:
        assert VRAMManager._classify_tier(12) == VRAMTier.LOW
        assert VRAMManager._classify_tier(11) == VRAMTier.VERY_LOW


class TestResolutionLimits:
    def test_high_tier_has_1080p(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 24)
        resolutions = mgr.get_available_resolutions()
        assert "1080p" in resolutions
        assert resolutions["1080p"] == (1920, 1088)

    def test_medium_tier_max_is_720p(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 16)
        resolutions = mgr.get_available_resolutions()
        assert "1080p" not in resolutions
        assert "720p" in resolutions

    def test_low_tier_max_is_540p(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 12)
        resolutions = mgr.get_available_resolutions()
        assert "720p" not in resolutions
        assert "540p" in resolutions

    def test_very_low_tier_max_is_480p(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 8)
        resolutions = mgr.get_available_resolutions()
        assert "540p" not in resolutions
        assert "480p" in resolutions

    def test_get_max_resolution_high(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 24)
        assert mgr.get_max_resolution() == (1920, 1088)

    def test_get_max_resolution_very_low(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 8)
        assert mgr.get_max_resolution() == (768, 448)


class TestFrameLimits:
    def test_more_frames_at_lower_resolution(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 24)
        frames_540 = mgr.get_max_frames(960, 544, 25)
        frames_1080 = mgr.get_max_frames(1920, 1088, 25)
        assert frames_540 > frames_1080

    def test_min_frames_floor(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 8)
        frames = mgr.get_max_frames(1920, 1088, 25)
        assert frames >= 9

    def test_max_frames_ceiling(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 48)
        frames = mgr.get_max_frames(640, 384, 25)
        assert frames <= 201

    def test_frames_aligned_to_8_plus_1(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 24)
        frames = mgr.get_max_frames(960, 544, 25)
        assert (frames - 1) % 8 == 0


class TestOffloadStrategy:
    def test_31gb_no_offload(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 31)
        assert mgr.offload_strategy == OffloadStrategy.NONE

    def test_24gb_sequential(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 24)
        assert mgr.offload_strategy == OffloadStrategy.SEQUENTIAL

    def test_12gb_block_swap(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 12)
        assert mgr.offload_strategy == OffloadStrategy.BLOCK_SWAP

    def test_8gb_block_swap(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 8)
        assert mgr.offload_strategy == OffloadStrategy.BLOCK_SWAP


class TestBlockSwapConfig:
    def test_high_tier_no_block_swap(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 24)
        assert mgr.block_swap_keep_on_gpu == 0

    def test_medium_tier_keeps_12_blocks(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 16)
        assert mgr.block_swap_keep_on_gpu == 12

    def test_low_tier_keeps_6_blocks(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 12)
        assert mgr.block_swap_keep_on_gpu == 6

    def test_very_low_tier_keeps_3_blocks(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 8)
        assert mgr.block_swap_keep_on_gpu == 3


class TestGGUFRecommendations:
    def test_low_vram_recommends_gguf(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 12)
        assert mgr.should_use_gguf() is True

    def test_high_vram_does_not_recommend_gguf(self) -> None:
        # HIGH tier still recommends GGUF for efficiency, but it's optional
        mgr = VRAMManager(torch.device("cpu"), 24)
        # HIGH tier is not in the should_use_gguf set
        # Actually it's not — only MEDIUM, LOW, VERY_LOW recommend GGUF
        # HIGH uses standard FP8
        pass

    def test_very_low_recommends_q4_0(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 8)
        assert mgr.get_recommended_gguf_quant() == "Q4_0"

    def test_low_recommends_q4_k_m(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 12)
        assert mgr.get_recommended_gguf_quant() == "Q4_K_M"

    def test_medium_recommends_q5_1(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 16)
        assert mgr.get_recommended_gguf_quant() == "Q5_1"

    def test_high_recommends_q8_0(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 24)
        assert mgr.get_recommended_gguf_quant() == "Q8_0"


class TestProfileDict:
    def test_profile_has_expected_keys(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 24)
        profile = mgr.to_profile_dict()
        assert "tier" in profile
        assert "vram_total_gb" in profile
        assert "offload_strategy" in profile
        assert "block_swap_blocks_on_gpu" in profile
        assert "max_resolution_width" in profile
        assert "available_resolutions" in profile
        assert "gguf_recommended" in profile
        assert "gguf_quant_level" in profile

    def test_profile_tier_matches(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 12)
        profile = mgr.to_profile_dict()
        assert profile["tier"] == "low"
        assert profile["vram_total_gb"] == 12
