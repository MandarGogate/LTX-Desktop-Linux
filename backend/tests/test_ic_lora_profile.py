from __future__ import annotations

from threading import RLock
from types import SimpleNamespace

import torch

from handlers.ic_lora_handler import IcLoraHandler


class _DummyGenerationHandler:
    pass


class _DummyPipelinesHandler:
    pass


class _DummyTextHandler:
    pass


class _DummyVideoProcessor:
    pass


def _make_handler() -> IcLoraHandler:
    state = SimpleNamespace(
        app_settings=SimpleNamespace(
            num_blocks_to_swap=-1,
            run_mode="auto",
            seed_locked=False,
            locked_seed=0,
        )
    )
    config = SimpleNamespace(device=torch.device("cpu"))
    return IcLoraHandler(
        state,
        RLock(),
        _DummyGenerationHandler(),
        _DummyPipelinesHandler(),
        _DummyTextHandler(),
        _DummyVideoProcessor(),
        config,
    )


def test_ic_lora_720p_profile_uses_conservative_frame_cap() -> None:
    handler = _make_handler()

    width, height, max_frames = handler._select_generation_profile(
        resolution="720p",
        aspect_ratio="16:9",
        fps=24.0,
    )

    assert (width, height) == (1280, 768)
    assert max_frames == 113


def test_ic_lora_1080p_profile_uses_conservative_frame_cap() -> None:
    handler = _make_handler()

    width, height, max_frames = handler._select_generation_profile(
        resolution="1080p",
        aspect_ratio="16:9",
        fps=24.0,
    )

    assert (width, height) == (1792, 1024)
    assert max_frames == 81


def test_ic_lora_1080p_long_chunk_uses_conservative_stage2_block_swap() -> None:
    handler = _make_handler()

    policy = handler._resolve_stage_2_block_swap_policy(
        width=1792,
        height=1024,
        num_frames=81,
    )

    assert policy == (1, 1, "ic-lora-stage2-1080p-long")


def test_ic_lora_shorter_chunk_skips_conservative_stage2_block_swap() -> None:
    handler = _make_handler()

    policy = handler._resolve_stage_2_block_swap_policy(
        width=1792,
        height=1024,
        num_frames=57,
    )

    assert policy is None
