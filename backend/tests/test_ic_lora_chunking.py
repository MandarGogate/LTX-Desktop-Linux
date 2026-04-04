from __future__ import annotations

from api_types import ImageConditioningInput
from handlers.ic_lora_handler import _chunk_image_inputs


def test_chunk_image_inputs_carries_prior_reference_image_into_later_chunks() -> None:
    images = [ImageConditioningInput(path="ref.png", frame_idx=0, strength=1.0)]

    localized = _chunk_image_inputs(
        images,
        chunk_start_frame=96,
        chunk_frame_count=113,
    )

    assert localized == [
        ImageConditioningInput(path="ref.png", frame_idx=0, strength=1.0)
    ]


def test_chunk_image_inputs_keeps_in_chunk_images_at_local_frame_offsets() -> None:
    images = [
        ImageConditioningInput(path="ref_a.png", frame_idx=0, strength=1.0),
        ImageConditioningInput(path="ref_b.png", frame_idx=120, strength=0.75),
        ImageConditioningInput(path="future.png", frame_idx=240, strength=0.5),
    ]

    localized = _chunk_image_inputs(
        images,
        chunk_start_frame=96,
        chunk_frame_count=113,
    )

    assert localized == [
        ImageConditioningInput(path="ref_a.png", frame_idx=0, strength=1.0),
        ImageConditioningInput(path="ref_b.png", frame_idx=24, strength=0.75),
    ]
