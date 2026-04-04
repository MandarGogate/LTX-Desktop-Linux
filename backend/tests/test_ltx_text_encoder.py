from __future__ import annotations

import torch

from services.text_encoder.ltx_text_encoder import LTXTextEncoder


class _CallableCachedEncoder:
    def __call__(self, prompt):
        del prompt
        return (
            torch.ones((1, 1, 4), dtype=torch.bfloat16),
            torch.zeros((1, 1, 2), dtype=torch.bfloat16),
            None,
        )

    def to(self, device: torch.device):
        del device
        return self


def test_patched_encode_text_uses_cached_encoder_when_text_encoder_is_none(test_state, fake_services):
    encoder = LTXTextEncoder(
        device=torch.device("cpu"),
        http=fake_services.http,
        ltx_api_base_url="https://api.ltx.video",
    )
    encoder.install_patches(lambda: test_state.state)

    assert test_state.state.text_encoder is not None
    test_state.state.text_encoder.api_embeddings = None
    test_state.state.text_encoder.cached_encoder = _CallableCachedEncoder()

    from ltx_core.text_encoders import gemma as text_enc_module

    result = text_enc_module.encode_text(None, prompts=["hello world"])

    assert len(result) == 1
    video_context, audio_context = result[0]
    assert tuple(video_context.shape) == (1, 1, 4)
    assert audio_context is not None
    assert tuple(audio_context.shape) == (1, 1, 2)
