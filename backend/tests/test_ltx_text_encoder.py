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


class _GGUFCachedEncoder(_CallableCachedEncoder):
    def __init__(self) -> None:
        class _Gemma:
            pass

        self.model = _Gemma()
        self._ltx_gguf_text_encoder = True
        self.moves: list[str] = []

    def __call__(self, prompt):
        assert self.model.device == torch.device("cpu")
        return super().__call__(prompt)

    def to(self, device: torch.device):
        self.moves.append(str(device))
        return self


class _ProvidedEncoder(_CallableCachedEncoder):
    def __init__(self) -> None:
        class _Gemma:
            pass

        self.model = _Gemma()
        self.moves: list[str] = []

    def to(self, device: torch.device):
        self.moves.append(str(device))
        return self


def test_patched_encode_text_uses_cached_encoder_when_text_encoder_is_none(test_state, fake_services):
    from ltx_core.text_encoders import gemma as text_enc_module

    original_encode_text = text_enc_module.encode_text
    try:
        encoder = LTXTextEncoder(
            device=torch.device("cpu"),
            http=fake_services.http,
            ltx_api_base_url="https://api.ltx.video",
        )
        encoder.install_patches(lambda: test_state.state)

        assert test_state.state.text_encoder is not None
        test_state.state.text_encoder.api_embeddings = None
        test_state.state.text_encoder.cached_encoder = _CallableCachedEncoder()

        result = text_enc_module.encode_text(None, prompts=["hello world"])

        assert len(result) == 1
        video_context, audio_context = result[0]
        assert tuple(video_context.shape) == (1, 1, 4)
        assert audio_context is not None
        assert tuple(audio_context.shape) == (1, 1, 2)
    finally:
        text_enc_module.encode_text = original_encode_text


def test_patched_encode_text_normalizes_cached_gguf_encoder_to_cpu(test_state, fake_services):
    from ltx_core.text_encoders import gemma as text_enc_module

    original_encode_text = text_enc_module.encode_text
    try:
        encoder = LTXTextEncoder(
            device=torch.device("cpu"),
            http=fake_services.http,
            ltx_api_base_url="https://api.ltx.video",
        )
        encoder.install_patches(lambda: test_state.state)

        assert test_state.state.text_encoder is not None
        test_state.state.text_encoder.api_embeddings = None
        cached = _GGUFCachedEncoder()
        test_state.state.text_encoder.cached_encoder = cached

        result = text_enc_module.encode_text(None, prompts=["hello world"])

        assert len(result) == 1
        assert cached.moves[-1] == "cpu"
    finally:
        text_enc_module.encode_text = original_encode_text


def test_patched_encode_text_does_not_retarget_caller_supplied_encoder(test_state, fake_services):
    from ltx_core.text_encoders import gemma as text_enc_module

    original_encode_text = text_enc_module.encode_text
    try:
        encoder = LTXTextEncoder(
            device=torch.device("cpu"),
            http=fake_services.http,
            ltx_api_base_url="https://api.ltx.video",
        )
        encoder.install_patches(lambda: test_state.state)

        provided = _ProvidedEncoder()
        result = text_enc_module.encode_text(provided, prompts=["hello world"])

        assert len(result) == 1
        assert provided.moves == []
    finally:
        text_enc_module.encode_text = original_encode_text
