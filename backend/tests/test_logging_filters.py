from __future__ import annotations

import os

os.environ.setdefault("LTX_APP_DATA_DIR", "/tmp/ltx-logging-filter-test")

from ltx2_server import _is_known_harmless_uninitialized_name


class TestLoggingFilters:
    def test_vision_tower_names_are_suppressed(self) -> None:
        assert _is_known_harmless_uninitialized_name(
            "model.model.vision_tower.vision_model.embeddings.patch_embedding.weight"
        )
        assert _is_known_harmless_uninitialized_name("vision_model.encoder.layers.0.self_attn.q_proj.weight")

    def test_non_vision_names_are_not_suppressed(self) -> None:
        assert not _is_known_harmless_uninitialized_name("model.model.language_model.embed_tokens.weight")
        assert not _is_known_harmless_uninitialized_name("conv_in.conv.weight")
