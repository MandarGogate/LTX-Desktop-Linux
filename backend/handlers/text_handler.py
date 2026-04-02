"""Text encoding cache handler — local-only."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from handlers.base import StateHandlerBase, with_state_lock
from runtime_config.model_download_specs import resolve_model_path
from state.app_state_types import AppState, TextEncodingResult

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)

# Tokenizer config files required by module_ops_from_gemma_root().
# These are small JSON/model files from the google/gemma-3-12b-it repo.
_TOKENIZER_FILES = [
    "tokenizer.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
]
_TOKENIZER_REPO = "google/gemma-3-12b-it"


class TextHandler(StateHandlerBase):
    def __init__(self, state: AppState, lock: RLock, config: RuntimeConfig) -> None:
        super().__init__(state, lock, config)

    @with_state_lock
    def _get_cached_prompt(self, prompt: str, enhance_prompt: bool) -> TextEncodingResult | None:
        te = self.state.text_encoder
        if te is None:
            return None
        return te.prompt_cache.get((prompt.strip(), enhance_prompt))

    @with_state_lock
    def _cache_prompt(self, prompt: str, enhance_prompt: bool, result: TextEncodingResult) -> None:
        te = self.state.text_encoder
        if te is None:
            return

        max_size = self.state.app_settings.prompt_cache_size
        if max_size <= 0:
            return

        key = (prompt.strip(), enhance_prompt)
        if key in te.prompt_cache:
            del te.prompt_cache[key]
        elif len(te.prompt_cache) >= max_size:
            oldest = next(iter(te.prompt_cache))
            del te.prompt_cache[oldest]
        te.prompt_cache[key] = result

    @with_state_lock
    def _set_api_embeddings(self, result: TextEncodingResult | None) -> None:
        if self.state.text_encoder is not None:
            self.state.text_encoder.api_embeddings = result

    def clear_api_embeddings(self) -> None:
        self._set_api_embeddings(None)

    # ------------------------------------------------------------------
    # Gemma root / tokenizer config
    # ------------------------------------------------------------------

    def _default_text_encoder_dir(self) -> Path:
        return resolve_model_path(
            self.models_dir, self.config.model_download_specs, "text_encoder"
        )

    def _text_encoders_dir(self) -> Path:
        return self.models_dir / "text_encoders"

    def _find_gemma_root_dir(self) -> Path | None:
        """Return a directory containing tokenizer.model + preprocessor_config.json.

        Search order:
        1. The default text encoder folder (gemma-3-12b-it-qat-q4_0-unquantized/)
        2. The text_encoders/ directory itself (tokenizer files stored alongside variants)
        """
        default_dir = self._default_text_encoder_dir()
        if default_dir.exists() and (default_dir / "tokenizer.model").exists():
            return default_dir

        te_dir = self._text_encoders_dir()
        if te_dir.exists() and (te_dir / "tokenizer.model").exists():
            return te_dir

        return None

    def _ensure_tokenizer_files(self) -> Path | None:
        """Ensure tokenizer config files exist, downloading them if needed.

        Returns the directory containing the tokenizer files, or None if
        they could not be obtained.
        """
        existing = self._find_gemma_root_dir()
        if existing is not None:
            return existing

        # Download tokenizer files into text_encoders/
        te_dir = self._text_encoders_dir()
        te_dir.mkdir(parents=True, exist_ok=True)

        try:
            from huggingface_hub import hf_hub_download  # type: ignore[reportUnknownVariableType]

            for filename in _TOKENIZER_FILES:
                target = te_dir / filename
                if target.exists():
                    continue
                logger.info("Downloading tokenizer file: %s from %s", filename, _TOKENIZER_REPO)
                downloaded = Path(str(hf_hub_download(  # type: ignore[reportUnknownMemberType]
                    repo_id=_TOKENIZER_REPO,
                    filename=filename,
                )))
                # hf_hub_download returns a cache path; copy to te_dir
                if downloaded != target and downloaded.exists():
                    import shutil
                    shutil.copy2(str(downloaded), str(target))

            logger.info("Tokenizer files ready at %s", te_dir)
            return te_dir
        except Exception:
            logger.warning("Failed to download tokenizer files", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Text encoder availability
    # ------------------------------------------------------------------

    def _preferred_variant_path(self) -> Path | None:
        preferred = self.state.app_settings.preferred_text_encoder_path.strip()
        if preferred:
            preferred_path = self.models_dir / preferred
            if preferred_path.exists() and preferred_path.is_file():
                return preferred_path

        te_dir = self._text_encoders_dir()
        if te_dir.exists():
            for path in sorted(te_dir.iterdir()):
                if path.is_file() and path.suffix == ".safetensors":
                    return path
        return None

    def _ensure_model_alias_for_builder(self, gemma_root: Path) -> None:
        """Create a model*.safetensors alias so ModelLedger's default builder initializes.

        ModelLedger expects gemma_root to contain a file matching `model*.safetensors`.
        Quantized variants like `gemma_3_12B_it_fp4_mixed.safetensors` don't match,
        so we create a symlink/hardlink/copy alias.
        """
        if any(gemma_root.glob("model*.safetensors")):
            return

        variant = self._preferred_variant_path()
        if variant is None or not variant.exists() or variant.parent != gemma_root:
            return

        alias_path = gemma_root / "model_variant.safetensors"
        if alias_path.exists():
            return

        try:
            os.symlink(variant.name, alias_path)
            logger.info("Created text encoder symlink alias: %s -> %s", alias_path, variant.name)
            return
        except Exception:
            pass

        try:
            os.link(variant, alias_path)
            logger.info("Created text encoder hardlink alias: %s -> %s", alias_path, variant.name)
            return
        except Exception:
            pass

        try:
            import shutil
            shutil.copy2(str(variant), str(alias_path))
            logger.info("Copied text encoder alias: %s -> %s", alias_path, variant.name)
        except Exception:
            logger.warning("Failed to create text encoder alias for %s", variant, exc_info=True)

    def _is_local_text_encoder_available(self) -> bool:
        """Check if any local text encoder is available.

        Checks (in order):
        1. User-preferred variant path from settings
        2. Any .safetensors or .gguf file in text_encoders/
        3. The default text encoder folder (gemma-3-12b-it-qat-q4_0-unquantized)
        """
        preferred_path = self._preferred_variant_path()
        if preferred_path is not None:
            return True

        te_dir = self._text_encoders_dir()
        if te_dir.exists():
            for path in te_dir.iterdir():
                if path.is_file() and path.suffix in (".safetensors", ".gguf"):
                    return True
                if path.is_dir() and any(path.iterdir()):
                    return True

        default_dir = self._default_text_encoder_dir()
        if default_dir.exists() and any(default_dir.iterdir()):
            return True

        return False

    def should_use_local_encoding(self) -> bool:
        return self._is_local_text_encoder_available()

    def prepare_text_encoding(self, prompt: str, enhance_prompt: bool) -> None:
        """Validate that local text encoder is available.

        Raises RuntimeError if no local text encoder can be found.
        """
        del prompt, enhance_prompt

        if not self._is_local_text_encoder_available():
            raise RuntimeError(
                "TEXT_ENCODER_MISSING: No text encoder found. "
                "Download a text encoder from Settings → Model Downloads."
            )

        self.clear_api_embeddings()

    def resolve_gemma_root(self) -> str | None:
        """Return the gemma root directory for the text encoder.

        The gemma root must contain tokenizer.model and preprocessor_config.json.
        If these files are missing, they are auto-downloaded from HuggingFace.
        """
        if not self._is_local_text_encoder_available():
            return None

        root = self._ensure_tokenizer_files()
        if root is not None:
            self._ensure_model_alias_for_builder(root)
            return str(root)
        return None
