"""Canonical app settings schema and patch models."""

from __future__ import annotations

from typing import Any, Literal, TypeGuard, TypeVar, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator


def _to_camel_case(field_name: str) -> str:
    special_aliases = {
        "prompt_enhancer_enabled_t2v": "promptEnhancerEnabledT2V",
        "prompt_enhancer_enabled_i2v": "promptEnhancerEnabledI2V",
    }
    if field_name in special_aliases:
        return special_aliases[field_name]

    head, *tail = field_name.split("_")
    return head + "".join(part.title() for part in tail)


def _clamp_int(value: Any, minimum: int, maximum: int, default: int) -> int:
    if value is None:
        return default

    parsed = int(value)
    return max(minimum, min(maximum, parsed))


class SettingsBaseModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel_case,
        populate_by_name=True,
        validate_assignment=True,
        extra="ignore",
    )


class SettingsPatchModel(SettingsBaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel_case,
        populate_by_name=True,
        validate_assignment=True,
        extra="forbid",
    )


class FastModelSettings(SettingsBaseModel):
    use_upscaler: bool = True


class ProModelSettings(SettingsBaseModel):
    steps: int = 30
    use_upscaler: bool = True

    @field_validator("steps", mode="before")
    @classmethod
    def _clamp_steps(cls, value: Any) -> int:
        return _clamp_int(value, minimum=1, maximum=100, default=30)


class CustomModelSettings(SettingsBaseModel):
    steps: int = 20
    use_upscaler: bool = True

    @field_validator("steps", mode="before")
    @classmethod
    def _clamp_steps(cls, value: Any) -> int:
        return _clamp_int(value, minimum=1, maximum=100, default=20)


class SelectedLoRASettings(SettingsBaseModel):
    path: str = ""
    strength: float = 0.8

    @field_validator("strength", mode="before")
    @classmethod
    def _clamp_strength(cls, value: Any) -> float:
        if value is None:
            return 0.8
        parsed = float(value)
        return max(0.0, min(2.0, parsed))


RunModeLiteral = Literal["auto", "high_vram", "medium_vram", "low_vram", "very_low_vram"]
A2VDecodeTilingLiteral = Literal["auto", "high", "medium", "low", "very_low", "none"]


class AppSettings(SettingsBaseModel):
    use_torch_compile: bool = False
    load_on_startup: bool = False
    ltx_api_key: str = ""
    user_prefers_ltx_api_video_generations: bool = False
    fal_api_key: str = ""
    use_local_text_encoder: bool = False
    fast_model: FastModelSettings = Field(default_factory=FastModelSettings)
    pro_model: ProModelSettings = Field(default_factory=ProModelSettings)
    custom_model: CustomModelSettings = Field(default_factory=CustomModelSettings)
    prompt_cache_size: int = 100
    prompt_enhancer_enabled_t2v: bool = True
    prompt_enhancer_enabled_i2v: bool = False
    gemini_api_key: str = ""
    seed_locked: bool = False
    locked_seed: int = 42
    models_dir: str = ""
    project_assets_dir: str = ""
    preferred_model_path: str = ""
    selected_loras: list[SelectedLoRASettings] = Field(default_factory=list)
    preferred_gguf_path: str = ""
    preferred_lora_path: str = ""
    preferred_lora_strength: float = 0.8
    num_blocks_to_swap: int = -1
    run_mode: RunModeLiteral = "auto"
    a2v_decode_tiling: A2VDecodeTilingLiteral = "auto"
    preferred_zit_model_path: str = ""
    preferred_text_encoder_path: str = ""

    @field_validator("prompt_cache_size", mode="before")
    @classmethod
    def _clamp_prompt_cache_size(cls, value: Any) -> int:
        return _clamp_int(value, minimum=0, maximum=1000, default=100)

    @field_validator("locked_seed", mode="before")
    @classmethod
    def _clamp_locked_seed(cls, value: Any) -> int:
        return _clamp_int(value, minimum=0, maximum=2_147_483_647, default=42)

    @field_validator("preferred_lora_strength", mode="before")
    @classmethod
    def _clamp_preferred_lora_strength(cls, value: Any) -> float:
        if value is None:
            return 0.8
        parsed = float(value)
        return max(0.0, min(2.0, parsed))

    @field_validator("num_blocks_to_swap", mode="before")
    @classmethod
    def _clamp_num_blocks_to_swap(cls, value: Any) -> int:
        if value is None:
            return -1
        parsed = int(value)
        return max(-1, min(48, parsed))


SettingsModelT = TypeVar("SettingsModelT", bound=SettingsBaseModel)
_PARTIAL_MODEL_CACHE: dict[type[SettingsBaseModel], type[SettingsPatchModel]] = {}


def _wrap_optional(annotation: Any) -> Any:
    if type(None) in get_args(annotation):
        return annotation
    return annotation | None


def _to_partial_annotation(annotation: Any) -> Any:
    if _is_settings_model_annotation(annotation):
        return make_partial_model(annotation)
    return annotation


def make_partial_model(model: type[SettingsModelT]) -> type[SettingsPatchModel]:
    cached = _PARTIAL_MODEL_CACHE.get(model)
    if cached is not None:
        return cached

    fields: dict[str, tuple[Any, Any]] = {}
    for field_name, field_info in model.model_fields.items():
        partial_annotation = _wrap_optional(_to_partial_annotation(field_info.annotation))
        fields[field_name] = (partial_annotation, Field(default=None))

    partial_model = create_model(
        f"{model.__name__}Patch",
        __base__=SettingsPatchModel,
        **cast(Any, fields),
    )

    _PARTIAL_MODEL_CACHE[model] = partial_model
    return partial_model


def _is_settings_model_annotation(annotation: object) -> TypeGuard[type[SettingsBaseModel]]:
    return isinstance(annotation, type) and issubclass(annotation, SettingsBaseModel)


AppSettingsPatch = make_partial_model(AppSettings)
UpdateSettingsRequest = AppSettingsPatch


class SettingsResponse(SettingsBaseModel):
    use_torch_compile: bool = False
    load_on_startup: bool = False
    has_ltx_api_key: bool = False
    user_prefers_ltx_api_video_generations: bool = False
    has_fal_api_key: bool = False
    use_local_text_encoder: bool = False
    fast_model: FastModelSettings = Field(default_factory=FastModelSettings)
    pro_model: ProModelSettings = Field(default_factory=ProModelSettings)
    custom_model: CustomModelSettings = Field(default_factory=CustomModelSettings)
    prompt_cache_size: int = 100
    prompt_enhancer_enabled_t2v: bool = True
    prompt_enhancer_enabled_i2v: bool = False
    has_gemini_api_key: bool = False
    seed_locked: bool = False
    locked_seed: int = 42
    models_dir: str = ""
    project_assets_dir: str = ""
    preferred_model_path: str = ""
    selected_loras: list[SelectedLoRASettings] = Field(default_factory=list)
    preferred_gguf_path: str = ""
    preferred_lora_path: str = ""
    preferred_lora_strength: float = 0.8
    num_blocks_to_swap: int = -1
    run_mode: RunModeLiteral = "auto"
    a2v_decode_tiling: A2VDecodeTilingLiteral = "auto"
    preferred_zit_model_path: str = ""
    preferred_text_encoder_path: str = ""


def to_settings_response(settings: AppSettings) -> SettingsResponse:
    data = settings.model_dump(by_alias=False)
    ltx_key = data.pop("ltx_api_key", "")
    fal_key = data.pop("fal_api_key", "")
    gemini_key = data.pop("gemini_api_key", "")
    data["has_ltx_api_key"] = bool(ltx_key)
    data["has_fal_api_key"] = bool(fal_key)
    data["has_gemini_api_key"] = bool(gemini_key)
    # models_dir passes through as-is (not secret)
    return SettingsResponse.model_validate(data)


def should_video_generate_with_ltx_api(*, force_api_generations: bool, settings: AppSettings) -> bool:
    del force_api_generations, settings
    return False
