"""FastAPI app factory decoupled from runtime bootstrap side effects."""

from __future__ import annotations

import base64
import hmac
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.responses import Response as StarletteResponse

from _routes._errors import HTTPError
from _routes.generation import router as generation_router
from _routes.health import router as health_router
from _routes.ic_lora import router as ic_lora_router
from _routes.image_gen import router as image_gen_router
from _routes.models import router as models_router
from _routes.suggest_gap_prompt import router as suggest_gap_prompt_router
from _routes.retake import router as retake_router
from _routes.runtime_policy import router as runtime_policy_router
from _routes.settings import router as settings_router
from _routes.web_routes import configure_allowed_paths, router as web_router
from _routes.gpu import router as gpu_router
from logging_policy import log_http_error, log_unhandled_exception
from state import get_state_service, init_state_service

if TYPE_CHECKING:
    from app_handler import AppHandler

import os

DEFAULT_ALLOWED_ORIGINS: list[str] = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def _get_allowed_origins() -> list[str]:
    env_origins = os.environ.get("CORS_ORIGINS", "")
    if env_origins == "*":
        return ["*"]
    if env_origins:
        return [o.strip() for o in env_origins.split(",") if o.strip()]
    return DEFAULT_ALLOWED_ORIGINS


def create_app(
    *,
    handler: "AppHandler",
    allowed_origins: list[str] | None = None,
    title: str = "LTX-2 Video Generation Server",
    auth_token: str = "",
    admin_token: str = "",
) -> FastAPI:
    """Create a configured FastAPI app bound to the provided handler."""
    init_state_service(handler)

    app = FastAPI(title=title)
    app.state.admin_token = admin_token  # type: ignore[attr-defined]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins or _get_allowed_origins(),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _auth_middleware(  # pyright: ignore[reportUnusedFunction]
        request: Request,
        call_next: Callable[[Request], Awaitable[StarletteResponse]],
    ) -> StarletteResponse:
        if not auth_token:
            return await call_next(request)
        if request.method == "OPTIONS":
            return await call_next(request)

        def _token_matches(candidate: str) -> bool:
            return hmac.compare_digest(candidate, auth_token)

        # WebSocket: check query param
        if request.headers.get("upgrade", "").lower() == "websocket":
            if _token_matches(request.query_params.get("token", "")):
                return await call_next(request)
            return JSONResponse(status_code=401, content={"error": "Unauthorized"})
        # HTTP: Bearer or Basic auth
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer ") and _token_matches(auth_header[7:]):
            return await call_next(request)
        if auth_header.startswith("Basic "):
            try:
                decoded = base64.b64decode(auth_header[6:]).decode()
                _, _, password = decoded.partition(":")
                if _token_matches(password):
                    return await call_next(request)
            except Exception:
                pass
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    _FALLBACK = "An unexpected error occurred"

    async def _route_http_error_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        if isinstance(exc, HTTPError):
            log_http_error(request, exc)
            return JSONResponse(
                status_code=exc.status_code, content={"error": exc.detail or _FALLBACK}
            )
        return JSONResponse(status_code=500, content={"error": str(exc) or _FALLBACK})

    async def _validation_error_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        if isinstance(exc, RequestValidationError):
            return JSONResponse(
                status_code=422, content={"error": str(exc) or _FALLBACK}
            )
        return JSONResponse(status_code=422, content={"error": str(exc) or _FALLBACK})

    async def _route_generic_error_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        log_unhandled_exception(request, exc)
        return JSONResponse(status_code=500, content={"error": str(exc) or _FALLBACK})

    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(HTTPError, _route_http_error_handler)
    app.add_exception_handler(Exception, _route_generic_error_handler)

    app.include_router(health_router)
    app.include_router(generation_router)
    app.include_router(models_router)
    app.include_router(settings_router)
    app.include_router(image_gen_router)
    app.include_router(suggest_gap_prompt_router)
    app.include_router(retake_router)
    app.include_router(ic_lora_router)
    app.include_router(runtime_policy_router)
    app.include_router(gpu_router)
    app.include_router(web_router)

    # Configure the web shim's filesystem allowlist and expose generated media.
    # These routes are still protected by the app's auth middleware when auth is enabled,
    # and the browser frontend depends on them for loading generated images/videos.
    outputs_dir = handler.config.outputs_dir
    app_data_dir = outputs_dir.parent
    configure_allowed_paths(
        models_dir=str(handler.config.default_models_dir),
        outputs_dir=str(outputs_dir),
        app_data_dir=str(app_data_dir),
    )

    from starlette.staticfiles import StaticFiles

    if Path(outputs_dir).exists():
        app.mount(
            "/serve/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs"
        )
    if Path(app_data_dir).exists():
        app.mount(
            "/serve/files", StaticFiles(directory=str(app_data_dir)), name="appdata"
        )

    @app.get("/serve/project-assets/{asset_path:path}")
    def serve_project_asset(
        asset_path: str,
        state_handler: "AppHandler" = Depends(get_state_service),
    ) -> FileResponse:
        configured = (
            state_handler.settings.get_settings_snapshot().project_assets_dir.strip()
        )
        project_assets_dir = (
            Path(configured).expanduser().resolve()
            if configured
            else state_handler.config.outputs_dir.parent.joinpath(
                "project-assets"
            ).resolve()
        )
        target = (project_assets_dir / asset_path).resolve()
        if not target.is_relative_to(project_assets_dir):
            raise HTTPException(status_code=404, detail="Not found")
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="Not found")
        return FileResponse(target)

    return app
