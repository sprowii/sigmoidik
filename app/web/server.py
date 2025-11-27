# Copyright (c) 2025 sprouee
import json
import re
import secrets
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    request,
    send_from_directory,
    session,
)

from app import config
from app.game.generator import GeneratedGame, generate_game, tweak_game
from app.logging_config import log
from app.storage.redis_store import (
    consume_login_code,
    list_games_for_author,
    list_recent_games,
    load_game_payload,
    redis_client,
)

from telegram import Update
from telegram.ext import Application
import asyncio

BASE_DIR = Path(__file__).resolve().parents[2]
WEBAPP_DIR = BASE_DIR / "webapp"

flask_app = Flask(__name__, static_folder=str(WEBAPP_DIR), static_url_path="/webapp")
flask_app.config["SECRET_KEY"] = config.FLASK_SECRET_KEY
flask_app.config["SESSION_COOKIE_NAME"] = config.SESSION_COOKIE_NAME
flask_app.permanent_session_lifetime = timedelta(days=14)

# Защита от CSRF и других атак
flask_app.config["SESSION_COOKIE_SECURE"] = True  # Только HTTPS
flask_app.config["SESSION_COOKIE_HTTPONLY"] = True  # Защита от XSS
flask_app.config["SESSION_COOKIE_SAMESITE"] = "Lax"  # Защита от CSRF


@flask_app.after_request
def add_security_headers(response):
    """Добавляет security headers ко всем ответам."""
    # Защита от clickjacking
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    # Защита от MIME-type sniffing
    response.headers["X-Content-Type-Options"] = "nosniff"
    # XSS Protection (для старых браузеров)
    response.headers["X-XSS-Protection"] = "1; mode=block"
    # Referrer Policy
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # Permissions Policy (отключаем ненужные API)
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    # Content Security Policy
    # Разрешаем: свои скрипты/стили, Google Fonts, CDN для Three.js
    csp = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data: https:; "
        "connect-src 'self'; "
        "frame-src 'self'; "
        "frame-ancestors 'self'; "
        "base-uri 'self'; "
        "form-action 'self'"
    )
    response.headers["Content-Security-Policy"] = csp
    return response


application: Optional[Application] = None
main_loop: Optional[asyncio.AbstractEventLoop] = None


def set_application(ptb_app: Application, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
    """Set the PTB application instance for webhook handling."""
    global application, main_loop
    application = ptb_app
    main_loop = loop


def _make_share_url(game_id: Optional[str]) -> Optional[str]:
    if not game_id:
        return None
    base = config.WEBAPP_BASE_URL
    if not base:
        return None
    return f"{base}/webapp/sandbox.html?game_id={game_id}"


def _serialize_game(payload: Dict[str, Any]) -> Dict[str, Any]:
    game_id = payload.get("id") or payload.get("game_id")
    return {
        "id": game_id,
        "title": payload.get("title"),
        "summary": payload.get("summary"),
        "model": payload.get("model"),
        "idea": payload.get("idea"),
        "created_at": payload.get("created_at"),
        "ttl": payload.get("ttl"),
        "revision": payload.get("revision", 1),
        "parent_id": payload.get("parent_id"),
        "author": {
            "id": payload.get("author_id"),
            "username": payload.get("author_username"),
            "name": payload.get("author_name"),
        },
        "share_url": _make_share_url(game_id),
    }


def _is_admin_id(raw_id: Any) -> bool:
    if raw_id is None or not config.ADMIN_ID:
        return False
    try:
        # Используем constant-time comparison для защиты от timing attacks
        return secrets.compare_digest(str(raw_id), str(config.ADMIN_ID))
    except Exception:
        return False


def _current_user() -> Optional[Dict[str, Any]]:
    user = session.get("user")
    if not user:
        return None
    user_id = user.get("user_id")
    return {
        "id": user_id,
        "username": user.get("username"),
        "name": user.get("display_name"),
        "chat_id": user.get("chat_id"),
        "is_admin": _is_admin_id(user_id),
    }


def _normalize_provider_choice(raw: Any) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    value = raw.strip().lower()
    if value in {"", "auto", "default"}:
        return None
    if value == "gemini" and config.API_KEYS:
        return "gemini"
    if value == "openrouter" and config.OPENROUTER_API_KEYS and config.OPENROUTER_MODELS:
        return "openrouter"
    if value == "pollinations" and getattr(config, "POLLINATIONS_TEXT_MODELS", None):
        return "pollinations"
    return None


def _normalize_pollinations_text_model(raw: Any) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    if not value:
        return None
    if value in config.POLLINATIONS_TEXT_MODELS:
        return value
    lowered_lookup = {model.lower(): model for model in config.POLLINATIONS_TEXT_MODELS}
    return lowered_lookup.get(value.lower())
    return None


def _require_user() -> Dict[str, Any]:
    user = _current_user()
    if not user:
        _abort_json(401, "Нужен код от бота, чтобы создавать и редактировать игры.", "auth_required")
    return user


def _abort_json(status_code: int, message: str, error_code: str) -> None:
    response = jsonify({"error": error_code, "message": message})
    response.status_code = status_code
    abort(response)


def _ensure_can_tweak(payload: Dict[str, Any], user: Dict[str, Any]) -> None:
    if user.get("is_admin"):
        return

    author_id = payload.get("author_id")
    if author_id is None:
        _abort_json(
            403,
            "Эта игра была создана без привязки к автору, поэтому доработки через веб недоступны.",
            "forbidden",
        )

    try:
        author_id_int = int(author_id)
    except (TypeError, ValueError):
        _abort_json(403, "Не удалось определить автора игры.", "forbidden")
        return

    user_id = user.get("id")
    try:
        user_id_int = int(user_id) if user_id is not None else None
    except (TypeError, ValueError):
        user_id_int = None

    if user_id_int is None or user_id_int != author_id_int:
        _abort_json(403, "Только автор игры может просить доработку.", "forbidden")


def _serialize_generated(game: GeneratedGame) -> Dict[str, Any]:
    return {
        "id": game.game_id,
        "title": game.title,
        "summary": game.summary,
        "model": game.model,
        "idea": game.idea,
        "created_at": game.created_at,
        "ttl": config.GAME_TTL_SECONDS,
        "revision": game.revision,
        "parent_id": game.parent_id,
        "author": {
            "id": game.author_id,
            "username": game.author_username,
            "name": game.author_name,
        },
        "share_url": game.share_url,
    }


def _ensure_web_chat_id() -> int:
    raw_value = session.get("web_chat_id")
    if raw_value is None:
        # отрицательные ids, чтобы не пересекаться с Telegram
        raw_value = -int(secrets.randbits(31))
        session["web_chat_id"] = raw_value
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        new_value = -int(secrets.randbits(31))
        session["web_chat_id"] = new_value
        return new_value


def _generation_context() -> Dict[str, Any]:
    user = _current_user()
    chat_id = None
    author_id: Optional[int] = None
    author_username: Optional[str] = None
    author_name: Optional[str] = None
    if user:
        author_username = user.get("username")
        author_name = user.get("name")
        chat_id = user.get("chat_id")
        try:
            if user.get("id") is not None:
                author_id = int(user["id"])
        except (TypeError, ValueError):
            author_id = None
    if chat_id is None:
        chat_id = _ensure_web_chat_id()
    return {
        "chat_id": chat_id,
        "author_id": author_id,
        "author_username": author_username,
        "author_name": author_name,
        "provider": session.get("llm_provider"),
        "pollinations_model": session.get("pollinations_text_model"),
    }


@flask_app.route("/")
def home():
    if not WEBAPP_DIR.exists():
        abort(404)
    return send_from_directory(flask_app.static_folder, "hub.html")


@flask_app.route("/admin/download/history")
def download_history():
    provided_key = request.args.get("key")
    # Используем constant-time comparison для защиты от timing attacks
    if not config.DOWNLOAD_KEY:
        abort(403)
    if not secrets.compare_digest(str(provided_key or ""), str(config.DOWNLOAD_KEY)):
        abort(403)
    try:
        history_snapshot: Dict[str, Any] = {}
        for key in redis_client.scan_iter(match=f"{config.HISTORY_KEY_PREFIX}*"):
            chat_id = key.split(":", 1)[1]
            raw_value = redis_client.get(key)
            if raw_value:
                history_snapshot[chat_id] = json.loads(raw_value)

        users_snapshot: Dict[str, Any] = {}
        for key in redis_client.scan_iter(match=f"{config.USER_KEY_PREFIX}*"):
            chat_id = key.split(":", 1)[1]
            raw_value = redis_client.get(key)
            if raw_value:
                users_snapshot[chat_id] = json.loads(raw_value)

        response_payload = {
            "history": history_snapshot,
            "users": users_snapshot,
        }
        response = Response(json.dumps(response_payload, ensure_ascii=False, indent=2), mimetype="application/json")
        response.headers["Content-Disposition"] = "attachment; filename=history.json"
        return response
    except Exception as exc:
        log.error(f"Не удалось выгрузить историю из Redis: {exc}", exc_info=True)
        abort(500)


@flask_app.route("/webapp/sandbox")
def sandbox_entrypoint():
    if not WEBAPP_DIR.exists():
        abort(404)
    return send_from_directory(flask_app.static_folder, "sandbox.html")


@flask_app.route("/webapp/hub")
def hub_entrypoint():
    if not WEBAPP_DIR.exists():
        abort(404)
    return send_from_directory(flask_app.static_folder, "hub.html")


@flask_app.route("/api/games/<string:game_id>")
def fetch_game(game_id: str):
    # Валидация game_id для предотвращения path traversal
    if not re.match(r'^[a-f0-9]{32}$', game_id):
        abort(400, description="Некорректный формат game_id")
    payload = load_game_payload(game_id)
    if not payload:
        abort(404)

    return jsonify(
        {
            "id": payload.get("id", game_id),
            "title": payload.get("title"),
            "summary": payload.get("summary"),
            "code": payload.get("code"),
            "model": payload.get("model"),
            "idea": payload.get("idea"),
            "created_at": payload.get("created_at"),
            "revision": payload.get("revision", 1),
            "parent_id": payload.get("parent_id"),
            "share_url": _make_share_url(payload.get("id", game_id)),
        }
    )


@flask_app.route("/api/auth/session", methods=["GET"])
def auth_session():
    user = _current_user()
    if not user:
        return jsonify(
            {
                "authenticated": False,
                "provider": "auto",
                "pollinations_model": config.POLLINATIONS_TEXT_DEFAULT,
            }
        )
    provider = session.get("llm_provider") or "auto"
    poll_model = session.get("pollinations_text_model") or config.POLLINATIONS_TEXT_DEFAULT
    return jsonify({"authenticated": True, "user": user, "provider": provider, "pollinations_model": poll_model})


@flask_app.route("/api/auth/login", methods=["POST"])
def auth_login():
    # Rate limiting для защиты от brute-force атак
    from app.middleware.rate_limit import check_login_rate_limit
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    if client_ip:
        client_ip = client_ip.split(",")[0].strip()  # Берём первый IP из цепочки
    allowed, message = check_login_rate_limit(client_ip or "unknown")
    if not allowed:
        return jsonify({"error": "rate_limit", "message": message}), 429
    
    data = request.get_json(silent=True) or {}
    code = (data.get("code") or "").strip().upper()
    if not code:
        abort(400, description="Код обязателен.")
    decoded = consume_login_code(code)
    if not decoded:
        # Логируем неудачную попытку входа для мониторинга
        log.warning(f"Failed login attempt from IP {client_ip} with code: {code[:2]}***")
        abort(400, description="Код недействителен или истёк.")
    session["user"] = {
        "user_id": decoded.get("user_id"),
        "username": decoded.get("username"),
        "display_name": decoded.get("display_name"),
        "chat_id": decoded.get("chat_id"),
    }
    session.permanent = True
    user = _current_user()
    provider = session.get("llm_provider") or "auto"
    poll_model = session.get("pollinations_text_model") or config.POLLINATIONS_TEXT_DEFAULT
    return jsonify({"authenticated": True, "user": user, "provider": provider, "pollinations_model": poll_model})


@flask_app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    session.pop("user", None)
    return jsonify({"authenticated": False})


@flask_app.route("/api/models", methods=["GET"])
def list_models_meta():
    providers: List[str] = []
    if config.API_KEYS:
        providers.append("gemini")
    if config.OPENROUTER_API_KEYS and config.OPENROUTER_MODELS:
        providers.append("openrouter")
    if getattr(config, "POLLINATIONS_TEXT_MODELS", None):
        providers.append("pollinations")
    return jsonify(
        {
            "providers": providers,
            "pollinations_text_models": config.POLLINATIONS_TEXT_MODELS,
            "pollinations_text_default": config.POLLINATIONS_TEXT_DEFAULT,
            "openrouter_models": config.OPENROUTER_MODELS,
        }
    )


@flask_app.route("/api/games", methods=["GET"])
def list_games_api():
    limit_param = request.args.get("limit", "20")
    offset_param = request.args.get("offset", "0")
    scope = request.args.get("scope")
    author_param = request.args.get("author_id")
    try:
        limit = max(1, min(int(limit_param), 50))
    except ValueError:
        limit = 20
    try:
        offset = max(0, int(offset_param))
    except ValueError:
        offset = 0

    if scope == "mine":
        user = _current_user()
        if not user or not user.get("id"):
            abort(401)
        games = list_games_for_author(int(user["id"]), limit=limit, offset=offset)
    elif author_param:
        try:
            author_id = int(author_param)
        except ValueError:
            abort(400, description="Некорректный author_id.")
        games = list_games_for_author(author_id, limit=limit, offset=offset)
    else:
        games = list_recent_games(limit=limit, offset=offset)

    serialized = [_serialize_game(item) for item in games]
    return jsonify({"items": serialized, "limit": limit, "offset": offset})


@flask_app.route("/api/games", methods=["POST"])
def create_game_api():
    # Rate limiting для защиты от DoS через генерацию
    from app.middleware.rate_limit import check_web_rate_limit
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    if client_ip:
        client_ip = client_ip.split(",")[0].strip()
    allowed, message = check_web_rate_limit(client_ip or "unknown")
    if not allowed:
        return jsonify({"error": "rate_limit", "message": message}), 429
    
    _require_user()
    data = request.get_json(silent=True) or {}
    idea = (data.get("idea") or "").strip()
    if len(idea) < 4:
        _abort_json(400, "Опиши идею игры чуть подробнее.", "invalid_request")
    # Ограничение длины для предотвращения DoS
    if len(idea) > 5000:
        _abort_json(400, "Описание игры слишком длинное (макс. 5000 символов).", "invalid_request")
    provider_raw = data.get("provider")
    provider_choice: Optional[str] = None
    if isinstance(provider_raw, str):
        normalized = provider_raw.strip().lower()
        provider_choice = _normalize_provider_choice(provider_raw)
        if normalized in {"auto", "default", ""}:
            session.pop("llm_provider", None)
        elif provider_choice:
            session["llm_provider"] = provider_choice
    pollinations_model = _normalize_pollinations_text_model(data.get("pollinations_model"))
    if pollinations_model:
        session["pollinations_text_model"] = pollinations_model
    elif provider_choice == "pollinations" and "pollinations_text_model" not in session:
        session["pollinations_text_model"] = config.POLLINATIONS_TEXT_DEFAULT
    context = _generation_context()
    try:
        generated = generate_game(
            chat_id=context["chat_id"],
            idea=idea,
            author_id=context["author_id"],
            author_username=context["author_username"],
            author_name=context["author_name"],
            provider=context.get("provider"),
            pollinations_model=context.get("pollinations_model"),
        )
    except ValueError as exc:
        _abort_json(400, str(exc), "invalid_request")
    except Exception as exc:
        log.error("Не удалось сгенерировать игру через веб: %s", exc, exc_info=True)
        _abort_json(500, "Ошибка генерации игры. Попробуйте позже.", "server_error")

    return jsonify({"game": _serialize_generated(generated)}), 201


@flask_app.route("/api/games/<string:game_id>/tweak", methods=["POST"])
def tweak_game_api(game_id: str):
    # Валидация game_id для предотвращения path traversal
    if not re.match(r'^[a-f0-9]{32}$', game_id):
        abort(400, description="Некорректный формат game_id")
    payload = load_game_payload(game_id)
    if not payload:
        abort(404)
    user = _require_user()
    _ensure_can_tweak(payload, user)
    data = request.get_json(silent=True) or {}
    instructions = (data.get("instructions") or "").strip()
    if len(instructions) < 4:
        _abort_json(400, "Опиши, что нужно изменить хотя бы в нескольких словах.", "invalid_request")
    # Ограничение длины для предотвращения DoS
    if len(instructions) > 5000:
        _abort_json(400, "Описание изменений слишком длинное (макс. 5000 символов).", "invalid_request")
    provider_raw = data.get("provider")
    provider_choice: Optional[str] = None
    if isinstance(provider_raw, str):
        normalized = provider_raw.strip().lower()
        provider_choice = _normalize_provider_choice(provider_raw)
        if normalized in {"auto", "default", ""}:
            session.pop("llm_provider", None)
        elif provider_choice:
            session["llm_provider"] = provider_choice
    pollinations_model = _normalize_pollinations_text_model(data.get("pollinations_model"))
    if pollinations_model:
        session["pollinations_text_model"] = pollinations_model
    elif provider_choice == "pollinations" and "pollinations_text_model" not in session:
        session["pollinations_text_model"] = config.POLLINATIONS_TEXT_DEFAULT
    context = _generation_context()
    try:
        generated = tweak_game(
            payload,
            instructions=instructions,
            chat_id=context["chat_id"],
            author_id=context["author_id"],
            author_username=context["author_username"],
            author_name=context["author_name"],
            provider=context.get("provider"),
            pollinations_model=context.get("pollinations_model"),
        )
    except ValueError as exc:
        _abort_json(400, str(exc), "invalid_request")
    except Exception as exc:
        log.error("Не удалось обновить игру %s: %s", game_id, exc, exc_info=True)
        _abort_json(500, "Ошибка обновления игры. Попробуйте позже.", "server_error")

    return jsonify({"game": _serialize_generated(generated)}), 201


@flask_app.route("/telegram-webhook", methods=["POST"])
def telegram_webhook():
    """Handle Telegram webhook updates via Flask."""
    if application is None:
        return Response("Application not ready", status=503)
    
    # Проверяем секретный токен для защиты от поддельных запросов
    secret_token = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
    if not secret_token or not secrets.compare_digest(secret_token, config.WEBHOOK_SECRET_TOKEN):
        log.warning("Webhook request with invalid or missing secret token")
        return Response("Unauthorized", status=401)
    
    json_string = request.get_data().decode("utf-8")
    update = Update.de_json(json.loads(json_string), application.bot)
    if update:
        try:
            # Flask работает в другом потоке, используем run_coroutine_threadsafe
            # чтобы выполнить корутину в main event loop где живёт PTB Application
            if main_loop is not None and main_loop.is_running():
                future = asyncio.run_coroutine_threadsafe(application.process_update(update), main_loop)
                # Увеличиваем timeout для обработки видео (может занять до 2 минут)
                future.result(timeout=120)
            else:
                log.error("Main event loop not available or not running")
                return Response("Event loop not ready", status=503)
        except asyncio.TimeoutError:
            log.error("Webhook processing timeout (120s exceeded)")
            # Возвращаем OK чтобы Telegram не повторял запрос
            return Response("OK", status=200)
        except Exception as exc:
            log.error(f"Error processing Telegram update: {exc}", exc_info=True)
    return Response("OK", status=200)
