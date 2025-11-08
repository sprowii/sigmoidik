# Copyright (c) 2025 sprouee
import json
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

BASE_DIR = Path(__file__).resolve().parents[2]
WEBAPP_DIR = BASE_DIR / "webapp"

flask_app = Flask(__name__, static_folder=str(WEBAPP_DIR), static_url_path="/webapp")
flask_app.config["SECRET_KEY"] = config.FLASK_SECRET_KEY
flask_app.config["SESSION_COOKIE_NAME"] = config.SESSION_COOKIE_NAME
flask_app.permanent_session_lifetime = timedelta(days=14)


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


def _current_user() -> Optional[Dict[str, Any]]:
    user = session.get("user")
    if not user:
        return None
    return {
        "id": user.get("user_id"),
        "username": user.get("username"),
        "name": user.get("display_name"),
        "chat_id": user.get("chat_id"),
    }


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
    }


@flask_app.route("/")
def home():
    if not WEBAPP_DIR.exists():
        abort(404)
    return send_from_directory(flask_app.static_folder, "hub.html")


@flask_app.route("/admin/download/history")
def download_history():
    provided_key = request.args.get("key")
    if not config.DOWNLOAD_KEY or provided_key != config.DOWNLOAD_KEY:
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
        return jsonify({"authenticated": False})
    return jsonify({"authenticated": True, "user": user})


@flask_app.route("/api/auth/login", methods=["POST"])
def auth_login():
    data = request.get_json(silent=True) or {}
    code = (data.get("code") or "").strip().upper()
    if not code:
        abort(400, description="Код обязателен.")
    decoded = consume_login_code(code)
    if not decoded:
        abort(400, description="Код недействителен или истёк.")
    session["user"] = {
        "user_id": decoded.get("user_id"),
        "username": decoded.get("username"),
        "display_name": decoded.get("display_name"),
        "chat_id": decoded.get("chat_id"),
    }
    session.permanent = True
    user = _current_user()
    return jsonify({"authenticated": True, "user": user})


@flask_app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    session.pop("user", None)
    return jsonify({"authenticated": False})


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
    data = request.get_json(silent=True) or {}
    idea = (data.get("idea") or "").strip()
    if len(idea) < 4:
        abort(400, description="Опиши идею игры чуть подробнее.")
    context = _generation_context()
    try:
        generated = generate_game(
            chat_id=context["chat_id"],
            idea=idea,
            author_id=context["author_id"],
            author_username=context["author_username"],
            author_name=context["author_name"],
        )
    except ValueError as exc:
        abort(400, description=str(exc))
    except Exception as exc:
        log.error("Не удалось сгенерировать игру через веб: %s", exc, exc_info=True)
        abort(500, description="Ошибка генерации игры. Попробуйте позже.")

    return jsonify({"game": _serialize_generated(generated)}), 201


@flask_app.route("/api/games/<string:game_id>/tweak", methods=["POST"])
def tweak_game_api(game_id: str):
    payload = load_game_payload(game_id)
    if not payload:
        abort(404)
    data = request.get_json(silent=True) or {}
    instructions = (data.get("instructions") or "").strip()
    context = _generation_context()
    try:
        generated = tweak_game(
            payload,
            instructions=instructions,
            chat_id=context["chat_id"],
            author_id=context["author_id"],
            author_username=context["author_username"],
            author_name=context["author_name"],
        )
    except ValueError as exc:
        abort(400, description=str(exc))
    except Exception as exc:
        log.error("Не удалось обновить игру %s: %s", game_id, exc, exc_info=True)
        abort(500, description="Ошибка обновления игры. Попробуйте позже.")

    return jsonify({"game": _serialize_generated(generated)}), 201
