# Copyright (c) 2025 sprouee
"""Utilities that orchestrate AI-driven generation of Phaser games."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app import config
from app.llm.client import llm_request
from app.logging_config import log
from app.storage.redis_store import store_game_payload
from app.state import ChatConfig, configs

PROMPT_TEMPLATE = (
    """
    Ты — генератор игр на Phaser 3, работающий в песочнице Telegram WebApps.
    Код игры выполняется как `(async function(Phaser, sandbox) {{ ... }})`.

    Требования к выходному коду:
    1. Используй чистый JavaScript (ES6). Без `import`, `export`, `require` и внешних HTML/CSS.
    2. Не используй `eval`, `new Function`, `document.cookie`, `localStorage`, `sessionStorage`, `window.parent/top/opener`.
    3. Инициализируй игру через `new Phaser.Game({{...}})`, обязательно укажи `parent: sandbox.getContainer()`.
    4. Информируй игрока с помощью `sandbox.setStatus(...)` во время загрузки ресурсов и `sandbox.clearStatus()` после готовности.
    5. Можно использовать встроенные графические примитивы Phaser или публичные HTTPS-ресурсы.
    6. Финальный код должен быть полностью самодостаточным и готовым к выполнению.

    Тебе дана идея игры: «{idea}».

    Ответь JSON-объектом без дополнительных комментариев и форматирования Markdown.
    Структура JSON:
    {{
      "title": "Короткое название игры",
      "summary": "Краткое описание механики",
      "code": "Полный JavaScript-код игры"
    }}

    Значение "code" должно быть строкой с экранированными переводами строк (используй `\n`).
    Не добавляй ничего кроме указанного JSON.
    """
)

TWEAK_PROMPT_TEMPLATE = (
    """
    Ты — генератор и редактор игр на Phaser 3.
    Тебе дан исходный код игры и запрос пользователя на изменения.

    Исходная идея:
    «{idea}»

    Пояснение/описание:
    «{summary}»

    Текущий код:
    ```javascript
    {code}
    ```

    Запрос пользователя:
    «{instructions}»

    Требования:
    1. Верни обновлённый JavaScript-код игры, полностью готовый к запуску.
    2. Сохрани рабочую игру (никаких синтаксических ошибок).
    3. Не используй eval, new Function и доступ к чувствительным API.
    4. Не удаляй parent: sandbox.getContainer() и не меняй базовую инициализацию Phaser.

    Ответь JSON-объектом:
    {{
      "title": "Название (можно оставить прежним)",
      "summary": "Краткое описание изменений",
      "code": "Обновлённый JavaScript-код"
    }}
    """
)

JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
CODE_BLOCK_PATTERN = re.compile(r"```[a-zA-Z0-9]*\n|```")

_NODE_CHECK_SUPPORTED: Optional[bool] = None


@dataclass
class GeneratedGame:
    game_id: str
    title: str
    summary: str
    code: str
    idea: str
    model: str
    share_url: Optional[str]
    author_id: Optional[int]
    author_username: Optional[str]
    author_name: Optional[str]
    created_at: float
    parent_id: Optional[str] = None
    revision: int = 1


def _extract_json(payload: str) -> Dict[str, Any]:
    payload = payload.strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        match = JSON_PATTERN.search(payload)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                log.debug("JSON parse failed on extracted block: %s", exc)
    raise ValueError("Модель вернула ответ, который невозможно преобразовать в JSON.")


def _cleanup_code(code: str) -> str:
    cleaned = CODE_BLOCK_PATTERN.sub("", code or "").strip()
    return cleaned


def _escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}").strip()


def _is_node_available() -> bool:
    global _NODE_CHECK_SUPPORTED
    if _NODE_CHECK_SUPPORTED is not None:
        return _NODE_CHECK_SUPPORTED
    try:
        subprocess.run(["node", "--version"], capture_output=True, text=True, timeout=5, check=False)
    except FileNotFoundError:
        log.warning("Node.js не найден, синтаксическая проверка игр отключена.")
        _NODE_CHECK_SUPPORTED = False
        return False
    except Exception as exc:
        log.warning("Не удалось выполнить проверку наличия Node.js: %s", exc)
        _NODE_CHECK_SUPPORTED = False
        return False
    _NODE_CHECK_SUPPORTED = True
    return True


def _sanitize_js_error(raw_error: str) -> str:
    lines = [line.strip() for line in (raw_error or "").splitlines() if line.strip()]
    for line in lines:
        if any(keyword in line for keyword in ("SyntaxError", "ReferenceError", "TypeError")):
            return line
    if lines:
        return lines[-1]
    return "Код игры содержит синтаксическую ошибку."


def _validate_js_code(code: str) -> Optional[str]:
    if not code or not code.strip():
        return "Код игры пустой."
    if not _is_node_available():
        return None
    tmp_path = None
    result = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as tmp_file:
            tmp_file.write(code)
            tmp_path = tmp_file.name
        result = subprocess.run(
            ["node", "--check", tmp_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        log.warning("Синтаксическая проверка игры превысила допустимое время.")
        return "Проверка синтаксиса превысила лимит времени."
    except Exception as exc:
        log.error("Не удалось выполнить синтаксическую проверку игры: %s", exc, exc_info=True)
        return None
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    if result and result.returncode != 0:
        error_output = (result.stderr or result.stdout or "").strip()
        sanitized = _sanitize_js_error(error_output)
        log.warning("Синтаксическая проверка игры не пройдена: %s", sanitized or error_output)
        return sanitized or "Код игры содержит синтаксическую ошибку."
    return None


def _ensure_code_is_valid(code: str) -> None:
    validation_error = _validate_js_code(code)
    if validation_error:
        raise ValueError(
            f"Код игры не прошёл синтаксическую проверку: {validation_error}. "
            "Попробуйте уточнить запрос и повторить генерацию."
        )


def _build_prompt(idea: str) -> str:
    safe_idea = _escape_braces(idea)
    return PROMPT_TEMPLATE.format(idea=safe_idea)


def _build_tweak_prompt(idea: str, summary: str, code: str, instructions: str) -> str:
    return TWEAK_PROMPT_TEMPLATE.format(
        idea=_escape_braces(idea or "нет описания"),
        summary=_escape_braces(summary or "нет описания"),
        code=code,
        instructions=_escape_braces(instructions),
    )


def _build_share_url(game_id: str) -> Optional[str]:
    base = config.WEBAPP_BASE_URL
    if not base:
        return None
    return f"{base}/webapp/sandbox.html?game_id={game_id}"


def _normalize_chat_id(chat_id: Optional[int]) -> int:
    if chat_id is None:
        return 0
    try:
        return int(chat_id)
    except (TypeError, ValueError):
        return 0


def _resolve_provider(
    chat_id: int,
    provider: Optional[str],
    pollinations_model: Optional[str] = None,
) -> Optional[str]:
    cfg = configs.setdefault(chat_id, ChatConfig())
    normalized = (provider or "").strip().lower() if provider else ""

    if normalized in {"", "auto", "default"}:
        cfg.llm_provider = ""
        if pollinations_model and pollinations_model in config.POLLINATIONS_TEXT_MODELS:
            cfg.pollinations_text_model = pollinations_model
        return None

    if normalized in {"gemini", "openrouter", "pollinations"}:
        cfg.llm_provider = normalized
        if normalized == "pollinations" and pollinations_model in config.POLLINATIONS_TEXT_MODELS:
            cfg.pollinations_text_model = pollinations_model
        return normalized

    if pollinations_model and pollinations_model in config.POLLINATIONS_TEXT_MODELS:
        cfg.pollinations_text_model = pollinations_model

    return cfg.llm_provider or None


def generate_game(
    chat_id: Optional[int],
    idea: str,
    author_id: Optional[int] = None,
    author_username: Optional[str] = None,
    author_name: Optional[str] = None,
    provider: Optional[str] = None,
    pollinations_model: Optional[str] = None,
) -> GeneratedGame:
    if not idea or not idea.strip():
        raise ValueError("Описание игры не должно быть пустым.")

    prompt = _build_prompt(idea)
    normalized_chat_id = _normalize_chat_id(chat_id)
    provider_override = _resolve_provider(normalized_chat_id, provider, pollinations_model)
    response, model_name, _ = llm_request(normalized_chat_id, [{"text": prompt}], provider_override)
    if not response:
        raise RuntimeError("Модель не вернула код игры.")

    parsed = _extract_json(response)
    code = _cleanup_code(parsed.get("code", ""))
    if not code:
        raise ValueError("Модель не вернула JavaScript-код игры.")

    _ensure_code_is_valid(code)

    title = (parsed.get("title") or "Игра от Сигмоиды").strip()
    summary = (parsed.get("summary") or "").strip()

    game_id = uuid.uuid4().hex
    created_at = time.time()
    payload = {
        "id": game_id,
        "title": title,
        "summary": summary,
        "code": code,
        "idea": idea.strip(),
        "model": model_name,
        "created_at": created_at,
        "ttl": config.GAME_TTL_SECONDS,
        "chat_id": chat_id,
        "author_id": author_id,
        "author_username": author_username,
        "author_name": author_name,
        "parent_id": None,
        "revision": 1,
    }
    store_game_payload(game_id, payload)

    share_url = _build_share_url(game_id)

    log.info("Сгенерирована игра %s моделью %s", game_id, model_name)
    return GeneratedGame(
        game_id=game_id,
        title=title,
        summary=summary,
        code=code,
        idea=idea.strip(),
        model=model_name,
        share_url=share_url,
        author_id=author_id,
        author_username=author_username,
        author_name=author_name,
        created_at=created_at,
        parent_id=None,
        revision=1,
    )


def tweak_game(
    payload: Dict[str, Any],
    instructions: str,
    chat_id: Optional[int],
    author_id: Optional[int] = None,
    author_username: Optional[str] = None,
    author_name: Optional[str] = None,
    provider: Optional[str] = None,
    pollinations_model: Optional[str] = None,
) -> GeneratedGame:
    if not instructions or not instructions.strip():
        raise ValueError("Описание изменений не должно быть пустым.")

    base_code = payload.get("code", "")
    if not isinstance(base_code, str) or not base_code.strip():
        raise ValueError("Нельзя обновить игру без исходного кода.")

    base_idea = payload.get("idea", "")
    base_summary = payload.get("summary", "")

    prompt = _build_tweak_prompt(base_idea, base_summary, base_code, instructions)
    normalized_chat_id = _normalize_chat_id(chat_id)
    provider_override = _resolve_provider(normalized_chat_id, provider, pollinations_model)
    response, model_name, _ = llm_request(normalized_chat_id, [{"text": prompt}], provider_override)
    if not response:
        raise RuntimeError("Модель не вернула обновлённый код игры.")

    parsed = _extract_json(response)
    code = _cleanup_code(parsed.get("code", ""))
    if not code:
        raise ValueError("Модель не вернула JavaScript-код игры.")

    _ensure_code_is_valid(code)

    title = (parsed.get("title") or payload.get("title") or "Игра от Сигмоиды").strip()
    summary = (parsed.get("summary") or "").strip()

    created_at = time.time()
    game_id = uuid.uuid4().hex
    revision = int(payload.get("revision") or 1) + 1

    new_payload = {
        "id": game_id,
        "title": title,
        "summary": summary,
        "code": code,
        "idea": base_idea,
        "model": model_name,
        "created_at": created_at,
        "ttl": config.GAME_TTL_SECONDS,
        "chat_id": chat_id,
        "author_id": author_id,
        "author_username": author_username,
        "author_name": author_name,
        "parent_id": payload.get("id"),
        "revision": revision,
        "instructions": instructions.strip(),
    }
    store_game_payload(game_id, new_payload)

    share_url = _build_share_url(game_id)
    log.info("Создана новая версия игры %s -> %s", payload.get("id"), game_id)
    return GeneratedGame(
        game_id=game_id,
        title=title,
        summary=summary,
        code=code,
        idea=base_idea,
        model=model_name,
        share_url=share_url,
        author_id=author_id,
        author_username=author_username,
        author_name=author_name,
        created_at=created_at,
        parent_id=payload.get("id"),
        revision=revision,
    )
