"""Utilities that orchestrate AI-driven generation of Phaser games."""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app import config
from app.llm.client import llm_request
from app.logging_config import log
from app.storage.redis_store import store_game_payload

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

JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
CODE_BLOCK_PATTERN = re.compile(r"```[a-zA-Z0-9]*\n|```")


@dataclass
class GeneratedGame:
    game_id: str
    title: str
    summary: str
    code: str
    idea: str
    model: str
    share_url: Optional[str]


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


def _build_prompt(idea: str) -> str:
    safe_idea = _escape_braces(idea)
    return PROMPT_TEMPLATE.format(idea=safe_idea)


def _build_share_url(game_id: str) -> Optional[str]:
    base = config.WEBAPP_BASE_URL
    if not base:
        return None
    return f"{base}/webapp/sandbox.html?game_id={game_id}"


def generate_game(chat_id: int, idea: str) -> GeneratedGame:
    if not idea or not idea.strip():
        raise ValueError("Описание игры не должно быть пустым.")

    prompt = _build_prompt(idea)
    response, model_name, _ = llm_request(chat_id, [{"text": prompt}])
    if not response:
        raise RuntimeError("Модель не вернула код игры.")

    parsed = _extract_json(response)
    code = _cleanup_code(parsed.get("code", ""))
    if not code:
        raise ValueError("Модель не вернула JavaScript-код игры.")

    title = (parsed.get("title") or "Игра от Сигмоиды").strip()
    summary = (parsed.get("summary") or "").strip()

    game_id = uuid.uuid4().hex
    payload = {
        "id": game_id,
        "title": title,
        "summary": summary,
        "code": code,
        "idea": idea.strip(),
        "model": model_name,
        "created_at": time.time(),
        "ttl": config.GAME_TTL_SECONDS,
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
    )
