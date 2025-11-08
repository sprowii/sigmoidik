# Copyright (c) 2025 sprouee
import os
import secrets
from typing import List

from dotenv import load_dotenv

load_dotenv()


def _resolve_redis_url(raw_url: str) -> str:
    if ".upstash.io" in raw_url and raw_url.startswith("redis://"):
        return "rediss" + raw_url[len("redis") :]
    return raw_url


REDIS_URL = os.getenv("REDIS_URL")
if not REDIS_URL:
    raise RuntimeError("Переменная окружения REDIS_URL должна быть установлена")
REDIS_URL = _resolve_redis_url(REDIS_URL)

TG_TOKEN = os.getenv("TG_TOKEN")
ADMIN_ID = os.getenv("ADMIN_ID")
DOWNLOAD_KEY = os.getenv("DOWNLOAD_KEY")


def _load_api_keys() -> List[str]:
    keys: List[str] = []
    for idx in (1, 2):
        key = os.getenv(f"GEMINI_API_KEY_{idx}")
        if key:
            keys.append(key)
    return keys


API_KEYS = _load_api_keys()
if not API_KEYS:
    raise RuntimeError(
        "Необходимо установить хотя бы одну переменную окружения GEMINI_API_KEY_1 или GEMINI_API_KEY_2"
    )

MODELS: List[str] = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-preview",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-lite-preview",
]

IMAGE_MODEL_NAME = os.getenv("IMAGE_MODEL_NAME", "gemini-2.0-flash-preview-image")

POLLINATIONS_ENABLED = os.getenv("POLLINATIONS_ENABLED", "false").lower() in {"1", "true", "yes"}
POLLINATIONS_MODEL = os.getenv("POLLINATIONS_MODEL", "flux")
POLLINATIONS_WIDTH = int(os.getenv("POLLINATIONS_WIDTH", "1024"))
POLLINATIONS_HEIGHT = int(os.getenv("POLLINATIONS_HEIGHT", "1024"))
POLLINATIONS_BASE_URL = os.getenv("POLLINATIONS_BASE_URL", "https://pollinations.ai")
POLLINATIONS_SEED = os.getenv("POLLINATIONS_SEED")
POLLINATIONS_TIMEOUT = float(os.getenv("POLLINATIONS_TIMEOUT", "30"))

MAX_HISTORY = 10
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB предел на сохранение изображений

BOT_PERSONA_PROMPT = """
Ты — умный и полезный ассистент по имени Сигмоида.
Не упоминай, что ты Google, Gemini или большая языковая модель.
Форматируй свои ответы, используя HTML-теги, совместимые с Telegram.
Используй <b>для жирного текста</b>, <i>для курсива</i>, <u>для подчеркнутого</u>, <s>для зачеркнутого</s>, <spoiler>для спойлеров</spoiler>, <code>для моноширинного текста</code> и <pre>для блоков кода</pre>.
Для ссылок используй <a href="URL">текст ссылки</a>.
""".strip()

HISTORY_KEY_PREFIX = "history:"
CONFIG_KEY_PREFIX = "config:"
USER_KEY_PREFIX = "users:"

FLASK_HOST = "0.0.0.0"
FLASK_PORT = int(os.getenv("PORT", 10000))

GAME_CODE_PREFIX = "games:"
GAME_TTL_SECONDS = int(os.getenv("GAME_TTL_SECONDS", 7 * 24 * 3600))
GAME_LIST_KEY = "games:list"
GAMES_BY_AUTHOR_PREFIX = "games:author:"
WEBAPP_BASE_URL = os.getenv("WEBAPP_BASE_URL")
if WEBAPP_BASE_URL and WEBAPP_BASE_URL.endswith("/"):
    WEBAPP_BASE_URL = WEBAPP_BASE_URL[:-1]

LOGIN_CODE_PREFIX = "login_codes:"
LOGIN_CODE_TTL_SECONDS = int(os.getenv("LOGIN_CODE_TTL_SECONDS", 10 * 60))

FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32)
SESSION_COOKIE_NAME = os.getenv("SESSION_COOKIE_NAME", "sig_session")


