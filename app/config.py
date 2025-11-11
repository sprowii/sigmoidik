# Copyright (c) 2025 sprouee
import os
import secrets
from typing import List

from dotenv import load_dotenv

load_dotenv()

from app.logging_config import log


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


def _load_openrouter_keys() -> List[str]:
    keys: List[str] = []
    base_key = os.getenv("OPENROUTER_API_KEY")
    if base_key:
        keys.append(base_key)
    idx = 1
    while True:
        key = os.getenv(f"OPENROUTER_API_KEY_{idx}")
        if not key:
            break
        keys.append(key)
        idx += 1
    return keys


API_KEYS = _load_api_keys()
if not API_KEYS:
    raise RuntimeError(
        "Необходимо установить хотя бы одну переменную окружения GEMINI_API_KEY_1 или GEMINI_API_KEY_2"
    )

MODELS: List[str] = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-lite-preview",
    "gemini-2.0-flash",
]

IMAGE_MODEL_NAME = os.getenv("IMAGE_MODEL_NAME", "gemini-2.0-flash-preview-image")

OPENROUTER_API_KEYS = _load_openrouter_keys()
OPENROUTER_MODELS: List[str] = [
    model.strip()
    for model in os.getenv(
        "OPENROUTER_MODELS",
        "deepseek/deepseek-chat-v3-0324:free,deepseek/deepseek-r1-0528:free,tngtech/deepseek-r1t2-chimera:free",
    ).split(",")
    if model.strip()
]
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL")
OPENROUTER_SITE_NAME = os.getenv("OPENROUTER_SITE_NAME")
OPENROUTER_TIMEOUT = float(os.getenv("OPENROUTER_TIMEOUT", "45"))

LLM_PROVIDER_ORDER: List[str] = [
    provider.strip().lower()
    for provider in os.getenv("LLM_PROVIDER_ORDER", "gemini,openrouter,pollinations").split(",")
    if provider.strip()
]

# ============================================================================
# POLLINATIONS CONFIGURATION
# ============================================================================

# --- Text Generation ---

# URL для текстовых моделей Pollinations
POLLINATIONS_TEXT_BASE_URL = os.getenv(
    "POLLINATIONS_TEXT_BASE_URL", 
    "https://text.pollinations.ai/openai"
)

# API ключ (токен) для Pollinations. Будет добавлен в поле 'token' в JSON.
POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_API_KEY")

# Список доступных текстовых моделей, через запятую.
# По умолчанию используются модели, которые вы просили: 'o4-mini' и 'gemini-search'
POLLINATIONS_TEXT_MODELS: List[str] = [
    model.strip()
    for model in os.getenv(
        "POLLINATIONS_TEXT_MODELS",
        "o4-mini,gemini-search",
    ).split(",")
    if model.strip()
]

# Запасной вариант, если переменная окружения пуста
if not POLLINATIONS_TEXT_MODELS:
    POLLINATIONS_TEXT_MODELS = ["o4-mini"]

# Модель по умолчанию
POLLINATIONS_TEXT_DEFAULT = os.getenv("POLLINATIONS_TEXT_DEFAULT", "o4-mini")
if POLLINATIONS_TEXT_DEFAULT not in POLLINATIONS_TEXT_MODELS:
    POLLINATIONS_TEXT_DEFAULT = POLLINATIONS_TEXT_MODELS[0]

# Температура для генерации ответа. 1.0 = более креативный, 0.1 = более точный.
POLLINATIONS_TEXT_TEMPERATURE = float(os.getenv("POLLINATIONS_TEXT_TEMPERATURE", "1.0"))

# Таймаут для текстовых запросов
POLLINATIONS_TEXT_TIMEOUT = float(os.getenv("POLLINATIONS_TEXT_TIMEOUT", "120.0"))


# --- Image Generation ---
POLLINATIONS_ENABLED = os.getenv("POLLINATIONS_ENABLED", "true").lower() in {"1", "true", "yes"}
POLLINATIONS_MODEL = os.getenv("POLLINATIONS_MODEL", "flux") # Модель для генерации изображений по умолчанию

_pollinations_models_raw = os.getenv("POLLINATIONS_MODELS")
if _pollinations_models_raw:
    POLLINATIONS_MODELS: List[str] = [
        model.strip()
        for model in _pollinations_models_raw.split(",")
        if model.strip()
    ]
else:
    POLLINATIONS_MODELS = [POLLINATIONS_MODEL]

POLLINATIONS_WIDTH = int(os.getenv("POLLINATIONS_WIDTH", "1024"))
POLLINATIONS_HEIGHT = int(os.getenv("POLLINATIONS_HEIGHT", "1024"))
POLLINATIONS_BASE_URL = os.getenv("POLLINATIONS_BASE_URL", "https://image.pollinations.ai")
POLLINATIONS_SEED = os.getenv("POLLINATIONS_SEED")
POLLINATIONS_TIMEOUT = float(os.getenv("POLLINATIONS_TIMEOUT", "300"))
POLLINATIONS_SAFE_MODE = os.getenv("POLLINATIONS_SAFE_MODE", "true").lower() in {"1", "true", "yes"}

POLLINATIONS_PRIVATE_IMAGES = os.getenv("POLLINATIONS_PRIVATE_IMAGES", "true").lower() in {"1", "true", "yes"}
POLLINATIONS_NO_LOGO = os.getenv("POLLINATIONS_NO_LOGO", "true").lower() in {"1", "true", "yes"}
# ============================================================================

MAX_HISTORY = 10
MAX_IMAGE_BYTES = 5 * 1024 * 1024

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

# Диагностика
log.info(f"Loaded {len(API_KEYS)} Gemini API keys")
log.info(f"Loaded {len(OPENROUTER_API_KEYS)} OpenRouter API keys")
log.info(f"Pollinations enabled: {POLLINATIONS_ENABLED}")
log.info(f"Pollinations authenticated: {bool(POLLINATIONS_API_KEY)}")
log.info(f"Pollinations safe mode (images): {POLLINATIONS_SAFE_MODE}")
log.info(f"LLM provider order: {LLM_PROVIDER_ORDER}")
log.info(f"Gemini models: {MODELS}")
log.info(f"OpenRouter models: {OPENROUTER_MODELS}")
log.info(f"Pollinations text models: {POLLINATIONS_TEXT_MODELS}, default: {POLLINATIONS_TEXT_DEFAULT}")