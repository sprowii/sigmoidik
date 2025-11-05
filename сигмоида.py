#!/usr/bin/env python3
# filename: wizard_bot.py
import os, asyncio, logging, time, io, re, json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from telegram import Update, ChatMember, User
from telegram.constants import ChatType, MessageEntityType, ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    ApplicationBuilder, ContextTypes,
    CommandHandler, MessageHandler, filters, CallbackContext
)
import google.generativeai as genai
from google.generativeai.types import ContentType, PartType, Tool
from flask import Flask, render_template_string, request, abort, Response
import threading
import requests
from dotenv import load_dotenv
import base64
import redis

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL")
if not REDIS_URL:
    raise RuntimeError("Переменная окружения REDIS_URL должна быть установлена")

# Upstash предоставляет URL redis://, но требует TLS-соединения.
# Библиотека `redis-py` автоматически включает TLS, если схема `rediss://`.
# Мы принудительно меняем схему для обеспечения безопасного соединения.
if ".upstash.io" in REDIS_URL and REDIS_URL.startswith("redis://"):
    REDIS_URL = "rediss" + REDIS_URL[len("redis"):]

try:
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
except Exception as exc:
    raise RuntimeError("Не удалось подключиться к Redis") from exc

# Создаем простое Flask-приложение для веб-сервера
flask_app = Flask(__name__)

# HTML-шаблон для основной страницы
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://xn--80aqtedp.xn--p1ai/skibidicss">
    <link href="https://unpkg.com/aos@2.3.1/dist/aos.css" rel="stylesheet">
    <title>щас всё заработает, слово пацана</title>
    <link rel="icon" href="https://xn--80aqtedp.xn--p1ai/favicon.ico" type="image/x-icon">
</head>
<body>
    <div class="hero">
        <div class="hero-content">
            <h1>Бро, я запустился</h1>
            <p>Если я правильно понял, то бот может работать ещё 15 минут, если тебе пришлось посетить этот сайт, а если он будет работать и больше, значит тебе его и не надо было посещать.\n</p>
            <div class="tenor-gif-embed" data-postid="13327394754582742145" data-share-method="host" data-aspect-ratio="1" data-width="100%">
                <a href="https://tenor.com/view/cologne-wear-i-buddy-home-gif-13327394754582742145">Cologne Wear GIF</a>
                from <a href="https://tenor.com/search/cologne-gifs">Cologne GIFs</a>
            </div>
            <script type="text/javascript" async src="https://tenor.com/embed.js"></script>
            <br>
            <a href="https://xn--80aqtedp.xn--p1ai/" target="_blank" class="button-link">Сайт создателей</a>
        </div>
    </div>
</body>
</html>
"""
def strip_html_tags(text: str) -> str:
    clean = re.compile(r'<.*?>')
    return re.sub(clean, '', text)

@flask_app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@flask_app.route('/admin/download/history')
def download_history():
    """Секретный эндпоинт для скачивания истории.
    Доступен только при передаче корректного ключа в query-параметре `key`.
    """
    provided_key = request.args.get('key')
    if not DOWNLOAD_KEY or provided_key != DOWNLOAD_KEY:
        abort(403)

    try:
        history_snapshot: Dict[str, Any] = {}
        for key in redis_client.scan_iter(match="history:*"):
            chat_id = key.split(":", 1)[1]
            raw_value = redis_client.get(key)
            if raw_value:
                history_snapshot[chat_id] = json.loads(raw_value)

        users_snapshot: Dict[str, Any] = {}
        for key in redis_client.scan_iter(match=f"{USER_KEY_PREFIX}*"):
            chat_id = key.split(":", 1)[1]
            raw_value = redis_client.get(key)
            if raw_value:
                users_snapshot[chat_id] = json.loads(raw_value)

        response_payload = {
            "history": history_snapshot,
            "users": users_snapshot,
        }

        response = Response(
            json.dumps(response_payload, ensure_ascii=False, indent=2),
            mimetype="application/json"
        )
        response.headers["Content-Disposition"] = "attachment; filename=history.json"
        return response
    except Exception as exc:
        log.error(f"Не удалось выгрузить историю из Redis: {exc}", exc_info=True)
        abort(500)

# ---------- Политика конфиденциальности ----------
PRIVACY_POLICY_TEXT = """
<b>Политика Конфиденциальности для бота "Сигмоида"</b>

<i>Дата последнего обновления: 4 ноября 2025 г.</i>

<b>Собираемые данные</b>
Бот хранит историю переписки (текст, фото, медиа), настройки и публичную информацию об участниках (username, имя, время последнего обращения) для поддержания контекста диалога и аналитики работы бота.

<b>Использование данных</b>
Данные используются для обеспечения непрерывности диалога и улучшения качества ответов.

<b>Хранение и безопасность</b>
Данные хранятся в удаленной базе данных Redis.

<b>Удаление ваших данных</b>
Вы можете удалить все свои данные в любой момент:
• <b>В личных чатах:</b> Отправьте команду <code>/delete_data</code>.
• <b>В групповых чатах:</b> Команду <code>/delete_data</code> могут использовать только администраторы группы.

После выполнения команды все данные для чата будут безвозвратно удалены.

<b>Сторонние сервисы</b>
Ваши сообщения обрабатываются через API Google Gemini.
"""

# Gemini API конфиг
API_KEYS = []
for i in [1, 2]:
    key = os.getenv(f"GEMINI_API_KEY_{i}")
    if key:
        API_KEYS.append(key)

if not API_KEYS:
    raise RuntimeError("Необходимо установить хотя бы одну переменную окружения GEMINI_API_KEY_1 или GEMINI_API_KEY_2")

MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-preview",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-lite-preview",
]
MAX_HISTORY = 10
current_key_idx = 0
current_model_idx = 0
available_models: List[str] = MODELS.copy()
last_model_check_ts: float = 0.0

# ----------- Персона бота и инструкции для LLM -----------
BOT_PERSONA_PROMPT = """
Ты - умный и полезный ассистент по имени Сигмоида.
Не упоминай, что ты Google, Gemini или большая языковая модель.
Форматируй свои ответы, используя HTML-теги, совместимые с Telegram.
Используй <b>для жирного текста</b>, <i>для курсива</i>, <u>для подчеркнутого</u>, <s>для зачеркнутого</s>, <spoiler>для спойлеров</spoiler>, <code>для моноширинного текста</code> и <pre>для блоков кода</pre>.
Для ссылок используй <a href="URL">текст ссылки</a>.
"""

# ---------- Логи ----------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO
)
log = logging.getLogger("wizardbot")

# ---------- Константы и глобальные переменные ----------
ADMIN_ID = os.getenv("ADMIN_ID")
DOWNLOAD_KEY = os.getenv("DOWNLOAD_KEY")
HISTORY_KEY_PREFIX = "history:"
CONFIG_KEY_PREFIX = "config:"
USER_KEY_PREFIX = "users:"

# ---------- Конфиг на чат ----------
@dataclass
class ChatConfig:
    autopost_enabled: bool = False
    interval: int = 14400
    min_messages: int = 10
    msg_size: str = ""
    last_post_ts: float = 0.0
    new_msg_counter: int = 0

configs: Dict[int, ChatConfig] = {}
history: Dict[int, List[ContentType]] = {}
user_profiles: Dict[int, Dict[int, Dict[str, Any]]] = {}

# ---------- Сохранение и загрузка данных ----------
def convert_part_to_dict(part):
    """Конвертирует Part объект в словарь для JSON сериализации."""
    if hasattr(part, 'inline_data') and getattr(part.inline_data, 'data', None) is not None and getattr(part.inline_data, 'mime_type', None):
        # Кодируем бинарные данные изображения в base64 строку
        encoded_data = base64.b64encode(part.inline_data.data).decode('utf-8')
        return {'inline_data': {'mime_type': part.inline_data.mime_type, 'data': encoded_data}}
    if hasattr(part, 'text'):
        return {'text': part.text}
    elif isinstance(part, dict):
        inline_data = part.get('inline_data')
        if isinstance(inline_data, dict) and inline_data.get('mime_type') and inline_data.get('data') is not None:
            data_field = inline_data['data']
            if isinstance(data_field, str):
                encoded_data = data_field
            else:
                encoded_data = base64.b64encode(bytes(data_field)).decode('utf-8')
            return {'inline_data': {'mime_type': inline_data.get('mime_type'), 'data': encoded_data}}
        if 'mime_type' in part and part.get('data') is not None:
            data_field = part['data']
            if isinstance(data_field, str):
                encoded_data = data_field
            else:
                encoded_data = base64.b64encode(bytes(data_field)).decode('utf-8')
            return {'inline_data': {'mime_type': part.get('mime_type'), 'data': encoded_data}}
        return part
    elif isinstance(part, (bytes, bytearray, memoryview)):
        encoded_data = base64.b64encode(bytes(part)).decode('utf-8')
        return {'inline_data': {'mime_type': 'application/octet-stream', 'data': encoded_data}}
    return str(part)


def convert_history_to_dict(history_item):
    """Конвертирует объекты Content из Gemini API в словари для JSON сериализации."""
    if hasattr(history_item, 'role') and hasattr(history_item, 'parts'):
        # Это объект Content из google.generativeai
        return {
            'role': history_item.role,
            'parts': [convert_part_to_dict(part) for part in history_item.parts]
        }
    elif isinstance(history_item, dict):
        # Уже словарь, но нужно проверить parts
        if 'parts' in history_item:
            return {
                'role': history_item.get('role'),
                'parts': [convert_part_to_dict(part) for part in history_item['parts']]
            }
        return history_item
    return history_item


def _deserialize_part(part: Any):
    if isinstance(part, dict):
        if 'text' in part:
            return {'text': part['text']}
        inline_data = part.get('inline_data')
        if isinstance(inline_data, dict) and inline_data.get('mime_type') and inline_data.get('data'):
            try:
                return genai.types.Part(
                    inline_data=genai.types.Blob(
                        mime_type=inline_data['mime_type'],
                        data=base64.b64decode(inline_data['data'].encode('utf-8'))
                    )
                )
            except Exception as exc:
                log.warning(f"Не удалось десериализовать часть истории: {exc}")
                return {'inline_data': inline_data}
        if part.get('mime_type') and part.get('data'):
            try:
                return genai.types.Part(
                    inline_data=genai.types.Blob(
                        mime_type=part['mime_type'],
                        data=base64.b64decode(part['data'].encode('utf-8'))
                    )
                )
            except Exception as exc:
                log.warning(f"Не удалось десериализовать часть истории (плоская запись): {exc}")
                return {'inline_data': part}
    return part


def load_data():
    global history, configs
    log.info("Загрузка данных из Redis...")
    try:
        loaded_history: Dict[int, List[ContentType]] = {}
        for key in redis_client.scan_iter(match=f"{HISTORY_KEY_PREFIX}*"):
            chat_id_part = key.split(":", 1)[1]
            raw_value = redis_client.get(key)
            if not raw_value:
                continue
            try:
                chat_history = json.loads(raw_value)
            except json.JSONDecodeError as exc:
                log.warning(f"Некорректный JSON истории для чата {chat_id_part}: {exc}")
                continue
            try:
                chat_id = int(chat_id_part)
            except ValueError:
                log.warning(f"Пропускаем историю с некорректным chat_id: {chat_id_part}")
                continue
            loaded_history[chat_id] = [
                {
                    'role': item.get('role'),
                    'parts': [_deserialize_part(part) for part in item.get('parts', [])]
                }
                for item in chat_history
            ]
        history = loaded_history
        log.info(f"Загружено {len(history)} историй чатов из Redis.")
    except Exception as exc:
        log.error(f"Ошибка при загрузке историй из Redis: {exc}", exc_info=True)
        history = {}

    try:
        loaded_configs: Dict[int, ChatConfig] = {}
        for key in redis_client.scan_iter(match=f"{CONFIG_KEY_PREFIX}*"):
            chat_id_part = key.split(":", 1)[1]
            raw_value = redis_client.get(key)
            if not raw_value:
                continue
            try:
                config_payload = json.loads(raw_value)
            except json.JSONDecodeError as exc:
                log.warning(f"Некорректный JSON конфигурации для чата {chat_id_part}: {exc}")
                continue
            try:
                chat_id = int(chat_id_part)
            except ValueError:
                log.warning(f"Пропускаем конфигурацию с некорректным chat_id: {chat_id_part}")
                continue
            try:
                loaded_configs[chat_id] = ChatConfig(**config_payload)
            except TypeError as exc:
                log.warning(f"Некорректные данные конфигурации для чата {chat_id}: {exc}")
        configs = loaded_configs
        log.info(f"Загружено {len(configs)} конфигураций чатов из Redis.")
    except Exception as exc:
        log.error(f"Ошибка при загрузке конфигураций из Redis: {exc}", exc_info=True)
        configs = {}

    try:
        loaded_users: Dict[int, Dict[int, Dict[str, Any]]] = {}
        for key in redis_client.scan_iter(match=f"{USER_KEY_PREFIX}*"):
            chat_id_part = key.split(":", 1)[1]
            raw_value = redis_client.get(key)
            if not raw_value:
                continue
            try:
                users_payload = json.loads(raw_value)
            except json.JSONDecodeError as exc:
                log.warning(f"Некорректный JSON профилей пользователей для чата {chat_id_part}: {exc}")
                continue
            try:
                chat_id = int(chat_id_part)
            except ValueError:
                log.warning(f"Пропускаем профили с некорректным chat_id: {chat_id_part}")
                continue
            try:
                loaded_users[chat_id] = {
                    int(user_id): profile
                    for user_id, profile in users_payload.items()
                    if isinstance(profile, dict)
                }
            except Exception as exc:
                log.warning(f"Некорректные данные профилей для чата {chat_id}: {exc}")
        user_profiles.clear()
        user_profiles.update(loaded_users)
        log.info(f"Загружено профилей пользователей для {len(user_profiles)} чатов из Redis.")
    except Exception as exc:
        log.error(f"Ошибка при загрузке профилей пользователей из Redis: {exc}", exc_info=True)


def save_chat_data(chat_id: int):
    history_key = f"{HISTORY_KEY_PREFIX}{chat_id}"
    config_key = f"{CONFIG_KEY_PREFIX}{chat_id}"

    try:
        with redis_client.pipeline() as pipe:
            if chat_id in history:
                serialized_history = [
                    convert_history_to_dict(item) for item in history[chat_id]
                ]
                pipe.set(history_key, json.dumps(serialized_history, ensure_ascii=False))
            else:
                pipe.delete(history_key)

            if chat_id in configs:
                pipe.set(config_key, json.dumps(asdict(configs[chat_id]), ensure_ascii=False))
            else:
                pipe.delete(config_key)

            users_key = f"{USER_KEY_PREFIX}{chat_id}"
            if chat_id in user_profiles and user_profiles[chat_id]:
                serialized_users = {
                    str(uid): profile for uid, profile in user_profiles[chat_id].items()
                }
                pipe.set(users_key, json.dumps(serialized_users, ensure_ascii=False))
            else:
                pipe.delete(users_key)

            pipe.execute()
    except Exception as exc:
        log.error(f"Не удалось сохранить данные чата {chat_id} в Redis: {exc}", exc_info=True)


async def persist_chat_data(chat_id: int):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, save_chat_data, chat_id)


def record_user_profile(chat_id: int, user: Optional[User]) -> bool:
    if not user:
        return False

    profile: Dict[str, Any] = {
        "id": user.id,
        "username": user.username or None,
        "first_name": user.first_name or None,
        "last_name": user.last_name or None,
        "full_name": getattr(user, "full_name", None) or " ".join(filter(None, [user.first_name, user.last_name])) or None,
        "language_code": user.language_code or None,
        "is_bot": user.is_bot,
        "updated_at": time.time(),
    }

    cleaned_profile = {key: value for key, value in profile.items() if value is not None}
    chat_profiles = user_profiles.setdefault(chat_id, {})
    existing = chat_profiles.get(user.id)

    if existing != cleaned_profile:
        chat_profiles[user.id] = cleaned_profile
        return True
    return False


async def ensure_user_profile(update: Update):
    chat = update.effective_chat
    user = update.effective_user
    if not chat or not user:
        return
    if record_user_profile(chat.id, user):
        await persist_chat_data(chat.id)

# ---------- Вспомогалки ----------
def get_cfg(chat_id: int) -> ChatConfig:
    if chat_id not in configs:
        configs[chat_id] = ChatConfig()
    return configs[chat_id]

def llm_request(chat_id: int, prompt_parts: List[PartType]) -> Tuple[Optional[str], str, Optional[Any]]:
    global current_key_idx, current_model_idx
    chat_history = history.get(chat_id, [])

    if len(chat_history) > MAX_HISTORY:
        log.info(f"Summarizing history for chat {chat_id}...")
        try:
            summary_prompt = "Summarize this conversation in a concise paragraph for context."
            # Включаем system_instruction и используем start_chat, чтобы инструкция применялась корректно
            summary_model = genai.GenerativeModel(
                "gemini-2.5-flash-preview",
                api_key=API_KEYS[current_key_idx],
                system_instruction=BOT_PERSONA_PROMPT
            )
            summary_session = summary_model.start_chat(history=chat_history)
            response = summary_session.send_message(summary_prompt)
            summary = response.text
            new_history = [
                {'role': 'user', 'parts': [{'text': "Start of conversation."}]},
                {'role': 'model', 'parts': [{'text': f"Previously discussed: {summary}"}]}
            ]
            history[chat_id] = new_history
            chat_history = new_history
        except Exception as e:
            log.error(f"History summarization failed for chat {chat_id}: {e}")
            history[chat_id] = chat_history[-MAX_HISTORY:]
            chat_history = history[chat_id]

    models_to_try = available_models if available_models else MODELS
    for model_idx_offset in range(len(models_to_try)):
        model_idx = (current_model_idx + model_idx_offset) % len(models_to_try)
        model_name = models_to_try[model_idx]
        for key_try in range(len(API_KEYS)):
            key_idx = (current_key_idx + key_try) % len(API_KEYS)
            try:
                genai.configure(api_key=API_KEYS[key_idx])
                tools = [Tool(function_declarations=[{
                    "name": "generate_image",
                    "description": "Generates an image from a text description. Use for explicit requests to 'draw', 'create an image', etc.",
                    "parameters": {"type": "OBJECT", "properties": {"prompt": {"type": "STRING", "description": "The image description."}}, "required": ["prompt"]}
                }])]
                model = genai.GenerativeModel(model_name, tools=tools, system_instruction=BOT_PERSONA_PROMPT)
                # Используем чат-сессию, чтобы system_instruction применялась вместе с историей
                chat_session = model.start_chat(history=chat_history)
                response = chat_session.send_message(prompt_parts)
                
                if response.candidates and response.candidates[0].content.parts[0].function_call:
                    return None, model_name, response.candidates[0].content.parts[0].function_call
                
                answer = response.text
                # Обновляем историю из сессии для точного контекста
                history[chat_id] = chat_session.history
                current_key_idx, current_model_idx = key_idx, model_idx
                return answer, model_name, None
            except Exception as e:
                if "rate limit" in str(e).lower():
                    log.info(f"Rate limit on key {key_idx+1}, model {model_name}. Trying next...")
                else:
                    log.warning(f"Request failed: key {key_idx+1}, model {model_name}: {e}")
    raise Exception("All API keys/models failed")

async def llm_generate_image(prompt: str) -> Tuple[Optional[bytes], str]:
    global current_key_idx
    model_name = "gemini-2.5-flash-preview"
    for key_try in range(len(API_KEYS)):
        key_idx = (current_key_idx + key_try) % len(API_KEYS)
        try:
            genai.configure(api_key=API_KEYS[key_idx])
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(f"Draw: {prompt}", generation_config={"response_mime_type": "image/png"})
            if response.parts:
                current_key_idx = key_idx
                return response.parts[0].inline_data.data, model_name
        except Exception as e:
            log.warning(f"Image generation failed on key {key_idx+1}: {e}")
    return None, model_name

def check_available_models() -> List[str]:
    global available_models, last_model_check_ts
    log.info("Checking available models...")
    working_models = []
    for model_name in MODELS:
        for api_key in API_KEYS:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name)
                _ = model.generate_content("hi").text
                working_models.append(model_name)
                log.info(f"Model {model_name} is available")
                break
            except Exception:
                continue
    if working_models:
        available_models = working_models
        last_model_check_ts = time.time()
        log.info(f"Available models updated: {working_models}")
    else:
        available_models = MODELS.copy()
    return available_models

async def is_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    if str(update.effective_user.id) == ADMIN_ID:
        return True
    await update.message.reply_text("Эта команда доступна только администратору.")
    return False

def answer_size_prompt(size: str) -> str:
    return {"small": "Кратко:", "medium": "Ответь развернуто:", "large": "Ответь максимально подробно:"}.get(size, "")

def split_long_message(text: str, max_length: int = 4096) -> List[str]:
    if len(text) <= max_length:
        return [text]
    parts, current = [], ""
    # TODO: More intelligent splitting that respects HTML tags
    for line in text.split("\n"):
        if len(current) + len(line) + 1 <= max_length:
            current += (line + "\n")
        else:
            if current: parts.append(current.strip())
            current = line
    if current: parts.append(current.strip())
    return parts

# ---------- Команды ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    await update.message.reply_text("👋 Я Сигмоида бот. /help – справка\n\n"
                                    "⚠️ <b>Важно:</b> Ваши сообщения и медиафайлы обрабатываются через Google Gemini API. /privacy",
                                    parse_mode=ParseMode.HTML)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    await update.message.reply_text("<b>Команды:</b>\n"
                                    "/settings – показать текущие настройки\n"
                                    "/autopost on|off – вкл/выкл автопосты (админ)\n"
                                    "/set_interval &lt;сек&gt; – интервал автопоста (админ)\n"
                                    "/set_minmsgs &lt;n&gt; – минимум сообщений для автопоста (админ)\n"
                                    "/set_msgsize &lt;s|m|l&gt; – размер ответов (админ)\n"
                                    "/draw &lt;описание&gt; – нарисовать изображение\n"
                                    "/reset – очистить историю диалога\n"
                                    "/privacy – политика конфиденциальности",
                                    parse_mode=ParseMode.HTML)

async def privacy_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    await update.message.reply_text(PRIVACY_POLICY_TEXT, parse_mode=ParseMode.HTML, disable_web_page_preview=True)

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    chat_id = update.effective_chat.id
    history.pop(chat_id, None)
    await persist_chat_data(chat_id)
    await update.message.reply_text("История очищена ✅")
async def delete_data_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    if not update.message or not update.effective_chat or not update.effective_user: return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    username = update.effective_user.username or update.effective_user.first_name
    chat_type = update.effective_chat.type

    # Получаем ADMIN_ID из переменных окружения
    ADMIN_ID = os.getenv("ADMIN_ID")
    is_bot_admin = (str(user_id) == ADMIN_ID) # Проверяем, является ли текущий пользователь главным админом бота

    can_delete = False
    if chat_type == ChatType.PRIVATE:
        can_delete = True # В личных чатах любой пользователь может удалить свои данные
        log.info(f"User {username} ({user_id}) in private chat requested to delete their data.")
    elif chat_type in [ChatType.GROUP, ChatType.SUPERGROUP]:
        if is_bot_admin:
            can_delete = True # Главный админ бота может удалять данные в группах
            log.info(f"Bot admin {username} ({user_id}) in group chat ({chat_id}) requested to delete chat data.")
        else:
            try:
                # Проверяем, является ли пользователь администратором группы
                chat_member = await context.bot.get_chat_member(chat_id, user_id)
                if chat_member.status in [ChatMember.ADMINISTRATOR, ChatMember.CREATOR]:
                    can_delete = True
                    log.info(f"Group admin {username} ({user_id}) in group chat ({chat_id}) requested to delete chat data.")
                else:
                    log.warning(f"User {username} ({user_id}) tried to delete data in group chat ({chat_id}) without admin rights.")
                    await update.message.reply_html("<b>Эту команду могут использовать только администраторы группы.</b>")
                    return
            except Exception as e:
                log.error(f"Error checking chat member status in group {chat_id}: {e}")
                await update.message.reply_html("<b>Произошла ошибка при проверке ваших прав администратора.</b>")
                return
    else:
        log.warning(f"User {username} ({user_id}) tried to delete data in unsupported chat type: {chat_type}.")
        await update.message.reply_html("<b>Эта команда не поддерживается в данном типе чата.</b>")
        return

    if can_delete:
        if chat_id in history:
            del history[chat_id]
            log.info(f"Deleted history for chat_id {chat_id}.")
        if chat_id in configs:
            del configs[chat_id]
            log.info(f"Deleted configs for chat_id {chat_id}.")
        if chat_id in user_profiles:
            del user_profiles[chat_id]
            log.info(f"Deleted user profiles for chat_id {chat_id}.")

        try:
            redis_client.delete(f"{HISTORY_KEY_PREFIX}{chat_id}", f"{CONFIG_KEY_PREFIX}{chat_id}", f"{USER_KEY_PREFIX}{chat_id}")
            log.info(f"Удалены ключи Redis для чата {chat_id}.")
        except Exception as exc:
            log.error(f"Не удалось удалить ключи Redis для чата {chat_id}: {exc}", exc_info=True)

        await update.message.reply_html(
            "<b>Все данные для этого чата (история переписки и настройки) были успешно удалены.</b>\n"
            "Если вы продолжите использовать бота, начнется новая история."
        )
async def delete_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    if not await is_admin(update, context): return
    if not context.args: return await update.message.reply_text("Укажите ID чата.")
    try:
        target_id = int(context.args[0])
        history.pop(target_id, None)
        configs.pop(target_id, None)
        user_profiles.pop(target_id, None)
        try:
            redis_client.delete(f"{HISTORY_KEY_PREFIX}{target_id}", f"{CONFIG_KEY_PREFIX}{target_id}", f"{USER_KEY_PREFIX}{target_id}")
        except Exception as exc:
            log.error(f"Не удалось удалить данные чата {target_id} из Redis: {exc}", exc_info=True)
        await update.message.reply_text(f"Данные для ID {target_id} удалены.")
    except (ValueError, IndexError):
        await update.message.reply_text("Неверный ID.")

async def settings_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    cfg = get_cfg(update.effective_chat.id)
    await update.message.reply_text(f"<b>Автопосты:</b> {'вкл' if cfg.autopost_enabled else 'выкл'}.\n"
                                    f"<b>Интервал:</b> {cfg.interval} сек, <b>мин. сообщений:</b> {cfg.min_messages}.\n"
                                    f"<b>Размер ответов:</b> {cfg.msg_size or 'default'}.",
                                    parse_mode=ParseMode.HTML)

async def autopost_switch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    if not await is_admin(update, context): return
    if not context.args or context.args[0] not in {"on", "off"}: return await update.message.reply_text("Пример: /autopost on")
    cfg = get_cfg(update.effective_chat.id)
    cfg.autopost_enabled = (context.args[0] == "on")
    await persist_chat_data(update.effective_chat.id)
    await update.message.reply_text(f"Автопосты {'включены' if cfg.autopost_enabled else 'выключены'}.")

async def set_interval(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    if not await is_admin(update, context): return
    try:
        cfg = get_cfg(update.effective_chat.id)
        cfg.interval = max(300, int(context.args[0]))
        await persist_chat_data(update.effective_chat.id)
        await update.message.reply_text(f"Интервал автопоста = {cfg.interval} сек.")
    except (IndexError, ValueError):
        await update.message.reply_text("Пример: /set_interval 7200")

async def set_minmsgs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    if not await is_admin(update, context): return
    try:
        cfg = get_cfg(update.effective_chat.id)
        cfg.min_messages = max(1, int(context.args[0]))
        await persist_chat_data(update.effective_chat.id)
        await update.message.reply_text(f"Минимум сообщений = {cfg.min_messages}.")
    except (IndexError, ValueError):
        await update.message.reply_text("Пример: /set_minmsgs 10")

async def set_msgsize(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    if not await is_admin(update, context): return
    size = (context.args or [""])[0].lower()
    if size not in {"small", "medium", "large", "s", "m", "l", ""}:
        return await update.message.reply_text("Варианты: small, medium, large или пусто (default)")
    cfg = get_cfg(update.effective_chat.id)
    if size in {"s", "m", "l"}:
        cfg.msg_size = size
    elif size:
        cfg.msg_size = size[0]
    else:
        cfg.msg_size = ""
    await persist_chat_data(update.effective_chat.id)
    await update.message.reply_text(f"Размер ответов = {cfg.msg_size or 'default'}.")

async def draw_image_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    if not context.args: return await update.message.reply_text("Пример: /draw кот в скафандре")
    await generate_and_send_image(update, context, ' '.join(context.args))

async def generate_and_send_image(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt: str):
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="upload_photo")
    try:
        image_bytes, model_used = await asyncio.get_running_loop().run_in_executor(None, llm_generate_image, prompt)
        if image_bytes:
            model_display = model_used.replace("gemini-", "").replace("-latest", "").title()
            caption = f"🎨 «{prompt}»\n\n<b>Generated by {model_display}</b>"
            await update.message.reply_photo(photo=image_bytes, caption=caption, parse_mode=ParseMode.HTML)
        else:
            await update.message.reply_text("⚠️ Не удалось создать изображение.")
    except Exception as e:
        log.exception(e)
        await update.message.reply_text("⚠️ Ошибка при генерации изображения.")

# ---------- Основной обработчик ----------
async def send_bot_response(update: Update, context: ContextTypes.DEFAULT_TYPE, chat_id: int, prompt_parts: List[PartType]):
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    data_updated = False
    try:
        reply, model_used, function_call = await asyncio.get_running_loop().run_in_executor(None, llm_request, chat_id, prompt_parts)

        if function_call and function_call.name == "generate_image":
            await generate_and_send_image(update, context, function_call.args.get("prompt", ""))
            data_updated = True
        elif reply:
            model_display = model_used.replace("gemini-", "").replace("-latest", "").title()
            full_reply = f"<b>{model_display}</b>\n\n{reply}"
            for chunk in split_long_message(full_reply):
                try:
                    await update.message.reply_text(chunk, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
                except BadRequest as e:
                    log.warning(f"HTML parse failed, sending plain text. Error: {e}")
                    # Strip HTML tags before sending as plain text
                    plain_text_chunk = strip_html_tags(chunk)
                    await update.message.reply_text(plain_text_chunk, disable_web_page_preview=True)
            data_updated = True
    except Exception as e:
        log.exception(e)
        await update.message.reply_text("⚠️ Ошибка модели.")
    finally:
        if data_updated:
            await persist_chat_data(chat_id)

async def handle_text_and_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message: return
    chat_id = update.effective_chat.id
    text = update.message.text or update.message.caption or ""
    record_user_profile(chat_id, update.effective_user)
    cfg = get_cfg(chat_id)

    if update.message.chat.type in (ChatType.GROUP, ChatType.SUPERGROUP):
        bot_mentioned = any(
            text[e.offset:e.offset+e.length].lstrip('@').lower() == context.bot.username.lower()
            for e in (update.message.entities or []) if e.type == MessageEntityType.MENTION
        )
        is_reply_to_bot = update.message.reply_to_message and update.message.reply_to_message.from_user.username == context.bot.username
        
        if not (bot_mentioned or is_reply_to_bot):
            cfg.new_msg_counter += 1
            await persist_chat_data(chat_id)
            return
        
        for e in reversed(update.message.entities or []):
            if e.type == MessageEntityType.MENTION and text[e.offset:e.offset+e.length].lstrip('@').lower() == context.bot.username.lower():
                text = (text[:e.offset] + text[e.offset+e.length:]).strip()
    
    cfg.new_msg_counter += 1
    await persist_chat_data(chat_id)
    prompt_parts = []
    if text: prompt_parts.append(answer_size_prompt(cfg.msg_size) + text)
    if update.message.photo:
        photo_size = update.message.photo[-1]
        file = await photo_size.get_file()
        image_buffer = io.BytesIO()
        await file.download_to_memory(out=image_buffer)
        file_bytes = image_buffer.getvalue()
        mime_type = getattr(photo_size, "mime_type", None) or getattr(file, "mime_type", None) or "image/jpeg"
        prompt_parts.insert(0, genai.types.Part(inline_data=genai.types.Blob(mime_type=mime_type, data=file_bytes)))

    if not prompt_parts: return
    await send_bot_response(update, context, chat_id, prompt_parts)

async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message: return
    await ensure_user_profile(update)

    # Бот пока не умеет...
    await update.message.reply_text(
        "😔 Извините, я пока не умею обрабатывать голосовые сообщения, видео и видео-кружочки.\n\n"
        "Пожалуйста, опишите ваш вопрос текстом или отправьте фото — с ними я работаю отлично!"
    )
# ---------- Задачи ----------
async def check_models_job(context: CallbackContext):
    await asyncio.get_running_loop().run_in_executor(None, check_available_models)

async def autopost_job(context: CallbackContext):
    for chat_id, cfg in list(configs.items()):
        if not (cfg.autopost_enabled and cfg.new_msg_counter >= cfg.min_messages and time.time() - cfg.last_post_ts > cfg.interval):
            continue
        prompt = f"Сделай краткий дайджест последних {cfg.new_msg_counter} сообщений чата. Выдели основные темы."
        log.info(f"Autopost in chat {chat_id}")
        try:
            summary, model_used, _ = await asyncio.get_running_loop().run_in_executor(None, llm_request, chat_id, [{"text": prompt}])
            if summary:
                model_display = model_used.replace("gemini-", "").replace("-latest", "").title()
                message_text = f"📰 <b>Автодайджест ({model_display}):</b>\n{summary}"
                for chunk in split_long_message(message_text):
                    try:
                        await context.bot.send_message(chat_id, chunk, parse_mode=ParseMode.HTML)
                    except BadRequest:
                        await context.bot.send_message(chat_id, chunk)
                cfg.last_post_ts, cfg.new_msg_counter = time.time(), 0
                await persist_chat_data(chat_id)
        except Exception as e:
            log.error(f"Autopost failed for chat {chat_id}: {e}")

# ---------- Main ----------
def main():
    load_data()
    token, admin_id = os.getenv("TG_TOKEN"), os.getenv("ADMIN_ID")
    if not token or not admin_id: raise RuntimeError("TG_TOKEN и ADMIN_ID должны быть установлены")
    if not DOWNLOAD_KEY:
        log.warning("DOWNLOAD_KEY не установлен. Скачивание истории через веб будет недоступно.")
    
    try:
        bot_info = requests.get(f"https://api.telegram.org/bot{token}/getMe").json().get('result', {})
        if not bot_info.get('username'): raise RuntimeError("Не удалось получить username бота.")
        log.info(f"Bot Username: @{bot_info['username']}")
    except Exception as e:
        raise RuntimeError(f"Ошибка при получении информации о боте: {e}")

    app = ApplicationBuilder().token(token).build()
    bot_username = bot_info['username']  # Сохраняем username для использования в регистрации команд

    command_handlers = {
        "start": start, "help": help_cmd, "privacy": privacy_cmd,
        "reset": reset, "draw": draw_image_cmd, "settings": settings_cmd,
        "delete_data": delete_data, "autopost": autopost_switch,
        "set_interval": set_interval, "set_minmsgs": set_minmsgs,
        "set_msgsize": set_msgsize
    }
    for command, callback in command_handlers.items():
        app.add_handler(CommandHandler(command, callback))
        # Добавляем алиасы для групп (e.g. /draw@botname)
        app.add_handler(CommandHandler(f"{command}_{bot_username}", callback))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_and_photo))
    app.add_handler(MessageHandler(filters.PHOTO, handle_text_and_photo))
    app.add_handler(MessageHandler(filters.VOICE | filters.VIDEO | filters.VIDEO_NOTE, handle_media))

    if app.job_queue:
        app.job_queue.run_repeating(check_models_job, 14400, 60)
        app.job_queue.run_repeating(autopost_job, 60, 60)
        log.info("JobQueue initialized")

    threading.Thread(target=lambda: flask_app.run(host='0.0.0.0', port=int(os.getenv("PORT", 10000))), daemon=True).start()
    log.info("Flask app started")
    log.info("Bot started 🚀")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()