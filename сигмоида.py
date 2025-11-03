#!/usr/bin/env python3
# filename: wizard_bot.py
import os, asyncio, logging, time, io, re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from PIL import Image
from telegram import Update
from telegram.constants import ChatType, MessageEntityType # <-- Добавляем импорт MessageEntityType
from telegram.error import BadRequest
from telegram.ext import (
    ApplicationBuilder, ContextTypes,
    CommandHandler, MessageHandler, filters, CallbackContext
)
import google.generativeai as genai
from flask import Flask, render_template_string
import threading
import requests # <-- Добавляем импорт requests

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
    <div class="hero"> <!-- Используем hero класс для центрирования и стилизации -->
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


@flask_app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

# ---------- Политика конфиденциальности ----------
PRIVACY_POLICY_TEXT = """
<b>Политика конфиденциальности и обработки данных Сигмоида</b>

⚠️ <b>Важно:</b> Ваши сообщения и изображения, отправленные этому боту, передаются в Google Gemini API для обработки. Это необходимо для функционирования бота и генерации ответов.
История диалога хранится только временно в оперативной памяти бота и не сохраняется на сервере на постоянной основе.
Используя бота, вы соглашаетесь с передачей данных в Google для их обработки.

<b>Согласие:</b> Продолжая использовать бота в личных сообщениях или отправляя упоминания боту в групповых чатах, вы подтверждаете свое согласие с данной политикой. Администраторы групповых чатов несут ответственность за информирование участников о том, что их сообщения могут обрабатываться ботом через сторонний API.

<b>Дополнительно:</b> Наш бот также подпадает под действие <b>Стандартной политики конфиденциальности Telegram для ботов</b>. Ознакомиться с ней можно по ссылке: <a href="https://telegram.org/privacy-tpa">https://telegram.org/privacy-tpa</a>. Эта политика является общей, и мы рекомендуем вам ознакомиться с ней для полного понимания того, как Telegram регулирует работу ботов на своей платформе.

Для удаления ваших данных, пожалуйста, свяжитесь с разработчиками бота.
"""

# Gemini API конфиг
API_KEYS = []
for i in [1, 2]:
    key = os.getenv(f"GEMINI_API_KEY_{i}")
    if key:
        API_KEYS.append(key)

if not API_KEYS:
    raise RuntimeError("Необходимо установить хотя бы одну переменную окружения GEMINI_API_KEY_1 или GEMINI_API_KEY_2")
# Список моделей по убыванию (от лучшей к худшей)
MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-preview",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-lite-preview",
    "gemini-2.0-flash"
]
MAX_HISTORY = 12                    # сколько пар вопрос-ответ храним
current_key_idx = 0
current_model_idx = 0
# Доступные модели (обновляется каждые 4 часа)
available_models: List[str] = MODELS.copy()
last_model_check_ts: float = 0.0

# ----------- Персона бота и инструкции для LLM -----------
BOT_PERSONA_PROMPT = """
Ты - умный и полезный ассистент по имени Сигмоида. 
Не упоминай, что ты Google, Gemini или большая языковая модель. 
Форматируй свои ответы, используя Telegram-совместимый MarkdownV2. 
Будь внимателен к правилам экранирования специальных символов в MarkdownV2 (например, '.', '!', '-', '+', '(', ')', '[', ']', '_', '*', '`', '~', '>', '#', '=', '|', '{', '}').
"""

# ---------- Логи ----------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO
)
log = logging.getLogger("wizardbot")

# ---------- Константы и глобальные переменные ----------
ADMIN_ID = os.getenv("ADMIN_ID")

# ---------- Конфиг на чат ----------
@dataclass
class ChatConfig:
    autopost_enabled: bool = False
    interval: int = 14400          # 4 ч
    min_messages: int = 10
    msg_size: str = ""      # small/medium/large/"" (по умолчанию)
    last_post_ts: float = 0.0
    new_msg_counter: int = 0

# chat_id -> ChatConfig
configs: Dict[int, ChatConfig] = {}
# chat_id -> history (для LLM)
history: Dict[int, List[Dict[str, str]]] = {}


# ---------- Вспомогалки ----------
def sanitize_telegram_markdown(text: str) -> str:
    """Очищает Markdown от неподдерживаемых Telegram элементов."""
    # Заменяем жирный текст с двойными ** на одинарные *
    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)
    # Удаляем заголовки (#, ##, ...)
    text = re.sub(r'^\s*#+\s+', '', text, flags=re.MULTILINE)
    # Удаляем горизонтальные линии (---, ***, ___), т.к. они не поддерживаются
    text = re.sub(r'^\s*[-*=_]{3,}\s*$', '', text, flags=re.MULTILINE)
    return text

def get_cfg(chat_id: int) -> ChatConfig:
    if chat_id not in configs:
        configs[chat_id] = ChatConfig()
    return configs[chat_id]

def llm_request(chat_id: int, prompt: str, image: Optional[Image.Image] = None) -> Tuple[str, str]:
    global current_key_idx, current_model_idx

    chat_history = history.get(chat_id, [])
    models_to_try = available_models if available_models else MODELS

    for model_idx_offset in range(len(models_to_try)):
        model_idx = (current_model_idx + model_idx_offset) % len(models_to_try)
        model_name = models_to_try[model_idx]

        model_failed_on_all_keys = True
        for key_try in range(len(API_KEYS)):
            key_idx = (current_key_idx + key_try) % len(API_KEYS)
            api_key = API_KEYS[key_idx]

            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name)
                
                # Модели с картинками плохо работают с историей, так что для них начинаем чат заново
                current_chat_history = chat_history if not image else []
                chat_session = model.start_chat(history=current_chat_history)
                
                content_to_send = [prompt]
                if image:
                    content_to_send.insert(0, image)

                response = chat_session.send_message(content_to_send)
                answer = response.text
                
                # Не сохраняем историю для запросов с картинками, чтобы избежать проблем
                if not image:
                    history[chat_id] = chat_session.history

                current_key_idx = key_idx
                current_model_idx = model_idx
                model_failed_on_all_keys = False
                return (answer, model_name)
            except Exception as e:
                error_msg = str(e).lower()
                if any(phrase in error_msg for phrase in ["resource exhausted", "quota exceeded", "rate limit", "exceeded", "ограничение", "limit"]):
                    log.info(f"Rate limit: key {key_idx+1}, model {model_name}, trying next key...")
                    continue
                else:
                    log.warning(f"Request failed: key {key_idx+1}, model {model_name}: {e}")
                    continue
        if model_failed_on_all_keys:
            log.info(f"Model {model_name} failed on all keys, trying next model...")
            continue
    raise Exception("All API keys/models failed")

def check_available_models() -> List[str]:
    global available_models, last_model_check_ts
    log.info("Checking available models...")
    working_models = []
    for model_name in MODELS:
        model_works = False
        for api_key in API_KEYS:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name)
                response = model.generate_content("hi")
                # Убедимся, что ответ валидный
                _ = response.text
                model_works = True
                break
            except Exception as e:
                error_msg = str(e).lower()
                if "not found" in error_msg or "invalid model" in error_msg or "does not exist" in error_msg:
                    log.warning(f"Model {model_name} not found or invalid.")
                    break 
                continue
        if model_works:
            working_models.append(model_name)
            log.info(f"Model {model_name} is available")
    
    if working_models:
        available_models = working_models
        last_model_check_ts = time.time()
        log.info(f"Available models updated: {working_models}")
    else:
        log.warning("No models available, using fallback list")
        available_models = MODELS.copy()
    return available_models

async def is_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Проверяет, является ли пользователь админом."""
    if update.message and str(update.message.from_user.id) == ADMIN_ID:
        return True
    
    # Отправляем сообщение только если это прямой вызов команды, а не фоновая проверка
    if update.message:
        await update.message.reply_text("Эта команда доступна только администратору.")
    return False

async def delete_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """(Админ) Удаляет данные чата по его ID."""
    if not await is_admin(update, context): return

    if not context.args:
        await update.message.reply_text("Нужно указать ID чата/пользователя. Пример: /delete_data 123456789")
        return

    try:
        target_id = int(context.args[0])
        deleted_from_history = history.pop(target_id, None) is not None
        deleted_from_settings = configs.pop(target_id, None) is not None

        if deleted_from_history or deleted_from_settings:
            await update.message.reply_text(f"Данные для ID {target_id} были удалены из памяти бота.")
        else:
            await update.message.reply_text(f"Данные для ID {target_id} не найдены в памяти бота.")

    except (ValueError, IndexError):
        await update.message.reply_text("Неверный ID. ID должен быть числом.")

def answer_size_prompt(size: str) -> str:
    mapping = {
        "small":   "Кратко:",
        "medium":  "Ответь развернуто:",
        "large":   "Ответь максимально подробно, с примерами кода и пояснениями:"
    }
    return mapping.get(size, "")

def split_long_message(text: str, max_length: int = 4096) -> List[str]:
    """Разбивает длинное сообщение на части по max_length символов."""
    if len(text) <= max_length:
        return [text]
    parts = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 <= max_length:
            current += (line + "\n" if current else line)
        else:
            if current:
                parts.append(current.strip())
            current = line
    if current:
        parts.append(current.strip())
    return parts

# ---------- Команды ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Я Gemini 2.5 бот. /help – справка\n\n"
        "⚠️ <b>Важно:</b> Ваши сообщения и изображения обрабатываются через Google Gemini API. "
        "Используя бота, вы соглашаетесь с передачей данных в Google для обработки. "
        "Полная информация: /privacy",
        parse_mode="HTML"
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "<b>Команды:</b>\n"
        "/settings – показать текущие настройки\n"
        "/autopost on|off – включить/выключить автопосты\n"
        "/set_interval &lt;сек&gt; – интервал автопоста\n"
        "/set_minmsgs &lt;n&gt; – минимум новых сообщений перед автопостом\n"
        "/set_msgsize &lt;small|medium|large|default&gt; – размер ответов бота\n"
        "/reset – очистить историю диалога\n"
        "/privacy – политика конфиденциальности\n\n" # <-- Добавлена новая команда
        "⚠️ <b>Конфиденциальность:</b> Ваши сообщения и изображения передаются в Google Gemini API для обработки. "
        "История диалога хранится только в памяти и не сохраняется на сервере. "
        "Полная информация: /privacy", # <-- Ссылка на новую команду
        parse_mode="HTML"
    )

async def privacy_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        PRIVACY_POLICY_TEXT,
        parse_mode="HTML",
        disable_web_page_preview=True
    )

async def settings_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cfg = get_cfg(update.effective_chat.id)
    txt = (
        f"Автопосты: {'включены' if cfg.autopost_enabled else 'выключены'}.\n"
        f"Интервал автопоста: {cfg.interval//3600} ч, "
        f"минимум новых сообщений: {cfg.min_messages}.\n"
        f"Размер ответов: {cfg.msg_size}."
    )
    await update.message.reply_text(txt)

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    history.pop(update.effective_chat.id, None)
    await update.message.reply_text("История очищена ✅")

async def autopost_switch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, context):
        return
    args = (context.args or [])[:1]
    if not args or args[0] not in {"on", "off"}:
        await update.message.reply_text("Используйте: /autopost on|off")
        return
    cfg = get_cfg(update.effective_chat.id)
    cfg.autopost_enabled = args[0] == "on"
    await update.message.reply_text(f"Автопосты {'включены' if cfg.autopost_enabled else 'выключены'}")

async def set_interval(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, context): return
    try:
        sec = int(context.args[0])
        cfg = get_cfg(update.effective_chat.id)
        cfg.interval = max(300, sec)
        await update.message.reply_text(f"Интервал автопоста = {cfg.interval} сек")
    except (IndexError, ValueError):
        await update.message.reply_text("Пример: /set_interval 7200")

async def set_minmsgs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, context): return
    try:
        n = int(context.args[0])
        cfg = get_cfg(update.effective_chat.id)
        cfg.min_messages = max(1, n)
        await update.message.reply_text(f"Минимум сообщений = {cfg.min_messages}")
    except (IndexError, ValueError):
        await update.message.reply_text("Пример: /set_minmsgs 10")

async def set_msgsize(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, context): return
    size = (context.args or [""])[0].lower()
    if size not in {"small", "medium", "large"}:
        await update.message.reply_text("Варианты: small | medium | large")
        return
    cfg = get_cfg(update.effective_chat.id)
    cfg.msg_size = size
    await update.message.reply_text(f"Размер ответов = {size}")

# ---------- Основной обработчик текста ----------
async def send_bot_response(update: Update, context: ContextTypes.DEFAULT_TYPE, chat_id: int, prompt: str, image: Optional[Image.Image] = None):
    """Общая логика для отправки ответа от LLM."""
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    loop = asyncio.get_running_loop()
    try:
        reply, model_used = await loop.run_in_executor(None, llm_request, chat_id, prompt, image)
        model_display = model_used.replace("gemini-", "").replace("-", " ").title()
        full_reply = f"*{model_display}*\n\n{reply}"
        
        sanitized_reply = sanitize_telegram_markdown(full_reply)
        
        for chunk in split_long_message(sanitized_reply):
            try:
                await update.message.reply_text(chunk, parse_mode="MarkdownV2", disable_web_page_preview=True)
            except BadRequest as e:
                log.warning(f"MarkdownV2 failed, sending plain text: {e}")
                await update.message.reply_text(chunk, disable_web_page_preview=True)

    except Exception as e:
        log.exception(e)
        await update.message.reply_text("⚠️ Ошибка модели.", disable_web_page_preview=True)

async def handle_msg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    
    chat_id = update.effective_chat.id
    text = update.message.text
    
    # Проверка на упоминание бота в группах
    if update.message.chat.type in (ChatType.GROUP, ChatType.SUPERGROUP):
        # Проверяем, есть ли упоминания в сообщении
        if not update.message.entities:
            return  # Нет entities, значит нет упоминаний
        
        bot_mentioned = False
        for entity in update.message.entities:
            if entity.type == MessageEntityType.MENTION:
                # Проверяем, что упоминание указывает на нашего бота
                mention_text = text[entity.offset:entity.offset + entity.length]
                # Убираем @ для сравнения
                mention_username = mention_text.lstrip('@')
                # Получаем username бота из контекста (сохраняем его глобально или передаём через context)
                # Или используем bot.username напрямую
                if mention_username.lower() == context.bot.username.lower():
                    bot_mentioned = True
                    break
        
        if not bot_mentioned:
            return  # Бот не упомянут, игнорируем
    
    # счётчик для автопоста
    cfg = get_cfg(chat_id)
    cfg.new_msg_counter += 1

    # LLM prompt
    sys_prompt = answer_size_prompt(cfg.msg_size)
    # Убираем упоминание бота из текста, чтобы оно не попало в промпт
    if update.message.chat.type in (ChatType.GROUP, ChatType.SUPERGROUP):
        # Удаляем упоминание бота из текста
        for entity in reversed(update.message.entities or []):
            if entity.type == MessageEntityType.MENTION:
                mention_text = text[entity.offset:entity.offset + entity.length]
                mention_username = mention_text.lstrip('@')
                if mention_username.lower() == context.bot.username.lower():
                    # Удаляем упоминание и лишние пробелы
                    text = (text[:entity.offset] + text[entity.offset + entity.length:]).strip()
    
    prompt = f"{sys_prompt}\n{text}" if sys_prompt else text

    await send_bot_response(update, context, chat_id, prompt)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    chat_id = update.effective_chat.id
    text = update.message.caption or "Опиши это изображение"
    
    # счётчик для автопоста
    cfg = get_cfg(chat_id)
    cfg.new_msg_counter += 1
    
    # Получаем фото - берем самое большое
    photos = update.message.photo
    if not photos:
        return
    photo = photos[-1]
    
    await send_bot_response(update, context, chat_id, text, image)

# ---------- JOB для проверки моделей ----------
async def check_models_job(context: CallbackContext):
    """Проверяет доступные модели каждые 4 часа (14400 сек)"""
    log.info("Checking available models...")
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, check_available_models)
    except Exception as e:
        log.exception(f"Error checking models: {e}")

# ---------- JOB для автопостов ----------
async def autopost_job(context: CallbackContext):
    for chat_id, cfg in list(configs.items()):
        if not cfg.autopost_enabled:
            continue
        if cfg.new_msg_counter < cfg.min_messages:
            continue
        if time.time() - cfg.last_post_ts < cfg.interval:
            continue

        prompt = (
            f"Сделай краткий дайджест последних {cfg.new_msg_counter} сообщений "
            "из группового чата. Выдели основные вопросы и идеи."
        )
        log.info(f"Autopost in chat {chat_id}")
        try:
            loop = asyncio.get_event_loop()
            summary, model_used = await loop.run_in_executor(None, llm_request, chat_id, prompt)
            model_display = model_used.replace("gemini-", "").replace("-", " ").title()
            
            message_text = f"📰 Автодайджест ({model_display}):\n{summary}"
            sanitized_text = sanitize_telegram_markdown(message_text)
            message_parts = split_long_message(sanitized_text)
            for part in message_parts:
                try:
                    await context.bot.send_message(chat_id, part, parse_mode="Markdown")
                except BadRequest as e:
                    if "entities" in str(e).lower() or "parse" in str(e).lower():
                        log.warning("Markdown parse failed for autopost, sending plain text. Error: %s", e)
                        await context.bot.send_message(chat_id, part)
                    elif "too long" in str(e).lower():
                        plain_parts = split_long_message(part, max_length=4000)
                        for plain_part in plain_parts:
                            await context.bot.send_message(chat_id, plain_part)
                    else:
                        log.error("Failed to send autopost message: %s", e)

        except Exception as e:
            log.error(f"Autopost failed for chat {chat_id}: {e}")

# ---------- Main ----------
def main(): # <--- Снова делаем main синхронной
    token = os.getenv("TG_TOKEN")
    if not token:
        raise RuntimeError("TG_TOKEN env not set")

    # Проверка наличия ADMIN_ID при запуске
    if not ADMIN_ID:
        raise RuntimeError("ADMIN_ID env not set. Бот не сможет определить администратора.")

    # Получаем ID и username бота с помощью прямого HTTP-запроса
    try:
        response = requests.get(f"https://api.telegram.org/bot{token}/getMe")
        response.raise_for_status() # Вызовет ошибку для плохих статусов HTTP
        bot_info = response.json().get('result', {})
        bot_id = bot_info.get('id')
        bot_username = bot_info.get('username')
        if not bot_id or not bot_username:
            raise RuntimeError("Не удалось получить информацию о боте от Telegram.")
        log.info(f"Получен ID бота: {bot_id}, Username: @{bot_username}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ошибка при получении информации о боте от Telegram: {e}")

    app = ApplicationBuilder().token(token).build()

    # Инициализация для JobQueue и других компонентов
    # app.initialize() не вызываем здесь напрямую, так как main синхронная.
    # run_polling() сама вызывает инициализацию внутренних компонентов.
    # Bot ID мы уже получили выше.

    # Добавляем обработчики команд
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("settings", settings_cmd))
    app.add_handler(CommandHandler("autopost", autopost_switch))
    app.add_handler(CommandHandler(f"autopost_{bot_username}", autopost_switch)) # Для совместимости с упоминаниями
    app.add_handler(CommandHandler("set_interval", set_interval))
    app.add_handler(CommandHandler(f"set_interval_{bot_username}", set_interval)) # Для совместимости с упоминаниями
    app.add_handler(CommandHandler("set_msgsize", set_msgsize))
    app.add_handler(CommandHandler(f"set_msgsize_{bot_username}", set_msgsize)) # Для совместимости с упоминаниями
    app.add_handler(CommandHandler("privacy", privacy_cmd))
    app.add_handler(CommandHandler(f"privacy_{bot_username}", privacy_cmd)) # Для совместимости с упоминаниями
    app.add_handler(CommandHandler("delete_data", delete_data)) # <-- Добавляем новую команду
    app.add_handler(CommandHandler(f"delete_data_{bot_username}", delete_data)) # Для совместимости с упоминаниями

    # Обработка текстовых сообщений (теперь проверяем упоминания внутри handle_msg)
    # В группах будем проверять упоминания внутри функции, а в личных - отвечать на любой текст
    app.add_handler(MessageHandler(filters.TEXT & filters.ChatType.GROUPS & ~filters.COMMAND, handle_msg))
    app.add_handler(MessageHandler(filters.TEXT & filters.ChatType.PRIVATE & ~filters.COMMAND, handle_msg))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Запускаем Flask приложение в отдельном потоке
    threading.Thread(target=lambda: flask_app.run(host='0.0.0.0', port=int(os.getenv("PORT", 10000)))).start()
    log.info("Flask app started on port 10000")


    if app.job_queue:
        app.job_queue.run_repeating(check_models_job, interval=14400, first=60) # Каждые 4 часа
        app.job_queue.run_repeating(autopost_job, interval=60, first=60) # Каждую минуту
        log.info("JobQueue initialized")
    else:
        log.warning("JobQueue not available - scheduled jobs disabled")

    log.info("Bot started 🚀")
    app.run_polling(allowed_updates=Update.ALL_TYPES, stop_signals=None) # <--- Запускаем run_polling синхронно

if __name__ == "__main__":
    main() # <--- Вызываем синхронную main