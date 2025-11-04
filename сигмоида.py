#!/usr/bin/env python3
# filename: wizard_bot.py
import os, asyncio, logging, time, io, re, json, atexit
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from PIL import Image
from telegram import Update, File
from telegram.constants import ChatType, MessageEntityType, ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    ApplicationBuilder, ContextTypes,
    CommandHandler, MessageHandler, filters, CallbackContext
)
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, ContentType, PartType, Tool, FunctionCall
from flask import Flask, render_template_string
import threading
import requests

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


@flask_app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

# ---------- Политика конфиденциальности ----------
PRIVACY_POLICY_TEXT = """
<b>Политика конфиденциальности и обработки данных Сигмоида</b>

⚠️ <b>Важно:</b> Ваши сообщения, изображения и другие медиафайлы, отправленные этому боту, передаются в Google Gemini API для обработки. Это необходимо для функционирования бота и генерации ответов.
История диалога и настройки чатов сохраняются в JSON-файлах на сервере для обеспечения непрерывности диалога между перезапусками.
Используя бота, вы соглашаетесь с передачей данных в Google для их обработки и хранением данных на сервере.

<b>Согласие:</b> Продолжая использовать бота в личных сообщениях или отправляя упоминания боту в групповых чатах, вы подтверждаете свое согласие с данной политикой. Администраторы групповых чатов несут ответственность за информирование участников о том, что их сообщения могут обрабатываться ботом через сторонний API.

<b>Дополнительно:</b> Наш бот также подпадает под действие <b>Стандартной политики конфиденциальности Telegram для ботов</b>. Ознакомиться с ней можно по ссылке: <a href="https://telegram.org/privacy-tpa">https://telegram.org/privacy-tpa</a>.

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

MODELS = [
    "gemini-1.5-pro-latest", "gemini-1.5-flash-latest",
]
MAX_HISTORY = 20
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
DATA_DIR = "data"
HISTORY_FILE = os.path.join(DATA_DIR, "history.json")
CONFIGS_FILE = os.path.join(DATA_DIR, "configs.json")

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

# ---------- Сохранение и загрузка данных ----------
def load_data():
    global history, configs
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            history = {int(k): v for k, v in json.load(f).items()}
            log.info(f"Loaded {len(history)} chat histories.")
    except (FileNotFoundError, json.JSONDecodeError):
        history = {}
    try:
        with open(CONFIGS_FILE, 'r', encoding='utf-8') as f:
            configs = {int(k): ChatConfig(**v) for k, v in json.load(f).items()}
            log.info(f"Loaded {len(configs)} chat configs.")
    except (FileNotFoundError, json.JSONDecodeError):
        configs = {}

def save_data():
    log.info("Saving data...")
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        configs_to_save = {cid: asdict(cfg) for cid, cfg in configs.items()}
        with open(CONFIGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(configs_to_save, f, ensure_ascii=False, indent=2)
        log.info("Data saved.")
    except Exception as e:
        log.error(f"Failed to save data: {e}", exc_info=True)

async def save_data_job(context: CallbackContext):
    await asyncio.get_running_loop().run_in_executor(None, save_data)

# ---------- Вспомогалки ----------
def get_cfg(chat_id: int) -> ChatConfig:
    if chat_id not in configs:
        configs[chat_id] = ChatConfig()
    return configs[chat_id]

def llm_request(chat_id: int, prompt_parts: List[PartType]) -> Tuple[Optional[str], str, Optional[FunctionCall]]:
    global current_key_idx, current_model_idx
    chat_history = history.get(chat_id, [])

    if len(chat_history) > MAX_HISTORY:
        log.info(f"Summarizing history for chat {chat_id}...")
        try:
            summary_prompt = "Summarize this conversation in a concise paragraph for context."
            summary_model = genai.GenerativeModel("gemini-1.5-flash-latest", api_key=API_KEYS[current_key_idx])
            response = summary_model.generate_content(chat_history + [{'role': 'user', 'parts': [{'text': summary_prompt}]}])
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
                contents = chat_history + [{'role': 'user', 'parts': prompt_parts}]
                response = model.generate_content(contents)
                
                if response.candidates and response.candidates[0].content.parts[0].function_call:
                    return None, model_name, response.candidates[0].content.parts[0].function_call
                
                answer = response.text
                history[chat_id] = contents + [{'role': 'model', 'parts': [{'text': answer}]}]
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
    model_name = "gemini-1.5-flash-latest"
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
    await update.message.reply_text("👋 Я Gemini бот. /help – справка\n\n"
                                    "⚠️ <b>Важно:</b> Ваши сообщения и медиафайлы обрабатываются через Google Gemini API. /privacy",
                                    parse_mode=ParseMode.HTML)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
    await update.message.reply_text(PRIVACY_POLICY_TEXT, parse_mode=ParseMode.HTML, disable_web_page_preview=True)

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    history.pop(update.effective_chat.id, None)
    await update.message.reply_text("История очищена ✅")
    
async def delete_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, context): return
    if not context.args: return await update.message.reply_text("Укажите ID чата.")
    try:
        target_id = int(context.args[0])
        history.pop(target_id, None)
        configs.pop(target_id, None)
        await update.message.reply_text(f"Данные для ID {target_id} удалены.")
    except (ValueError, IndexError):
        await update.message.reply_text("Неверный ID.")

async def settings_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cfg = get_cfg(update.effective_chat.id)
    await update.message.reply_text(f"<b>Автопосты:</b> {'вкл' if cfg.autopost_enabled else 'выкл'}.\n"
                                    f"<b>Интервал:</b> {cfg.interval} сек, <b>мин. сообщений:</b> {cfg.min_messages}.\n"
                                    f"<b>Размер ответов:</b> {cfg.msg_size or 'default'}.",
                                    parse_mode=ParseMode.HTML)

async def autopost_switch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, context): return
    if not context.args or context.args[0] not in {"on", "off"}: return await update.message.reply_text("Пример: /autopost on")
    cfg = get_cfg(update.effective_chat.id)
    cfg.autopost_enabled = (context.args[0] == "on")
    await update.message.reply_text(f"Автопосты {'включены' if cfg.autopost_enabled else 'выключены'}.")

async def set_interval(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, context): return
    try:
        cfg = get_cfg(update.effective_chat.id)
        cfg.interval = max(300, int(context.args[0]))
        await update.message.reply_text(f"Интервал автопоста = {cfg.interval} сек.")
    except (IndexError, ValueError):
        await update.message.reply_text("Пример: /set_interval 7200")

async def set_minmsgs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, context): return
    try:
        cfg = get_cfg(update.effective_chat.id)
        cfg.min_messages = max(1, int(context.args[0]))
        await update.message.reply_text(f"Минимум сообщений = {cfg.min_messages}.")
    except (IndexError, ValueError):
        await update.message.reply_text("Пример: /set_minmsgs 10")

async def set_msgsize(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
    await update.message.reply_text(f"Размер ответов = {cfg.msg_size or 'default'}.")

async def draw_image_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
    try:
        reply, model_used, function_call = await asyncio.get_running_loop().run_in_executor(None, llm_request, chat_id, prompt_parts)

        if function_call and function_call.name == "generate_image":
            return await generate_and_send_image(update, context, function_call.args.get("prompt", ""))

        if reply:
            model_display = model_used.replace("gemini-", "").replace("-latest", "").title()
            full_reply = f"<b>{model_display}</b>\n\n{reply}"
            for chunk in split_long_message(full_reply):
                try:
                    await update.message.reply_text(chunk, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
                except BadRequest as e:
                    log.warning(f"HTML parse failed, sending plain text. Error: {e}")
                    await update.message.reply_text(chunk, disable_web_page_preview=True)
    except Exception as e:
        log.exception(e)
        await update.message.reply_text("⚠️ Ошибка модели.")

async def handle_text_and_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message: return
    chat_id = update.effective_chat.id
    text = update.message.text or update.message.caption or ""

    if update.message.chat.type in (ChatType.GROUP, ChatType.SUPERGROUP):
        bot_mentioned = any(
            text[e.offset:e.offset+e.length].lstrip('@').lower() == context.bot.username.lower()
            for e in (update.message.entities or []) if e.type == MessageEntityType.MENTION
        )
        is_reply_to_bot = update.message.reply_to_message and update.message.reply_to_message.from_user.username == context.bot.username
        
        if not (bot_mentioned or is_reply_to_bot):
            get_cfg(chat_id).new_msg_counter += 1
            return
        
        for e in reversed(update.message.entities or []):
            if e.type == MessageEntityType.MENTION and text[e.offset:e.offset+e.length].lstrip('@').lower() == context.bot.username.lower():
                text = (text[:e.offset] + text[e.offset+e.length:]).strip()
    
    cfg = get_cfg(chat_id)
    cfg.new_msg_counter += 1
    prompt_parts = []
    if text: prompt_parts.append(answer_size_prompt(cfg.msg_size) + text)
    if update.message.photo:
        file = await update.message.photo[-1].get_file()
        file_bytes = await file.download_as_bytearray()
        prompt_parts.insert(0, Image.open(io.BytesIO(file_bytes)))

    if not prompt_parts: return
    await send_bot_response(update, context, chat_id, prompt_parts)

async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message: return
    media, prompt = None, ""
    if update.message.voice: media, prompt = update.message.voice, "Расшифруй это голосовое сообщение:"
    elif update.message.video: media, prompt = update.message.video, "Опиши, что происходит на этом видео:"
    elif update.message.video_note: media, prompt = update.message.video_note, "Опиши это видео-сообщение:"
    
    if not media: return
    chat_id = update.effective_chat.id
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    file = await media.get_file()
    prompt_parts = [{"mime_type": file.mime_type, "data": await file.download_as_bytearray()}, {"text": prompt}]
    await send_bot_response(update, context, chat_id, prompt_parts)

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
        except Exception as e:
            log.error(f"Autopost failed for chat {chat_id}: {e}")

# ---------- Main ----------
def main():
    load_data()
    atexit.register(save_data)

    token, admin_id = os.getenv("TG_TOKEN"), os.getenv("ADMIN_ID")
    if not token or not admin_id: raise RuntimeError("TG_TOKEN и ADMIN_ID должны быть установлены")
    
    try:
        bot_info = requests.get(f"https://api.telegram.org/bot{token}/getMe").json().get('result', {})
        if not bot_info.get('username'): raise RuntimeError("Не удалось получить username бота.")
        log.info(f"Bot Username: @{bot_info['username']}")
    except Exception as e:
        raise RuntimeError(f"Ошибка при получении информации о боте: {e}")

    app = ApplicationBuilder().token(token).build()
    app.bot.username = bot_info['username']

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
        app.add_handler(CommandHandler(f"{command}_{app.bot.username}", callback))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_and_photo))
    app.add_handler(MessageHandler(filters.PHOTO, handle_text_and_photo))
    app.add_handler(MessageHandler(filters.VOICE | filters.VIDEO | filters.VIDEO_NOTE, handle_media))

    if app.job_queue:
        app.job_queue.run_repeating(save_data_job, 60, 60)
        app.job_queue.run_repeating(check_models_job, 14400, 60)
        app.job_queue.run_repeating(autopost_job, 60, 60)
        log.info("JobQueue initialized")

    threading.Thread(target=lambda: flask_app.run(host='0.0.0.0', port=int(os.getenv("PORT", 10000))), daemon=True).start()
    log.info("Flask app started")
    log.info("Bot started 🚀")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()