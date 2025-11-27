# Copyright (c) 2025 sprouee
import asyncio
import html
import io
from functools import partial
import secrets
from typing import List, Optional
from telegram import ChatMember, InlineKeyboardButton, InlineKeyboardMarkup, Update, WebAppInfo
from telegram.constants import ChatAction, ChatType, MessageEntityType, ParseMode
from telegram.error import BadRequest
from telegram.ext import ContextTypes
from app import config
from app.config import OPENROUTER_MODELS
from app.llm.client import llm_generate_image, llm_request
from app.logging_config import log
from app.security.privacy import PRIVACY_POLICY_TEXT
from app.state import ChatConfig, configs, history
from app.storage.redis_store import create_login_code, persist_chat_data, record_user_profile, redis_client, user_profiles
from app.utils.text import answer_size_prompt, split_long_message, strip_html_tags
from app.game.generator import GeneratedGame, generate_game
from app.middleware.rate_limit import check_rate_limit, get_user_stats
from app.middleware.cache import get_cached_response, cache_response, get_cache_stats
from app.features.translator import translate_text, detect_language
from app.features.summarizer import summarize_text, summarize_url

MAX_IMAGE_BYTES = config.MAX_IMAGE_BYTES
MAX_VIDEO_BYTES = getattr(config, "ZAI_MAX_VIDEO_BYTES", 200 * 1024 * 1024)
async def ensure_user_profile(update: Update):
    chat = update.effective_chat
    user = update.effective_user
    if not chat or not user:
        return
    if record_user_profile(chat.id, user):
        await persist_chat_data(chat.id)
def get_cfg(chat_id: int) -> ChatConfig:
    if chat_id not in configs:
        configs[chat_id] = ChatConfig()
    return configs[chat_id]
async def is_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    # Используем constant-time comparison для защиты от timing attacks
    if config.ADMIN_ID and secrets.compare_digest(str(update.effective_user.id), str(config.ADMIN_ID)):
        return True
    await update.message.reply_text("Эта команда доступна только администратору.")
    return False
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    await update.message.reply_text(
        "👋 Я Сигмоид бот. /help – справка\n\n"
        "⚠️ <b>Важно:</b> Ваши сообщения обрабатываются через AI (Gemini/OpenRouter/Pollinations).\n\n"
        "⚠️ <b>Disclaimer:</b> Весь контент генерируется AI. Разработчик не несет ответственности за сгенерированный контент.\n\n"
        "/privacy – политика конфиденциальности",
        parse_mode=ParseMode.HTML,
    )
async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    poll_models = ", ".join(config.POLLINATIONS_MODELS) if getattr(config, "POLLINATIONS_MODELS", None) else config.POLLINATIONS_MODEL
    poll_text_models = ", ".join(config.POLLINATIONS_TEXT_MODELS) if getattr(config, "POLLINATIONS_TEXT_MODELS", None) else config.POLLINATIONS_TEXT_DEFAULT
    provider_options = ["gemini"]
    if getattr(config, "ZAI_API_KEY", None):
        provider_options.append("zai")
    if config.OPENROUTER_API_KEYS and config.OPENROUTER_MODELS:
        provider_options.append("openrouter")
    if getattr(config, "POLLINATIONS_TEXT_MODELS", None):
        provider_options.append("pollinations")
    provider_hint = ", ".join(provider_options + ["auto"])
    
    video_hint = "🎬 Видео поддерживается через Z.AI!\n\n" if getattr(config, "ZAI_API_KEY", None) else ""
    
    await update.message.reply_text(
        f"{video_hint}"
        "<b>Основные:</b>\n"
        "/tr [язык] текст – перевести 🌍\n"
        "/sum текст/url – краткое содержание 📝\n"
        "/draw описание – нарисовать 🎨\n"
        "/game идея – сгенерировать игру 🎮\n"
        "/stats – статистика 📊\n\n"
        "<b>Настройки:</b>\n"
        "/settings – текущие настройки\n"
        f"/set_provider &lt;{provider_hint}&gt; – выбрать LLM\n"
        "/set_or_model – модель OpenRouter\n"
        "/set_zai_model – модель Z.AI\n"
        "/set_msgsize &lt;s|m|l&gt; – размер ответа\n\n"
        "<b>Админ:</b>\n"
        "/autopost on|off – автопосты\n"
        "/set_interval – интервал\n"
        "/set_minmsgs – мин. сообщений\n\n"
        "/reset – очистить историю\n"
        "/privacy – политика конфиденциальности",
        parse_mode=ParseMode.HTML,
    )
async def privacy_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    await update.message.reply_text(PRIVACY_POLICY_TEXT, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    chat_id = update.effective_chat.id
    history.pop(chat_id, None)
    await persist_chat_data(chat_id)
    await update.message.reply_text("История очищена ✓")
async def delete_data_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    if not update.message or not update.effective_chat or not update.effective_user:
        return
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    username = update.effective_user.username or update.effective_user.first_name
    chat_type = update.effective_chat.type
    # Используем constant-time comparison для защиты от timing attacks
    is_bot_admin = config.ADMIN_ID and secrets.compare_digest(str(user_id), str(config.ADMIN_ID))
    can_delete = False
    if chat_type == ChatType.PRIVATE:
        can_delete = True
        log.info(f"User {username} ({user_id}) in private chat requested to delete their data.")
    elif chat_type in [ChatType.GROUP, ChatType.SUPERGROUP]:
        if is_bot_admin:
            can_delete = True
            log.info(f"Bot admin {username} ({user_id}) in group chat ({chat_id}) requested to delete chat data.")
        else:
            try:
                chat_member = await context.bot.get_chat_member(chat_id, user_id)
                if chat_member.status in [ChatMember.ADMINISTRATOR, ChatMember.CREATOR]:
                    can_delete = True
                    log.info(f"Group admin {username} ({user_id}) in group chat ({chat_id}) requested to delete chat data.")
                else:
                    log.warning(
                        f"User {username} ({user_id}) tried to delete data in group chat ({chat_id}) without admin rights."
                    )
                    await update.message.reply_html("<b>Эту команду могут использовать только администраторы группы.</b>")
                    return
            except Exception as exc:
                log.error(f"Error checking chat member status in group {chat_id}: {exc}")
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
            redis_client.delete(
                f"{config.HISTORY_KEY_PREFIX}{chat_id}",
                f"{config.CONFIG_KEY_PREFIX}{chat_id}",
                f"{config.USER_KEY_PREFIX}{chat_id}",
            )
            log.info(f"Удалены ключи Redis для чата {chat_id}.")
        except Exception as exc:
            log.error(f"Не удалось удалить ключи Redis для чата {chat_id}: {exc}", exc_info=True)
        await update.message.reply_html(
            "<b>Все данные для этого чата (история переписки и настройки) были успешно удалены.</b>\n"
            "Если вы продолжите использовать бота, начнется новая история."
        )
async def delete_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    if not await is_admin(update, context):
        return
    if not context.args:
        await update.message.reply_text("Укажите ID чата.")
        return
    try:
        target_id = int(context.args[0])
        history.pop(target_id, None)
        configs.pop(target_id, None)
        user_profiles.pop(target_id, None)
        try:
            redis_client.delete(
                f"{config.HISTORY_KEY_PREFIX}{target_id}",
                f"{config.CONFIG_KEY_PREFIX}{target_id}",
                f"{config.USER_KEY_PREFIX}{target_id}",
            )
        except Exception as exc:
            log.error(f"Не удалось удалить данные чата {target_id} из Redis: {exc}", exc_info=True)
        await update.message.reply_text(f"Данные для ID {target_id} удалены.")
    except (ValueError, IndexError):
        await update.message.reply_text("Неверный ID.")
async def settings_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    cfg = get_cfg(update.effective_chat.id)
    provider = cfg.llm_provider or "auto"
    pollinations_text = cfg.pollinations_text_model or config.POLLINATIONS_TEXT_DEFAULT
    openrouter_model = cfg.openrouter_model or "ротация"
    zai_model = getattr(cfg, "zai_model", None) or getattr(config, "ZAI_DEFAULT_MODEL", "glm-4.6")
    
    provider_line = f"<b>LLM:</b> {html.escape(provider)}"
    if provider == "pollinations" and pollinations_text:
        provider_line += f" (Pollinations → {html.escape(pollinations_text)})"
    elif provider == "openrouter":
        provider_line += f" (OpenRouter → {html.escape(openrouter_model)})"
    elif provider == "zai":
        provider_line += f" (Z.AI → {html.escape(zai_model)}, 🎬 видео)"
    
    zai_status = "✅ Z.AI (видео)" if getattr(config, "ZAI_API_KEY", None) else "❌ Z.AI"
    
    await update.message.reply_text(
        f"<b>Автопосты:</b> {'вкл' if cfg.autopost_enabled else 'выкл'}.\n"
        f"<b>Интервал:</b> {cfg.interval} сек, <b>мин. сообщений:</b> {cfg.min_messages}.\n"
        f"<b>Размер ответа:</b> {cfg.msg_size or 'default'}.\n"
        f"{provider_line}\n"
        f"<b>Провайдеры:</b> {zai_status}",
        parse_mode=ParseMode.HTML,
    )
async def autopost_switch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    if not await is_admin(update, context):
        return
    if not context.args or context.args[0] not in {"on", "off"}:
        await update.message.reply_text("Пример: /autopost on")
        return
    cfg = get_cfg(update.effective_chat.id)
    cfg.autopost_enabled = context.args[0] == "on"
    await persist_chat_data(update.effective_chat.id)
    await update.message.reply_text(f"Автопосты {'включены' if cfg.autopost_enabled else 'выключены'}.")
async def set_interval(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    if not await is_admin(update, context):
        return
    try:
        cfg = get_cfg(update.effective_chat.id)
        cfg.interval = max(300, int(context.args[0]))
        await persist_chat_data(update.effective_chat.id)
        await update.message.reply_text(f"Интервал автопоста = {cfg.interval} сек.")
    except (IndexError, ValueError):
        await update.message.reply_text("Пример: /set_interval 7200")
async def set_minmsgs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    if not await is_admin(update, context):
        return
    try:
        cfg = get_cfg(update.effective_chat.id)
        cfg.min_messages = max(1, int(context.args[0]))
        await persist_chat_data(update.effective_chat.id)
        await update.message.reply_text(f"Минимум сообщений = {cfg.min_messages}.")
    except (IndexError, ValueError):
        await update.message.reply_text("Пример: /set_minmsgs 10")
async def set_msgsize(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    if not await is_admin(update, context):
        return
    size = (context.args or [""])[0].lower()
    if size not in {"small", "medium", "large", "s", "m", "l", ""}:
        await update.message.reply_text("Варианты: small, medium, large или пусто (default)")
        return
    cfg = get_cfg(update.effective_chat.id)
    if size in {"s", "m", "l"}:
        cfg.msg_size = size
    elif size:
        cfg.msg_size = size[0]
    else:
        cfg.msg_size = ""
    await persist_chat_data(update.effective_chat.id)
    await update.message.reply_text(f"Размер ответа = {cfg.msg_size or 'default'}.")
async def generate_and_send_image(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt: str):
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="upload_photo")
    try:
        chat_id = update.effective_chat.id if update.effective_chat else None
        poll_model: Optional[str] = None
        if chat_id is not None:
            cfg = get_cfg(chat_id)
            poll_model = cfg.pollinations_model or None
        loop = asyncio.get_running_loop()
        image_bytes, model_used = await loop.run_in_executor(
            None, partial(llm_generate_image, prompt, poll_model)
        )
        if image_bytes:
            model_display = model_used.replace("gemini-", "").replace("-latest", "").title()
            caption = f"🖼️ «{prompt}»\n\n<b>Generated by {model_display}</b>"
            await update.message.reply_photo(photo=image_bytes, caption=caption, parse_mode=ParseMode.HTML)
        else:
            await update.message.reply_text("⚠️ Не удалось создать изображение.")
    except Exception as exc:
        log.exception(exc)
        await update.message.reply_text("⚠️ Ошибка при генерации изображения.")
async def draw_image_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    if not context.args:
        await update.message.reply_text("Пример: /draw кот в скафандре")
        return
    await generate_and_send_image(update, context, " ".join(context.args))


async def set_draw_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    message = update.message
    if not message:
        return
    if not config.POLLINATIONS_ENABLED:
        await message.reply_text("Pollinations сейчас отключён. Включите интеграцию через переменные окружения.")
        return
    chat_id = update.effective_chat.id
    cfg = get_cfg(chat_id)
    available = config.POLLINATIONS_MODELS or [config.POLLINATIONS_MODEL]
    lookup = {item.lower(): item for item in available}
    if not context.args:
        current = cfg.pollinations_model or config.POLLINATIONS_MODEL
        await message.reply_text(
            "Текущая модель Pollinations: <b>{current}</b>\n"
            "Доступные: {choices}\n"
            "Команда: <code>/set_draw_model flux</code> или <code>/set_draw_model default</code> для сброса."
            .format(current=html.escape(current), choices=", ".join(available)),
            parse_mode=ParseMode.HTML,
        )
        return
    raw_value = context.args[0].strip()
    value_lower = raw_value.lower()
    if value_lower in {"default", "reset"}:
        cfg.pollinations_model = ""
        await persist_chat_data(chat_id)
        await message.reply_text(
            f"Модель Pollinations сброшена. Теперь используется значение по умолчанию: {config.POLLINATIONS_MODEL}."
        )
        return
    if value_lower not in lookup:
        await message.reply_text(
            "Неизвестная модель. Доступные варианты: {choices}".format(choices=", ".join(available))
        )
        return
    selected = lookup[value_lower]
    cfg.pollinations_model = selected
    await persist_chat_data(chat_id)
    await message.reply_text(f"Для этого чата теперь используется Pollinations модель: {selected}")


async def set_pollinations_text_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    message = update.message
    if not message:
        return
    if not getattr(config, "POLLINATIONS_TEXT_MODELS", None):
        await message.reply_text("Текстовые модели Pollinations не настроены.")
        return
    chat_id = update.effective_chat.id
    cfg = get_cfg(chat_id)
    available = config.POLLINATIONS_TEXT_MODELS
    lookup = {model.lower(): model for model in available}
    if not context.args:
        current = cfg.pollinations_text_model or config.POLLINATIONS_TEXT_DEFAULT or available[0]
        await message.reply_text(
            "Текущая текстовая модель Pollinations: <b>{}</b>\nДоступные: {}".format(
                html.escape(current), ", ".join(available)
            ),
            parse_mode=ParseMode.HTML,
        )
        return
    value = context.args[0].strip().lower()
    selected = lookup.get(value)
    if not selected:
        await message.reply_text(
            "Неизвестная модель. Доступные варианты: {}.".format(", ".join(available))
        )
        return
    cfg.pollinations_text_model = selected
    await persist_chat_data(chat_id)
    await message.reply_text(f"Генерация текста Pollinations теперь использует модель: {selected}")


async def set_provider(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    message = update.message
    if not message:
        return
    available: List[str] = []
    if config.API_KEYS:
        available.append("gemini")
    if getattr(config, "ZAI_API_KEY", None):
        available.append("zai")
    if config.OPENROUTER_API_KEYS and config.OPENROUTER_MODELS:
        available.append("openrouter")
    if getattr(config, "POLLINATIONS_TEXT_MODELS", None):
        available.append("pollinations")
    if not available:
        await message.reply_text("Нет доступных провайдеров LLM. Проверь ключи в настройках.")
        return
    chat_id = update.effective_chat.id
    cfg = get_cfg(chat_id)
    if not context.args:
        current = cfg.llm_provider or "auto"
        await message.reply_text(
            "Текущий провайдер: <b>{current}</b>\n"
            "Доступные варианты: {choices}, auto (используй <code>/set_provider auto</code> для автоматического выбора)."
            .format(current=html.escape(current), choices=", ".join(available)),
            parse_mode=ParseMode.HTML,
        )
        return
    value = context.args[0].strip().lower()
    if value in {"auto", "default"}:
        cfg.llm_provider = ""
        await persist_chat_data(chat_id)
        await message.reply_text("Буду автоматически переключаться между доступными провайдерами.")
        return
    if value not in available:
        await message.reply_text(
            "Неизвестный провайдер. Доступные варианты: {choices}, auto."
            .format(choices=", ".join(available))
        )
        return
    cfg.llm_provider = value
    if value == "pollinations":
        if not cfg.pollinations_text_model or cfg.pollinations_text_model not in config.POLLINATIONS_TEXT_MODELS:
            cfg.pollinations_text_model = (
                config.POLLINATIONS_TEXT_DEFAULT or config.POLLINATIONS_TEXT_MODELS[0]
            )
    await persist_chat_data(chat_id)
    if value == "pollinations":
        await message.reply_text(
            "Для этого чата выбран провайдер LLM: pollinations.\n"
            f"Используется текстовая модель Pollinations: {cfg.pollinations_text_model}.\n"
            "Чтобы сменить модель, воспользуйся командой /set_pollinations_text_model <название>."
        )
    elif value == "zai":
        zai_model = getattr(cfg, "zai_model", None) or config.ZAI_DEFAULT_MODEL
        await message.reply_text(
            f"Для этого чата выбран провайдер LLM: zai (Z.AI/ZhipuAI).\n"
            f"Используется модель: {zai_model}.\n"
            "🎬 Z.AI поддерживает видео! Отправь видео и я его проанализирую.\n"
            "Чтобы сменить модель, воспользуйся командой /set_zai_model <название>."
        )
    else:
        await message.reply_text(f"Для этого чата выбран провайдер LLM: {value}")
async def send_bot_response(update: Update, context: ContextTypes.DEFAULT_TYPE, chat_id: int, prompt_parts: List):
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    data_updated = False
    cfg = get_cfg(chat_id)
    provider_override = cfg.llm_provider or None
    
    # Проверяем кэш для текстовых запросов
    cached_result = None
    if len(prompt_parts) == 1 and isinstance(prompt_parts[0], str):
        cached_result = get_cached_response(chat_id, prompt_parts[0])
    
    if cached_result:
        reply, model_used = cached_result
        function_call = None
        log.info(f"Using cached response for chat {chat_id}")
    else:
        try:
            reply, model_used, function_call = await asyncio.get_running_loop().run_in_executor(
                None, llm_request, chat_id, prompt_parts, provider_override
            )
            # Кэшируем ответ
            if reply and len(prompt_parts) == 1 and isinstance(prompt_parts[0], str):
                cache_response(chat_id, prompt_parts[0], reply, model_used)
        except Exception as exc:
            log.exception(exc)
            await update.message.reply_text("⚠️ Ошибка модели.")
            return
    
    try:
        if function_call and function_call.name == "generate_image":
            await generate_and_send_image(update, context, function_call.args.get("prompt", ""))
            data_updated = True
        elif reply:
            model_display = model_used.replace("gemini-", "").replace("-latest", "").title()
            full_reply = f"<b>{model_display}</b>\n\n{reply}"
            for chunk in split_long_message(full_reply):
                try:
                    await update.message.reply_text(chunk, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
                except BadRequest as exc:
                    log.warning(f"HTML parse failed, sending plain text. Error: {exc}")
                    plain_text_chunk = strip_html_tags(chunk)
                    await update.message.reply_text(plain_text_chunk, disable_web_page_preview=True)
            data_updated = True
        else:
            await update.message.reply_text(
                "⚠️ Модель не вернула ответа. Попробуй переформулировать запрос или сменить провайдера через /set_provider."
            )
            data_updated = True
    except Exception as exc:
        log.exception(exc)
        await update.message.reply_text("⚠️ Ошибка модели.")
    finally:
        if data_updated:
            await persist_chat_data(chat_id)
async def handle_text_and_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    
    # Rate limiting
    user_id = update.effective_user.id
    allowed, message = check_rate_limit(user_id)
    if not allowed:
        await update.message.reply_text(message)
        return
    
    chat_id = update.effective_chat.id
    text = update.message.text or update.message.caption or ""
    record_user_profile(chat_id, update.effective_user)
    cfg = get_cfg(chat_id)
    if update.message.chat.type in (ChatType.GROUP, ChatType.SUPERGROUP):
        bot_username = context.bot.username.lower()
        bot_mentioned = any(
            text[e.offset : e.offset + e.length].lstrip("@").lower() == bot_username
            for e in (update.message.entities or [])
            if e.type == MessageEntityType.MENTION
        )
        is_reply_to_bot = (
            update.message.reply_to_message
            and update.message.reply_to_message.from_user
            and update.message.reply_to_message.from_user.username
            and update.message.reply_to_message.from_user.username.lower() == bot_username
        )
        if not (bot_mentioned or is_reply_to_bot):
            cfg.new_msg_counter += 1
            await persist_chat_data(chat_id)
            return
        for e in reversed(update.message.entities or []):
            if (
                e.type == MessageEntityType.MENTION
                and text[e.offset : e.offset + e.length].lstrip("@").lower() == bot_username
            ):
                text = (text[: e.offset] + text[e.offset + e.length :]).strip()
    cfg.new_msg_counter += 1
    await persist_chat_data(chat_id)
    prompt_parts: List = []
    if text:
        prompt_parts.append(answer_size_prompt(cfg.msg_size) + text)
    if update.message.photo:
        photo_size = update.message.photo[-1]
        if photo_size.file_size and photo_size.file_size > MAX_IMAGE_BYTES:
            await update.message.reply_text("⚠️ Изображение слишком большое. Принимаю файлы до 5 МБ.")
            return
        file = await photo_size.get_file()
        image_buffer = io.BytesIO()
        await file.download_to_memory(out=image_buffer)
        file_bytes = image_buffer.getvalue()
        if len(file_bytes) > MAX_IMAGE_BYTES:
            await update.message.reply_text("⚠️ Изображение слишком большое. Принимаю файлы до 5 МБ.")
            return
        mime_type = getattr(photo_size, "mime_type", None) or getattr(file, "mime_type", None) or "image/jpeg"
        if not mime_type.lower().startswith("image/"):
            log.warning(f"Отфильтрован файл с неподдерживаемым MIME типом: {mime_type}")
            await update.message.reply_text(
                "⚠️ Пока принимаю только изображения (image/*). Пожалуйста, отправьте картинку."
            )
            return
        prompt_parts.insert(0, {"inline_data": {"mime_type": mime_type, "data": file_bytes}})
    if not prompt_parts:
        return
    await send_bot_response(update, context, chat_id, prompt_parts)
async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    await ensure_user_profile(update)
    
    # Проверяем, есть ли Z.AI для обработки видео
    zai_available = bool(getattr(config, "ZAI_API_KEY", None))
    
    # Обработка видео и видео-кружочков
    video = update.message.video or update.message.video_note
    if video and zai_available:
        # Rate limiting
        user_id = update.effective_user.id
        allowed, message = check_rate_limit(user_id)
        if not allowed:
            await update.message.reply_text(message)
            return
        
        chat_id = update.effective_chat.id
        record_user_profile(chat_id, update.effective_user)
        
        # Проверяем размер видео
        if video.file_size and video.file_size > MAX_VIDEO_BYTES:
            await update.message.reply_text(
                f"⚠️ Видео слишком большое. Максимум {MAX_VIDEO_BYTES // (1024*1024)} МБ."
            )
            return
        
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        
        try:
            file = await video.get_file()
            video_buffer = io.BytesIO()
            await file.download_to_memory(out=video_buffer)
            video_bytes = video_buffer.getvalue()
            
            if len(video_bytes) > MAX_VIDEO_BYTES:
                await update.message.reply_text(
                    f"⚠️ Видео слишком большое. Максимум {MAX_VIDEO_BYTES // (1024*1024)} МБ."
                )
                return
            
            # Определяем MIME тип
            mime_type = getattr(video, "mime_type", None) or "video/mp4"
            
            # Формируем prompt
            caption = update.message.caption or "Опиши, что происходит на этом видео"
            cfg = get_cfg(chat_id)
            
            prompt_parts: List = [
                {"inline_data": {"mime_type": mime_type, "data": video_bytes}},
                {"text": answer_size_prompt(cfg.msg_size) + caption}
            ]
            
            # Принудительно используем Z.AI для видео
            cfg.llm_provider = "zai"
            await persist_chat_data(chat_id)
            
            await send_bot_response(update, context, chat_id, prompt_parts)
            
        except Exception as exc:
            log.error(f"Video processing error: {exc}", exc_info=True)
            await update.message.reply_text("⚠️ Не удалось обработать видео. Попробуйте позже.")
        return
    
    # Для голосовых сообщений и видео без Z.AI
    await update.message.reply_text(
        "😔 Извините, я пока не умею обрабатывать голосовые сообщения.\n\n"
        + ("Для видео нужен Z.AI провайдер (не настроен).\n\n" if not zai_available else "")
        + "Пожалуйста, опишите ваш вопрос текстом или отправьте фото — с ними я работаю отлично!"
    )
async def login_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    message = update.message
    if not message:
        return
    effective_user = update.effective_user
    if not effective_user:
        await message.reply_text("Не удалось определить пользователя.")
        return
    chat_id = update.effective_chat.id
    try:
        code = create_login_code(
            user_id=effective_user.id,
            chat_id=chat_id,
            username=effective_user.username,
            display_name=(
                getattr(effective_user, "full_name", None)
                or " ".join(filter(None, [effective_user.first_name, effective_user.last_name]))
                or effective_user.username
                or str(effective_user.id)
            ),
        )
    except Exception as exc:
        log.error("Не удалось создать код входа: %s", exc, exc_info=True)
        await message.reply_text("⚠️ Не удалось создать код входа. Попробуйте позже.")
        return
    await message.reply_text(
        f"🔑 Ваш код для входа на сайт: <code>{code}</code>\n\n"
        f"Используйте его на {config.WEBAPP_BASE_URL or 'сайте'} для доступа к личным играм.\n"
        f"Код действителен 10 минут.",
        parse_mode=ParseMode.HTML,
    )
async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает статистику использования бота."""
    await ensure_user_profile(update)
    
    user_id = update.effective_user.id
    user_stats = get_user_stats(user_id)
    cache_stats = get_cache_stats()
    
    stats_text = (
        "📊 <b>Статистика</b>\n\n"
        f"<b>Твои запросы:</b>\n"
        f"• За последний час: {user_stats['requests']}\n"
        f"• Последний запрос: {user_stats['time_window']} сек назад\n\n"
        f"<b>Кэш ответов:</b>\n"
        f"• Размер: {cache_stats['size']}/{cache_stats['max_size']}\n"
        f"• TTL: {cache_stats['ttl_seconds']} сек\n\n"
        f"<b>Лимиты:</b>\n"
        f"• 10 запросов в минуту\n"
        f"• 100 запросов в час"
    )
    
    await update.message.reply_text(stats_text, parse_mode=ParseMode.HTML)


async def summarize_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Создает краткое содержание текста или статьи."""
    await ensure_user_profile(update)
    
    text_to_summarize = None
    is_url = False
    
    # Проверяем аргументы
    if context.args:
        arg = " ".join(context.args)
        # Это URL?
        if arg.startswith(('http://', 'https://')):
            text_to_summarize = arg
            is_url = True
        else:
            text_to_summarize = arg
    
    # Или это ответ на сообщение?
    if not text_to_summarize and update.message.reply_to_message:
        text_to_summarize = update.message.reply_to_message.text
        # Проверяем, может это URL в сообщении
        if text_to_summarize and text_to_summarize.startswith(('http://', 'https://')):
            is_url = True
    
    if not text_to_summarize:
        await update.message.reply_text(
            "Использование:\n"
            "<code>/sum текст для саммаризации</code>\n"
            "<code>/sum https://example.com/article</code>\n"
            "Или ответь командой /sum на сообщение\n\n"
            "Минимум 200 символов для текста",
            parse_mode=ParseMode.HTML
        )
        return
    
    await update.message.reply_text("⏳ Анализирую...")
    
    chat_id = update.effective_chat.id
    
    if is_url:
        summary = summarize_url(chat_id, text_to_summarize)
    else:
        summary = summarize_text(chat_id, text_to_summarize)
    
    if summary:
        await update.message.reply_text(
            f"📝 <b>Краткое содержание:</b>\n\n{summary}",
            parse_mode=ParseMode.HTML
        )
    else:
        await update.message.reply_text("❌ Ошибка саммаризации")


async def translate_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Переводит текст."""
    await ensure_user_profile(update)
    
    # Определяем целевой язык и текст
    target_lang = "ru"
    text_to_translate = None
    
    if context.args:
        # Проверяем, первый аргумент - язык?
        first_arg = context.args[0].lower()
        if first_arg in ["ru", "en", "es", "fr", "de", "it", "ja", "ko", "zh"]:
            target_lang = first_arg
            text_to_translate = " ".join(context.args[1:])
        else:
            text_to_translate = " ".join(context.args)
    
    # Или это ответ на сообщение?
    if not text_to_translate and update.message.reply_to_message:
        text_to_translate = update.message.reply_to_message.text
    
    if not text_to_translate:
        await update.message.reply_text(
            "Использование:\n"
            "<code>/tr текст</code> - перевести на русский\n"
            "<code>/tr en текст</code> - перевести на английский\n"
            "Или ответь командой /tr на сообщение\n\n"
            "Языки: ru, en, es, fr, de, it, ja, ko, zh",
            parse_mode=ParseMode.HTML
        )
        return
    
    # Автоопределение: если текст на русском, переводим на английский
    detected = detect_language(text_to_translate)
    if detected == "ru" and target_lang == "ru":
        target_lang = "en"
    
    await update.message.reply_text("⏳ Перевожу...")
    
    chat_id = update.effective_chat.id
    translation = translate_text(chat_id, text_to_translate, target_lang)
    
    if translation:
        await update.message.reply_text(
            f"🌍 <b>Перевод ({target_lang}):</b>\n{translation}",
            parse_mode=ParseMode.HTML
        )
    else:
        await update.message.reply_text("❌ Ошибка перевода")


async def set_openrouter_model_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Устанавливает или показывает предпочитаемую модель OpenRouter.
    При вызове без аргументов показывает текущее значение и список доступных моделей.
    """
    await ensure_user_profile(update)
    if not update.message or not update.effective_chat:
        return

    chat_id = update.effective_chat.id
    cfg = get_cfg(chat_id)
    args = context.args
    
    # ЕСЛИ КОМАНДА ВЫЗВАНА БЕЗ АРГУМЕНТОВ
    if not args:
        # Безопасно получаем текущее значение
        current_model = getattr(cfg, 'openrouter_model', 'по умолчанию (ротация)')
        
        # Формируем красивый список доступных моделей
        available_models_text = "\n".join([f"• <code>{model}</code>" for model in OPENROUTER_MODELS])
        
        # Отправляем пользователю справку
        await update.message.reply_html(
            f"Текущая модель OpenRouter: <b>{current_model}</b>\n\n"
            f"Чтобы изменить, используйте команду с названием модели, например:\n"
            f"<code>/set_or_model {OPENROUTER_MODELS[0]}</code>\n\n"
            f"<b>Доступные модели:</b>\n{available_models_text}"
        )
        return

    # Если аргумент есть, пытаемся его установить
    chosen_model = args[0].strip()

    if chosen_model not in OPENROUTER_MODELS:
        await update.message.reply_html(
            f"❌ <b>Ошибка:</b> Модель '<code>{chosen_model}</code>' не найдена в списке доступных.\n"
            f"Используйте команду <code>/set_or_model</code> без параметров, чтобы увидеть список."
        )
        return

    # Сохраняем выбор пользователя
    cfg.openrouter_model = chosen_model
    await persist_chat_data(chat_id)
    
    await update.message.reply_html(
        f"✅ Готово! Ваша модель OpenRouter установлена на:\n<b>{chosen_model}</b>"
    )


async def set_zai_model_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Устанавливает или показывает предпочитаемую модель Z.AI.
    При вызове без аргументов показывает текущее значение и список доступных моделей.
    """
    await ensure_user_profile(update)
    if not update.message or not update.effective_chat:
        return
    
    zai_models = getattr(config, "ZAI_TEXT_MODELS", [])
    zai_vision = getattr(config, "ZAI_VISION_MODEL", "glm-4.5v")
    
    if not getattr(config, "ZAI_API_KEY", None):
        await update.message.reply_text("Z.AI не настроен. Установите ZAI_API_KEY.")
        return

    chat_id = update.effective_chat.id
    cfg = get_cfg(chat_id)
    args = context.args
    
    if not args:
        current_model = getattr(cfg, 'zai_model', None) or config.ZAI_DEFAULT_MODEL
        
        available_models_text = "\n".join([f"• <code>{model}</code>" for model in zai_models])
        
        await update.message.reply_html(
            f"Текущая модель Z.AI: <b>{current_model}</b>\n"
            f"Модель для видео: <b>{zai_vision}</b> (автоматически)\n\n"
            f"Чтобы изменить, используйте команду с названием модели, например:\n"
            f"<code>/set_zai_model glm-4.6</code>\n\n"
            f"<b>Доступные текстовые модели:</b>\n{available_models_text}"
        )
        return

    chosen_model = args[0].strip().lower()
    
    # Ищем модель (case-insensitive)
    matched = None
    for model in zai_models:
        if model.lower() == chosen_model:
            matched = model
            break
    
    if not matched:
        await update.message.reply_html(
            f"❌ <b>Ошибка:</b> Модель '<code>{chosen_model}</code>' не найдена.\n"
            f"Используйте команду <code>/set_zai_model</code> без параметров, чтобы увидеть список."
        )
        return

    cfg.zai_model = matched
    await persist_chat_data(chat_id)
    
    await update.message.reply_html(
        f"✅ Готово! Ваша модель Z.AI установлена на:\n<b>{matched}</b>\n\n"
        f"🎬 Для видео автоматически используется {zai_vision}"
    )


async def game_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    message = update.message
    if not message:
        return
    idea_text = " ".join(context.args).strip() if context.args else ""
    if not idea_text and message.reply_to_message:
        idea_text = (message.reply_to_message.text or "").strip()
    if not idea_text:
        await message.reply_text(
            "Пришлите идею игры, например:\n"
            "<code>/game аркада про прыгающего кота</code>\n"
            "Или ответьте этой командой на сообщение с описанием.",
            parse_mode=ParseMode.HTML,
        )
        return
    chat_id = update.effective_chat.id
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    loop = asyncio.get_running_loop()
    effective_user = update.effective_user
    author_id = effective_user.id if effective_user else None
    author_username = effective_user.username if effective_user else None
    author_name = None
    if effective_user:
        author_name = (
            getattr(effective_user, "full_name", None)
            or " ".join(filter(None, [effective_user.first_name, effective_user.last_name]))
            or None
        )
    cfg = get_cfg(chat_id)
    provider_override = cfg.llm_provider or None
    try:
        generated: GeneratedGame = await loop.run_in_executor(
            None,
            partial(
                generate_game,
                chat_id,
                idea_text,
                author_id,
                author_username,
                author_name,
                provider_override,
                cfg.pollinations_text_model or None,
            ),
        )
    except ValueError as exc:
        await message.reply_text(str(exc))
        return
    except Exception as exc:
        log.error("Не удалось сгенерировать игру: %s", exc, exc_info=True)
        await message.reply_text("⚠️ Не удалось сгенерировать игру. Попробуйте уточнить описание или повторите позже.")
        return
    title_text = html.escape(generated.title)
    summary_text = html.escape(generated.summary) if generated.summary else "Игра готова!"
    model_text = html.escape(generated.model)
    ttl_days = max(1, config.GAME_TTL_SECONDS // 86400) if config.GAME_TTL_SECONDS else 7
    message_lines = [f"🎮 <b>{title_text}</b>", summary_text, f"Модель: <code>{model_text}</code>"]
    message_lines.append(f"Игра хранится {ttl_days} дн. после генерации.")
    reply_markup = None
    if generated.share_url:
        share_url = html.escape(generated.share_url)
        message_lines.append(f"<a href=\"{share_url}\">Открыть в песочнице</a>")
        reply_markup = InlineKeyboardMarkup(
            [[InlineKeyboardButton("Играть", web_app=WebAppInfo(url=generated.share_url))]]
        )
    else:
        message_lines.append(
            "🚧 Установите переменную окружения WEBAPP_BASE_URL, чтобы получать ссылку на игру."
        )
    await message.reply_text(
        "\n".join(message_lines),
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=False,
        reply_markup=reply_markup,
    )