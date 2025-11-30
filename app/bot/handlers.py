# Copyright (c) 2025 sprouee
import asyncio
import html
import io
import time
from functools import partial
import secrets
from typing import List, Optional
from telegram import ChatMember, InlineKeyboardButton, InlineKeyboardMarkup, Update, WebAppInfo
from telegram.constants import ChatAction, ChatType, MessageEntityType, ParseMode
from telegram.error import BadRequest, TelegramError
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
    if config.OPENROUTER_API_KEYS and config.OPENROUTER_MODELS:
        provider_options.append("openrouter")
    if getattr(config, "POLLINATIONS_TEXT_MODELS", None):
        provider_options.append("pollinations")
    provider_hint = ", ".join(provider_options + ["auto"])
    
    await update.message.reply_text(
        "🎬 Видео поддерживается!\n\n"
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
        "/set_pollinations_text_model – модель Pollinations\n"
        "/set_msgsize &lt;s|m|l&gt; – размер ответа\n\n"
        "<b>Модерация (для групп):</b>\n"
        "/warn @user причина – предупреждение ⚠️\n"
        "/warns @user – список предупреждений\n"
        "/clearwarns @user – очистить предупреждения\n"
        "/ban @user причина – забанить 🚫\n"
        "/unban @user – разбанить ✅\n"
        "/mute @user время – замутить 🔇\n"
        "/unmute @user – размутить 🔊\n"
        "/kick @user – кикнуть 👢\n"
        "/addfilter слово – добавить в фильтр\n"
        "/removefilter слово – убрать из фильтра\n"
        "/filters – список фильтров\n"
        "/modsettings – настройки модерации ⚙️\n"
        "/modlog – лог действий 📋\n"
        "/exportsettings – экспорт настроек\n"
        "/importsettings – импорт настроек\n\n"
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
    
    provider_line = f"<b>LLM:</b> {html.escape(provider)}"
    if provider == "pollinations" and pollinations_text:
        provider_line += f" (Pollinations → {html.escape(pollinations_text)})"
    elif provider == "openrouter":
        provider_line += f" (OpenRouter → {html.escape(openrouter_model)})"
    
    providers_status = []
    if config.API_KEYS:
        providers_status.append("✅ Gemini")
    if config.OPENROUTER_API_KEYS and config.OPENROUTER_MODELS:
        providers_status.append("✅ OpenRouter")
    if getattr(config, "POLLINATIONS_TEXT_MODELS", None):
        providers_status.append("✅ Pollinations")
    
    await update.message.reply_text(
        f"<b>Автопосты:</b> {'вкл' if cfg.autopost_enabled else 'выкл'}.\n"
        f"<b>Интервал:</b> {cfg.interval} сек, <b>мин. сообщений:</b> {cfg.min_messages}.\n"
        f"<b>Размер ответа:</b> {cfg.msg_size or 'default'}.\n"
        f"{provider_line}\n"
        f"<b>Провайдеры:</b> {', '.join(providers_status)}",
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
    
    # Обработка видео и видео-кружочков через Gemini
    video = update.message.video or update.message.video_note
    if video:
        # Rate limiting
        user_id = update.effective_user.id
        allowed, message = check_rate_limit(user_id)
        if not allowed:
            await update.message.reply_text(message)
            return
        
        chat_id = update.effective_chat.id
        record_user_profile(chat_id, update.effective_user)
        
        # Telegram Bot API лимит на скачивание — 20MB
        TELEGRAM_FILE_LIMIT = 20 * 1024 * 1024
        
        if video.file_size and video.file_size > TELEGRAM_FILE_LIMIT:
            await update.message.reply_text(
                "⚠️ Видео слишком большое для скачивания через Telegram (лимит 20 МБ).\n\n"
                "Попробуйте:\n"
                "• Сжать видео перед отправкой\n"
                "• Отправить более короткий клип\n"
                "• Использовать видео-кружочек (до 1 мин)"
            )
            return
        
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        
        try:
            file = await video.get_file()
            video_buffer = io.BytesIO()
            await file.download_to_memory(out=video_buffer)
            video_bytes = video_buffer.getvalue()
            
            # Определяем MIME тип
            mime_type = getattr(video, "mime_type", None) or "video/mp4"
            
            # Формируем prompt
            caption = update.message.caption or "Опиши, что происходит на этом видео"
            cfg = get_cfg(chat_id)
            
            prompt_parts: List = [
                {"inline_data": {"mime_type": mime_type, "data": video_bytes}},
                {"text": answer_size_prompt(cfg.msg_size) + caption}
            ]
            
            await send_bot_response(update, context, chat_id, prompt_parts)
            
        except Exception as exc:
            error_msg = str(exc).lower()
            if "too big" in error_msg or "file is too big" in error_msg:
                await update.message.reply_text(
                    "⚠️ Видео слишком большое для Telegram Bot API (лимит ~20 МБ).\n\n"
                    "Попробуйте сжать видео или отправить более короткий клип."
                )
            else:
                log.error(f"Video processing error: {exc}", exc_info=True)
                await update.message.reply_text("⚠️ Не удалось обработать видео. Попробуйте позже.")
        return
    
    # Для голосовых сообщений
    await update.message.reply_text(
        "😔 Извините, я пока не умею обрабатывать голосовые сообщения.\n\n"
        "Пожалуйста, опишите ваш вопрос текстом или отправьте фото/видео — с ними я работаю отлично!"
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



# ============================================================================
# MODERATION HANDLERS
# ============================================================================

async def handle_new_chat_members(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик входа новых участников в чат.
    
    Requirements:
    - 1.1: Send customizable welcome message within 3 seconds
    - 1.3: Respect welcome_enabled setting
    - 2.3: Record join time for newbie link filter
    - 6.1: Send captcha challenge when captcha is enabled
    - 6.3: Grant full chat permissions and send welcome message on captcha success
    - 6.4: Allow immediate participation when captcha is disabled
    """
    if not update.message or not update.message.new_chat_members:
        return
    
    chat = update.effective_chat
    if not chat:
        return
    
    # Импортируем здесь чтобы избежать циклических импортов
    from app.moderation.storage import load_settings_async
    from app.moderation.welcome import WelcomeManager
    from app.moderation.spam import record_user_join_async
    from app.moderation.captcha import CaptchaManager
    
    # Загружаем настройки модерации для чата
    settings = await load_settings_async(chat.id)
    
    # Создаём менеджеры
    welcome_manager = WelcomeManager(context.bot)
    captcha_manager = CaptchaManager(context.bot, settings)
    
    # Обрабатываем каждого нового участника
    for new_member in update.message.new_chat_members:
        # Пропускаем ботов
        if new_member.is_bot:
            continue
        
        # Записываем время входа для фильтра ссылок новичков (Requirement 2.3)
        await record_user_join_async(chat.id, new_member.id)
        
        # Проверяем, включена ли captcha (Requirement 6.1, 6.4)
        if settings.captcha_enabled:
            # Отправляем captcha challenge
            # Приветствие будет отправлено после успешного прохождения captcha
            await captcha_manager.create_captcha(
                chat_id=chat.id,
                user=new_member,
                settings=settings
            )
        else:
            # Captcha выключена - сразу отправляем приветствие (Requirement 6.4)
            if settings.welcome_enabled:
                await welcome_manager.send_welcome(
                    chat_id=chat.id,
                    user=new_member,
                    chat=chat,
                    settings=settings
                )



async def check_spam_moderation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Проверка сообщения на спам перед обработкой.
    
    Requirements:
    - 2.1: Mute user for 5 minutes if they send more than 5 messages within 10 seconds
    - 2.2: Delete messages with known spam patterns
    - 2.3: Hold messages with links from newbies
    - 2.4: Log all spam detection actions
    
    Returns:
        True если сообщение было заблокировано (спам), False иначе
    """
    if not update.message or not update.effective_chat or not update.effective_user:
        return False
    
    # Спам-фильтр работает только в группах
    if update.message.chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        return False
    
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    message_id = update.message.message_id
    text = update.message.text or update.message.caption or ""
    
    # Импортируем здесь чтобы избежать циклических импортов
    from app.moderation.storage import load_settings_async, save_mod_action_async
    from app.moderation.spam import SpamFilter, SpamAction, get_spam_reason_message
    from app.moderation.models import ModAction
    from app.moderation.content_filter import ContentFilter, notify_user_violation
    
    # Загружаем настройки модерации
    settings = await load_settings_async(chat_id)
    
    # Проверяем, является ли пользователь админом (админов не проверяем на контент-фильтр и спам)
    try:
        member = await context.bot.get_chat_member(chat_id, user_id)
        if member.status in (ChatMember.ADMINISTRATOR, ChatMember.OWNER):
            return False
    except TelegramError as exc:
        log.warning(f"Не удалось проверить статус пользователя {user_id}: {exc}")
    
    # Проверка контент-фильтра (Requirement 5.1)
    if settings.filter_words:
        content_filter = ContentFilter(settings)
        filter_result = content_filter.check(text)
        
        if filter_result.is_filtered:
            # Удаляем сообщение
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
                log.info(f"Удалено сообщение {message_id} от {user_id} по фильтру: {filter_result.matched_word}")
            except TelegramError as exc:
                log.warning(f"Не удалось удалить сообщение {message_id}: {exc}")
            
            # Отправляем приватное уведомление (Requirement 5.2)
            if settings.filter_notify_user:
                chat_title = update.effective_chat.title or "чат"
                await notify_user_violation(
                    bot=context.bot,
                    user=update.effective_user,
                    matched_word=filter_result.matched_word or "",
                    chat_title=chat_title
                )
            
            # Логируем действие
            try:
                mod_action = ModAction.create(
                    chat_id=chat_id,
                    action_type="filter",
                    target_user_id=user_id,
                    reason=filter_result.reason,
                    admin_id=None,
                    auto=True
                )
                await save_mod_action_async(mod_action)
            except Exception as exc:
                log.error(f"Не удалось залогировать действие фильтра: {exc}")
            
            return True
    
    # Если спам-фильтр выключен, пропускаем
    if not settings.spam_enabled and not settings.link_filter_enabled:
        return False
    
    # Создаём фильтр и проверяем сообщение
    spam_filter = SpamFilter(settings)
    result = await spam_filter.check_message(
        user_id=user_id,
        text=text,
        message_id=message_id,
        timestamp=time.time()
    )
    
    # Если спама нет, пропускаем
    if result.action == SpamAction.NONE:
        return False
    
    # Обрабатываем результат
    reason_message = get_spam_reason_message(result.reason)
    
    # Удаляем сообщение если нужно
    if result.should_delete:
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
            log.info(f"Удалено спам-сообщение {message_id} от {user_id} в чате {chat_id}: {result.reason}")
        except TelegramError as exc:
            log.warning(f"Не удалось удалить сообщение {message_id}: {exc}")
    
    # Применяем действие
    if result.action == SpamAction.MUTE:
        # Мутим пользователя (Requirement 2.1)
        try:
            until_date = int(time.time()) + (result.mute_duration_min * 60)
            await context.bot.restrict_chat_member(
                chat_id=chat_id,
                user_id=user_id,
                permissions={"can_send_messages": False},
                until_date=until_date
            )
            
            # Уведомляем пользователя
            username = update.effective_user.username or update.effective_user.first_name
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"🔇 @{username} замучен на {result.mute_duration_min} мин.\n{reason_message}",
                parse_mode=ParseMode.HTML
            )
            
            log.info(f"Пользователь {user_id} замучен в чате {chat_id} на {result.mute_duration_min} мин: {result.reason}")
        except TelegramError as exc:
            log.error(f"Не удалось замутить пользователя {user_id}: {exc}")
    
    elif result.action == SpamAction.WARN:
        # Предупреждаем пользователя (Requirement 2.2)
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"⚠️ {reason_message}",
                reply_to_message_id=message_id if not result.should_delete else None
            )
        except TelegramError as exc:
            log.warning(f"Не удалось отправить предупреждение: {exc}")
    
    elif result.action == SpamAction.HOLD:
        # Задерживаем для проверки админом (Requirement 2.3)
        try:
            # Удаляем сообщение
            await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
            
            # Уведомляем пользователя приватно
            try:
                await context.bot.send_message(
                    chat_id=user_id,
                    text=f"⏳ Ваше сообщение в чате задержано для проверки.\n{reason_message}\n\n"
                         f"Новые участники не могут отправлять ссылки в первые {settings.link_newbie_hours} часов."
                )
            except TelegramError:
                # Пользователь не начал диалог с ботом
                pass
            
            log.info(f"Сообщение {message_id} от новичка {user_id} задержано: {result.reason}")
        except TelegramError as exc:
            log.warning(f"Не удалось задержать сообщение: {exc}")
    
    # Логируем действие (Requirement 2.4)
    try:
        action_type_map = {
            SpamAction.MUTE: "mute",
            SpamAction.DELETE: "delete",
            SpamAction.WARN: "warn",
            SpamAction.HOLD: "hold",
        }
        mod_action = ModAction.create(
            chat_id=chat_id,
            action_type=action_type_map.get(result.action, "spam"),
            target_user_id=user_id,
            reason=result.reason,
            admin_id=None,  # Автоматическое действие
            auto=True
        )
        await save_mod_action_async(mod_action)
    except Exception as exc:
        log.error(f"Не удалось залогировать действие модерации: {exc}")
    
    return True


async def handle_group_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик сообщений в группах с проверкой на спам.
    
    Этот обработчик вызывается для всех сообщений в группах и выполняет:
    1. Проверку на спам
    2. Запись времени входа новых пользователей
    3. Передачу управления основному обработчику если спама нет
    """
    if not update.message:
        return
    
    # Проверяем на спам
    is_spam = await check_spam_moderation(update, context)
    if is_spam:
        return  # Сообщение заблокировано
    
    # Если не спам, передаём управление основному обработчику
    # (он будет вызван через цепочку handlers в main.py)


# ============================================================================
# WARN COMMANDS
# ============================================================================

async def is_chat_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Проверить, является ли пользователь админом чата.
    
    Requirement 4.5: Non-admin users cannot use moderation commands
    
    Returns:
        True если пользователь админ чата или бот-админ, False иначе
    """
    if not update.message or not update.effective_chat or not update.effective_user:
        return False
    
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    
    # Проверяем, является ли пользователь глобальным админом бота
    if config.ADMIN_ID and secrets.compare_digest(str(user_id), str(config.ADMIN_ID)):
        return True
    
    # В приватных чатах команды модерации не работают
    if update.message.chat.type == ChatType.PRIVATE:
        await update.message.reply_text("⚠️ Команды модерации работают только в группах.")
        return False
    
    # Проверяем статус в чате
    try:
        member = await context.bot.get_chat_member(chat_id, user_id)
        if member.status in (ChatMember.ADMINISTRATOR, ChatMember.OWNER):
            return True
    except TelegramError as exc:
        log.error(f"Ошибка проверки статуса админа: {exc}")
    
    await update.message.reply_text("⚠️ Эта команда доступна только администраторам чата.")
    return False


def _extract_user_from_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> tuple[Optional[int], Optional[str], str]:
    """Извлечь пользователя из команды.
    
    Поддерживает:
    - /command @username reason
    - /command user_id reason
    - /command (в ответ на сообщение) reason
    
    Returns:
        Tuple (user_id, username/mention, reason)
    """
    user_id = None
    user_mention = None
    reason = ""
    
    # Проверяем reply
    if update.message.reply_to_message and update.message.reply_to_message.from_user:
        target_user = update.message.reply_to_message.from_user
        user_id = target_user.id
        user_mention = f"@{target_user.username}" if target_user.username else target_user.first_name
        reason = " ".join(context.args) if context.args else ""
        return user_id, user_mention, reason
    
    # Проверяем аргументы команды
    if not context.args:
        return None, None, ""
    
    first_arg = context.args[0]
    remaining_args = context.args[1:] if len(context.args) > 1 else []
    
    # Пробуем как @username
    if first_arg.startswith("@"):
        user_mention = first_arg
        reason = " ".join(remaining_args)
        # user_id будет None - нужно будет получить через API
        return None, user_mention, reason
    
    # Пробуем как user_id
    try:
        user_id = int(first_arg)
        user_mention = str(user_id)
        reason = " ".join(remaining_args)
        return user_id, user_mention, reason
    except ValueError:
        pass
    
    # Не удалось распознать пользователя
    return None, None, ""


async def _resolve_user_id(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    username: str
) -> Optional[int]:
    """Получить user_id по username через Telegram API.
    
    Args:
        context: Контекст бота
        chat_id: ID чата
        username: Username без @
        
    Returns:
        user_id или None если не найден
    """
    # Убираем @ если есть
    clean_username = username.lstrip("@").lower()
    
    # Пробуем найти в кэше профилей
    if chat_id in user_profiles:
        for uid, profile in user_profiles[chat_id].items():
            if profile.get("username", "").lower() == clean_username:
                return uid
    
    # К сожалению, Telegram Bot API не позволяет получить user_id по username напрямую
    # Можно только если пользователь есть в чате и мы его видели
    return None


async def warn_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Выдать предупреждение пользователю.
    
    Requirement 3.1: Record warning with reason, timestamp, and issuing admin
    Requirement 3.2: Auto-mute after N warnings
    Requirement 3.3: Auto-ban after M warnings
    
    Usage:
        /warn @username причина
        /warn user_id причина
        /warn (в ответ на сообщение) причина
    """
    await ensure_user_profile(update)
    
    if not await is_chat_admin(update, context):
        return
    
    chat_id = update.effective_chat.id
    admin_id = update.effective_user.id
    
    # Извлекаем пользователя
    user_id, user_mention, reason = _extract_user_from_command(update, context)
    
    # Если username, пробуем получить user_id
    if user_id is None and user_mention and user_mention.startswith("@"):
        user_id = await _resolve_user_id(context, chat_id, user_mention)
        if user_id is None:
            await update.message.reply_text(
                f"⚠️ Не удалось найти пользователя {user_mention}.\n"
                "Попробуйте ответить на сообщение пользователя или указать его ID."
            )
            return
    
    if user_id is None:
        await update.message.reply_text(
            "⚠️ Укажите пользователя:\n"
            "<code>/warn @username причина</code>\n"
            "<code>/warn user_id причина</code>\n"
            "Или ответьте на сообщение пользователя.",
            parse_mode=ParseMode.HTML
        )
        return
    
    # Нельзя варнить самого себя
    if user_id == admin_id:
        await update.message.reply_text("⚠️ Нельзя выдать предупреждение самому себе.")
        return
    
    # Нельзя варнить бота
    if user_id == context.bot.id:
        await update.message.reply_text("⚠️ Нельзя выдать предупреждение боту.")
        return
    
    # Проверяем, не админ ли целевой пользователь
    try:
        target_member = await context.bot.get_chat_member(chat_id, user_id)
        if target_member.status in (ChatMember.ADMINISTRATOR, ChatMember.OWNER):
            await update.message.reply_text("⚠️ Нельзя выдать предупреждение администратору.")
            return
    except TelegramError:
        pass  # Продолжаем, даже если не удалось проверить
    
    # Причина по умолчанию
    if not reason:
        reason = "Нарушение правил чата"
    
    # Импортируем систему предупреждений
    from app.moderation.warns import WarnSystem, WarnEscalation, format_warns_list
    from app.moderation.storage import save_mod_action_async
    from app.moderation.models import ModAction
    
    # Добавляем предупреждение
    warn_system = WarnSystem()
    result = await warn_system.add_warn_async(
        chat_id=chat_id,
        user_id=user_id,
        admin_id=admin_id,
        reason=reason
    )
    
    # Формируем ответ
    response_lines = [
        f"⚠️ <b>Предупреждение #{result.total_warns}</b>",
        f"👤 Пользователь: {html.escape(user_mention or str(user_id))}",
        f"📝 Причина: {html.escape(reason)}",
    ]
    
    # Логируем действие
    mod_action = ModAction.create(
        chat_id=chat_id,
        action_type="warn",
        target_user_id=user_id,
        admin_id=admin_id,
        reason=reason,
        auto=False
    )
    await save_mod_action_async(mod_action)
    
    # Обрабатываем эскалацию
    if result.escalation == WarnEscalation.BAN:
        # Баним пользователя (Requirement 3.3)
        try:
            await context.bot.ban_chat_member(chat_id=chat_id, user_id=user_id)
            response_lines.append(f"\n🚫 <b>Пользователь забанен</b> (достигнут лимит предупреждений)")
            
            # Логируем бан
            ban_action = ModAction.create(
                chat_id=chat_id,
                action_type="ban",
                target_user_id=user_id,
                admin_id=None,
                reason=f"Автобан: {result.total_warns} предупреждений",
                auto=True
            )
            await save_mod_action_async(ban_action)
        except TelegramError as exc:
            response_lines.append(f"\n⚠️ Не удалось забанить: {exc}")
            log.error(f"Не удалось забанить пользователя {user_id}: {exc}")
    
    elif result.escalation == WarnEscalation.MUTE:
        # Мутим пользователя (Requirement 3.2)
        try:
            until_date = int(time.time()) + (result.mute_duration_hours * 3600)
            await context.bot.restrict_chat_member(
                chat_id=chat_id,
                user_id=user_id,
                permissions={"can_send_messages": False},
                until_date=until_date
            )
            response_lines.append(
                f"\n🔇 <b>Пользователь замучен на {result.mute_duration_hours} ч.</b> "
                f"(достигнут порог предупреждений)"
            )
            
            # Логируем мут
            mute_action = ModAction.create(
                chat_id=chat_id,
                action_type="mute",
                target_user_id=user_id,
                admin_id=None,
                reason=f"Автомут: {result.total_warns} предупреждений",
                auto=True
            )
            await save_mod_action_async(mute_action)
        except TelegramError as exc:
            response_lines.append(f"\n⚠️ Не удалось замутить: {exc}")
            log.error(f"Не удалось замутить пользователя {user_id}: {exc}")
    
    await update.message.reply_text("\n".join(response_lines), parse_mode=ParseMode.HTML)


async def warns_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать предупреждения пользователя.
    
    Requirement 3.4: Display all warnings for that user with dates and reasons
    
    Usage:
        /warns @username
        /warns user_id
        /warns (в ответ на сообщение)
    """
    await ensure_user_profile(update)
    
    if not await is_chat_admin(update, context):
        return
    
    chat_id = update.effective_chat.id
    
    # Извлекаем пользователя
    user_id, user_mention, _ = _extract_user_from_command(update, context)
    
    # Если username, пробуем получить user_id
    if user_id is None and user_mention and user_mention.startswith("@"):
        user_id = await _resolve_user_id(context, chat_id, user_mention)
        if user_id is None:
            await update.message.reply_text(
                f"⚠️ Не удалось найти пользователя {user_mention}.\n"
                "Попробуйте ответить на сообщение пользователя или указать его ID."
            )
            return
    
    if user_id is None:
        await update.message.reply_text(
            "⚠️ Укажите пользователя:\n"
            "<code>/warns @username</code>\n"
            "<code>/warns user_id</code>\n"
            "Или ответьте на сообщение пользователя.",
            parse_mode=ParseMode.HTML
        )
        return
    
    # Импортируем систему предупреждений
    from app.moderation.warns import WarnSystem, format_warns_list
    
    # Получаем предупреждения
    warn_system = WarnSystem()
    warns = await warn_system.get_warns_async(chat_id, user_id)
    
    # Форматируем и отправляем
    response = format_warns_list(warns, user_mention or str(user_id))
    await update.message.reply_text(response, parse_mode=ParseMode.HTML)


async def clearwarns_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Очистить все предупреждения пользователя.
    
    Requirement 3.5: Remove all warnings for that user
    
    Usage:
        /clearwarns @username
        /clearwarns user_id
        /clearwarns (в ответ на сообщение)
    """
    await ensure_user_profile(update)
    
    if not await is_chat_admin(update, context):
        return
    
    chat_id = update.effective_chat.id
    admin_id = update.effective_user.id
    
    # Извлекаем пользователя
    user_id, user_mention, _ = _extract_user_from_command(update, context)
    
    # Если username, пробуем получить user_id
    if user_id is None and user_mention and user_mention.startswith("@"):
        user_id = await _resolve_user_id(context, chat_id, user_mention)
        if user_id is None:
            await update.message.reply_text(
                f"⚠️ Не удалось найти пользователя {user_mention}.\n"
                "Попробуйте ответить на сообщение пользователя или указать его ID."
            )
            return
    
    if user_id is None:
        await update.message.reply_text(
            "⚠️ Укажите пользователя:\n"
            "<code>/clearwarns @username</code>\n"
            "<code>/clearwarns user_id</code>\n"
            "Или ответьте на сообщение пользователя.",
            parse_mode=ParseMode.HTML
        )
        return
    
    # Импортируем систему предупреждений
    from app.moderation.warns import WarnSystem
    from app.moderation.storage import save_mod_action_async
    from app.moderation.models import ModAction
    
    # Очищаем предупреждения
    warn_system = WarnSystem()
    cleared_count = await warn_system.clear_warns_async(chat_id, user_id)
    
    if cleared_count > 0:
        # Логируем действие
        mod_action = ModAction.create(
            chat_id=chat_id,
            action_type="clearwarns",
            target_user_id=user_id,
            admin_id=admin_id,
            reason=f"Очищено {cleared_count} предупреждений",
            auto=False
        )
        await save_mod_action_async(mod_action)
        
        await update.message.reply_text(
            f"✅ Очищено {cleared_count} предупреждений для {html.escape(user_mention or str(user_id))}.",
            parse_mode=ParseMode.HTML
        )
    else:
        await update.message.reply_text(
            f"ℹ️ У {html.escape(user_mention or str(user_id))} нет предупреждений.",
            parse_mode=ParseMode.HTML
        )


# ============================================================================
# MODERATION COMMANDS: BAN, MUTE, UNMUTE, KICK
# ============================================================================

async def ban_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Забанить пользователя в чате.
    
    Requirement 4.1: Permanently ban the user and log the action
    
    Usage:
        /ban @username причина
        /ban user_id причина
        /ban (в ответ на сообщение) причина
    """
    await ensure_user_profile(update)
    
    if not await is_chat_admin(update, context):
        return
    
    chat_id = update.effective_chat.id
    admin_id = update.effective_user.id
    
    # Извлекаем пользователя
    user_id, user_mention, reason = _extract_user_from_command(update, context)
    
    # Если username, пробуем получить user_id
    if user_id is None and user_mention and user_mention.startswith("@"):
        user_id = await _resolve_user_id(context, chat_id, user_mention)
        if user_id is None:
            await update.message.reply_text(
                f"⚠️ Не удалось найти пользователя {user_mention}.\n"
                "Попробуйте ответить на сообщение пользователя или указать его ID."
            )
            return
    
    if user_id is None:
        await update.message.reply_text(
            "⚠️ Укажите пользователя:\n"
            "<code>/ban @username причина</code>\n"
            "<code>/ban user_id причина</code>\n"
            "Или ответьте на сообщение пользователя.",
            parse_mode=ParseMode.HTML
        )
        return
    
    # Нельзя банить самого себя
    if user_id == admin_id:
        await update.message.reply_text("⚠️ Нельзя забанить самого себя.")
        return
    
    # Нельзя банить бота
    if user_id == context.bot.id:
        await update.message.reply_text("⚠️ Нельзя забанить бота.")
        return
    
    # Проверяем, не админ ли целевой пользователь
    try:
        target_member = await context.bot.get_chat_member(chat_id, user_id)
        if target_member.status in (ChatMember.ADMINISTRATOR, ChatMember.OWNER):
            await update.message.reply_text("⚠️ Нельзя забанить администратора.")
            return
    except TelegramError:
        pass  # Продолжаем, даже если не удалось проверить
    
    # Причина по умолчанию
    if not reason:
        reason = "Нарушение правил чата"
    
    # Импортируем для логирования
    from app.moderation.storage import save_mod_action_async
    from app.moderation.models import ModAction
    
    # Баним пользователя
    try:
        await context.bot.ban_chat_member(chat_id=chat_id, user_id=user_id)
        
        # Логируем действие
        mod_action = ModAction.create(
            chat_id=chat_id,
            action_type="ban",
            target_user_id=user_id,
            admin_id=admin_id,
            reason=reason,
            auto=False
        )
        await save_mod_action_async(mod_action)
        
        await update.message.reply_text(
            f"🚫 <b>Пользователь забанен</b>\n"
            f"👤 {html.escape(user_mention or str(user_id))}\n"
            f"📝 Причина: {html.escape(reason)}",
            parse_mode=ParseMode.HTML
        )
        
        log.info(f"Пользователь {user_id} забанен в чате {chat_id} админом {admin_id}: {reason}")
        
    except TelegramError as exc:
        await update.message.reply_text(f"⚠️ Не удалось забанить пользователя: {exc}")
        log.error(f"Не удалось забанить пользователя {user_id} в чате {chat_id}: {exc}")


async def unban_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Разбанить пользователя в чате.
    
    Usage:
        /unban @username
        /unban user_id
        /unban (в ответ на сообщение)
    """
    await ensure_user_profile(update)
    
    if not await is_chat_admin(update, context):
        return
    
    chat_id = update.effective_chat.id
    admin_id = update.effective_user.id
    
    # Извлекаем пользователя
    user_id, user_mention, _ = _extract_user_from_command(update, context)
    
    # Если username, пробуем получить user_id
    if user_id is None and user_mention and user_mention.startswith("@"):
        user_id = await _resolve_user_id(context, chat_id, user_mention)
        if user_id is None:
            await update.message.reply_text(
                f"⚠️ Не удалось найти пользователя {user_mention}.\n"
                "Попробуйте указать ID пользователя напрямую."
            )
            return
    
    if user_id is None:
        await update.message.reply_text(
            "⚠️ Укажите пользователя:\n"
            "<code>/unban @username</code>\n"
            "<code>/unban user_id</code>\n"
            "Или ответьте на сообщение пользователя.",
            parse_mode=ParseMode.HTML
        )
        return
    
    # Импортируем для логирования
    from app.moderation.storage import save_mod_action_async
    from app.moderation.models import ModAction
    
    # Разбаниваем пользователя
    try:
        await context.bot.unban_chat_member(chat_id=chat_id, user_id=user_id, only_if_banned=True)
        
        # Логируем действие
        mod_action = ModAction.create(
            chat_id=chat_id,
            action_type="unban",
            target_user_id=user_id,
            admin_id=admin_id,
            reason="Снятие бана",
            auto=False
        )
        await save_mod_action_async(mod_action)
        
        await update.message.reply_text(
            f"✅ <b>Пользователь разбанен</b>\n"
            f"👤 {html.escape(user_mention or str(user_id))}\n"
            f"Теперь он может снова присоединиться к чату.",
            parse_mode=ParseMode.HTML
        )
        
        log.info(f"Пользователь {user_id} разбанен в чате {chat_id} админом {admin_id}")
        
    except TelegramError as exc:
        await update.message.reply_text(f"⚠️ Не удалось разбанить пользователя: {exc}")
        log.error(f"Не удалось разбанить пользователя {user_id} в чате {chat_id}: {exc}")


def _parse_duration(duration_str: str) -> Optional[int]:
    """Парсинг строки длительности в секунды.
    
    Поддерживаемые форматы:
    - 5m, 10m - минуты
    - 1h, 2h - часы
    - 1d, 7d - дни
    - 1w, 2w - недели
    
    Returns:
        Количество секунд или None если формат неверный
    """
    if not duration_str:
        return None
    
    duration_str = duration_str.lower().strip()
    
    # Пробуем распарсить число + единицу измерения
    import re
    match = re.match(r'^(\d+)([mhdw])$', duration_str)
    if not match:
        return None
    
    value = int(match.group(1))
    unit = match.group(2)
    
    if value <= 0:
        return None
    
    multipliers = {
        'm': 60,           # минуты
        'h': 3600,         # часы
        'd': 86400,        # дни
        'w': 604800,       # недели
    }
    
    return value * multipliers.get(unit, 0)


def _format_duration(seconds: int) -> str:
    """Форматирование длительности в читаемый вид."""
    if seconds >= 604800:  # недели
        weeks = seconds // 604800
        return f"{weeks} нед."
    elif seconds >= 86400:  # дни
        days = seconds // 86400
        return f"{days} дн."
    elif seconds >= 3600:  # часы
        hours = seconds // 3600
        return f"{hours} ч."
    else:  # минуты
        minutes = seconds // 60
        return f"{minutes} мин."


async def mute_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Замутить пользователя в чате.
    
    Requirement 4.2: Restrict the user from sending messages for the specified duration
    
    Usage:
        /mute @username 5m причина
        /mute @username 1h
        /mute user_id 1d причина
        /mute (в ответ на сообщение) 1w
    
    Форматы длительности: 5m (минуты), 1h (часы), 1d (дни), 1w (недели)
    """
    await ensure_user_profile(update)
    
    if not await is_chat_admin(update, context):
        return
    
    chat_id = update.effective_chat.id
    admin_id = update.effective_user.id
    
    # Извлекаем пользователя и аргументы
    user_id = None
    user_mention = None
    duration_str = None
    reason = ""
    
    # Проверяем reply
    if update.message.reply_to_message and update.message.reply_to_message.from_user:
        target_user = update.message.reply_to_message.from_user
        user_id = target_user.id
        user_mention = f"@{target_user.username}" if target_user.username else target_user.first_name
        
        # Первый аргумент - длительность, остальное - причина
        if context.args:
            duration_str = context.args[0]
            reason = " ".join(context.args[1:]) if len(context.args) > 1 else ""
    else:
        # Первый аргумент - пользователь, второй - длительность, остальное - причина
        if not context.args or len(context.args) < 2:
            await update.message.reply_text(
                "⚠️ Укажите пользователя и длительность:\n"
                "<code>/mute @username 5m причина</code>\n"
                "<code>/mute user_id 1h причина</code>\n"
                "Или ответьте на сообщение: <code>/mute 1d причина</code>\n\n"
                "Форматы: 5m (мин), 1h (час), 1d (день), 1w (неделя)",
                parse_mode=ParseMode.HTML
            )
            return
        
        first_arg = context.args[0]
        duration_str = context.args[1]
        reason = " ".join(context.args[2:]) if len(context.args) > 2 else ""
        
        # Пробуем как @username
        if first_arg.startswith("@"):
            user_mention = first_arg
            user_id = await _resolve_user_id(context, chat_id, first_arg)
            if user_id is None:
                await update.message.reply_text(
                    f"⚠️ Не удалось найти пользователя {first_arg}.\n"
                    "Попробуйте ответить на сообщение пользователя или указать его ID."
                )
                return
        else:
            # Пробуем как user_id
            try:
                user_id = int(first_arg)
                user_mention = str(user_id)
            except ValueError:
                await update.message.reply_text(
                    "⚠️ Неверный формат пользователя. Используйте @username или user_id."
                )
                return
    
    # Парсим длительность
    duration_seconds = _parse_duration(duration_str)
    if duration_seconds is None:
        await update.message.reply_text(
            "⚠️ Неверный формат длительности.\n"
            "Используйте: 5m (мин), 1h (час), 1d (день), 1w (неделя)"
        )
        return
    
    # Ограничиваем максимальную длительность (1 год)
    max_duration = 365 * 86400
    if duration_seconds > max_duration:
        await update.message.reply_text("⚠️ Максимальная длительность мута - 1 год.")
        return
    
    # Нельзя мутить самого себя
    if user_id == admin_id:
        await update.message.reply_text("⚠️ Нельзя замутить самого себя.")
        return
    
    # Нельзя мутить бота
    if user_id == context.bot.id:
        await update.message.reply_text("⚠️ Нельзя замутить бота.")
        return
    
    # Проверяем, не админ ли целевой пользователь
    try:
        target_member = await context.bot.get_chat_member(chat_id, user_id)
        if target_member.status in (ChatMember.ADMINISTRATOR, ChatMember.OWNER):
            await update.message.reply_text("⚠️ Нельзя замутить администратора.")
            return
    except TelegramError:
        pass
    
    # Причина по умолчанию
    if not reason:
        reason = "Нарушение правил чата"
    
    # Импортируем для логирования
    from app.moderation.storage import save_mod_action_async
    from app.moderation.models import ModAction
    
    # Мутим пользователя
    try:
        until_date = int(time.time()) + duration_seconds
        await context.bot.restrict_chat_member(
            chat_id=chat_id,
            user_id=user_id,
            permissions={"can_send_messages": False},
            until_date=until_date
        )
        
        # Логируем действие
        duration_formatted = _format_duration(duration_seconds)
        mod_action = ModAction.create(
            chat_id=chat_id,
            action_type="mute",
            target_user_id=user_id,
            admin_id=admin_id,
            reason=f"{reason} ({duration_formatted})",
            auto=False
        )
        await save_mod_action_async(mod_action)
        
        await update.message.reply_text(
            f"🔇 <b>Пользователь замучен</b>\n"
            f"👤 {html.escape(user_mention or str(user_id))}\n"
            f"⏱ Длительность: {duration_formatted}\n"
            f"📝 Причина: {html.escape(reason)}",
            parse_mode=ParseMode.HTML
        )
        
        log.info(f"Пользователь {user_id} замучен в чате {chat_id} на {duration_formatted} админом {admin_id}: {reason}")
        
    except TelegramError as exc:
        await update.message.reply_text(f"⚠️ Не удалось замутить пользователя: {exc}")
        log.error(f"Не удалось замутить пользователя {user_id} в чате {chat_id}: {exc}")


async def unmute_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Размутить пользователя в чате.
    
    Requirement 4.3: Restore the user's messaging permissions
    
    Usage:
        /unmute @username
        /unmute user_id
        /unmute (в ответ на сообщение)
    """
    await ensure_user_profile(update)
    
    if not await is_chat_admin(update, context):
        return
    
    chat_id = update.effective_chat.id
    admin_id = update.effective_user.id
    
    # Извлекаем пользователя
    user_id, user_mention, _ = _extract_user_from_command(update, context)
    
    # Если username, пробуем получить user_id
    if user_id is None and user_mention and user_mention.startswith("@"):
        user_id = await _resolve_user_id(context, chat_id, user_mention)
        if user_id is None:
            await update.message.reply_text(
                f"⚠️ Не удалось найти пользователя {user_mention}.\n"
                "Попробуйте ответить на сообщение пользователя или указать его ID."
            )
            return
    
    if user_id is None:
        await update.message.reply_text(
            "⚠️ Укажите пользователя:\n"
            "<code>/unmute @username</code>\n"
            "<code>/unmute user_id</code>\n"
            "Или ответьте на сообщение пользователя.",
            parse_mode=ParseMode.HTML
        )
        return
    
    # Импортируем для логирования
    from app.moderation.storage import save_mod_action_async
    from app.moderation.models import ModAction
    
    # Размучиваем пользователя - восстанавливаем все права
    try:
        await context.bot.restrict_chat_member(
            chat_id=chat_id,
            user_id=user_id,
            permissions={
                "can_send_messages": True,
                "can_send_media_messages": True,
                "can_send_polls": True,
                "can_send_other_messages": True,
                "can_add_web_page_previews": True,
                "can_change_info": False,
                "can_invite_users": True,
                "can_pin_messages": False,
            }
        )
        
        # Логируем действие
        mod_action = ModAction.create(
            chat_id=chat_id,
            action_type="unmute",
            target_user_id=user_id,
            admin_id=admin_id,
            reason="Снятие мута",
            auto=False
        )
        await save_mod_action_async(mod_action)
        
        await update.message.reply_text(
            f"🔊 <b>Пользователь размучен</b>\n"
            f"👤 {html.escape(user_mention or str(user_id))}",
            parse_mode=ParseMode.HTML
        )
        
        log.info(f"Пользователь {user_id} размучен в чате {chat_id} админом {admin_id}")
        
    except TelegramError as exc:
        await update.message.reply_text(f"⚠️ Не удалось размутить пользователя: {exc}")
        log.error(f"Не удалось размутить пользователя {user_id} в чате {chat_id}: {exc}")


async def kick_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Кикнуть пользователя из чата (без бана).
    
    Requirement 4.4: Remove the user from chat without banning
    
    Usage:
        /kick @username причина
        /kick user_id причина
        /kick (в ответ на сообщение) причина
    """
    await ensure_user_profile(update)
    
    if not await is_chat_admin(update, context):
        return
    
    chat_id = update.effective_chat.id
    admin_id = update.effective_user.id
    
    # Извлекаем пользователя
    user_id, user_mention, reason = _extract_user_from_command(update, context)
    
    # Если username, пробуем получить user_id
    if user_id is None and user_mention and user_mention.startswith("@"):
        user_id = await _resolve_user_id(context, chat_id, user_mention)
        if user_id is None:
            await update.message.reply_text(
                f"⚠️ Не удалось найти пользователя {user_mention}.\n"
                "Попробуйте ответить на сообщение пользователя или указать его ID."
            )
            return
    
    if user_id is None:
        await update.message.reply_text(
            "⚠️ Укажите пользователя:\n"
            "<code>/kick @username причина</code>\n"
            "<code>/kick user_id причина</code>\n"
            "Или ответьте на сообщение пользователя.",
            parse_mode=ParseMode.HTML
        )
        return
    
    # Нельзя кикать самого себя
    if user_id == admin_id:
        await update.message.reply_text("⚠️ Нельзя кикнуть самого себя.")
        return
    
    # Нельзя кикать бота
    if user_id == context.bot.id:
        await update.message.reply_text("⚠️ Нельзя кикнуть бота.")
        return
    
    # Проверяем, не админ ли целевой пользователь
    try:
        target_member = await context.bot.get_chat_member(chat_id, user_id)
        if target_member.status in (ChatMember.ADMINISTRATOR, ChatMember.OWNER):
            await update.message.reply_text("⚠️ Нельзя кикнуть администратора.")
            return
    except TelegramError:
        pass
    
    # Причина по умолчанию
    if not reason:
        reason = "Нарушение правил чата"
    
    # Импортируем для логирования
    from app.moderation.storage import save_mod_action_async
    from app.moderation.models import ModAction
    
    # Кикаем пользователя (ban + unban чтобы он мог вернуться)
    try:
        # Сначала баним
        await context.bot.ban_chat_member(chat_id=chat_id, user_id=user_id)
        # Сразу разбаниваем чтобы пользователь мог вернуться
        await context.bot.unban_chat_member(chat_id=chat_id, user_id=user_id, only_if_banned=True)
        
        # Логируем действие
        mod_action = ModAction.create(
            chat_id=chat_id,
            action_type="kick",
            target_user_id=user_id,
            admin_id=admin_id,
            reason=reason,
            auto=False
        )
        await save_mod_action_async(mod_action)
        
        await update.message.reply_text(
            f"👢 <b>Пользователь кикнут</b>\n"
            f"👤 {html.escape(user_mention or str(user_id))}\n"
            f"📝 Причина: {html.escape(reason)}",
            parse_mode=ParseMode.HTML
        )
        
        log.info(f"Пользователь {user_id} кикнут из чата {chat_id} админом {admin_id}: {reason}")
        
    except TelegramError as exc:
        await update.message.reply_text(f"⚠️ Не удалось кикнуть пользователя: {exc}")
        log.error(f"Не удалось кикнуть пользователя {user_id} из чата {chat_id}: {exc}")


# ============================================================================
# CONTENT FILTER COMMANDS
# ============================================================================

async def addfilter_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Добавить слово в фильтр чата.
    
    Requirement 5.1: Add word to blacklist via /addfilter
    
    Usage:
        /addfilter слово
        /addfilter несколько слов
    """
    await ensure_user_profile(update)
    
    if not update.message or not update.effective_chat or not update.effective_user:
        return
    
    chat_id = update.effective_chat.id
    
    # Проверяем права админа
    from app.moderation.permissions import check_admin_permission
    if not await check_admin_permission(update, context):
        return
    
    # Получаем слово для добавления
    if not context.args:
        await update.message.reply_text(
            "⚠️ Укажите слово для фильтрации:\n"
            "<code>/addfilter слово</code>",
            parse_mode=ParseMode.HTML
        )
        return
    
    word = " ".join(context.args).strip()
    if not word:
        await update.message.reply_text("⚠️ Укажите слово для фильтрации.")
        return
    
    # Добавляем слово в фильтр
    from app.moderation.content_filter import add_filter_word
    
    if add_filter_word(chat_id, word):
        await update.message.reply_text(
            f"✅ Слово «{html.escape(word)}» добавлено в фильтр.\n"
            f"Сообщения с этим словом будут автоматически удаляться.",
            parse_mode=ParseMode.HTML
        )
        log.info(f"Слово '{word}' добавлено в фильтр чата {chat_id} админом {update.effective_user.id}")
    else:
        await update.message.reply_text(
            f"⚠️ Слово «{html.escape(word)}» уже есть в фильтре.",
            parse_mode=ParseMode.HTML
        )


async def removefilter_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Удалить слово из фильтра чата.
    
    Requirement 5.4: Remove word from blacklist via /removefilter
    
    Usage:
        /removefilter слово
    """
    await ensure_user_profile(update)
    
    if not update.message or not update.effective_chat or not update.effective_user:
        return
    
    chat_id = update.effective_chat.id
    
    # Проверяем права админа
    from app.moderation.permissions import check_admin_permission
    if not await check_admin_permission(update, context):
        return
    
    # Получаем слово для удаления
    if not context.args:
        await update.message.reply_text(
            "⚠️ Укажите слово для удаления из фильтра:\n"
            "<code>/removefilter слово</code>",
            parse_mode=ParseMode.HTML
        )
        return
    
    word = " ".join(context.args).strip()
    if not word:
        await update.message.reply_text("⚠️ Укажите слово для удаления.")
        return
    
    # Удаляем слово из фильтра
    from app.moderation.content_filter import remove_filter_word
    
    if remove_filter_word(chat_id, word):
        await update.message.reply_text(
            f"✅ Слово «{html.escape(word)}» удалено из фильтра.",
            parse_mode=ParseMode.HTML
        )
        log.info(f"Слово '{word}' удалено из фильтра чата {chat_id} админом {update.effective_user.id}")
    else:
        await update.message.reply_text(
            f"⚠️ Слово «{html.escape(word)}» не найдено в фильтре.",
            parse_mode=ParseMode.HTML
        )


async def filters_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать список слов в фильтре чата.
    
    Requirement 5.3: Display all active filters for the chat
    
    Usage:
        /filters
    """
    await ensure_user_profile(update)
    
    if not update.message or not update.effective_chat or not update.effective_user:
        return
    
    chat_id = update.effective_chat.id
    
    # Проверяем права админа
    from app.moderation.permissions import check_admin_permission
    if not await check_admin_permission(update, context):
        return
    
    # Получаем список слов
    from app.moderation.content_filter import get_filter_words
    
    words = get_filter_words(chat_id)
    
    if not words:
        await update.message.reply_text(
            "📋 <b>Фильтр слов пуст</b>\n\n"
            "Добавьте слова командой:\n"
            "<code>/addfilter слово</code>",
            parse_mode=ParseMode.HTML
        )
        return
    
    # Форматируем список слов
    words_list = "\n".join(f"• {html.escape(w)}" for w in words)
    
    await update.message.reply_text(
        f"📋 <b>Фильтр слов ({len(words)})</b>\n\n"
        f"{words_list}\n\n"
        f"Удалить: <code>/removefilter слово</code>",
        parse_mode=ParseMode.HTML
    )


# ============================================================================
# CAPTCHA CALLBACK HANDLER
# ============================================================================

async def handle_captcha_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик нажатия кнопки captcha.
    
    Requirements:
    - 6.1: Verify captcha answer
    - 6.3: Grant full chat permissions and send welcome message on success
    """
    query = update.callback_query
    if not query or not query.data:
        return
    
    # Проверяем, что это callback от captcha
    if not query.data.startswith("captcha:"):
        return
    
    # Извлекаем ответ
    answer = query.data.replace("captcha:", "")
    
    user = query.from_user
    chat_id = query.message.chat_id if query.message else None
    
    if not user or not chat_id:
        await query.answer("⚠️ Ошибка обработки")
        return
    
    # Импортируем необходимые модули
    from app.moderation.captcha import CaptchaManager
    from app.moderation.storage import load_settings_async
    from app.moderation.welcome import WelcomeManager
    
    # Загружаем настройки
    settings = await load_settings_async(chat_id)
    
    # Создаём менеджер captcha
    captcha_manager = CaptchaManager(context.bot, settings)
    
    # Проверяем ответ
    is_correct = await captcha_manager.verify_answer(chat_id, user.id, answer)
    
    if is_correct:
        # Успешно! (Requirement 6.3)
        await query.answer("✅ Верно! Добро пожаловать!")
        
        # Отправляем приветствие если включено
        if settings.welcome_enabled:
            welcome_manager = WelcomeManager(context.bot)
            chat = await context.bot.get_chat(chat_id)
            await welcome_manager.send_welcome(
                chat_id=chat_id,
                user=user,
                chat=chat,
                settings=settings
            )
        
        log.info(f"Пользователь {user.id} успешно прошёл captcha в чате {chat_id}")
    else:
        # Неверный ответ
        await query.answer("❌ Неверно! Попробуй ещё раз.", show_alert=True)
        log.info(f"Пользователь {user.id} дал неверный ответ на captcha в чате {chat_id}")


# ============================================================================
# MODERATION SETTINGS MENU
# ============================================================================

# Callback data prefixes for settings menu
SETTINGS_PREFIX = "modsettings:"
SETTINGS_CATEGORY_PREFIX = "modcat:"
SETTINGS_TOGGLE_PREFIX = "modtoggle:"
SETTINGS_VALUE_PREFIX = "modval:"
SETTINGS_BACK = "modback"


def _build_main_settings_keyboard(settings) -> InlineKeyboardMarkup:
    """Построить главное меню настроек модерации.
    
    Requirement 7.1: Display interactive menu with all configurable parameters grouped by category
    """
    from app.moderation.models import ChatModSettings
    
    # Статусы категорий
    welcome_status = "✅" if settings.welcome_enabled else "❌"
    spam_status = "✅" if settings.spam_enabled else "❌"
    captcha_status = "✅" if settings.captcha_enabled else "❌"
    link_status = "✅" if settings.link_filter_enabled else "❌"
    filter_count = len(settings.filter_words)
    log_status = "✅" if settings.log_channel_id else "❌"
    
    keyboard = [
        [InlineKeyboardButton(f"👋 Приветствия {welcome_status}", callback_data=f"{SETTINGS_CATEGORY_PREFIX}welcome")],
        [InlineKeyboardButton(f"🛡 Антиспам {spam_status}", callback_data=f"{SETTINGS_CATEGORY_PREFIX}spam")],
        [InlineKeyboardButton(f"⚠️ Предупреждения", callback_data=f"{SETTINGS_CATEGORY_PREFIX}warns")],
        [InlineKeyboardButton(f"🔐 Captcha {captcha_status}", callback_data=f"{SETTINGS_CATEGORY_PREFIX}captcha")],
        [InlineKeyboardButton(f"🔗 Фильтр ссылок {link_status}", callback_data=f"{SETTINGS_CATEGORY_PREFIX}links")],
        [InlineKeyboardButton(f"🚫 Фильтр слов ({filter_count})", callback_data=f"{SETTINGS_CATEGORY_PREFIX}filters")],
        [InlineKeyboardButton(f"📋 Логирование {log_status}", callback_data=f"{SETTINGS_CATEGORY_PREFIX}logging")],
    ]
    
    return InlineKeyboardMarkup(keyboard)


def _build_welcome_settings_keyboard(settings) -> InlineKeyboardMarkup:
    """Построить меню настроек приветствий."""
    enabled_text = "✅ Включено" if settings.welcome_enabled else "❌ Выключено"
    private_text = "🔒 Приватно" if settings.welcome_private else "📢 В чат"
    
    keyboard = [
        [InlineKeyboardButton(enabled_text, callback_data=f"{SETTINGS_TOGGLE_PREFIX}welcome_enabled")],
        [InlineKeyboardButton(f"⏱ Задержка: {settings.welcome_delay_sec} сек", callback_data=f"{SETTINGS_VALUE_PREFIX}welcome_delay")],
        [InlineKeyboardButton(f"🗑 Автоудаление: {settings.welcome_auto_delete_sec} сек", callback_data=f"{SETTINGS_VALUE_PREFIX}welcome_autodelete")],
        [InlineKeyboardButton(private_text, callback_data=f"{SETTINGS_TOGGLE_PREFIX}welcome_private")],
        [InlineKeyboardButton("« Назад", callback_data=SETTINGS_BACK)],
    ]
    
    return InlineKeyboardMarkup(keyboard)


def _build_spam_settings_keyboard(settings) -> InlineKeyboardMarkup:
    """Построить меню настроек антиспама."""
    enabled_text = "✅ Включено" if settings.spam_enabled else "❌ Выключено"
    
    keyboard = [
        [InlineKeyboardButton(enabled_text, callback_data=f"{SETTINGS_TOGGLE_PREFIX}spam_enabled")],
        [InlineKeyboardButton(f"📨 Лимит сообщений: {settings.spam_message_limit}", callback_data=f"{SETTINGS_VALUE_PREFIX}spam_limit")],
        [InlineKeyboardButton(f"⏱ Окно времени: {settings.spam_time_window_sec} сек", callback_data=f"{SETTINGS_VALUE_PREFIX}spam_window")],
        [InlineKeyboardButton(f"🔇 Мут: {settings.spam_mute_duration_min} мин", callback_data=f"{SETTINGS_VALUE_PREFIX}spam_mute")],
        [InlineKeyboardButton("« Назад", callback_data=SETTINGS_BACK)],
    ]
    
    return InlineKeyboardMarkup(keyboard)


def _build_warns_settings_keyboard(settings) -> InlineKeyboardMarkup:
    """Построить меню настроек предупреждений."""
    keyboard = [
        [InlineKeyboardButton(f"🔇 Мут после: {settings.warn_mute_threshold} варнов", callback_data=f"{SETTINGS_VALUE_PREFIX}warn_mute_threshold")],
        [InlineKeyboardButton(f"🚫 Бан после: {settings.warn_ban_threshold} варнов", callback_data=f"{SETTINGS_VALUE_PREFIX}warn_ban_threshold")],
        [InlineKeyboardButton(f"⏱ Длительность мута: {settings.warn_mute_duration_hours} ч", callback_data=f"{SETTINGS_VALUE_PREFIX}warn_mute_duration")],
        [InlineKeyboardButton("« Назад", callback_data=SETTINGS_BACK)],
    ]
    
    return InlineKeyboardMarkup(keyboard)


def _build_captcha_settings_keyboard(settings) -> InlineKeyboardMarkup:
    """Построить меню настроек captcha."""
    enabled_text = "✅ Включено" if settings.captcha_enabled else "❌ Выключено"
    difficulty_map = {"easy": "🟢 Легко", "medium": "🟡 Средне", "hard": "🔴 Сложно"}
    difficulty_text = difficulty_map.get(settings.captcha_difficulty, settings.captcha_difficulty)
    fail_action_text = "👢 Кик" if settings.captcha_fail_action == "kick" else "🔇 Мут"
    
    keyboard = [
        [InlineKeyboardButton(enabled_text, callback_data=f"{SETTINGS_TOGGLE_PREFIX}captcha_enabled")],
        [InlineKeyboardButton(f"⏱ Таймаут: {settings.captcha_timeout_sec} сек", callback_data=f"{SETTINGS_VALUE_PREFIX}captcha_timeout")],
        [InlineKeyboardButton(f"📊 Сложность: {difficulty_text}", callback_data=f"{SETTINGS_VALUE_PREFIX}captcha_difficulty")],
        [InlineKeyboardButton(f"❌ При провале: {fail_action_text}", callback_data=f"{SETTINGS_VALUE_PREFIX}captcha_fail_action")],
        [InlineKeyboardButton("« Назад", callback_data=SETTINGS_BACK)],
    ]
    
    return InlineKeyboardMarkup(keyboard)


def _build_links_settings_keyboard(settings) -> InlineKeyboardMarkup:
    """Построить меню настроек фильтра ссылок."""
    enabled_text = "✅ Включено" if settings.link_filter_enabled else "❌ Выключено"
    action_map = {"delete": "🗑 Удалить", "warn": "⚠️ Предупредить", "hold": "⏳ Задержать"}
    action_text = action_map.get(settings.link_action, settings.link_action)
    
    keyboard = [
        [InlineKeyboardButton(enabled_text, callback_data=f"{SETTINGS_TOGGLE_PREFIX}link_filter_enabled")],
        [InlineKeyboardButton(f"⏱ Период новичка: {settings.link_newbie_hours} ч", callback_data=f"{SETTINGS_VALUE_PREFIX}link_newbie_hours")],
        [InlineKeyboardButton(f"⚡ Действие: {action_text}", callback_data=f"{SETTINGS_VALUE_PREFIX}link_action")],
        [InlineKeyboardButton("« Назад", callback_data=SETTINGS_BACK)],
    ]
    
    return InlineKeyboardMarkup(keyboard)


def _build_filters_settings_keyboard(settings) -> InlineKeyboardMarkup:
    """Построить меню настроек фильтра слов."""
    notify_text = "✅ Уведомлять" if settings.filter_notify_user else "❌ Не уведомлять"
    filter_count = len(settings.filter_words)
    
    keyboard = [
        [InlineKeyboardButton(f"📋 Слов в фильтре: {filter_count}", callback_data=f"{SETTINGS_CATEGORY_PREFIX}filters_list")],
        [InlineKeyboardButton(notify_text, callback_data=f"{SETTINGS_TOGGLE_PREFIX}filter_notify_user")],
        [InlineKeyboardButton("« Назад", callback_data=SETTINGS_BACK)],
    ]
    
    return InlineKeyboardMarkup(keyboard)


def _build_logging_settings_keyboard(settings) -> InlineKeyboardMarkup:
    """Построить меню настроек логирования."""
    log_channel = settings.log_channel_id
    log_text = f"📢 Канал: {log_channel}" if log_channel else "📢 Канал не настроен"
    
    keyboard = [
        [InlineKeyboardButton(log_text, callback_data=f"{SETTINGS_VALUE_PREFIX}log_channel")],
        [InlineKeyboardButton("🗑 Убрать канал", callback_data=f"{SETTINGS_VALUE_PREFIX}log_channel_remove")],
        [InlineKeyboardButton("« Назад", callback_data=SETTINGS_BACK)],
    ]
    
    return InlineKeyboardMarkup(keyboard)


async def mod_settings_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать интерактивное меню настроек модерации.
    
    Requirement 7.1: Display an interactive menu with all configurable parameters grouped by category
    
    Usage:
        /modsettings
    """
    await ensure_user_profile(update)
    
    if not update.message or not update.effective_chat:
        return
    
    chat_id = update.effective_chat.id
    
    # Проверяем права админа
    if not await is_chat_admin(update, context):
        return
    
    # Загружаем настройки
    from app.moderation.storage import load_settings_async
    settings = await load_settings_async(chat_id)
    
    # Строим клавиатуру
    keyboard = _build_main_settings_keyboard(settings)
    
    await update.message.reply_text(
        "⚙️ <b>Настройки модерации</b>\n\n"
        "Выберите категорию для настройки:",
        parse_mode=ParseMode.HTML,
        reply_markup=keyboard
    )


async def handle_settings_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик callback-кнопок меню настроек.
    
    Requirement 7.1: Interactive settings menu
    """
    query = update.callback_query
    if not query or not query.data:
        return
    
    data = query.data
    
    # Проверяем, что это наш callback
    if not any(data.startswith(prefix) for prefix in [
        SETTINGS_PREFIX, SETTINGS_CATEGORY_PREFIX, SETTINGS_TOGGLE_PREFIX, 
        SETTINGS_VALUE_PREFIX, SETTINGS_BACK
    ]) and data != SETTINGS_BACK:
        return
    
    user = query.from_user
    chat_id = query.message.chat_id if query.message else None
    
    if not user or not chat_id:
        await query.answer("⚠️ Ошибка")
        return
    
    # Проверяем права админа
    try:
        member = await context.bot.get_chat_member(chat_id, user.id)
        if member.status not in (ChatMember.ADMINISTRATOR, ChatMember.OWNER):
            # Проверяем глобального админа
            if not (config.ADMIN_ID and secrets.compare_digest(str(user.id), str(config.ADMIN_ID))):
                await query.answer("⚠️ Только для админов", show_alert=True)
                return
    except TelegramError:
        await query.answer("⚠️ Ошибка проверки прав")
        return
    
    # Загружаем настройки
    from app.moderation.storage import load_settings_async, save_settings_async
    settings = await load_settings_async(chat_id)
    
    # Обрабатываем разные типы callback
    if data == SETTINGS_BACK:
        # Возврат в главное меню
        keyboard = _build_main_settings_keyboard(settings)
        await query.edit_message_text(
            "⚙️ <b>Настройки модерации</b>\n\n"
            "Выберите категорию для настройки:",
            parse_mode=ParseMode.HTML,
            reply_markup=keyboard
        )
        await query.answer()
        return
    
    if data.startswith(SETTINGS_CATEGORY_PREFIX):
        # Переход в категорию
        category = data.replace(SETTINGS_CATEGORY_PREFIX, "")
        await _show_category_settings(query, settings, category)
        return
    
    if data.startswith(SETTINGS_TOGGLE_PREFIX):
        # Переключение boolean настройки
        setting_name = data.replace(SETTINGS_TOGGLE_PREFIX, "")
        await _toggle_setting(query, settings, setting_name, chat_id, save_settings_async)
        return
    
    if data.startswith(SETTINGS_VALUE_PREFIX):
        # Изменение значения настройки
        setting_name = data.replace(SETTINGS_VALUE_PREFIX, "")
        await _handle_value_setting(query, settings, setting_name, chat_id, context, save_settings_async)
        return
    
    await query.answer()


async def _show_category_settings(query, settings, category: str):
    """Показать настройки категории."""
    category_info = {
        "welcome": ("👋 <b>Настройки приветствий</b>", _build_welcome_settings_keyboard),
        "spam": ("🛡 <b>Настройки антиспама</b>", _build_spam_settings_keyboard),
        "warns": ("⚠️ <b>Настройки предупреждений</b>", _build_warns_settings_keyboard),
        "captcha": ("🔐 <b>Настройки Captcha</b>", _build_captcha_settings_keyboard),
        "links": ("🔗 <b>Фильтр ссылок для новичков</b>", _build_links_settings_keyboard),
        "filters": ("🚫 <b>Фильтр слов</b>", _build_filters_settings_keyboard),
        "logging": ("📋 <b>Настройки логирования</b>", _build_logging_settings_keyboard),
    }
    
    if category == "filters_list":
        # Показываем список слов в фильтре
        words = settings.filter_words
        if words:
            words_text = "\n".join(f"• {html.escape(w)}" for w in words[:20])
            if len(words) > 20:
                words_text += f"\n... и ещё {len(words) - 20}"
            text = f"🚫 <b>Слова в фильтре ({len(words)})</b>\n\n{words_text}"
        else:
            text = "🚫 <b>Фильтр слов пуст</b>\n\nДобавьте слова командой /addfilter"
        
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("« Назад", callback_data=f"{SETTINGS_CATEGORY_PREFIX}filters")]])
        await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
        await query.answer()
        return
    
    if category not in category_info:
        await query.answer("⚠️ Неизвестная категория")
        return
    
    title, keyboard_builder = category_info[category]
    keyboard = keyboard_builder(settings)
    
    await query.edit_message_text(title, parse_mode=ParseMode.HTML, reply_markup=keyboard)
    await query.answer()


async def _toggle_setting(query, settings, setting_name: str, chat_id: int, save_func):
    """Переключить boolean настройку."""
    toggle_map = {
        "welcome_enabled": ("welcome_enabled", "welcome", "Приветствия"),
        "welcome_private": ("welcome_private", "welcome", "Приватная отправка"),
        "spam_enabled": ("spam_enabled", "spam", "Антиспам"),
        "captcha_enabled": ("captcha_enabled", "captcha", "Captcha"),
        "link_filter_enabled": ("link_filter_enabled", "links", "Фильтр ссылок"),
        "filter_notify_user": ("filter_notify_user", "filters", "Уведомления о фильтре"),
    }
    
    if setting_name not in toggle_map:
        await query.answer("⚠️ Неизвестная настройка")
        return
    
    attr_name, category, display_name = toggle_map[setting_name]
    
    # Переключаем значение
    current_value = getattr(settings, attr_name)
    new_value = not current_value
    setattr(settings, attr_name, new_value)
    
    # Сохраняем
    await save_func(settings)
    
    # Обновляем меню
    await _show_category_settings(query, settings, category)
    
    status = "включено" if new_value else "выключено"
    await query.answer(f"{display_name}: {status}")


async def _handle_value_setting(query, settings, setting_name: str, chat_id: int, context, save_func):
    """Обработать изменение значения настройки."""
    # Для некоторых настроек показываем подсказку
    hints = {
        "welcome_delay": "Используйте команду:\n/setmodvalue welcome_delay <0-30>",
        "welcome_autodelete": "Используйте команду:\n/setmodvalue welcome_autodelete <0-3600>",
        "spam_limit": "Используйте команду:\n/setmodvalue spam_limit <1-20>",
        "spam_window": "Используйте команду:\n/setmodvalue spam_window <5-60>",
        "spam_mute": "Используйте команду:\n/setmodvalue spam_mute <1-1440>",
        "warn_mute_threshold": "Используйте команду:\n/setmodvalue warn_mute <1-10>",
        "warn_ban_threshold": "Используйте команду:\n/setmodvalue warn_ban <1-20>",
        "warn_mute_duration": "Используйте команду:\n/setmodvalue warn_duration <1-168>",
        "captcha_timeout": "Используйте команду:\n/setmodvalue captcha_timeout <30-600>",
        "link_newbie_hours": "Используйте команду:\n/setmodvalue link_hours <0-168>",
        "log_channel": "Используйте команду:\n/setlogchannel <channel_id>",
        "log_channel_remove": None,  # Специальная обработка
    }
    
    # Циклические настройки (переключаются по кругу)
    cycle_settings = {
        "captcha_difficulty": (["easy", "medium", "hard"], "captcha_difficulty", "captcha"),
        "captcha_fail_action": (["kick", "mute"], "captcha_fail_action", "captcha"),
        "link_action": (["delete", "warn", "hold"], "link_action", "links"),
    }
    
    if setting_name == "log_channel_remove":
        # Удаляем канал логирования
        settings.log_channel_id = None
        await save_func(settings)
        await _show_category_settings(query, settings, "logging")
        await query.answer("Канал логирования удалён")
        return
    
    if setting_name in cycle_settings:
        # Циклическое переключение
        values, attr_name, category = cycle_settings[setting_name]
        current = getattr(settings, attr_name)
        try:
            current_idx = values.index(current)
            next_idx = (current_idx + 1) % len(values)
        except ValueError:
            next_idx = 0
        
        new_value = values[next_idx]
        setattr(settings, attr_name, new_value)
        await save_func(settings)
        await _show_category_settings(query, settings, category)
        await query.answer(f"Установлено: {new_value}")
        return
    
    if setting_name in hints:
        hint = hints[setting_name]
        if hint:
            await query.answer(hint, show_alert=True)
        return
    
    await query.answer("⚠️ Неизвестная настройка")


async def setmodvalue_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Установить числовое значение настройки модерации.
    
    Usage:
        /setmodvalue <setting> <value>
    """
    await ensure_user_profile(update)
    
    if not update.message or not update.effective_chat:
        return
    
    chat_id = update.effective_chat.id
    
    # Проверяем права админа
    if not await is_chat_admin(update, context):
        return
    
    if not context.args or len(context.args) < 2:
        await update.message.reply_text(
            "⚠️ Использование:\n"
            "<code>/setmodvalue &lt;настройка&gt; &lt;значение&gt;</code>\n\n"
            "Настройки:\n"
            "• welcome_delay (0-30)\n"
            "• welcome_autodelete (0-3600)\n"
            "• spam_limit (1-20)\n"
            "• spam_window (5-60)\n"
            "• spam_mute (1-1440)\n"
            "• warn_mute (1-10)\n"
            "• warn_ban (1-20)\n"
            "• warn_duration (1-168)\n"
            "• captcha_timeout (30-600)\n"
            "• link_hours (0-168)",
            parse_mode=ParseMode.HTML
        )
        return
    
    setting_name = context.args[0].lower()
    try:
        value = int(context.args[1])
    except ValueError:
        await update.message.reply_text("⚠️ Значение должно быть числом.")
        return
    
    # Маппинг настроек
    settings_map = {
        "welcome_delay": ("welcome_delay_sec", 0, 30),
        "welcome_autodelete": ("welcome_auto_delete_sec", 0, 3600),
        "spam_limit": ("spam_message_limit", 1, 20),
        "spam_window": ("spam_time_window_sec", 5, 60),
        "spam_mute": ("spam_mute_duration_min", 1, 1440),
        "warn_mute": ("warn_mute_threshold", 1, 10),
        "warn_ban": ("warn_ban_threshold", 1, 20),
        "warn_duration": ("warn_mute_duration_hours", 1, 168),
        "captcha_timeout": ("captcha_timeout_sec", 30, 600),
        "link_hours": ("link_newbie_hours", 0, 168),
    }
    
    if setting_name not in settings_map:
        await update.message.reply_text("⚠️ Неизвестная настройка.")
        return
    
    attr_name, min_val, max_val = settings_map[setting_name]
    
    if not (min_val <= value <= max_val):
        await update.message.reply_text(f"⚠️ Значение должно быть от {min_val} до {max_val}.")
        return
    
    # Загружаем и обновляем настройки
    from app.moderation.storage import load_settings_async, save_settings_async
    settings = await load_settings_async(chat_id)
    
    # Дополнительная проверка для warn thresholds
    if setting_name == "warn_mute" and value >= settings.warn_ban_threshold:
        await update.message.reply_text("⚠️ Порог мута должен быть меньше порога бана.")
        return
    if setting_name == "warn_ban" and value <= settings.warn_mute_threshold:
        await update.message.reply_text("⚠️ Порог бана должен быть больше порога мута.")
        return
    
    setattr(settings, attr_name, value)
    await save_settings_async(settings)
    
    await update.message.reply_text(f"✅ Настройка {setting_name} установлена: {value}")


async def setlogchannel_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Установить канал для логирования действий модерации.
    
    Requirement 8.4: Forward all moderation actions to log channel in real-time
    
    Usage:
        /setlogchannel <channel_id>
        /setlogchannel (в ответ на пересланное сообщение из канала)
    """
    await ensure_user_profile(update)
    
    if not update.message or not update.effective_chat:
        return
    
    chat_id = update.effective_chat.id
    
    # Проверяем права админа
    if not await is_chat_admin(update, context):
        return
    
    channel_id = None
    
    # Проверяем аргументы
    if context.args:
        try:
            channel_id = int(context.args[0])
        except ValueError:
            await update.message.reply_text("⚠️ ID канала должен быть числом.")
            return
    elif update.message.reply_to_message and update.message.reply_to_message.forward_from_chat:
        channel_id = update.message.reply_to_message.forward_from_chat.id
    
    if channel_id is None:
        await update.message.reply_text(
            "⚠️ Укажите ID канала:\n"
            "<code>/setlogchannel -1001234567890</code>\n\n"
            "Или перешлите сообщение из канала и ответьте на него этой командой.",
            parse_mode=ParseMode.HTML
        )
        return
    
    # Проверяем, что бот может писать в канал
    try:
        await context.bot.send_message(
            chat_id=channel_id,
            text="✅ Канал настроен для логирования модерации."
        )
    except TelegramError as exc:
        await update.message.reply_text(
            f"⚠️ Не удалось отправить сообщение в канал.\n"
            f"Убедитесь, что бот добавлен в канал как администратор.\n\n"
            f"Ошибка: {exc}"
        )
        return
    
    # Сохраняем настройку
    from app.moderation.storage import load_settings_async, save_settings_async
    settings = await load_settings_async(chat_id)
    settings.log_channel_id = channel_id
    await save_settings_async(settings)
    
    await update.message.reply_text(f"✅ Канал логирования установлен: {channel_id}")



# ============================================================================
# SETTINGS EXPORT/IMPORT
# ============================================================================

async def exportsettings_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Экспортировать настройки модерации в JSON.
    
    Requirement 7.8: Provide a JSON configuration that can be imported to another chat
    
    Usage:
        /exportsettings
    """
    await ensure_user_profile(update)
    
    if not update.message or not update.effective_chat:
        return
    
    chat_id = update.effective_chat.id
    
    # Проверяем права админа
    if not await is_chat_admin(update, context):
        return
    
    # Экспортируем настройки
    from app.moderation.storage import export_settings
    
    try:
        json_str = export_settings(chat_id)
        
        # Отправляем как документ
        json_bytes = json_str.encode('utf-8')
        json_file = io.BytesIO(json_bytes)
        json_file.name = f"mod_settings_{chat_id}.json"
        
        await update.message.reply_document(
            document=json_file,
            filename=f"mod_settings_{chat_id}.json",
            caption=(
                "📤 <b>Настройки модерации экспортированы</b>\n\n"
                "Используйте /importsettings в другом чате для импорта."
            ),
            parse_mode=ParseMode.HTML
        )
        
        log.info(f"Настройки модерации экспортированы для чата {chat_id}")
        
    except Exception as exc:
        log.error(f"Ошибка экспорта настроек для чата {chat_id}: {exc}")
        await update.message.reply_text(f"⚠️ Ошибка экспорта: {exc}")


async def importsettings_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Импортировать настройки модерации из JSON.
    
    Requirement 7.8: Import settings from JSON configuration
    
    Usage:
        /importsettings (в ответ на JSON файл)
        /importsettings <json_string>
    """
    await ensure_user_profile(update)
    
    if not update.message or not update.effective_chat:
        return
    
    chat_id = update.effective_chat.id
    
    # Проверяем права админа
    if not await is_chat_admin(update, context):
        return
    
    json_str = None
    
    # Проверяем, есть ли ответ на сообщение с документом
    if update.message.reply_to_message:
        reply = update.message.reply_to_message
        
        # Проверяем документ
        if reply.document:
            try:
                file = await reply.document.get_file()
                file_buffer = io.BytesIO()
                await file.download_to_memory(out=file_buffer)
                json_str = file_buffer.getvalue().decode('utf-8')
            except Exception as exc:
                await update.message.reply_text(f"⚠️ Не удалось прочитать файл: {exc}")
                return
        
        # Проверяем текст сообщения
        elif reply.text:
            json_str = reply.text
    
    # Проверяем аргументы команды
    if not json_str and context.args:
        json_str = " ".join(context.args)
    
    if not json_str:
        await update.message.reply_text(
            "⚠️ Укажите JSON настроек:\n\n"
            "1. Ответьте на сообщение с JSON файлом\n"
            "2. Ответьте на сообщение с JSON текстом\n"
            "3. Передайте JSON как аргумент команды\n\n"
            "Получить JSON: /exportsettings",
            parse_mode=ParseMode.HTML
        )
        return
    
    # Импортируем настройки
    from app.moderation.storage import import_settings
    
    try:
        settings = import_settings(chat_id, json_str)
        
        await update.message.reply_text(
            "✅ <b>Настройки модерации импортированы</b>\n\n"
            f"• Приветствия: {'✅' if settings.welcome_enabled else '❌'}\n"
            f"• Антиспам: {'✅' if settings.spam_enabled else '❌'}\n"
            f"• Captcha: {'✅' if settings.captcha_enabled else '❌'}\n"
            f"• Фильтр ссылок: {'✅' if settings.link_filter_enabled else '❌'}\n"
            f"• Слов в фильтре: {len(settings.filter_words)}",
            parse_mode=ParseMode.HTML
        )
        
        log.info(f"Настройки модерации импортированы для чата {chat_id}")
        
    except ValueError as exc:
        await update.message.reply_text(f"⚠️ Ошибка импорта: {exc}")
    except Exception as exc:
        log.error(f"Ошибка импорта настроек для чата {chat_id}: {exc}")
        await update.message.reply_text(f"⚠️ Ошибка импорта: {exc}")



# ============================================================================
# MODERATION LOG COMMAND
# ============================================================================

async def modlog_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать лог действий модерации.
    
    Requirement 8.2: Display the last 20 moderation actions
    Requirement 8.3: Display all actions involving that user when @user is specified
    
    Usage:
        /modlog - показать последние 20 действий
        /modlog @username - показать действия для пользователя
        /modlog user_id - показать действия для пользователя
        /modlog (в ответ на сообщение) - показать действия для автора сообщения
    """
    await ensure_user_profile(update)
    
    if not update.message or not update.effective_chat:
        return
    
    chat_id = update.effective_chat.id
    
    # Проверяем права админа
    if not await is_chat_admin(update, context):
        return
    
    # Определяем, нужна ли фильтрация по пользователю
    target_user_id = None
    user_mention = None
    
    # Проверяем reply
    if update.message.reply_to_message and update.message.reply_to_message.from_user:
        target_user = update.message.reply_to_message.from_user
        target_user_id = target_user.id
        user_mention = f"@{target_user.username}" if target_user.username else target_user.first_name
    
    # Проверяем аргументы
    elif context.args:
        first_arg = context.args[0]
        
        # Пробуем как @username
        if first_arg.startswith("@"):
            user_mention = first_arg
            target_user_id = await _resolve_user_id(context, chat_id, first_arg)
            if target_user_id is None:
                await update.message.reply_text(
                    f"⚠️ Не удалось найти пользователя {first_arg}.\n"
                    "Попробуйте указать user_id или ответить на сообщение пользователя."
                )
                return
        else:
            # Пробуем как user_id
            try:
                target_user_id = int(first_arg)
                user_mention = str(target_user_id)
            except ValueError:
                await update.message.reply_text("⚠️ Неверный формат. Используйте @username или user_id.")
                return
    
    # Загружаем лог
    from app.moderation.storage import load_mod_log_async
    from app.moderation.logger import format_mod_log_entry
    
    limit = 20
    actions = await load_mod_log_async(chat_id, limit=limit, user_id=target_user_id)
    
    if not actions:
        if target_user_id:
            await update.message.reply_text(
                f"📋 <b>Лог модерации</b>\n\n"
                f"Нет действий для пользователя {html.escape(user_mention or str(target_user_id))}.",
                parse_mode=ParseMode.HTML
            )
        else:
            await update.message.reply_text(
                "📋 <b>Лог модерации</b>\n\n"
                "Лог пуст. Действия модерации будут записываться автоматически.",
                parse_mode=ParseMode.HTML
            )
        return
    
    # Форматируем лог
    log_entries = []
    for action in actions:
        entry = format_mod_log_entry(action)
        log_entries.append(entry)
    
    log_text = "\n\n".join(log_entries)
    
    # Формируем заголовок
    if target_user_id:
        header = f"📋 <b>Лог модерации для {html.escape(user_mention or str(target_user_id))}</b>"
    else:
        header = f"📋 <b>Лог модерации (последние {len(actions)})</b>"
    
    # Отправляем (разбиваем на части если слишком длинный)
    full_message = f"{header}\n\n{log_text}"
    
    if len(full_message) > 4000:
        # Разбиваем на части
        await update.message.reply_text(header, parse_mode=ParseMode.HTML)
        
        current_chunk = ""
        for entry in log_entries:
            if len(current_chunk) + len(entry) + 2 > 4000:
                await update.message.reply_text(current_chunk)
                current_chunk = entry
            else:
                current_chunk += ("\n\n" if current_chunk else "") + entry
        
        if current_chunk:
            await update.message.reply_text(current_chunk)
    else:
        await update.message.reply_text(full_message, parse_mode=ParseMode.HTML)
