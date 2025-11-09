# Copyright (c) 2025 sprouee
import asyncio
import html
import io
from typing import List, Optional
from telegram import ChatMember, InlineKeyboardButton, InlineKeyboardMarkup, Update, WebAppInfo
from telegram.constants import ChatAction, ChatType, MessageEntityType, ParseMode
from telegram.error import BadRequest
from telegram.ext import ContextTypes
from app import config
from app.llm.client import llm_generate_image, llm_request
from app.logging_config import log
from app.security.privacy import PRIVACY_POLICY_TEXT
from app.state import ChatConfig, configs, history
from app.storage.redis_store import create_login_code, persist_chat_data, record_user_profile, redis_client, user_profiles
from app.utils.text import answer_size_prompt, split_long_message, strip_html_tags
from app.game.generator import GeneratedGame, generate_game
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
    if str(update.effective_user.id) == config.ADMIN_ID:
        return True
    await update.message.reply_text("Эта команда доступна только администратору.")
    return False
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    await update.message.reply_text(
        "👋 Я Сигмоид бот. /help – справка\n\n"
        "⚠️ <b>Важно:</b> Ваши сообщения и медиафайлы обрабатываются через Google Gemini API. /privacy",
        parse_mode=ParseMode.HTML,
    )
async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    await update.message.reply_text(
        "<b>Команды:</b>\n"
        "/settings – показать текущие настройки\n"
        "/autopost on|off – вкл/выкл автопосты (админ)\n"
        "/set_interval &lt;сек&gt; – интервал автопоста (админ)\n"
        "/set_minmsgs &lt;n&gt; – минимум сообщений для автопоста (админ)\n"
        "/set_msgsize &lt;s|m|l&gt; – размер ответа (админ)\n"
        "/draw <описание> – нарисовать изображение\n"
        "/game <идея> – сгенерировать игру на Phaser через ИИ\n"
        "/login – получить код для входа на сайт\n"
        "/reset – очистить историю диалога\n"
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
    is_bot_admin = str(user_id) == config.ADMIN_ID
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
    await update.message.reply_text(
        f"<b>Автопосты:</b> {'вкл' if cfg.autopost_enabled else 'выкл'}.\n"
        f"<b>Интервал:</b> {cfg.interval} сек, <b>мин. сообщений:</b> {cfg.min_messages}.\n"
        f"<b>Размер ответа:</b> {cfg.msg_size or 'default'}.",
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
        image_bytes, model_used = await asyncio.get_running_loop().run_in_executor(None, llm_generate_image, prompt)
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
async def send_bot_response(update: Update, context: ContextTypes.DEFAULT_TYPE, chat_id: int, prompt_parts: List):
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    data_updated = False
    try:
        reply, model_used, function_call = await asyncio.get_running_loop().run_in_executor(
            None, llm_request, chat_id, prompt_parts
        )
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
    except Exception as exc:
        log.exception(exc)
        await update.message.reply_text("⚠️ Ошибка модели.")
    finally:
        if data_updated:
            await persist_chat_data(chat_id)
async def handle_text_and_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
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
    await update.message.reply_text(
        "😔 Извините, я пока не умею обрабатывать голосовые сообщения, видео и видео-кружочки.\n\n"
        "Пожалуйста, опишите ваш вопрос текстом или отправьте фото — с ними я работаю отлично!"
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
    try:
        generated: GeneratedGame = await loop.run_in_executor(
            None, generate_game, chat_id, idea_text, author_id, author_username, author_name
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