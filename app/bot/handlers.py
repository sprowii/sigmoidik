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
from app.storage.redis_store import persist_chat_data, record_user_profile, redis_client, user_profiles
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
    await update.message.reply_text("Р­С‚Р° РєРѕРјР°РЅРґР° РґРѕСЃС‚СѓРїРЅР° С‚РѕР»СЊРєРѕ Р°РґРјРёРЅРёСЃС‚СЂР°С‚РѕСЂСѓ.")
    return False


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    await update.message.reply_text(
        "рџ‘‹ РЇ РЎРёРіРјРѕРёРґР° Р±РѕС‚. /help вЂ“ СЃРїСЂР°РІРєР°\n\n"
        "вљ пёЏ <b>Р’Р°Р¶РЅРѕ:</b> Р’Р°С€Рё СЃРѕРѕР±С‰РµРЅРёСЏ Рё РјРµРґРёР°С„Р°Р№Р»С‹ РѕР±СЂР°Р±Р°С‚С‹РІР°СЋС‚СЃСЏ С‡РµСЂРµР· Google Gemini API. /privacy",
        parse_mode=ParseMode.HTML,
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    await update.message.reply_text(
        "<b>РљРѕРјР°РЅРґС‹:</b>\n"
        "/settings вЂ“ РїРѕРєР°Р·Р°С‚СЊ С‚РµРєСѓС‰РёРµ РЅР°СЃС‚СЂРѕР№РєРё\n"
        "/autopost on|off вЂ“ РІРєР»/РІС‹РєР» Р°РІС‚РѕРїРѕСЃС‚С‹ (Р°РґРјРёРЅ)\n"
        "/set_interval &lt;СЃРµРє&gt; вЂ“ РёРЅС‚РµСЂРІР°Р» Р°РІС‚РѕРїРѕСЃС‚Р° (Р°РґРјРёРЅ)\n"
        "/set_minmsgs &lt;n&gt; вЂ“ РјРёРЅРёРјСѓРј СЃРѕРѕР±С‰РµРЅРёР№ РґР»СЏ Р°РІС‚РѕРїРѕСЃС‚Р° (Р°РґРјРёРЅ)\n"
        "/set_msgsize &lt;s|m|l&gt; вЂ“ СЂР°Р·РјРµСЂ РѕС‚РІРµС‚РѕРІ (Р°РґРјРёРЅ)\n"
        "/draw &lt;РѕРїРёСЃР°РЅРёРµ&gt; вЂ“ РЅР°СЂРёСЃРѕРІР°С‚СЊ РёР·РѕР±СЂР°Р¶РµРЅРёРµ\n"
        "/game &lt;РёРґРµСЏ&gt; – СЃРіРµРЅРµСЂРёСЂРѕРІР°С‚СЊ РёРіСЂСѓ РЅР° Phaser С‡РµСЂРµР· РёРё\n"
        "/reset вЂ“ РѕС‡РёСЃС‚РёС‚СЊ РёСЃС‚РѕСЂРёСЋ РґРёР°Р»РѕРіР°\n"
        "/privacy вЂ“ РїРѕР»РёС‚РёРєР° РєРѕРЅС„РёРґРµРЅС†РёР°Р»СЊРЅРѕСЃС‚Рё",
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
    await update.message.reply_text("РСЃС‚РѕСЂРёСЏ РѕС‡РёС‰РµРЅР° вњ…")


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
                    await update.message.reply_html("<b>Р­С‚Сѓ РєРѕРјР°РЅРґСѓ РјРѕРіСѓС‚ РёСЃРїРѕР»СЊР·РѕРІР°С‚СЊ С‚РѕР»СЊРєРѕ Р°РґРјРёРЅРёСЃС‚СЂР°С‚РѕСЂС‹ РіСЂСѓРїРїС‹.</b>")
                    return
            except Exception as exc:
                log.error(f"Error checking chat member status in group {chat_id}: {exc}")
                await update.message.reply_html("<b>РџСЂРѕРёР·РѕС€Р»Р° РѕС€РёР±РєР° РїСЂРё РїСЂРѕРІРµСЂРєРµ РІР°С€РёС… РїСЂР°РІ Р°РґРјРёРЅРёСЃС‚СЂР°С‚РѕСЂР°.</b>")
                return
    else:
        log.warning(f"User {username} ({user_id}) tried to delete data in unsupported chat type: {chat_type}.")
        await update.message.reply_html("<b>Р­С‚Р° РєРѕРјР°РЅРґР° РЅРµ РїРѕРґРґРµСЂР¶РёРІР°РµС‚СЃСЏ РІ РґР°РЅРЅРѕРј С‚РёРїРµ С‡Р°С‚Р°.</b>")
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
            log.info(f"РЈРґР°Р»РµРЅС‡Рё РєР»СЋС‡Рё Redis РґР»СЏ С‡Р°С‚Р° {chat_id}.")
        except Exception as exc:
            log.error(f"РќРµ СѓРґР°Р»РѕСЃСЊ СѓРґР°Р»РёС‚СЊ РєР»СЋС‡Рё Redis РґР»СЏ С‡Р°С‚Р° {chat_id}: {exc}", exc_info=True)

        await update.message.reply_html(
            "<b>Р’СЃРµ РґР°РЅРЅС‹Рµ РґР»СЏ СЌС‚РѕРіРѕ С‡Р°С‚Р° (РёСЃС‚РѕСЂРёСЏ РїРµСЂРµРїРёСЃРєРё Рё РЅР°СЃС‚СЂРѕР№РєРё) Р±С‹Р»Рё СѓСЃРїРµС€РЅРѕ СѓРґР°Р»РµРЅС‹.</b>\n"
            "Р•СЃР»Рё РІС‹ РїСЂРѕРґРѕР»Р¶РёС‚Рµ РёСЃРїРѕР»СЊР·РѕРІР°С‚СЊ Р±РѕС‚Р°, РЅР°С‡РЅРµС‚СЃСЏ РЅРѕРІР°СЏ РёСЃС‚РѕСЂРёСЏ."
        )


async def delete_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    if not await is_admin(update, context):
        return
    if not context.args:
        await update.message.reply_text("РЈРєР°Р¶РёС‚Рµ ID С‡Р°С‚Р°.")
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
            log.error(f"РќРµ СѓРґР°Р»РѕСЃСЊ СѓРґР°Р»РёС‚СЊ РґР°РЅРЅС‹Рµ С‡Р°С‚Р° {target_id} РёР· Redis: {exc}", exc_info=True)
        await update.message.reply_text(f"Р”Р°РЅРЅС‹Рµ РґР»СЏ ID {target_id} СѓРґР°Р»РµРЅС‹.")
    except (ValueError, IndexError):
        await update.message.reply_text("РќРµРІРµСЂРЅС‹Р№ ID.")


async def settings_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    cfg = get_cfg(update.effective_chat.id)
    await update.message.reply_text(
        f"<b>РђРІС‚РѕРїРѕСЃС‚С‹:</b> {'РІРєР»' if cfg.autopost_enabled else 'РІС‹РєР»'}.\n"
        f"<b>РРЅС‚РµСЂРІР°Р»:</b> {cfg.interval} СЃРµРє, <b>РјРёРЅ. СЃРѕРѕР±С‰РµРЅРёР№:</b> {cfg.min_messages}.\n"
        f"<b>Р Р°Р·РјРµСЂ РѕС‚РІРµС‚РѕРІ:</b> {cfg.msg_size or 'default'}.",
        parse_mode=ParseMode.HTML,
    )


async def autopost_switch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    if not await is_admin(update, context):
        return
    if not context.args or context.args[0] not in {"on", "off"}:
        await update.message.reply_text("РџСЂРёРјРµСЂ: /autopost on")
        return
    cfg = get_cfg(update.effective_chat.id)
    cfg.autopost_enabled = context.args[0] == "on"
    await persist_chat_data(update.effective_chat.id)
    await update.message.reply_text(f"РђРІС‚РѕРїРѕСЃС‚С‹ {'РІРєР»СЋС‡РµРЅС‹' if cfg.autopost_enabled else 'РІС‹РєР»СЋС‡РµРЅС‹'}.")


async def set_interval(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    if not await is_admin(update, context):
        return
    try:
        cfg = get_cfg(update.effective_chat.id)
        cfg.interval = max(300, int(context.args[0]))
        await persist_chat_data(update.effective_chat.id)
        await update.message.reply_text(f"РРЅС‚РµСЂРІР°Р» Р°РІС‚РѕРїРѕСЃС‚Р° = {cfg.interval} СЃРµРє.")
    except (IndexError, ValueError):
        await update.message.reply_text("РџСЂРёРјРµСЂ: /set_interval 7200")


async def set_minmsgs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    if not await is_admin(update, context):
        return
    try:
        cfg = get_cfg(update.effective_chat.id)
        cfg.min_messages = max(1, int(context.args[0]))
        await persist_chat_data(update.effective_chat.id)
        await update.message.reply_text(f"РњРёРЅРёРјСѓРј СЃРѕРѕР±С‰РµРЅРёР№ = {cfg.min_messages}.")
    except (IndexError, ValueError):
        await update.message.reply_text("РџСЂРёРјРµСЂ: /set_minmsgs 10")


async def set_msgsize(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    if not await is_admin(update, context):
        return
    size = (context.args or [""])[0].lower()
    if size not in {"small", "medium", "large", "s", "m", "l", ""}:
        await update.message.reply_text("Р’Р°СЂРёР°РЅС‚С‹: small, medium, large РёР»Рё РїСѓСЃС‚Рѕ (default)")
        return
    cfg = get_cfg(update.effective_chat.id)
    if size in {"s", "m", "l"}:
        cfg.msg_size = size
    elif size:
        cfg.msg_size = size[0]
    else:
        cfg.msg_size = ""
    await persist_chat_data(update.effective_chat.id)
    await update.message.reply_text(f"Р Р°Р·РјРµСЂ РѕС‚РІРµС‚РѕРІ = {cfg.msg_size or 'default'}.")


async def generate_and_send_image(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt: str):
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="upload_photo")
    try:
        image_bytes, model_used = await asyncio.get_running_loop().run_in_executor(None, llm_generate_image, prompt)
        if image_bytes:
            model_display = model_used.replace("gemini-", "").replace("-latest", "").title()
            caption = f"рџЋЁ В«{prompt}В»\n\n<b>Generated by {model_display}</b>"
            await update.message.reply_photo(photo=image_bytes, caption=caption, parse_mode=ParseMode.HTML)
        else:
            await update.message.reply_text("вљ пёЏ РќРµ СѓРґР°Р»РѕСЃСЊ СЃРѕР·РґР°С‚СЊ РёР·РѕР±СЂР°Р¶РµРЅРёРµ.")
    except Exception as exc:
        log.exception(exc)
        await update.message.reply_text("вљ пёЏ РћС€РёР±РєР° РїСЂРё РіРµРЅРµСЂР°С†РёРё РёР·РѕР±СЂР°Р¶РµРЅРёСЏ.")


async def draw_image_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_user_profile(update)
    if not context.args:
        await update.message.reply_text("РџСЂРёРјРµСЂ: /draw РєРѕС‚ РІ СЃРєР°С„Р°РЅРґСЂРµ")
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
        await update.message.reply_text("вљ пёЏ РћС€РёР±РєР° РјРѕРґРµР»Рё.")
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
            await update.message.reply_text("вљ пёЏ РР·РѕР±СЂР°Р¶РµРЅРёРµ СЃР»РёС€РєРѕРј Р±РѕР»СЊС€РѕРµ. РџСЂРёРЅРёРјР°СЋ С„Р°Р№Р»С‹ РґРѕ 5 РњР‘.")
            return
        file = await photo_size.get_file()
        image_buffer = io.BytesIO()
        await file.download_to_memory(out=image_buffer)
        file_bytes = image_buffer.getvalue()
        if len(file_bytes) > MAX_IMAGE_BYTES:
            await update.message.reply_text("вљ пёЏ РР·РѕР±СЂР°Р¶РµРЅРёРµ СЃР»РёС€РєРѕРј Р±РѕР»СЊС€РѕРµ. РџСЂРёРЅРёРјР°СЋ С„Р°Р№Р»С‹ РґРѕ 5 РњР‘.")
            return
        mime_type = getattr(photo_size, "mime_type", None) or getattr(file, "mime_type", None) or "image/jpeg"
        if not mime_type.lower().startswith("image/"):
            log.warning(f"РћС‚С„РёР»СЊС‚СЂРѕРІР°РЅ С„Р°Р№Р» СЃ РЅРµРїРѕРґРґРµСЂР¶РёРІР°РµРјС‹Рј MIME С‚РёРїРѕРј: {mime_type}")
            await update.message.reply_text(
                "вљ пёЏ РџРѕРєР° РїСЂРёРЅРёРјР°СЋ С‚РѕР»СЊРєРѕ РёР·РѕР±СЂР°Р¶РµРЅРёСЏ (image/*). РџРѕР¶Р°Р»СѓР№СЃС‚Р°, РѕС‚РїСЂР°РІСЊС‚Рµ РєР°СЂС‚РёРЅРєСѓ."
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
        "рџ” РР·РІРёРЅРёС‚Рµ, СЏ РїРѕРєР° РЅРµ СѓРјРµСЋ РѕР±СЂР°Р±Р°С‚С‹РІР°С‚СЊ РіРѕР»РѕСЃРѕРІС‹Рµ СЃРѕРѕР±С‰РµРЅРёСЏ, РІРёРґРµРѕ Рё РІРёРґРµРѕ-РєСЂСѓР¶РѕС‡РєРё.\n\n"
        "РџРѕР¶Р°Р»СѓР№СЃС‚Р°, РѕРїРёС€РёС‚Рµ РІР°С€ РІРѕРїСЂРѕСЃ С‚РµРєСЃС‚РѕРј РёР»Рё РѕС‚РїСЂР°РІСЊС‚Рµ С„РѕС‚Рѕ вЂ” СЃ РЅРёРјРё СЏ СЂР°Р±РѕС‚Р°СЋ РѕС‚Р»РёС‡РЅРѕ!"
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


