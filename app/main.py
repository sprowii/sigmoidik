# Copyright (c) 2025 sprouee
import threading
from typing import Dict, Tuple

import requests
import time
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from app import config
from app.bot import handlers
from app.bot.jobs import autopost_job, check_models_job
from app.logging_config import log
from app.moderation import init_moderation_controller
from app.storage.redis_store import load_data
from app.web.server import flask_app
from app.web.webhook import get_webhook_url, setup_webhook


def _ensure_env() -> Tuple[str, str]:
    if not config.TG_TOKEN or not config.ADMIN_ID:
        raise RuntimeError("TG_TOKEN и ADMIN_ID должны быть установлены")
    if not config.DOWNLOAD_KEY:
        log.warning("DOWNLOAD_KEY не установлен. Скачивание истории через веб будет недоступно.")
    if not config.WEBAPP_BASE_URL:
        log.warning("WEBAPP_BASE_URL не установлен. Ссылки на игры работать не будут.")
    return config.TG_TOKEN, config.ADMIN_ID


def _fetch_bot_info(token: str) -> Dict:
    try:
        bot_info = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10).json().get("result", {})
        if not bot_info.get("username"):
            raise RuntimeError("Не удалось получить username бота.")
        log.info(f"Bot Username: @{bot_info['username']}")
        return bot_info
    except Exception as exc:
        raise RuntimeError(f"Ошибка при получении информации о боте: {exc}")


def build_application(token: str, bot_username: str):
    app = ApplicationBuilder().token(token).build()
    command_handlers = {
        "start": handlers.start,
        "help": handlers.help_cmd,
        "privacy": handlers.privacy_cmd,
        "reset": handlers.reset,
        "tr": handlers.translate_cmd,
        "sum": handlers.summarize_cmd,
        "draw": handlers.draw_image_cmd,
        "game": handlers.game_cmd,
        "login": handlers.login_cmd,
        "settings": handlers.settings_cmd,
        "stats": handlers.stats_cmd,
        "delete_data": handlers.delete_data,
        "autopost": handlers.autopost_switch,
        "set_interval": handlers.set_interval,
        "set_minmsgs": handlers.set_minmsgs,
        "set_msgsize": handlers.set_msgsize,
        "set_draw_model": handlers.set_draw_model,
        "set_pollinations_text_model": handlers.set_pollinations_text_model,
        "set_or_model": handlers.set_openrouter_model_handler,
        "set_provider": handlers.set_provider,
        # Moderation commands (Requirements 3.1, 3.4, 3.5, 4.1, 4.2, 4.3, 4.4)
        "warn": handlers.warn_cmd,
        "warns": handlers.warns_cmd,
        "clearwarns": handlers.clearwarns_cmd,
        "ban": handlers.ban_cmd,
        "unban": handlers.unban_cmd,
        "mute": handlers.mute_cmd,
        "unmute": handlers.unmute_cmd,
        "kick": handlers.kick_cmd,
        # Content filter commands (Requirement 5)
        "addfilter": handlers.addfilter_cmd,
        "removefilter": handlers.removefilter_cmd,
        "filters": handlers.filters_cmd,
        # Moderation settings commands (Requirement 7)
        "modsettings": handlers.mod_settings_cmd,
        "setmodvalue": handlers.setmodvalue_cmd,
        "setlogchannel": handlers.setlogchannel_cmd,
        "exportsettings": handlers.exportsettings_cmd,
        "importsettings": handlers.importsettings_cmd,
        # Moderation log command (Requirement 8)
        "modlog": handlers.modlog_cmd,
    }
    for command, callback in command_handlers.items():
        app.add_handler(CommandHandler(command, callback))

    # Moderation handlers - должны быть первыми для проверки спама
    app.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, handlers.handle_new_chat_members))
    
    # Captcha callback handler (Requirement 6.1, 6.3)
    app.add_handler(CallbackQueryHandler(handlers.handle_captcha_callback, pattern=r"^captcha:"))
    
    # Moderation settings callback handler (Requirement 7.1)
    # Паттерн ловит все callback для настроек: modsettings:, modcat:, modtoggle:, modval:, modback
    app.add_handler(CallbackQueryHandler(handlers.handle_settings_callback, pattern=r"^(modsettings:|modcat:|modtoggle:|modval:|modback)"))
    
    # Spam check handler for group messages (runs before main handlers)
    # Используем group=-1 чтобы обработчик запускался раньше основных
    from telegram.ext import ApplicationHandlerStop
    
    async def spam_check_wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Wrapper для проверки спама. Возвращает True если сообщение заблокировано."""
        is_spam = await handlers.check_spam_moderation(update, context)
        if is_spam:
            # Останавливаем дальнейшую обработку
            raise ApplicationHandlerStop()
    
    # Добавляем проверку спама для текстовых сообщений в группах
    app.add_handler(
        MessageHandler(
            (filters.TEXT | filters.PHOTO | filters.CAPTION) & 
            (filters.ChatType.GROUP | filters.ChatType.SUPERGROUP) & 
            ~filters.COMMAND,
            spam_check_wrapper
        ),
        group=-1  # Запускается раньше основных обработчиков
    )
    
    # Main message handlers
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handlers.handle_text_and_photo))
    app.add_handler(MessageHandler(filters.PHOTO, handlers.handle_text_and_photo))
    app.add_handler(MessageHandler(filters.VOICE | filters.VIDEO | filters.VIDEO_NOTE, handlers.handle_media))

    if app.job_queue:
        app.job_queue.run_repeating(check_models_job, 14400, 60)
        app.job_queue.run_repeating(autopost_job, 60, 60)
        log.info("JobQueue initialized")
    
    # Initialize ModerationController
    init_moderation_controller(app.bot)

    return app


def main():
    load_data()
    token, _ = _ensure_env()
    bot_info = _fetch_bot_info(token)
    app = build_application(token, bot_info["username"])
    
    from app.web.server import set_application
    set_application(app, None)  # Для polling режима loop не нужен

    # Запускаем Flask в отдельном потоке
    threading.Thread(
        target=lambda: flask_app.run(host=config.FLASK_HOST, port=config.FLASK_PORT),
        daemon=True,
    ).start()
    log.info("Flask app started")
    
    # Проверяем, нужен ли webhook (для Render)
    webhook_url = get_webhook_url()
    
    if webhook_url:
        log.info(f"Using webhook mode: {webhook_url}")
        # Настраиваем webhook асинхронно
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        webhook_ok = loop.run_until_complete(setup_webhook(app, webhook_url, config.FLASK_PORT))
        
        if webhook_ok:
            loop.run_until_complete(app.initialize())
            loop.run_until_complete(app.start())
            
            # Передаём loop в server.py для обработки вебхуков из Flask-потока
            from app.web.server import set_application
            set_application(app, loop)
            
            log.info("Bot started with webhook (Flask handling) 🚀")
            # Держим event loop запущенным, чтобы Flask мог отправлять корутины
            loop.run_forever()
        else:
            log.warning("Webhook setup failed, falling back to polling")
            log.info("Bot started with polling 🚀")
            app.run_polling(allowed_updates=Update.ALL_TYPES)
    else:
        log.info("Bot started with polling 🚀")
        app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
