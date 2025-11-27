# Copyright (c) 2025 sprouee
"""Webhook support для Telegram бота на Render."""

import asyncio
from typing import Optional

from telegram import Update
from telegram.ext import Application

from app.logging_config import log


async def setup_webhook(app: Application, webhook_url: str, port: int) -> bool:
    """
    Настраивает webhook для бота.
    
    Args:
        app: Telegram Application
        webhook_url: URL для webhook (например, https://yourapp.onrender.com)
        port: Порт для webhook сервера
    
    Returns:
        True если успешно настроен
    """
    try:
        # Удаляем старый webhook
        await app.bot.delete_webhook(drop_pending_updates=True)
        log.info("Deleted old webhook")
        
        # Устанавливаем новый webhook с секретным токеном для верификации
        from app import config
        webhook_path = f"{webhook_url}/telegram-webhook"
        success = await app.bot.set_webhook(
            url=webhook_path,
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=False,
            secret_token=config.WEBHOOK_SECRET_TOKEN,
        )
        
        if success:
            log.info(f"Webhook set successfully: {webhook_path}")
            return True
        else:
            log.error("Failed to set webhook")
            return False
            
    except Exception as exc:
        log.error(f"Error setting up webhook: {exc}", exc_info=True)
        return False


async def remove_webhook(app: Application) -> bool:
    """Удаляет webhook и переключается на polling."""
    try:
        await app.bot.delete_webhook(drop_pending_updates=False)
        log.info("Webhook removed, switching to polling")
        return True
    except Exception as exc:
        log.error(f"Error removing webhook: {exc}", exc_info=True)
        return False


def get_webhook_url() -> Optional[str]:
    """
    Определяет URL для webhook из переменных окружения.
    
    Для Render используется RENDER_EXTERNAL_URL.
    """
    import os
    
    # Render автоматически устанавливает эту переменную
    render_url = os.getenv("RENDER_EXTERNAL_URL")
    if render_url:
        return render_url
    
    # Fallback на ручную настройку
    webhook_url = os.getenv("WEBHOOK_URL")
    if webhook_url:
        return webhook_url
    
    return None
