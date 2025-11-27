# Copyright (c) 2025 sprouee
"""Rate limiting middleware для защиты от спама."""

import time
from typing import Dict, Tuple

from app.logging_config import log

# Хранилище: {user_id: (timestamp, count)}
_rate_limits: Dict[int, Tuple[float, int]] = {}

# Хранилище для веб-запросов по IP: {ip: (timestamp, count)}
_web_rate_limits: Dict[str, Tuple[float, int]] = {}

# Хранилище для login попыток по IP: {ip: (timestamp, count)}
_login_rate_limits: Dict[str, Tuple[float, int]] = {}

# Настройки
MAX_REQUESTS_PER_MINUTE = 10
MAX_REQUESTS_PER_HOUR = 100
CLEANUP_INTERVAL = 300  # Очистка каждые 5 минут
_last_cleanup = time.time()

# Настройки для веб-запросов
WEB_MAX_REQUESTS_PER_MINUTE = 30
WEB_MAX_REQUESTS_PER_HOUR = 300

# Настройки для login (более строгие для защиты от brute-force)
LOGIN_MAX_ATTEMPTS_PER_MINUTE = 5
LOGIN_MAX_ATTEMPTS_PER_HOUR = 20


def _cleanup_old_entries():
    """Удаляет старые записи для экономии памяти."""
    global _last_cleanup
    now = time.time()
    if now - _last_cleanup < CLEANUP_INTERVAL:
        return
    
    cutoff = now - 3600  # Удаляем записи старше часа
    to_remove = [uid for uid, (ts, _) in _rate_limits.items() if ts < cutoff]
    for uid in to_remove:
        del _rate_limits[uid]
    
    _last_cleanup = now
    if to_remove:
        log.debug(f"Cleaned up {len(to_remove)} old rate limit entries")


def check_rate_limit(user_id: int) -> Tuple[bool, str]:
    """
    Проверяет, не превышен ли лимит запросов.
    
    Returns:
        (allowed, message): allowed=True если можно, message - причина отказа
    """
    _cleanup_old_entries()
    
    now = time.time()
    
    if user_id not in _rate_limits:
        _rate_limits[user_id] = (now, 1)
        return True, ""
    
    last_time, count = _rate_limits[user_id]
    time_diff = now - last_time
    
    # Проверка минутного лимита
    if time_diff < 60:
        if count >= MAX_REQUESTS_PER_MINUTE:
            wait_time = int(60 - time_diff)
            return False, f"⏱️ Слишком много запросов. Подожди {wait_time} сек."
        _rate_limits[user_id] = (last_time, count + 1)
        return True, ""
    
    # Проверка часового лимита
    if time_diff < 3600:
        if count >= MAX_REQUESTS_PER_HOUR:
            wait_time = int((3600 - time_diff) / 60)
            return False, f"⏱️ Превышен часовой лимит. Подожди {wait_time} мин."
        _rate_limits[user_id] = (last_time, count + 1)
        return True, ""
    
    # Сброс счетчика после часа
    _rate_limits[user_id] = (now, 1)
    return True, ""


def get_user_stats(user_id: int) -> Dict[str, int]:
    """Возвращает статистику пользователя."""
    if user_id not in _rate_limits:
        return {"requests": 0, "time_window": 0}
    
    last_time, count = _rate_limits[user_id]
    time_diff = int(time.time() - last_time)
    
    return {
        "requests": count,
        "time_window": time_diff,
    }


def _cleanup_web_entries():
    """Удаляет старые записи веб rate limits."""
    global _last_cleanup
    now = time.time()
    if now - _last_cleanup < CLEANUP_INTERVAL:
        return
    
    cutoff = now - 3600
    
    # Очистка веб-лимитов
    to_remove = [ip for ip, (ts, _) in _web_rate_limits.items() if ts < cutoff]
    for ip in to_remove:
        del _web_rate_limits[ip]
    
    # Очистка login-лимитов
    to_remove = [ip for ip, (ts, _) in _login_rate_limits.items() if ts < cutoff]
    for ip in to_remove:
        del _login_rate_limits[ip]


def check_web_rate_limit(ip_address: str) -> Tuple[bool, str]:
    """
    Проверяет rate limit для веб-запросов по IP.
    
    Returns:
        (allowed, message): allowed=True если можно, message - причина отказа
    """
    _cleanup_web_entries()
    
    now = time.time()
    
    if ip_address not in _web_rate_limits:
        _web_rate_limits[ip_address] = (now, 1)
        return True, ""
    
    last_time, count = _web_rate_limits[ip_address]
    time_diff = now - last_time
    
    if time_diff < 60:
        if count >= WEB_MAX_REQUESTS_PER_MINUTE:
            return False, "Too many requests. Please wait."
        _web_rate_limits[ip_address] = (last_time, count + 1)
        return True, ""
    
    if time_diff < 3600:
        if count >= WEB_MAX_REQUESTS_PER_HOUR:
            return False, "Hourly limit exceeded. Please wait."
        _web_rate_limits[ip_address] = (last_time, count + 1)
        return True, ""
    
    _web_rate_limits[ip_address] = (now, 1)
    return True, ""


def check_login_rate_limit(ip_address: str) -> Tuple[bool, str]:
    """
    Проверяет rate limit для login попыток (более строгий).
    
    Returns:
        (allowed, message): allowed=True если можно, message - причина отказа
    """
    _cleanup_web_entries()
    
    now = time.time()
    
    if ip_address not in _login_rate_limits:
        _login_rate_limits[ip_address] = (now, 1)
        return True, ""
    
    last_time, count = _login_rate_limits[ip_address]
    time_diff = now - last_time
    
    if time_diff < 60:
        if count >= LOGIN_MAX_ATTEMPTS_PER_MINUTE:
            log.warning(f"Login rate limit exceeded for IP: {ip_address}")
            return False, "Too many login attempts. Please wait 1 minute."
        _login_rate_limits[ip_address] = (last_time, count + 1)
        return True, ""
    
    if time_diff < 3600:
        if count >= LOGIN_MAX_ATTEMPTS_PER_HOUR:
            log.warning(f"Hourly login rate limit exceeded for IP: {ip_address}")
            return False, "Too many login attempts. Please wait 1 hour."
        _login_rate_limits[ip_address] = (last_time, count + 1)
        return True, ""
    
    _login_rate_limits[ip_address] = (now, 1)
    return True, ""
