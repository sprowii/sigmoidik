# Copyright (c) 2025 sprouee
"""Саммаризация текста и статей."""

import ipaddress
import re
import socket
from typing import Optional
from urllib.parse import urlparse

import requests

from app.llm.client import llm_request
from app.logging_config import log


# Блокируем внутренние сети для защиты от SSRF
BLOCKED_IP_RANGES = [
    ipaddress.ip_network("127.0.0.0/8"),       # Localhost
    ipaddress.ip_network("10.0.0.0/8"),        # Private
    ipaddress.ip_network("172.16.0.0/12"),     # Private
    ipaddress.ip_network("192.168.0.0/16"),    # Private
    ipaddress.ip_network("169.254.0.0/16"),    # Link-local (AWS metadata!)
    ipaddress.ip_network("::1/128"),           # IPv6 localhost
    ipaddress.ip_network("fc00::/7"),          # IPv6 private
    ipaddress.ip_network("fe80::/10"),         # IPv6 link-local
]


def _is_safe_url(url: str) -> bool:
    """Проверяет, что URL не ведёт на внутренние ресурсы (защита от SSRF)."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        
        if not hostname:
            return False
        
        # Блокируем явные localhost варианты
        if hostname.lower() in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
            return False
        
        # Резолвим DNS и проверяем IP
        try:
            ip_addresses = socket.getaddrinfo(hostname, None)
            for family, _, _, _, sockaddr in ip_addresses:
                ip_str = sockaddr[0]
                ip = ipaddress.ip_address(ip_str)
                
                for blocked_range in BLOCKED_IP_RANGES:
                    if ip in blocked_range:
                        log.warning(f"SSRF blocked: {url} resolves to {ip_str}")
                        return False
        except socket.gaierror:
            # DNS resolution failed - блокируем на всякий случай
            return False
        
        return True
    except Exception as exc:
        log.warning(f"URL validation error: {exc}")
        return False


def summarize_text(chat_id: int, text: str, max_length: int = 500) -> Optional[str]:
    """
    Создает краткое содержание текста.
    
    Args:
        chat_id: ID чата
        text: Текст для саммаризации
        max_length: Максимальная длина саммари
    
    Returns:
        Краткое содержание или None
    """
    if len(text) < 200:
        return "Текст слишком короткий для саммаризации (минимум 200 символов)"
    
    prompt = (
        f"Сделай краткое содержание следующего текста (максимум {max_length} символов). "
        f"Выдели главные мысли:\n\n{text[:10000]}"  # Лимит на входной текст
    )
    
    try:
        response, _, _ = llm_request(chat_id, [{"text": prompt}], None)
        return response
    except Exception as exc:
        log.error(f"Summarization error: {exc}")
        return None


def extract_text_from_url(url: str) -> Optional[str]:
    """
    Извлекает текст из URL (простая версия).
    
    Args:
        url: URL статьи
    
    Returns:
        Извлеченный текст или None
    """
    try:
        # Проверяем URL
        if not url.startswith(('http://', 'https://')):
            return None
        
        # Защита от SSRF - проверяем, что URL не ведёт на внутренние ресурсы
        if not _is_safe_url(url):
            log.warning(f"Blocked potentially unsafe URL: {url}")
            return None
        
        # Получаем контент
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Простое извлечение текста (убираем HTML теги)
        html = response.text
        
        # Убираем скрипты и стили
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Убираем все HTML теги
        text = re.sub(r'<[^>]+>', ' ', html)
        
        # Убираем лишние пробелы
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Берем первые 15000 символов
        return text[:15000] if text else None
        
    except Exception as exc:
        log.error(f"URL extraction error: {exc}")
        return None


def summarize_url(chat_id: int, url: str) -> Optional[str]:
    """
    Создает саммари статьи по URL.
    
    Args:
        chat_id: ID чата
        url: URL статьи
    
    Returns:
        Краткое содержание или None
    """
    text = extract_text_from_url(url)
    
    if not text:
        return "Не удалось извлечь текст из URL"
    
    if len(text) < 200:
        return "Текст на странице слишком короткий"
    
    return summarize_text(chat_id, text, max_length=800)
