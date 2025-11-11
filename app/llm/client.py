# Copyright (c) 2025 sprouee
import base64
import random
import time
import urllib.parse
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types
import requests

from app.config import (
    POLLINATIONS_PRIVATE_IMAGES,
    POLLINATIONS_NO_LOGO,
    API_KEYS,
    BOT_PERSONA_PROMPT, 
    IMAGE_MODEL_NAME,
    LLM_PROVIDER_ORDER,
    MAX_HISTORY,
    MODELS as GEMINI_MODELS,
    OPENROUTER_API_KEYS,
    OPENROUTER_MODELS,
    OPENROUTER_SITE_NAME,
    OPENROUTER_SITE_URL,
    OPENROUTER_TIMEOUT,
    POLLINATIONS_API_KEY,
    POLLINATIONS_BASE_URL,
    POLLINATIONS_ENABLED,
    POLLINATIONS_HEIGHT,
    POLLINATIONS_MODEL,
    POLLINATIONS_SAFE_MODE,
    POLLINATIONS_SEED,
    POLLINATIONS_TEXT_BASE_URL,
    POLLINATIONS_TEXT_DEFAULT,
    POLLINATIONS_TEXT_MODELS,
    POLLINATIONS_TEXT_TEMPERATURE,
    POLLINATIONS_TEXT_TIMEOUT,
    POLLINATIONS_TIMEOUT,
    POLLINATIONS_WIDTH,
)
from app.logging_config import log
from app.state import configs, history

current_key_idx = 0
current_model_idx = 0
available_models: List[str] = GEMINI_MODELS.copy()
last_model_check_ts: float = 0.0
_clients: Dict[int, genai.Client] = {}

current_or_key_idx = 0
current_or_model_idx = 0

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SERVICE_UNAVAILABLE_DELAY = 2.0
MODEL_CHECK_INTERVAL = 3600.0


def _get_client(idx: int) -> genai.Client:
    if not API_KEYS:
        raise RuntimeError("Не заданы API ключи для Gemini")
    idx = idx % len(API_KEYS)
    if idx not in _clients:
        _clients[idx] = genai.Client(api_key=API_KEYS[idx])
    return _clients[idx]


def _normalize_provider_name(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = value.strip().lower()
    if normalized in {"", "auto", "default"}:
        return None
    if normalized == "gemini":
        return "gemini" if API_KEYS else None
    if normalized == "openrouter":
        return "openrouter" if OPENROUTER_API_KEYS and OPENROUTER_MODELS else None
    if normalized == "pollinations":
        return "pollinations" if POLLINATIONS_TEXT_MODELS and POLLINATIONS_TEXT_BASE_URL else None
    return None


def _provider_sequence(preferred: Optional[str] = None) -> List[str]:
    normalized_preferred = _normalize_provider_name(preferred)
    ordered_config = [
        provider for provider in LLM_PROVIDER_ORDER 
        if provider in {"gemini", "openrouter", "pollinations"}
    ]
    if not ordered_config:
        ordered_config = ["gemini", "openrouter", "pollinations"]

    if normalized_preferred:
        ordered = [normalized_preferred] + [p for p in ordered_config if p != normalized_preferred]
    else:
        ordered = ordered_config

    sequence: List[str] = []
    for provider in ordered:
        if provider == "gemini" and API_KEYS and GEMINI_MODELS:
            sequence.append("gemini")
        elif provider == "openrouter" and OPENROUTER_API_KEYS and OPENROUTER_MODELS:
            sequence.append("openrouter")
        elif provider == "pollinations" and POLLINATIONS_TEXT_MODELS and POLLINATIONS_TEXT_BASE_URL:
            sequence.append("pollinations")
    
    if not sequence:
        if API_KEYS:
            sequence.append("gemini")
        if OPENROUTER_API_KEYS and OPENROUTER_MODELS:
            sequence.append("openrouter")
        if POLLINATIONS_TEXT_MODELS and POLLINATIONS_TEXT_BASE_URL:
            sequence.append("pollinations")
    
    return sequence


def _parts_to_text(parts: List[Dict[str, Any]]) -> str:
    texts: List[str] = []
    for part in parts:
        if isinstance(part, dict):
            text_val = part.get("text")
            if text_val:
                texts.append(str(text_val))
    return "\n".join(texts).strip()


def _message_has_inline_data(parts: List[Dict[str, Any]]) -> bool:
    for part in parts:
        if isinstance(part, dict) and ("inline_data" in part or "inlineData" in part):
            return True
    return False


def _chat_provider_preference(chat_id: Optional[int], override: Optional[str] = None) -> Optional[str]:
    candidate = _normalize_provider_name(override)
    if candidate:
        return candidate
    if chat_id is None:
        return None
    cfg = configs.get(chat_id)
    if cfg and getattr(cfg, "llm_provider", None):
        return _normalize_provider_name(cfg.llm_provider)
    return None


def _can_use_text_only_provider(message: Dict[str, Any]) -> bool:
    parts = message.get("parts", [])
    if _message_has_inline_data(parts):
        return False
    text = _parts_to_text(parts)
    return bool(text.strip())


def _prepare_openai_compatible_messages(
    stored_history: List[Dict[str, Any]],
    user_text: str,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    
    if BOT_PERSONA_PROMPT:
        messages.append({"role": "system", "content": BOT_PERSONA_PROMPT})
    
    for message in stored_history:
        parts = message.get("parts", [])
        text = _parts_to_text(parts)
        if not text:
            continue
        role = message.get("role", "user")
        if role == "model":
            role = "assistant"
        elif role not in {"assistant", "system"}:
            role = "user"
        messages.append({"role": role, "content": text})
    
    if user_text.strip():
        messages.append({"role": "user", "content": user_text})
    
    return messages


def _pollinations_text_model_for_chat(chat_id: Optional[int]) -> str:
    if not POLLINATIONS_TEXT_MODELS:
        return "openai"
    
    if chat_id is not None:
        cfg = configs.get(chat_id)
        if cfg and getattr(cfg, "pollinations_text_model", None) in POLLINATIONS_TEXT_MODELS:
            return cfg.pollinations_text_model
    
    if POLLINATIONS_TEXT_DEFAULT and POLLINATIONS_TEXT_DEFAULT in POLLINATIONS_TEXT_MODELS:
        return POLLINATIONS_TEXT_DEFAULT
    
    return POLLINATIONS_TEXT_MODELS[0]


def _openai_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        pieces: List[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                pieces.append(str(item["text"]))
            elif isinstance(item, str):
                pieces.append(item)
        return "\n".join(pieces).strip()
    if isinstance(content, dict):
        return str(content.get("text", "")).strip()
    return str(content).strip()


def _openrouter_model_for_chat(chat_id: Optional[int]) -> Optional[str]:
    """Возвращает предпочтительную модель OpenRouter для пользователя, если она задана и валидна."""
    if chat_id is None:
        return None

    cfg = configs.get(chat_id)
    if cfg:
        preferred_model = getattr(cfg, "openrouter_model", None)
        if preferred_model and preferred_model in OPENROUTER_MODELS:
            return preferred_model

    return None


def _is_service_unavailable_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "503" in text or "service unavailable" in text


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "rate limit" in text or "429" in text or "quota" in text


def _to_base64(data: Any) -> str:
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    if isinstance(data, bytes):
        return base64.b64encode(data).decode("utf-8")
    if isinstance(data, bytearray):
        return base64.b64encode(bytes(data)).decode("utf-8")
    if isinstance(data, memoryview):
        return base64.b64encode(data.tobytes()).decode("utf-8")
    return base64.b64encode(str(data).encode("utf-8")).decode("utf-8")


def _from_base64_maybe(data: Any) -> Any:
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, str):
        try:
            return base64.b64decode(data)
        except Exception:
            return data
    return data


def _normalize_prompt_parts(prompt_parts: List[Any]) -> Dict[str, Any]:
    normalized: List[Dict[str, Any]] = []
    for part in prompt_parts:
        if isinstance(part, dict) and "inline_data" in part:
            inline = part["inline_data"]
            normalized.append({
                "inline_data": {
                    "mime_type": inline.get("mime_type") or inline.get("mimeType") or "application/octet-stream",
                    "data": inline.get("data"),
                }
            })
        elif isinstance(part, dict) and "text" in part:
            normalized.append({"text": str(part["text"])})
        elif isinstance(part, (bytes, bytearray, memoryview)):
            normalized.append({"inline_data": {"mime_type": "application/octet-stream", "data": bytes(part)}})
        else:
            normalized.append({"text": str(part)})
    return {"role": "user", "parts": normalized}


def _api_part(part: Dict[str, Any]) -> Dict[str, Any]:
    if "text" in part and part["text"] is not None:
        return {"text": str(part["text"])}
    
    if "function_call" in part:
        fn = part["function_call"]
        if not fn or not fn.get("name"):
            return {}
        return {
            "functionCall": {
                "name": fn.get("name"),
                "args": fn.get("args", {}),
            }
        }
    
    if "functionCall" in part:
        if not part["functionCall"] or not part["functionCall"].get("name"):
            return {}
        return part
    
    if "inline_data" in part:
        inline = part["inline_data"]
        return {
            "inlineData": {
                "mimeType": inline.get("mime_type") or inline.get("mimeType") or "application/octet-stream",
                "data": _to_base64(inline.get("data")),
            }
        }
    
    if "inlineData" in part:
        return {
            "inlineData": {
                "mimeType": part["inlineData"].get("mimeType"),
                "data": _to_base64(part["inlineData"].get("data")),
            }
        }
    
    return {"text": str(part)}


def _api_content(message: Dict[str, Any]) -> Dict[str, Any]:
    formatted_parts: List[Dict[str, Any]] = []
    for raw_part in message.get("parts", []):
        mapped = _api_part(raw_part)
        if mapped:
            formatted_parts.append(mapped)
    return {
        "role": message.get("role", "user"),
        "parts": formatted_parts or [{"text": ""}],
    }


def _part_from_any(part: Any) -> Dict[str, Any]:
    # ПРЕОБРАЗУЕМ В СЛОВАРЬ, ЕСЛИ ЭТО ОБЪЕКТ БИБЛИОТЕКИ
    if hasattr(part, '_raw_part'):
        part = part._raw_part

    if isinstance(part, dict):
        if "functionCall" in part:
            fc = part["functionCall"]
            if not fc or not fc.get("name"):
                return {}
            return {"function_call": {"name": fc.get("name"), "args": fc.get("args", {})}}
        if "function_call" in part:
            if not part["function_call"] or not part["function_call"].get("name"):
                return {}
            return {"function_call": part["function_call"]}
        if "inlineData" in part:
            inline = part["inlineData"]
            return {
                "inline_data": {
                    "mime_type": inline.get("mimeType"),
                    "data": _from_base64_maybe(inline.get("data")),
                }
            }
        if "inline_data" in part:
            inline = part["inline_data"]
            return {
                "inline_data": {
                    "mime_type": inline.get("mime_type"),
                    "data": _from_base64_maybe(inline.get("data")),
                }
            }
        if "text" in part:
            return {"text": part["text"]}

    # Остальная логика без изменений
    if hasattr(part, "function_call"):
        function_call = part.function_call
        args = getattr(function_call, "args", {}) or {}
        if hasattr(args, "items"):
            args = dict(args)
        name = getattr(function_call, "name", "")
        if not name:
            return {}
        return {"function_call": {"name": name, "args": args}}

    if hasattr(part, "text"):
        return {"text": part.text}

    if hasattr(part, "inline_data"):
        inline = part.inline_data
        return {
            "inline_data": {
                "mime_type": getattr(inline, "mime_type", None),
                "data": _from_base64_maybe(getattr(inline, "data", None)),
            }
        }

    if isinstance(part, str):
        return {"text": part}
    if isinstance(part, (bytes, bytearray, memoryview)):
        return {"inline_data": {"mime_type": "application/octet-stream", "data": bytes(part)}}
    return {"text": str(part)}


def _response_parts(response: Any) -> List[Dict[str, Any]]:
    candidates = getattr(response, "candidates", None)
    if not candidates and isinstance(response, dict):
        candidates = response.get("candidates")
    
    if candidates:
        candidate = candidates[0]
        content = getattr(candidate, "content", None)
        if content is None and isinstance(candidate, dict):
            content = candidate.get("content")
        
        parts = getattr(content, "parts", None)
        if parts is None and isinstance(content, dict):
            parts = content.get("parts", [])
        
        if parts:
            cleaned_parts = []
            for part in parts:
                mapped = _part_from_any(part)
                if mapped:
                    cleaned_parts.append(mapped)
            return cleaned_parts
    
    text = getattr(response, "text", None)
    if text:
        return [{"text": text}]
    return []


def _extract_text_from_parts(parts: List[Dict[str, Any]]) -> str:
    texts = [part["text"] for part in parts if isinstance(part, dict) and part.get("text")]
    return "\n".join(texts).strip()


def _extract_function_call(parts: List[Dict[str, Any]]) -> Optional[SimpleNamespace]:
    for part in parts:
        fn = part.get("function_call")
        if isinstance(fn, dict):
            name = fn.get("name")
            if not name:
                continue
            args = fn.get("args") or {}
            if hasattr(args, "items"):
                args = dict(args)
            return SimpleNamespace(name=name, args=args)
    return None


def _history_to_text(chat_history: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for message in chat_history:
        role = message.get("role", "user")
        texts = [
            part.get("text") 
            for part in message.get("parts", []) 
            if isinstance(part, dict) and part.get("text")
        ]
        if texts:
            lines.append(f"{role}: {' '.join(texts)}")
    return "\n".join(lines)


def _request_config() -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    
    if BOT_PERSONA_PROMPT:
        config["system_instruction"] = {"parts": [{"text": BOT_PERSONA_PROMPT}]}
    
    config["tools"] = [
        {
            "function_declarations": [
                {
                    "name": "generate_image",
                    "description": "Generates an image from a text description. Use for explicit image requests.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "prompt": {
                                "type": "STRING",
                                "description": "The image description.",
                            }
                        },
                        "required": ["prompt"],
                    },
                }
            ]
        }
    ]
    
    return config


def _send_gemini_request(
    stored_history: List[Dict[str, Any]],
    user_message: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    log.info("Attempting Gemini request")
    global current_key_idx, current_model_idx

    models_to_try = available_models if available_models else GEMINI_MODELS
    if not models_to_try or not API_KEYS:
        log.warning("No Gemini models or API keys available")
        return None

    for model_offset in range(len(models_to_try)):
        model_idx = (current_model_idx + model_offset) % len(models_to_try)
        model_name = models_to_try[model_idx]
        
        for key_attempt in range(len(API_KEYS)):
            key_idx = (current_key_idx + key_attempt) % len(API_KEYS)
            try:
                client = _get_client(key_idx)
                contents_payload = [_api_content(item) for item in stored_history + [user_message]]
                contents_payload = [item for item in contents_payload if item.get("parts")]
                
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents_payload,
                    config=_request_config(),
                )

                parts = _response_parts(response)
                reply_text = _extract_text_from_parts(parts)
                fn_call = _extract_function_call(parts)
                
                current_key_idx, current_model_idx = key_idx, model_idx
                
                return {
                    "parts": parts,
                    "reply_text": reply_text,
                    "fn_call": fn_call,
                    "model_name": model_name,
                    "provider": "gemini",
                }
                
            except Exception as exc:
                if _is_service_unavailable_error(exc):
                    log.warning(
                        "Model service unavailable (key %s, model %s). Retrying after %.1fs...",
                        key_idx + 1,
                        model_name,
                        SERVICE_UNAVAILABLE_DELAY,
                    )
                    time.sleep(SERVICE_UNAVAILABLE_DELAY)
                    continue
                
                if _is_rate_limit_error(exc):
                    log.info("Rate limit on key %s, model %s. Trying next...", key_idx + 1, model_name)
                    continue
                
                log.warning("Request failed: key %s, model %s: %s", key_idx + 1, model_name, exc)
    
    return None


def _send_openrouter_request(
    chat_id: Optional[int],
    stored_history: List[Dict[str, Any]],
    user_message: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    log.info("Attempting OpenRouter request")
    global current_or_key_idx, current_or_model_idx

    if not OPENROUTER_API_KEYS or not OPENROUTER_MODELS:
        log.warning("No OpenRouter API keys or models available")
        return None

    if not _can_use_text_only_provider(user_message):
        log.info("Message contains non-text content, skipping OpenRouter")
        return None

    user_text = _parts_to_text(user_message.get("parts", []))
    messages = _prepare_openai_compatible_messages(stored_history, user_text)

    # Логика выбора модели
    models_to_iterate: List[str]
    preferred_model = _openrouter_model_for_chat(chat_id)

    if preferred_model:
        models_to_iterate = [preferred_model]
        log.info(f"Using user-preferred OpenRouter model for chat {chat_id}: {preferred_model}")
    else:
        # Стандартная ротация (round-robin), если модель не выбрана
        models_to_iterate = [
            OPENROUTER_MODELS[(current_or_model_idx + i) % len(OPENROUTER_MODELS)]
            for i in range(len(OPENROUTER_MODELS))
        ]

    for model_name in models_to_iterate:
        for key_attempt in range(len(OPENROUTER_API_KEYS)):
            key_idx = (current_or_key_idx + key_attempt) % len(OPENROUTER_API_KEYS)
            api_key = OPENROUTER_API_KEYS[key_idx]

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            if OPENROUTER_SITE_URL:
                headers["HTTP-Referer"] = OPENROUTER_SITE_URL
            if OPENROUTER_SITE_NAME:
                headers["X-Title"] = OPENROUTER_SITE_NAME

            payload = {"model": model_name, "messages": messages}

            try:
                response = requests.post(
                    OPENROUTER_URL,
                    json=payload,
                    headers=headers,
                    timeout=OPENROUTER_TIMEOUT,
                )

                if response.status_code in {429, 503}:
                    log.warning(
                        "OpenRouter returned %s for model %s (key %s). Retrying after %.1fs...",
                        response.status_code,
                        model_name,
                        key_idx + 1,
                        SERVICE_UNAVAILABLE_DELAY,
                    )
                    time.sleep(SERVICE_UNAVAILABLE_DELAY)
                    continue

                response.raise_for_status()
                data = response.json()
                choices = data.get("choices") or []

                if not choices:
                    raise ValueError("OpenRouter response contains no choices")

                message = choices[0].get("message") or {}
                reply_text = _openai_content_to_text(message.get("content"))

                if not reply_text:
                    raise ValueError("OpenRouter response has empty content")

                # Обновляем глобальные индексы
                current_or_key_idx = key_idx
                # Обновляем индекс модели, только если это была ротация, а не выбор пользователя
                if not preferred_model:
                    try:
                        successful_model_index = OPENROUTER_MODELS.index(model_name)
                        current_or_model_idx = successful_model_index
                    except ValueError:
                        # Модель не найдена в списке, ничего не делаем с индексом
                        pass

                return {
                    "parts": [{"text": reply_text}],
                    "reply_text": reply_text,
                    "fn_call": None,
                    "model_name": f"openrouter:{model_name}",
                    "provider": "openrouter",
                }

            except requests.RequestException as exc:
                log.warning("OpenRouter request failed (model %s, key %s): %s", model_name, key_idx + 1, exc)
                time.sleep(SERVICE_UNAVAILABLE_DELAY)
                continue
            except ValueError as exc:
                log.warning("OpenRouter response error (model %s): %s", model_name, exc)
                continue

    return None


def _send_pollinations_request(
    chat_id: Optional[int],
    stored_history: List[Dict[str, Any]],
    user_message: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    log.info("Attempting Pollinations request")
    
    if not POLLINATIONS_TEXT_MODELS or not POLLINATIONS_TEXT_BASE_URL:
        log.warning("Pollinations not configured properly")
        return None
    
    if not _can_use_text_only_provider(user_message):
        log.info("Message contains non-text content, skipping Pollinations")
        return None

    model_name = _pollinations_text_model_for_chat(chat_id)
    if model_name not in POLLINATIONS_TEXT_MODELS:
        model_name = POLLINATIONS_TEXT_MODELS[0]
    
    log.info(f"Using Pollinations model: {model_name}")

    user_text = _parts_to_text(user_message.get("parts", []))
    messages = _prepare_openai_compatible_messages(stored_history, user_text)
    
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": POLLINATIONS_TEXT_TEMPERATURE,
    }
    
    if POLLINATIONS_API_KEY:
        payload["token"] = POLLINATIONS_API_KEY
        log.debug("Using Pollinations API key in request payload (as 'token')")

    headers = {"Content-Type": "application/json"}
    
    try:
        log.debug(f"Sending Pollinations request to {POLLINATIONS_TEXT_BASE_URL}")
        response = requests.post(
            POLLINATIONS_TEXT_BASE_URL,
            json=payload,
            headers=headers,
            timeout=POLLINATIONS_TEXT_TIMEOUT,
        )
        
        log.debug(f"Pollinations response status: {response.status_code}")
        
        if response.status_code in {429, 503}:
            log.warning(
                "Pollinations returned %s for model %s. Retrying other providers...",
                response.status_code,
                model_name,
            )
            time.sleep(SERVICE_UNAVAILABLE_DELAY)
            return None
        
        response.raise_for_status()
        
        data = response.json()
        choices = data.get("choices") or []
        
        if not choices:
            raise ValueError("Pollinations response contains no choices")
        
        message = choices[0].get("message") or {}
        reply_text = _openai_content_to_text(message.get("content"))
        
        if not reply_text:
            raise ValueError("Pollinations response has empty content")

        return {
            "parts": [{"text": reply_text}],
            "reply_text": reply_text,
            "fn_call": None,
            "model_name": f"pollinations:{model_name}",
            "provider": "pollinations",
        }
        
    except requests.RequestException as exc:
        log.warning("Pollinations request failed (model %s): %s", model_name, exc)
    except ValueError as exc:
        log.warning("Pollinations response error (model %s): %s", model_name, exc)
    except Exception as exc:
        log.error("Unexpected Pollinations error (model %s): %s", model_name, exc)
    
    return None


def _summarize_history(chat_id: int, provider_override: Optional[str] = None) -> None:
    chat_history = history.get(chat_id, [])
    if len(chat_history) <= MAX_HISTORY:
        return
    
    log.info("Summarizing history for chat %s...", chat_id)
    conversation_text = _history_to_text(chat_history)
    
    if not conversation_text:
        history[chat_id] = chat_history[-MAX_HISTORY:]
        return

    prompt_text = (
        "Summarize the following conversation in a concise paragraph for future context:\n"
        f"{conversation_text}\nSummary:"
    )
    summary_message = {"role": "user", "parts": [{"text": prompt_text}]}

    providers = _provider_sequence(_chat_provider_preference(chat_id, provider_override))
    log.info("History summary provider order: %s", providers)

    for provider in providers:
        try:
            result = None
            if provider == "gemini":
                result = _send_gemini_request([], summary_message)
            elif provider == "openrouter":
                result = _send_openrouter_request(chat_id, [], summary_message)
            elif provider == "pollinations":
                result = _send_pollinations_request(chat_id, [], summary_message)

            if result:
                summary_text = result.get("reply_text") or ""
                if summary_text.strip():
                    history[chat_id] = [
                        {"role": "user", "parts": [{"text": "Start of conversation."}]},
                        {"role": "model", "parts": [{"text": f"Previously discussed: {summary_text}"}]},
                    ]
                    log.info("Summary generated via %s", result.get("model_name"))
                    return
                    
        except Exception as exc:
            log.warning("History summarization attempt via %s failed: %s", provider, exc)

    log.error("History summarization failed for chat %s: all providers exhausted", chat_id)
    history[chat_id] = chat_history[-MAX_HISTORY:]


def llm_request(
    chat_id: int, 
    prompt_parts: List[Any], 
    provider_override: Optional[str] = None
) -> Tuple[Optional[str], str, Optional[Any]]:
    preferred_provider = _chat_provider_preference(chat_id, provider_override)
    _summarize_history(chat_id, provider_override)
    stored_history = history.get(chat_id, [])
    user_message = _normalize_prompt_parts(prompt_parts)

    providers = _provider_sequence(preferred_provider)
    log.info("Provider sequence: %s", providers)

    for provider in providers:
        log.info(f"Trying provider: {provider}")
        
        result = None
        try:
            if provider == "gemini":
                result = _send_gemini_request(stored_history, user_message)
            elif provider == "openrouter":
                result = _send_openrouter_request(chat_id, stored_history, user_message)
            elif provider == "pollinations":
                result = _send_pollinations_request(chat_id, stored_history, user_message)
        except Exception as exc:
            log.error(f"Provider {provider} raised exception: {exc}")
            continue
        
        if result:
            parts = result.get("parts") or []
            reply_text = (result.get("reply_text") or "").strip()
            fn_call = result.get("fn_call")
            model_name = result.get("model_name", provider)

            new_history = stored_history + [user_message]
            if parts:
                new_history.append({"role": "model", "parts": parts})
            history[chat_id] = new_history

            log.info("Response generated via %s", model_name)
            return reply_text if reply_text else None, model_name, fn_call

    raise Exception("All providers failed")


def llm_generate_image(prompt: str, pollinations_model: Optional[str] = None) -> Tuple[Optional[bytes], str]:
    global current_key_idx
    
    if POLLINATIONS_ENABLED and POLLINATIONS_BASE_URL:
        image_bytes, provider = _generate_image_via_pollinations(prompt, pollinations_model)
        if image_bytes:
            return image_bytes, provider

    if not API_KEYS:
        log.warning("No API keys available for Gemini image generation")
        return None, "gemini"
        
    model_name = IMAGE_MODEL_NAME
    for attempt in range(len(API_KEYS)):
        key_idx = (current_key_idx + attempt) % len(API_KEYS)
        try:
            client = _get_client(key_idx)
            image_bytes = _generate_image_via_gemini(client, model_name, prompt)
            if image_bytes:
                current_key_idx = key_idx
                return image_bytes, model_name
        except Exception as exc:
            log.warning(f"Image generation failed on key {key_idx + 1}: {exc}")
    
    return None, model_name


def _generate_image_via_pollinations(
    prompt: str, 
    model_override: Optional[str] = None
) -> Tuple[Optional[bytes], str]:
    seed_value = POLLINATIONS_SEED or str(random.randint(0, 1_000_000_000))
    encoded_prompt = urllib.parse.quote_plus(prompt)
    model_name = model_override or POLLINATIONS_MODEL
    
    params = {
        "width": str(POLLINATIONS_WIDTH),
        "height": str(POLLINATIONS_HEIGHT),
        "seed": seed_value,
        "model": model_name,
    }
    
    # Обновленная логика добавления параметров
    if POLLINATIONS_SAFE_MODE:
        params["safe"] = "true"
    
    if POLLINATIONS_PRIVATE_IMAGES: # <-- ДОБАВЛЕНО
        params["private"] = "true"
        
    if POLLINATIONS_NO_LOGO: # <-- ДОБАВЛЕНО
        params["nologo"] = "true"
    
    query = "&".join(f"{key}={urllib.parse.quote_plus(value)}" for key, value in params.items())
    url = f"{POLLINATIONS_BASE_URL.rstrip('/')}/prompt/{encoded_prompt}?{query}"
    
    headers = {}
    if POLLINATIONS_API_KEY:
        headers["Authorization"] = f"Bearer {POLLINATIONS_API_KEY}"
        log.debug("Using Pollinations API key for image generation")
    
    try:
        log.info(f"Generating image via Pollinations with params: {params}") # Улучшенное логирование
        response = requests.get(
            url, 
            headers=headers if headers else None,
            timeout=POLLINATIONS_TIMEOUT
        )
        response.raise_for_status()
        
        if response.content:
            # Формируем имя провайдера с учетом всех флагов
            provider_tags = [f"pollinations:{model_name}"]
            if POLLINATIONS_SAFE_MODE: provider_tags.append("safe")
            if POLLINATIONS_PRIVATE_IMAGES: provider_tags.append("private")
            if POLLINATIONS_NO_LOGO: provider_tags.append("nologo")
            provider_name = ":".join(provider_tags)
            
            log.info(f"Generated image via {provider_name}")
            return response.content, provider_name
        
        log.warning("Pollinations returned empty image content")
    except Exception as exc:
        log.warning(f"Pollinations image generation failed: {exc}")
    
    return None, f"pollinations:{model_name}"


def _generate_image_via_gemini(
    client: genai.Client, 
    model_name: str, 
    prompt: str
) -> Optional[bytes]:
    images_api = getattr(client, "images", None)
    if images_api and hasattr(images_api, "generate"):
        try:
            response = images_api.generate(model=model_name, prompt=prompt)
            generated = getattr(response, "generated_images", None) or getattr(response, "images", None)
            if generated:
                first = generated[0]
                data = (
                    getattr(first, "data", None) or 
                    getattr(first, "image", None) or 
                    getattr(first, "bytes", None)
                )
                if isinstance(data, str):
                    return base64.b64decode(data)
                if isinstance(data, (bytes, bytearray)):
                    return bytes(data)
        except Exception as exc:
            log.warning(f"Gemini image generation via images.generate failed: {exc}")
    
    try:
        generate_config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio="1:1"),
        )
        response = client.models.generate_content(
            model=model_name,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config=generate_config,
        )
        parts = _response_parts(response)
        for part in parts:
            inline = part.get("inline_data")
            if inline:
                data = inline.get("data")
                if isinstance(data, bytes):
                    return data
                if isinstance(data, str):
                    try:
                        return base64.b64decode(data)
                    except Exception:
                        continue
    except Exception as exc:
        log.warning(f"Gemini image generation fallback failed: {exc}")
    
    return None


def check_available_models() -> List[str]:
    global available_models, last_model_check_ts
    
    current_time = time.time()
    if current_time - last_model_check_ts < MODEL_CHECK_INTERVAL:
        return available_models
    
    log.info("Checking available Gemini models...")
    working_models: List[str] = []
    
    for model_name in GEMINI_MODELS:
        model_found = False
        for key_idx in range(len(API_KEYS)):
            if model_found:
                break
            try:
                client = _get_client(key_idx)
                response = client.models.generate_content(
                    model=model_name,
                    contents=[{"role": "user", "parts": [{"text": "Reply with OK"}]}],
                    config={"temperature": 0.1}
                )
                
                if hasattr(response, 'candidates') or hasattr(response, 'text'):
                    working_models.append(model_name)
                    log.info("Model %s is available with key #%s", model_name, key_idx + 1)
                    model_found = True
                    
            except Exception as e:
                log.debug(f"Model {model_name} test failed with key #{key_idx + 1}: {e}")
                continue
    
    if working_models:
        available_models = working_models
        last_model_check_ts = current_time
        log.info(f"Available Gemini models: {working_models}")
    else:
        available_models = GEMINI_MODELS.copy()
        log.warning("Could not verify models, using all configured models")
    
    return available_models