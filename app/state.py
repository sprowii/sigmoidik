# Copyright (c) 2025 sprouee
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ChatConfig:
    autopost_enabled: bool = False
    interval: int = 14400
    min_messages: int = 10
    msg_size: str = ""
    last_post_ts: float = 0.0
    new_msg_counter: int = 0
    pollinations_model: str = ""


configs: Dict[int, ChatConfig] = {}
history: Dict[int, List[Dict[str, Any]]] = {}
user_profiles: Dict[int, Dict[int, Dict[str, Any]]] = {}


