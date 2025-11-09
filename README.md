# Сигмоида

Телеграм-бот, который общается через Google Gemini API. Понимает текст, голосовые, картинки и видео. Генерирует изображения. Создаёт Phaser-игры по описанию.

## Что умеет

- Общается в группах и личке, запоминает контекст диалога
- Рисует картинки по команде `/draw` или автоматически, если модель решит
- Создаёт мини-игры на Phaser по идее из команды `/game`
- Понимает голосовые и видеосообщения
- Автоматически сокращает длинную историю (по умолчанию через DeepSeek на OpenRouter), чтобы не терять контекст
- Автопосты в группы (если включить)
- Сохраняет историю и настройки в Redis
- Форматирует ответы через HTML-теги Telegram
- Может использовать как модели Google Gemini, так и модели через OpenRouter (DeepSeek и др.)

## Что нужно для запуска

- Пайтон в районе 3.12
- Токен бота от @BotFather в Telegram
- API-ключ от Google AI Studio (минимум один, можно несколько для ротации)
- Redis (локально или в облаке)
-ну, ещё нужно быть крутым(шучу)

## Как запустить?

1. Склонируй репозиторий и перейди в папку:

```bash
git clone <адрес_репозитория>
cd сигмоида
```

2. Создай виртуальное окружение и установи зависимости:

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

3. Создай файл `.env` со своими ключами:

```env
TG_TOKEN=твой_токен_бота
ADMIN_ID=твой_telegram_id
GEMINI_API_KEY_1=первый_ключ_gemini
GEMINI_API_KEY_2=второй_ключ_gemini
REDIS_URL=redis://localhost:6379
PORT=10000
# Опционально: OpenRouter
OPENROUTER_API_KEY=твой_openrouter_api_key
OPENROUTER_MODELS=deepseek/deepseek-chat-v3-0324:free,deepseek/deepseek-r1-0528:free,tngtech/deepseek-r1t2-chimera:free
OPENROUTER_SITE_URL=https://example.com
OPENROUTER_SITE_NAME=Sigmoida
LLM_PROVIDER_ORDER=gemini,openrouter
# Опционально: Pollinations
POLLINATIONS_MODEL=flux
POLLINATIONS_MODELS=flux,flux-realistic,flux-anime
```

4. Запусти:

```bash
python сигмоида.py
```

## Деплой на Render

В проекте есть `render.yaml` — создаёшь Blueprint-сервис, подключаешь репозиторий, прописываешь переменные окружения (включая ключи Gemini и, при необходимости, OpenRouter) и ждёшь деплой. Бесплатного тарифа достаточно для тестов.

## Команды бота

**Для всех:**
- `/start` — запустить бота
- `/help` — справка по командам
- `/draw описание` — нарисовать картинку
- `/set_draw_model название` — выбрать модель Pollinations для /draw и автогенерации
- `/game идея` — создать игру на Phaser
- `/login` — получить код для входа на сайт
- `/reset` — очистить историю
- `/privacy` — политика конфиденциальности

**Для админа:**
- `/settings` — посмотреть настройки чата
- `/delete_data chat_id` — удалить данные чата
- `/autopost on|off` — включить/выключить автопосты
- `/set_interval секунды` — интервал между автопостами
- `/set_minmsgs число` — минимум сообщений для автопоста
- `/set_msgsize small|medium|large` — размер ответов бота

## Как сжимается история

- Когда переписка в чате становится длиннее `MAX_HISTORY` (сейчас 10 сообщений), бот готовит текстовую выжимку из текущего диалога.
- Само сжатие сначала пытается сделать модель DeepSeek через OpenRouter. Если OpenRouter недоступен, бот переключается на ближайший доступный ключ Gemini.
- Полученная выжимка сохраняется как сжатый «пролог» истории, а старые сообщения удаляются. Так сохраняется контекст без перерасхода токенов.

## Структура проекта

```
app/
├── bot/handlers.py        # обработчики команд и сообщений
├── game/generator.py      # генерация игр через LLM
├── llm/client.py          # работа с Gemini API
├── security/privacy.py    # текст политики конфиденциальности
├── storage/redis_store.py # сохранение в Redis
├── web/server.py          # Flask-сервер для веб-части
├── config.py              # настройки из .env
└── state.py               # состояние в памяти

webapp/
├── hub.html               # главная страница с играми
├── sandbox.html           # песочница для запуска игр
└── *.css, *.js            # стили и скрипты

сигмоида.py                # точка входа
render.yaml                # конфиг для Render
```

## Лицензия

**Modified MIT License**

Copyright 2025 sprouee

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

**Commercial use requires written permission from the copyright holder.**