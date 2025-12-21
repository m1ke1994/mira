# Голосовой ассистент Mira (Windows, Python)

Standalone ассистент, который слушает микрофон, ловит wakeword «Привет, Мира» (Porcupine) или работает в push‑to‑talk, отправляет речь в Gemini Native Audio (`gemini-2.5-flash-native-audio-dialog`) и воспроизводит голосовой ответ.

## Установка
- Python 3.11+ (проверено на 3.12)
- Установите зависимости: `python -m pip install -r requirements.txt`
- Положите `.env` в корень (см. пример ниже)
- На Windows потребуется установленный портaudio (PyAudio ставится через pip wheel)

## Конфигурация (`.env`)
```
GEMINI_API_KEY=your_key
GEMINI_MODEL_NAME=gemini-2.5-flash-native-audio-dialog

# Wakeword (Porcupine)
PICOVOICE_ACCESS_KEY=...
WAKEWORD_KEYWORD_PATH=path\to\mirra.ppn
WAKEWORD_SENSITIVITY=0.6

# Аудио
INPUT_DEVICE_INDEX=0          # опционально
INPUT_TARGET_RATE=16000
OUTPUT_TARGET_RATE=24000
MAX_UTTERANCE_SECONDS=15
SILENCE_MS=1400

# Поведение
LOG_LEVEL=INFO
LANG=ru
SYSTEM_PROMPT=Ты голосовой ассистент Мира. Отвечай по-русски, кратко и дружелюбно.
VOICE_NAME=ru-RU-Standard-A   # если модель поддерживает выбор голоса
```

Пример `.env` лежит в `.env.example`.

## Запуск
```
python mira_assistant.py
```

- При наличии `WAKEWORD_KEYWORD_PATH` ассистент стартует в wakeword-режиме. После срабатывания скажет «Да?»/короткий бип и начнёт запись реплики.
- Если модели нет — fallback: в консоли появится `Нажмите Enter и говорите...`.

## Диагностика
- Список микрофонов: `python tools/list_mics.py`
- Проверка wakeword: `python tools/test_wakeword.py`
- Проверка вывода звука: `python tools/test_audio_out.py`
- Логи: `logs/mira.log` (rotating)

## Получение wakeword `.ppn`
Сгенерируйте модель в Picovoice Console с фразой «Привет, Мира» (ru-RU), скачайте `.ppn` и пропишите путь в `.env` (`WAKEWORD_KEYWORD_PATH`). Sensitivity подберите (0.5–0.7).

## Что есть под капотом
- PyAudio захватывает микрофон, нормализует в mono int16 и ресэмплит к 16 кГц.
- Porcupine слушает wakeword в фоне; при проигрывании ответа wakeword не активен.
- После захвата реплики аудио отправляется в Gemini Native Audio, ответ стримится и сразу воспроизводится (24 кГц).
- FSM: ожидание wakeword/PTT → запись пользователя (VAD по энергии/тайм-аут) → запрос к модели → воспроизведение → возврат в ожидание.

## Известные ограничения
- Нужен интернет-доступ к Gemini API и доступная модель `gemini-2.5-flash-native-audio-dialog`.
- Голосовые параметры зависят от поддержки голосов в API; если `VOICE_NAME` не поддерживается, используется дефолт модели.
