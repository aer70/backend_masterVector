"""ИИ-ассистент: анализирует BMP через Ollama и предлагает настройки векторизации."""

from __future__ import annotations

import base64
import io
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np
from PIL import Image

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
OLLAMA_TIMEOUT_SEC = int(os.getenv("OLLAMA_TIMEOUT_SEC", "60"))

THUMBNAIL_SIZE = (512, 512)

# Модели, поддерживающие vision (передачу изображений base64)
_VISION_MODEL_PREFIXES = (
    "llava", "moondream", "bakllava", "llava-phi", "minicpm-v",
    "llama3.2-vision", "gemma3", "qwen2-vl", "cogvlm", "phi3.5-vision",
)


def _is_vision_model(model: str) -> bool:
    low = model.lower()
    return any(low.startswith(p) or p in low for p in _VISION_MODEL_PREFIXES)


# ─── Описание параметров для промпта ────────────────────────────────────────
_PARAM_DOCS = """
Параметры векторизации:
- method: "rle" (точный, больше узлов) или "contours" (меньше узлов, проще).
- pre_scale_factor: масштаб перед векторизацией (float, 1.0–4.0). Больше = лучше сглаживание ступенек.
- min_area: минимальная площадь объекта (int, ≥1). Больше = убирает шум.
- color_merge_step: слияние близких цветов (int, 1=нет, 2–8=антиалиасинг/градиент).
- connect_same_color: true/false — склеивать фрагменты одного цвета.
- connect_radius_px: радиус склейки (float, 0–6).
- connect_op: "close" (аккуратно), "dilate" (агрессивно), "none".
- connect_4neighbors: true=4-связность, false=8-связность.
- enable_smoothing: true/false — сглаживание лесенок.
- snap_90deg_corners: true/false — выравнивание прямых углов (для схем).
- simplify_collinear: true/false — удалять точки на одной прямой.
- collinear_tol: допуск (float, 0–2.0).
- stair4_enable: true/false — убирать углы 135° (для пиксельной графики).
- ra90_enable: true/false — упрощение серий прямых углов.
- ignore_background: true/false — не векторизовывать фон.
- draw_background: true/false — добавить фоновый прямоугольник.
- black_threshold: порог чёрного (int, 0–30).
""".strip()

_SYSTEM_PROMPT_VISION = f"""Ты — эксперт по векторизации растровых изображений. Проанализируй изображение и подбери оптимальные настройки для алгоритма BMP→SVG.

{_PARAM_DOCS}

Ответ верни СТРОГО в формате:
```json
{{
  "method": "rle",
  "pre_scale_factor": 3.0,
  "min_area": 2,
  "color_merge_step": 1,
  "connect_same_color": true,
  "connect_radius_px": 1.0,
  "connect_op": "close",
  "connect_4neighbors": true,
  "enable_smoothing": false,
  "snap_90deg_corners": true,
  "simplify_collinear": true,
  "collinear_tol": 0.0,
  "stair4_enable": true,
  "ra90_enable": true,
  "ignore_background": true,
  "draw_background": true,
  "black_threshold": 0
}}
```
ОБЪЯСНЕНИЕ: <1–3 предложения почему эти настройки оптимальны>"""

_SYSTEM_PROMPT_TEXT = f"""Ты — эксперт по векторизации растровых изображений. На основе метрик изображения подбери оптимальные настройки для алгоритма BMP→SVG.

{_PARAM_DOCS}

Ответ верни СТРОГО в формате:
```json
{{
  "method": "rle",
  "pre_scale_factor": 3.0,
  "min_area": 2,
  "color_merge_step": 1,
  "connect_same_color": true,
  "connect_radius_px": 1.0,
  "connect_op": "close",
  "connect_4neighbors": true,
  "enable_smoothing": false,
  "snap_90deg_corners": true,
  "simplify_collinear": true,
  "collinear_tol": 0.0,
  "stair4_enable": true,
  "ra90_enable": true,
  "ignore_background": true,
  "draw_background": true,
  "black_threshold": 0
}}
```
ОБЪЯСНЕНИЕ: <1–3 предложения почему эти настройки оптимальны для изображения с такими характеристиками>"""


@dataclass
class AISuggestion:
    ok: bool
    settings: dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    model: str = ""
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "settings": self.settings,
            "explanation": self.explanation,
            "model": self.model,
            "error": self.error,
        }


# ─── Локальный анализ изображения (PIL + numpy) ──────────────────────────────

def _analyze_image(image_bytes: bytes) -> dict[str, Any]:
    """Извлекает числовые метрики из изображения для текстового промпта."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        orig_w, orig_h = img.size
        img_rgb = img.convert("RGB")
        img_small = img_rgb.copy()
        img_small.thumbnail((256, 256), Image.LANCZOS)
        arr = np.array(img_small)

    h, w, _ = arr.shape
    pixels = arr.reshape(-1, 3)

    # Число уникальных цветов
    unique_colors = len(set(map(tuple, pixels.tolist())))

    # Топ-3 цвета по частоте
    color_counts = Counter(map(tuple, pixels.tolist()))
    top_colors = [list(c) for c, _ in color_counts.most_common(3)]

    # Оценка "резкости" — средний градиент по яркости
    gray = np.mean(arr, axis=2).astype(np.float32)
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    sharpness = float(np.mean(gx) + np.mean(gy))

    # Оценка наличия чёткой пиксельной структуры (ступеньки)
    # Резкость > 5 обычно означает чёткие края
    is_pixel_art = sharpness > 8.0 and unique_colors < 64

    # Наличие почти-чёрных пикселей
    dark_ratio = float(np.mean(np.all(arr < 30, axis=2)))

    # Примерный тип изображения
    if unique_colors <= 8:
        image_type = "схема/диаграмма с малым числом цветов (≤8)"
    elif unique_colors <= 32:
        image_type = "пиксельная графика или иконка"
    elif is_pixel_art:
        image_type = "пиксельная графика с чёткими краями"
    elif unique_colors > 1000:
        image_type = "фотография или изображение с градиентами"
    else:
        image_type = "растровая графика со средним числом цветов"

    return {
        "width_px": orig_w,
        "height_px": orig_h,
        "unique_colors": unique_colors,
        "sharpness_score": round(sharpness, 2),
        "is_pixel_art": is_pixel_art,
        "dark_pixel_ratio": round(dark_ratio, 3),
        "top3_colors_rgb": top_colors,
        "image_type": image_type,
    }


def _build_text_user_prompt(metrics: dict[str, Any]) -> str:
    return (
        f"Характеристики изображения:\n"
        f"- Размер: {metrics['width_px']}×{metrics['height_px']} пикселей\n"
        f"- Уникальных цветов: {metrics['unique_colors']}\n"
        f"- Оценка резкости: {metrics['sharpness_score']} (>8 = чёткие края)\n"
        f"- Пиксельная графика: {'да' if metrics['is_pixel_art'] else 'нет'}\n"
        f"- Доля тёмных пикселей (<30): {metrics['dark_pixel_ratio']:.1%}\n"
        f"- Топ-3 цвета RGB: {metrics['top3_colors_rgb']}\n"
        f"- Тип изображения: {metrics['image_type']}\n\n"
        "Подбери оптимальные настройки векторизации для этого изображения."
    )


def _image_to_base64_png(image_bytes: bytes) -> str:
    """Конвертирует изображение в PNG-thumbnail и кодирует в base64."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        img.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")


def _parse_response(text: str) -> tuple[dict[str, Any], str]:
    """Извлекает JSON настроек и объяснение из ответа модели."""
    json_match = re.search(r"```json\s*([\s\S]+?)\s*```", text, re.IGNORECASE)
    if not json_match:
        json_match = re.search(r"(\{[\s\S]+\})", text)

    settings: dict[str, Any] = {}
    if json_match:
        try:
            settings = json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    explanation = text
    if json_match:
        after = text[json_match.end():]
        expl_match = re.search(r"ОБЪЯСНЕНИЕ:\s*(.+)", after, re.IGNORECASE | re.DOTALL)
        if expl_match:
            explanation = expl_match.group(1).strip()
        else:
            explanation = after.strip()

    if not explanation:
        explanation = re.sub(r"```json[\s\S]+?```", "", text, flags=re.IGNORECASE).strip()

    return settings, explanation


async def suggest_settings(image_bytes: bytes) -> AISuggestion:
    """Запускает ollama run и возвращает рекомендованные настройки."""
    import asyncio as _asyncio
    import logging as _logging
    _log = _logging.getLogger("ai_service")

    vision = _is_vision_model(OLLAMA_MODEL)

    if vision:
        try:
            img_b64 = _image_to_base64_png(image_bytes)
        except Exception as exc:
            return AISuggestion(ok=False, error=f"Не удалось обработать изображение: {exc}")
        # Для vision-моделей: используем HTTP API (ollama run не поддерживает передачу изображений)
        prompt_text = "Проанализируй изображение и подбери настройки."
        use_subprocess = False
    else:
        try:
            metrics = _analyze_image(image_bytes)
        except Exception as exc:
            return AISuggestion(ok=False, error=f"Не удалось проанализировать изображение: {exc}")
        prompt_text = _SYSTEM_PROMPT_TEXT + "\n\n" + _build_text_user_prompt(metrics)
        use_subprocess = True

    if use_subprocess:
        # Вызываем `ollama run <model>` через subprocess — обходим HTTP API
        _log.info("Running ollama subprocess for model %s", OLLAMA_MODEL)
        try:
            proc = await _asyncio.create_subprocess_exec(
                "ollama", "run", OLLAMA_MODEL, prompt_text,
                stdout=_asyncio.subprocess.PIPE,
                stderr=_asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await _asyncio.wait_for(
                    proc.communicate(), timeout=OLLAMA_TIMEOUT_SEC
                )
            except _asyncio.TimeoutError:
                proc.kill()
                return AISuggestion(ok=False, error=f"Ollama не ответила за {OLLAMA_TIMEOUT_SEC} секунд.")
            raw_text = stdout.decode("utf-8", errors="replace").strip()
            if proc.returncode != 0 and not raw_text:
                err = stderr.decode("utf-8", errors="replace")[:300]
                return AISuggestion(ok=False, error=f"Ошибка ollama run: {err}")
            _log.info("ollama run returned %d chars", len(raw_text))
        except FileNotFoundError:
            return AISuggestion(ok=False, error="Команда 'ollama' не найдена. Установите Ollama и добавьте в PATH.")
        except Exception as exc:
            return AISuggestion(ok=False, error=f"Неожиданная ошибка: {exc}")
    else:
        # Vision-модели: HTTP API
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT_VISION},
                {"role": "user", "content": prompt_text, "images": [img_b64]},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        }
        try:
            async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT_SEC) as client:
                response = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
                if response.status_code == 404:
                    return AISuggestion(ok=False, error=f"Модель '{OLLAMA_MODEL}' не найдена.")
                if response.status_code != 200:
                    return AISuggestion(ok=False, error=f"Ошибка Ollama {response.status_code}: {response.text[:200]}")
                raw_text = response.json().get("message", {}).get("content", "")
        except httpx.ConnectError:
            return AISuggestion(ok=False, error=f"Не удалось подключиться к Ollama ({OLLAMA_URL}).")
        except httpx.TimeoutException:
            return AISuggestion(ok=False, error=f"Ollama не ответила за {OLLAMA_TIMEOUT_SEC} секунд.")
        except Exception as exc:
            return AISuggestion(ok=False, error=f"Неожиданная ошибка: {exc}")

    if not raw_text:
        return AISuggestion(ok=False, error="Ollama вернула пустой ответ.")

    settings, explanation = _parse_response(raw_text)

    if not settings:
        return AISuggestion(
            ok=False,
            error="Не удалось извлечь JSON настроек из ответа модели.",
            explanation=raw_text[:600],
            model=OLLAMA_MODEL,
        )

    return AISuggestion(ok=True, settings=settings, explanation=explanation, model=OLLAMA_MODEL)


_CHAT_SYSTEM_PROMPT = """Ты — эксперт по векторизации растровых изображений. Ты помогаешь пользователю подобрать настройки алгоритма BMP→SVG.

Текущие настройки и контекст переданы в начале разговора. Когда пользователь просит изменить или объяснить настройки:
- Отвечай по-русски, кратко и понятно
- Если предлагаешь новые настройки, верни JSON-блок в формате:
```json
{ "param": value, ... }
```
- Объясняй почему именно эти значения подходят
- Если пользователь просто задаёт вопрос — просто ответь без JSON

Параметры (краткая справка):
- method: "rle" (точный) / "contours" (проще)
- pre_scale_factor: масштаб 1.0–4.0 (больше = лучше сглаживание)
- min_area: минимальная площадь (убирает шум)
- color_merge_step: слияние цветов 1–8 (для антиалиасинга)
- enable_smoothing: сглаживание ступенек
- snap_90deg_corners: прямые углы (для схем)
- stair4_enable: убирать диагональные ступеньки
- connect_same_color / connect_radius_px: склейка фрагментов одного цвета
- ignore_background: не векторизовывать фон
- black_threshold: порог чёрного (0–30)
"""


async def chat_about_settings(
    messages: list[dict[str, str]],
    current_settings: dict[str, Any] | None = None,
    image_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ведёт диалог с LLM об настройках векторизации через subprocess."""
    import asyncio as _asyncio

    # Формируем полный промпт: системный + контекст + история
    context_parts = [_CHAT_SYSTEM_PROMPT]
    if current_settings:
        context_parts.append(f"\nТекущие настройки:\n```json\n{json.dumps(current_settings, ensure_ascii=False, indent=2)}\n```")
    if image_metrics:
        context_parts.append(
            f"\nХарактеристики изображения: размер {image_metrics.get('width_px')}×{image_metrics.get('height_px')}px, "
            f"цветов: {image_metrics.get('unique_colors')}, тип: {image_metrics.get('image_type')}"
        )

    # Строим текстовый диалог для передачи в ollama run
    dialog_lines = ["\n".join(context_parts)]
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            dialog_lines.append(f"\nПОЛЬЗОВАТЕЛЬ: {content}")
        else:
            dialog_lines.append(f"\nАССИСТЕНТ: {content}")

    full_prompt = "\n".join(dialog_lines)

    try:
        proc = await _asyncio.create_subprocess_exec(
            "ollama", "run", OLLAMA_MODEL, full_prompt,
            stdout=_asyncio.subprocess.PIPE,
            stderr=_asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await _asyncio.wait_for(proc.communicate(), timeout=OLLAMA_TIMEOUT_SEC)
        except _asyncio.TimeoutError:
            proc.kill()
            return {"ok": False, "error": f"Ollama не ответила за {OLLAMA_TIMEOUT_SEC} секунд."}

        raw_text = stdout.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0 and not raw_text:
            err = stderr.decode("utf-8", errors="replace")[:300]
            return {"ok": False, "error": f"Ошибка ollama: {err}"}
    except FileNotFoundError:
        return {"ok": False, "error": "Команда 'ollama' не найдена."}
    except Exception as exc:
        return {"ok": False, "error": f"Неожиданная ошибка: {exc}"}

    # Извлекаем JSON если есть (новые настройки)
    settings, _ = _parse_response(raw_text)
    # Убираем JSON-блок из текста ответа для чистого отображения
    reply_text = re.sub(r"```json[\s\S]+?```", "", raw_text, flags=re.IGNORECASE).strip()
    if not reply_text:
        reply_text = raw_text

    return {
        "ok": True,
        "reply": reply_text,
        "new_settings": settings if settings else None,
        "model": OLLAMA_MODEL,
    }
