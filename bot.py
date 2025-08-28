import asyncio
import csv
import re
from urllib.parse import unquote
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Optional
import html

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandObject
from aiogram.types import Message, User, FSInputFile

from config import get_settings
from rag.retrieve import retrieve, Chunk
from rag.llm import generate_answer


settings = get_settings()

# --- Images support settings ---
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
MAX_RETURN_PHOTOS = 3
TELEGRAM_MSG_LIMIT = 4096


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return path.read_text(errors="ignore")


IMG_MD_RE = re.compile(r"!\[[^\]]*\]\(([^)]+?\.(?:jpg|jpeg|png|webp|gif))\)", re.IGNORECASE)
IMG_PHOTO_RE = re.compile(r"(?:^|\s)(?:фото|photo)\s*:\s*(.+?\.(?:jpg|jpeg|png|webp|gif))", re.IGNORECASE)
# Markdown links like [Text](https://...)
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
# Raw links to catch presence even without markdown
LINK_RE = re.compile(r"https?://\S+", re.IGNORECASE)


def split_markdown_into_segments(md: str) -> List[Tuple[str, Optional[str]]]:
    """
    Разбивает Markdown на последовательность (text, image_path_or_None),
    сохраняя порядок появления картинок.
    Между картинками возвращаются текстовые сегменты (включая переносы).
    """
    segments: List[Tuple[str, Optional[str]]] = []

    # Список всех изображений в тексте: markdown и "фото: ..."
    matches: List[Tuple[int, int, str]] = []  # (start, end, raw_path)

    for m in IMG_MD_RE.finditer(md):
        matches.append((m.start(), m.end(), m.group(1).strip()))
    for m in IMG_PHOTO_RE.finditer(md):
        matches.append((m.start(), m.end(), m.group(1).strip()))

    if not matches:
        return [(md, None)]

    matches.sort(key=lambda x: x[0])
    cursor = 0
    for (s, e, raw) in matches:
        if s > cursor:
            segments.append((md[cursor:s], None))
        segments.append(("", raw))  # место изображения
        cursor = e
    if cursor < len(md):
        segments.append((md[cursor:], None))
    return segments


def extract_section_text(full_md: str, section: str) -> str:
    """
    Если есть название секции (## Заголовок), берём её содержимое до следующего заголовка того же/высшего уровня.
    Если секции нет — возвращаем весь текст.
    """
    if not section:
        return full_md
    pattern = re.compile(rf"(?mi)^##\s+{re.escape(section)}\s*$")
    m = pattern.search(full_md)
    if not m:
        return full_md
    start = m.end()
    m2 = re.search(r"(?m)^\s*#{1,2}\s+.+$", full_md[start:])
    end = start + m2.start() if m2 else len(full_md)
    return full_md[start:end].strip()


def select_primary_doc(chunks: List[Chunk]) -> Tuple[Optional[str], Optional[str]]:
    """Возвращает (relative_path, section_name) для top-1 чанка."""
    if not chunks:
        return None, None
    primary = chunks[0]
    return primary.path, (primary.section or "")


def _md_links_to_html(text: str) -> str:
    """Convert Markdown links [text](url) to HTML <a> while escaping other text safely."""
    out = []
    last = 0
    for m in MD_LINK_RE.finditer(text):
        s, e = m.span()
        # escape text before link
        out.append(html.escape(text[last:s]))
        link_text = html.escape(m.group(1))
        href = m.group(2)
        # minimal sanitization: disallow quotes in href
        href = href.replace('"', "&quot;")
        out.append(f'<a href="{href}">{link_text}</a>')
        last = e
    out.append(html.escape(text[last:]))
    return "".join(out)


async def send_text_chunks(message: Message, text: str):
    """Отправляет текст, деля на части не длиннее лимита Telegram, сохраняя гиперссылки."""
    # Convert markdown links to HTML anchors and escape the rest
    html_text = _md_links_to_html(text.strip())
    t = html_text
    while t:
        part = t[:TELEGRAM_MSG_LIMIT]
        if len(t) > TELEGRAM_MSG_LIMIT:
            # try not to break inside an anchor by cutting at a safe boundary
            cut = part.rfind("</a>")
            if cut != -1 and cut > TELEGRAM_MSG_LIMIT - 500:
                cut += len("</a>")
            else:
                cut = part.rfind("\n")
                if cut < 500:
                    cut = part.rfind(" ")
                if cut > 200:
                    part = part[:cut]
        await message.answer(part, parse_mode="HTML")
        t = t[len(part):]


async def render_markdown_with_inline_images(message: Message, rel_path: str, section: str):
    """
    Рендерит указанный Markdown (или его секцию) и отправляет последовательность сообщений:
    текст -> фото -> текст -> ...
    Также передаёт внешние ссылки как кликабельные гиперссылки в тексте.
    """
    base_dir = (settings.data_dir / Path(rel_path)).parent.resolve()
    file_path = (settings.data_dir / rel_path).resolve()
    if not file_path.is_file():
        return False
    md = read_text_file(file_path)
    section_md = extract_section_text(md, section)
    segments = split_markdown_into_segments(section_md)

    any_output = False
    for text_seg, image_raw in segments:
        if text_seg and text_seg.strip():
            await send_text_chunks(message, text_seg)
            any_output = True
        if image_raw:
            p = try_resolve_path(image_raw, base_dir)
            if p and p.suffix.lower() in IMAGE_EXTS and p.is_file():
                try:
                    await message.answer_photo(photo=FSInputFile(str(p)))
                    any_output = True
                except Exception:
                    pass
    return any_output


def normalize_name(text: str) -> List[str]:
    """Упрощённая нормализация ФИО/имен для поиска по названиям файлов."""
    t = text.lower()
    # оставить буквы/цифры/пробелы
    t = re.sub(r"[^a-zа-я0-9ё\s]+", " ", t)
    parts = [p for p in t.split() if len(p) >= 2]
    return parts


def extract_photo_paths_from_text(text: str) -> List[str]:
    """Ищет в тексте конструкции вида 'фото: /path/to/file.jpg' и markdown-изображения."""
    found: List[str] = []
    # фото: <путь>
    for m in re.finditer(r"(?:фото|photo)\s*:\s*(.+?\.(?:jpg|jpeg|png|webp|gif))", text, flags=re.IGNORECASE):
        found.append(m.group(1).strip())
    # Markdown ![alt](path)
    for m in re.finditer(r"!\[[^\]]*\]\((.+?\.(?:jpg|jpeg|png|webp|gif))\)", text, flags=re.IGNORECASE):
        found.append(m.group(1).strip())
    return found


def try_resolve_path(raw_path: str, base_dir: Path) -> Path | None:
    """Пытается привести путь к существующему файлу: абсолютный или относительно base_dir.
    Также декодирует URL-encoded пути из Markdown (%XX).
    """
    if not raw_path:
        return None
    raw_decoded = unquote(raw_path)
    candidates = [raw_decoded]
    # иногда в тексте встречается префикс './' или 'data/'
    if raw_decoded.startswith("./"):
        candidates.append(raw_decoded[2:])
    if raw_decoded.startswith("data/"):
        candidates.append(raw_decoded[5:])

    for cand in candidates:
        p = Path(cand)
        if p.is_file():
            return p
        p2 = (base_dir / cand).resolve()
        if p2.is_file():
            return p2
    return None


def find_photos(question: str, chunks: List[Chunk]) -> List[Path]:
    """Находит релевантные фото:
    1) парсит явные пути к фото в тексте чанков (фото: ... или ![](…));
    2) если не нашли — ищет по имени из вопроса в директориях рядом с найденными файлами.
    Возвращает существующие пути, максимум MAX_RETURN_PHOTOS.
    """
    settings_local = get_settings()
    results: List[Path] = []
    seen: set[Path] = set()

    # 1) явные упоминания
    for c in chunks:
        base_dir = (settings_local.data_dir / Path(c.path).parent).resolve()
        for raw in extract_photo_paths_from_text(c.text):
            p = try_resolve_path(raw, base_dir)
            if p and p.suffix.lower() in IMAGE_EXTS and p not in seen:
                seen.add(p)
                results.append(p)
                if len(results) >= MAX_RETURN_PHOTOS:
                    return results

    # 2) эвристика по имени из вопроса (например, ФИО)
    name_tokens = normalize_name(question)
    if name_tokens:
        tokens = set(name_tokens)
        # искать только в папках рядом с top-k найденными документами
        candidate_dirs: List[Path] = []
        for c in chunks:
            d = (settings_local.data_dir / Path(c.path).parent).resolve()
            if d not in candidate_dirs:
                candidate_dirs.append(d)
        for d in candidate_dirs:
            try:
                for entry in d.iterdir():
                    if entry.is_file() and entry.suffix.lower() in IMAGE_EXTS:
                        stem_norm = set(normalize_name(entry.stem))
                        if tokens.issubset(stem_norm):
                            if entry not in seen:
                                seen.add(entry)
                                results.append(entry)
                                if len(results) >= MAX_RETURN_PHOTOS:
                                    return results
            except FileNotFoundError:
                continue

    return results


def is_allowed(chat_id: int, user_id: int) -> bool:
    allow = settings.allowed_chat_ids
    return (chat_id in allow) or (user_id in allow)


def log_query(user: User, question: str, sources: List[str], answer: str, status: str, has_attachment: bool):
    settings.queries_log.parent.mkdir(parents=True, exist_ok=True)
    new_file = not settings.queries_log.exists()
    with open(settings.queries_log, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "user_id", "username", "question", "retrieved_sources", "answer_len", "status", "has_attachment"]) 
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            getattr(user, 'id', ''),
            getattr(user, 'username', ''),
            question,
            "; ".join(sorted(set(sources)))[:1000],
            len(answer),
            status,
            int(has_attachment),
        ])


def log_feedback(user: User, text: str):
    settings.feedback_log.parent.mkdir(parents=True, exist_ok=True)
    new_file = not settings.feedback_log.exists()
    with open(settings.feedback_log, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "user_id", "username", "feedback"]) 
        writer.writerow([datetime.now(timezone.utc).isoformat(), getattr(user, 'id', ''), getattr(user, 'username', ''), text])


async def handle_start(message: Message):
    if not is_allowed(message.chat.id, message.from_user.id):
        return await message.answer("Доступ ограничен. Обратитесь к администратору.")
    examples = (
        "Примеры вопросов:\n"
        "• Что делать, если пропал интернет?\n"
        "• Как правильно принимать наличные и проверять купюры?\n"
        "• Как включить кондиционер?\n"
        "• Звук Бизнес не играет — что сделать?\n"
        "• Дезинсекция — что подготовить?\n"
    )
    await message.answer(
        "Я отвечаю только на основе внутренней базы знаний. Прикреплённые фото/видео игнорирую, но логирую факт.\n"
        + examples
    )


async def handle_policy(message: Message):
    if not is_allowed(message.chat.id, message.from_user.id):
        return await message.answer("Доступ ограничен. Обратитесь к администратору.")
    await message.answer(
        "Политика: отвечаю только из локальной БЗ, без интернета. Если данных нет — честно откажу и подскажу эскалацию."
    )


async def handle_kb(message: Message, command: CommandObject):
    if not is_allowed(message.chat.id, message.from_user.id):
        return await message.answer("Доступ ограничен. Обратитесь к администратору.")
    q = (command.args or "").strip()
    if not q:
        return await message.answer("Использование: /kb <запрос>")
    chunks, _, status = retrieve(q)
    if not chunks:
        return await message.answer("Ничего не найдено.")
    lines = ["Топ разделов:"]
    seen = set()
    for c in chunks:
        if c.path in seen:
            continue
        seen.add(c.path)
        lines.append(f"• {c.path} (score={c.score:.3f})")
        if len(seen) >= 10:
            break
    await message.answer("\n".join(lines))


async def handle_feedback(message: Message, command: CommandObject):
    if not is_allowed(message.chat.id, message.from_user.id):
        return await message.answer("Доступ ограничен. Обратитесь к администратору.")
    text = (command.args or "").strip()
    if not text:
        return await message.answer("Опишите проблему после команды /feedback")
    log_feedback(message.from_user, text)
    await message.answer("Спасибо! Ваш фидбек записан.")


async def handle_question(message: Message):
    has_attachment = bool(message.photo or message.video or message.document or message.audio)
    if not is_allowed(message.chat.id, message.from_user.id):
        return await message.answer("Доступ ограничен. Обратитесь к администратору.")

    question = (message.text or message.caption or "").strip()
    if not question:
        # no text, but maybe just attachment — log and ignore
        log_query(message.from_user, "<no_text>", [], "", "NO_CONTEXT", has_attachment)
        return await message.answer("Я обрабатываю только текстовые вопросы. Прикрепления зафиксированы.")

    try:
        chunks, context, status = retrieve(question)
    except Exception as e:
        # Index not built or other error
        msg = "Индекс не готов. Попросите администратора выполнить построение индекса: python -m rag.index_build"
        log_query(message.from_user, question, [], msg, "ERROR", has_attachment)
        return await message.answer(msg)

    # Попытка "инлайн-рендера" основной секции Markdown, если там есть изображения
    inline_rendered = False
    if status == "OK" and chunks:
        rel_path, section = select_primary_doc(chunks)
        if rel_path:
            try_path = (settings.data_dir / rel_path).resolve()
            if try_path.is_file():
                full_md = read_text_file(try_path)
                sec_md = extract_section_text(full_md, section or "")
                if IMG_MD_RE.search(sec_md) or IMG_PHOTO_RE.search(sec_md) or MD_LINK_RE.search(sec_md) or LINK_RE.search(sec_md):
                    inline_rendered = await render_markdown_with_inline_images(
                        message, rel_path, section or ""
                    )
                    if inline_rendered:
                        from pathlib import Path as _P
                        _src = str(_P(rel_path).with_suffix(""))
                        await message.answer(f"Источник: {_src}")

    if not inline_rendered:
        # Стандартный режим: LLM-ответ + (опционально) фото после ответа
        answer = generate_answer(question, context, chunks, status)
        # Фото ищем только в основном (primary) документе, чтобы исключить «утечку» чужих изображений
        rel_path, _ = select_primary_doc(chunks)
        primary_chunks = [c for c in chunks if c.path == rel_path][:1] if rel_path else chunks[:1]
        photos = find_photos(question, primary_chunks)
        srcs = [c.path for c in chunks][:5]
        log_query(message.from_user, question, srcs, answer, status, has_attachment)
        await message.answer(answer)
        for p in photos[:MAX_RETURN_PHOTOS]:
            try:
                await message.answer_photo(photo=FSInputFile(str(p)))
            except Exception:
                # не падаем, если фото не отправилось
                pass
    else:
        # Логируем: мы отправили поток секции вместо единого LLM-текста
        srcs = [c.path for c in chunks][:5]
        log_query(message.from_user, question, srcs, "<inline_markdown>", status, has_attachment)


async def main():
    if not settings.telegram_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN/telegram_token is not set in .env")

    bot = Bot(settings.telegram_token)
    dp = Dispatcher()

    dp.message.register(handle_start, Command("start"))
    dp.message.register(handle_policy, Command("policy"))
    dp.message.register(handle_kb, Command("kb"))
    dp.message.register(handle_feedback, Command("feedback"))
    # Catch-all message handler after commands
    dp.message.register(handle_question)

    print("Bot started. Press Ctrl+C to stop.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("Bot stopped.")
