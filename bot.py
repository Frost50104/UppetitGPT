import asyncio
import csv
import re
from urllib.parse import unquote
from datetime import datetime, timezone
from pathlib import Path
from typing import List

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

    answer = generate_answer(question, context, chunks, status)

    # поиск фото по извлечённым фрагментам и вопросу
    photos = find_photos(question, chunks)

    srcs = [c.path for c in chunks][:5]
    log_query(message.from_user, question, srcs, answer, status, has_attachment)

    # Сначала отправляем текстовый ответ, затем фото отдельными сообщениями
    await message.answer(answer)
    for p in photos[:MAX_RETURN_PHOTOS]:
        try:
            await message.answer_photo(photo=FSInputFile(str(p)))
        except Exception:
            # не падаем, если фото не отправилось
            pass


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
