from typing import List
from openai import OpenAI

from config import get_settings
from rag.retrieve import Chunk

SYSTEM_PROMPT = (
    "Ты помощник для сотрудников розничной сети. Отвечай строго на основе предоставленного контекста. "
    "Если в контексте недостаточно информации для точного ответа — честно откажись и предложи эскалацию. "
    "Формат ответа: коротко (3–8 предложений), пошагово при необходимости. Обязательно укажи секцию 'Источник: …' с путями."
)

ANSWER_TEMPLATE = (
    "Вопрос (RU):\n{question}\n\n"
    "Контекст (фрагменты из базы знаний, используй только их):\n{context}\n\n"
    "Инструкция: отвечай только фактами из контекста. Если не хватает данных — напиши отказ вида: "
    "'В базе нет точного ответа. Обратитесь: {escalation}'. В конце добавь 'Источник: …' со списком путей (2–5)."
)


def generate_answer(question: str, context: str, chunks: List[Chunk], status: str) -> str:
    settings = get_settings()
    # Pre-check: if no context/LOW conf
    if status != "OK" or not chunks or not context.strip():
        return (
            "В базе нет точного ответа. Обратитесь: ответственный чат/лицо вашей сети.\n"
            "Источник: —"
        )

    client = OpenAI(api_key=settings.openai_api_key)

    # prepare sources list (unique paths)
    unique_paths = []
    for c in chunks:
        if c.path not in unique_paths:
            unique_paths.append(c.path)
        if len(unique_paths) >= 5:
            break

    escalation = f"служба поддержки {settings.org_name} / ваш ТУ"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": ANSWER_TEMPLATE.format(
                question=question.strip(), context=context, escalation=escalation
            ),
        },
    ]

    resp = client.chat.completions.create(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        messages=messages,
    )
    text = (resp.choices[0].message.content or "").strip()

    # Post-validate: ensure Источник exists; if absent, append
    if "Источник:" not in text:
        src = "; ".join(unique_paths[: max(2, min(5, len(unique_paths)))]) or "—"
        text = text + f"\nИсточник: {src}"
    return text
