import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import faiss
from openai import OpenAI

from config import get_settings


@dataclass
class Chunk:
    text: str
    path: str
    title: str
    section: str
    updated_at: str
    department: str
    score: float


KEYWORD_BOOSTS = {
    # keyword -> bonus
    "касса": 0.05,
    "x-отчёт": 0.05,
    "интернет": 0.05,
    "звук бизнес": 0.05,
    "кондицион": 0.05,
    "дезинсек": 0.05,
}


def normalize(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"[^а-яa-z0-9ё\-\s/#]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_index():
    settings = get_settings()
    index = faiss.read_index(str(settings.index_dir / "faiss.index"))
    meta_path = settings.index_dir / "meta.jsonl"
    meta: List[Dict] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return index, meta


def embed_query(client: OpenAI, text: str, model: str) -> np.ndarray:
    resp = client.embeddings.create(model=model, input=[text])
    vec = np.array(resp.data[0].embedding, dtype="float32")
    faiss.normalize_L2(vec.reshape(1, -1))
    return vec


def apply_bonuses(query_norm: str, meta_rec: Dict, base_score: float) -> float:
    score = float(base_score)
    title = (meta_rec.get("title") or "").lower()
    section = (meta_rec.get("section") or "").lower()
    path = (meta_rec.get("path") or "").lower()

    # exact term presence bonus in title/section/path (усилено)
    for token in set(query_norm.split()):
        if not token or len(token) < 3:
            continue
        if token in title or token in section or token in path:
            score += 0.05

    # keyword boosts (оставляем как было, с проверкой пути/заголовка/секции)
    for kw, bonus in KEYWORD_BOOSTS.items():
        if kw in query_norm:
            if kw.split()[0] in path or kw in title or kw in section:
                score += bonus
    return score


def retrieve(query: str) -> Tuple[List[Chunk], str, str]:
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    index, meta = load_index()

    q_norm = normalize(query)
    q_vec = embed_query(client, q_norm, settings.embedding_model)

    D, I = index.search(q_vec.reshape(1, -1), k=min(settings.top_k * 2, len(meta)))
    indices = I[0]
    sims = D[0]

    # build candidate chunks with bonuses
    candidates: List[Tuple[float, Dict]] = []
    for idx, base in zip(indices, sims):
        if idx < 0 or idx >= len(meta):
            continue
        boosted = apply_bonuses(q_norm, meta[idx], base)
        candidates.append((boosted, meta[idx]))

    # sort and take top_k
    candidates.sort(key=lambda x: x[0], reverse=True)
    top = candidates[: settings.top_k]

    chunks: List[Chunk] = []
    for score, rec in top:
        chunks.append(
            Chunk(
                text=rec["text"],
                path=rec["path"],
                title=rec.get("title", ""),
                section=rec.get("section", ""),
                updated_at=rec.get("updated_at", ""),
                department=rec.get("department", ""),
                score=float(score),
            )
        )

    # relevance gate
    if chunks:
        avg_score = float(np.mean([c.score for c in chunks]))
    else:
        avg_score = 0.0

    status = "OK" if avg_score >= settings.relevance_threshold else "NO_CONTEXT"

    # build context limited by chars
    ctx_parts: List[str] = []
    used = 0
    for c in chunks:
        if used >= settings.max_ctx_chars:
            break
        from pathlib import Path as _P
        _src = str(_P(c.path).with_suffix(""))
        piece = f"[Источник: {_src}]\n{c.text}\n"
        if used + len(piece) > settings.max_ctx_chars:
            piece = piece[: settings.max_ctx_chars - used]
        ctx_parts.append(piece)
        used += len(piece)

    context = "\n\n".join(ctx_parts)

    return chunks, context, status
