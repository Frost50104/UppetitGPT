import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import faiss
from openai import OpenAI

from config import get_settings


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return path.read_text(errors="ignore")


def split_into_chunks(text: str, min_len=800, max_len=1200, overlap=150) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    chunks: List[str] = []
    i = 0
    while i < len(text):
        end = min(i + max_len, len(text))
        chunk = text[i:end]
        if len(chunk) < min_len and end < len(text):
            # extend to min_len when possible
            end = min(i + min_len, len(text))
            chunk = text[i:end]
        chunks.append(chunk)
        if end == len(text):
            break
        i = end - overlap
        if i < 0:
            i = 0
    return chunks


def guess_title_and_section(path: Path, content: str) -> Tuple[str, str]:
    # try first markdown header
    m = re.search(r"^\s*#\s+(.+)$", content, flags=re.MULTILINE)
    title = m.group(1).strip() if m else path.stem
    # naive section: second-level header if present
    s = re.search(r"^\s*##\s+(.+)$", content, flags=re.MULTILINE)
    section = s.group(1).strip() if s else ""
    return title, section


def get_department(path: Path) -> str:
    # department by top-level subfolder name
    parts = path.parts
    if "data" in parts:
        idx = parts.index("data")
        if idx + 1 < len(parts):
            dept = parts[idx + 1]
            return dept
    return "general"


def build_embeddings(client: OpenAI, texts: List[str], model: str) -> np.ndarray:
    # batch embeddings
    resp = client.embeddings.create(model=model, input=texts)
    vectors = [np.array(d.embedding, dtype="float32") for d in resp.data]
    return np.vstack(vectors)


def main():
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    files: List[Path] = []
    for root, _, filenames in os.walk(settings.data_dir):
        for fn in filenames:
            if fn.lower().endswith((".md", ".txt")):
                files.append(Path(root) / fn)

    if not files:
        # Fallback: also look in project root for .md/.txt (MVP convenience)
        root = Path('.')
        for fn in root.iterdir():
            if fn.is_file() and fn.name.lower().endswith((".md", ".txt")):
                files.append(fn)
    if not files:
        print(f"No source files found under {settings.data_dir}. Place .md or .txt files there.")
        return

    meta_path = settings.index_dir / "meta.jsonl"
    index_path = settings.index_dir / "faiss.index"

    records: List[Dict] = []
    vectors: List[np.ndarray] = []

    for file_path in files:
        content = read_text_file(file_path)
        title, section = guess_title_and_section(file_path, content)
        chunks = split_into_chunks(content)
        timestamp = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        dept = get_department(file_path)

        # create records first to embed in batches per file
        batch_texts = []
        pending_records = []
        for idx, chunk in enumerate(chunks):
            try:
                rel_path = str(file_path.relative_to(settings.data_dir))
            except Exception:
                rel_path = file_path.name
            rec = {
                "path": rel_path,
                "title": title,
                "section": section,
                "updated_at": timestamp,
                "department": dept,
                "chunk_id": idx,
                "text": chunk,
            }
            pending_records.append(rec)
            batch_texts.append(chunk)

        if batch_texts:
            embs = build_embeddings(client, batch_texts, settings.embedding_model)
            for rec, vec in zip(pending_records, embs):
                records.append(rec)
                vectors.append(vec)

    if not records:
        print("No chunks produced. Index not built.")
        return

    dim = vectors[0].shape[0]
    index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors
    # normalize
    mat = np.vstack(vectors).astype("float32")
    faiss.normalize_L2(mat)
    index.add(mat)

    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Built index with {len(records)} chunks from {len(files)} files.")
    print(f"Index path: {index_path}")
    print(f"Meta path: {meta_path}")


if __name__ == "__main__":
    main()
