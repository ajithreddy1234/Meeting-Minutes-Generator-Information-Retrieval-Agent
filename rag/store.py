# rag/store.py
import os, time, logging
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np
# import your FAISS/Qdrant client here...

LOG = logging.getLogger("rag.store")
DEVICE = os.getenv("RAG_DEVICE", "mps")  # mps|cpu|cuda
MODEL_NAME = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

_model = None
def get_model():
    global _model
    if _model is None:
        t0 = time.time()
        LOG.info("embed: loading model=%s device=%s", MODEL_NAME, DEVICE)
        _model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        LOG.info("embed: model ready in %.2fs", time.time() - t0)
    return _model

def _chunk_text(s: str, max_chars: int = 1000, overlap: int = 100) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    chunks = []
    i = 0
    while i < len(s):
        end = min(len(s), i + max_chars)
        chunks.append(s[i:end])
        i = end - overlap
        if i < 0:
            i = 0
        if end == len(s):
            break
    return chunks

def _embed_texts(texts: List[str], batch_size: int = 16) -> np.ndarray:
    m = get_model()
    LOG.info("embed: encoding %d chunks (batch_size=%d)", len(texts), batch_size)
    t0 = time.time()
    embs = m.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    LOG.info("embed: done in %.2fs", time.time() - t0)
    return embs

def upsert_meeting(transcript: str, minutes: str, meta: Dict):
    t0 = time.time()
    LOG.info("upsert: start | title=%s date=%s", meta.get("title"), meta.get("date"))

    # 1) chunk
    tr_chunks = _chunk_text(transcript, max_chars=1200, overlap=150)
    mm_chunks = _chunk_text(minutes,    max_chars=1200, overlap=150)
    LOG.info("upsert: chunks | transcript=%d minutes=%d", len(tr_chunks), len(mm_chunks))

    # 2) embed
    texts = [f"[TRANSCRIPT] {c}" for c in tr_chunks] + [f"[MINUTES] {c}" for c in mm_chunks]
    if not texts:
        LOG.warning("upsert: no text to index; skipping.")
        return

    vecs = _embed_texts(texts, batch_size=8)  # small batches are safer on MPS/CPU

    # 3) write to vector store (replace with your FAISS/Qdrant upsert)
    # index.upsert(vectors=vecs, payloads=..., ids=...)
    LOG.info("upsert: vector_store write start (n=%d)", len(vecs))
    # ... your write code ...
    LOG.info("upsert: vector_store write done")

    LOG.info("upsert: end in %.2fs", time.time() - t0)
