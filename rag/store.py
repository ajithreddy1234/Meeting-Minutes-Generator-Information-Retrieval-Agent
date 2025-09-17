# rag/store.py
import os, json
from typing import List, Dict, Tuple
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from config import EMBED_MODEL, INDEX_DIR, MEETINGS_DIR

# from langchain_community.embeddings import HuggingFaceEmbeddings   # old (deprecated)
from langchain_huggingface import HuggingFaceEmbeddings               # new

def _emb():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"batch_size": 64, "show_progress_bar": True, "normalize_embeddings": True}
    )

def _index_path():
    return os.path.join(INDEX_DIR, "faiss_idx")

def _chunk(text: str, size: int = 800, overlap: int = 80) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0: start = 0
    return chunks

def build_docs(transcript: str, minutes_md: str, meta: Dict) -> List[Document]:
    docs: List[Document] = []
    for ch in _chunk(transcript):
        docs.append(Document(page_content=ch, metadata={**meta, "source":"transcript"}))
    for ch in _chunk(minutes_md):
        docs.append(Document(page_content=ch, metadata={**meta, "source":"minutes"}))
    return docs

def upsert_meeting(transcript: str, minutes_md: str, meta: Dict):
    emb = _emb()
    path = _index_path()
    docs = build_docs(transcript, minutes_md, meta)

    if os.path.exists(path):
        vs = FAISS.load_local(path, emb, allow_dangerous_deserialization=True)
        vs.add_documents(docs)
        vs.save_local(path)
    else:
        vs = FAISS.from_documents(docs, emb)
        os.makedirs(INDEX_DIR, exist_ok=True)
        vs.save_local(path)

def upsert_meeting_dir(meeting_dir: str):
    """meeting_dir must contain transcript.txt, minutes.md, meta.json"""
    with open(os.path.join(meeting_dir, "transcript.txt"), "r", encoding="utf-8") as f:
        transcript = f.read()
    with open(os.path.join(meeting_dir, "minutes.md"), "r", encoding="utf-8") as f:
        minutes_md = f.read()
    with open(os.path.join(meeting_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    upsert_meeting(transcript, minutes_md, meta)

def reindex_all():
    """Rebuild FAISS from MEETINGS_DIR"""
    emb = _emb()
    path = _index_path()
    if os.path.exists(path):
        # remove old index to ensure clean rebuild
        for f in os.listdir(INDEX_DIR):
            os.remove(os.path.join(INDEX_DIR, f))

    all_docs: List[Document] = []
    for mid in sorted(os.listdir(MEETINGS_DIR)):
        mdir = os.path.join(MEETINGS_DIR, mid)
        if not os.path.isdir(mdir): continue
        try:
            with open(os.path.join(mdir, "transcript.txt"), "r", encoding="utf-8") as f:
                transcript = f.read()
            with open(os.path.join(mdir, "minutes.md"), "r", encoding="utf-8") as f:
                minutes_md = f.read()
            with open(os.path.join(mdir, "meta.json"), "r", encoding="utf-8") as f:
                meta = json.load(f)
            all_docs += build_docs(transcript, minutes_md, meta)
        except FileNotFoundError:
            continue

    if not all_docs:
        raise RuntimeError("No meetings found in storage/meetings to index.")

    vs = FAISS.from_documents(all_docs, emb)
    os.makedirs(INDEX_DIR, exist_ok=True)
    vs.save_local(path)

def search(query: str, k: int = 8, scope: str = "all", meeting_id: str = "") -> List[Tuple[Document, float]]:
    emb = _emb()
    vs = FAISS.load_local(_index_path(), emb, allow_dangerous_deserialization=True)
    results = vs.similarity_search_with_score(query, k=k)
    if scope == "this" and meeting_id:
        results = [(d, s) for (d, s) in results if d.metadata.get("meeting_id") == meeting_id]
    return results

def keyword_rerank(query: str, hits: List[Tuple[Document, float]], topn: int = 4):
    q = [w.lower() for w in query.split() if len(w) > 3]
    scored = []
    for d, s in hits:
        t = d.page_content.lower()
        kw = sum(t.count(w) for w in q)
        scored.append((d, s, kw))
    scored.sort(key=lambda x: (-x[2], x[1]))  # more keyword hits, then lower distance
    return [(d, s) for d, s, _ in scored[:topn]]

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--upsert_dir", help="Path to storage/meetings/<meeting_id>")
    p.add_argument("--reindex_all", action="store_true")
    args = p.parse_args()
    if args.reindex_all:
        reindex_all()
        print("Rebuilt FAISS index.")
    elif args.upsert_dir:
        upsert_meeting_dir(args.upsert_dir)
        print(f"Upserted {args.upsert_dir}")
    else:
        print("Use --reindex_all or --upsert_dir <path>")
