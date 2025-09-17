# scripts/ingest_bulk_minutes.py
import os
import re
import json
import uuid
import argparse
import logging
from pathlib import Path

# Project root for local imports
ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(ROOT))

from config import MEETINGS_DIR
from rag.store import upsert_meeting  # assumes (transcript, minutes, meta) signature


# ---------- Logging ----------
def setup_logging():
    logdir = ROOT / "logs" / "ingest"
    logdir.mkdir(parents=True, exist_ok=True)
    logfile = logdir / "bulk_ingest.txt"

    for h in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(h)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(logfile, encoding="utf-8"),
            logging.StreamHandler()
        ],
    )
    logging.info("Logging to %s", logfile)
    return logfile


# ---------- Parsing helpers ----------

# Split blocks on a clear meeting separator line
BLOCK_RE = re.compile(r"^[ \t]*###[ \t]*MEETING[ \t]*\r?$", re.M | re.I)

# Accept multiple minutes delimiters
MINUTES_SPLIT_RE = re.compile(
    r"(?im)^[ \t]*(?:---\s*MINUTES\s*---|###\s*MINUTES)\s*\r?$"
)

FIELD_LINE_RE = re.compile(r"^(?P<key>Title|Date|MeetingID):\s*(?P<val>.*)$", re.M)

def parse_blocks(text: str):
    """
    Yields raw meeting blocks. Anything before the first ### MEETING is ignored.
    """
    parts = BLOCK_RE.split(text)
    for raw in parts:
        raw = raw.strip()
        if not raw:
            continue
        yield raw

def extract_fields(head: str) -> dict:
    """Extract Title/Date/MeetingID from the header text (above minutes)."""
    fields = {"Title": "", "Date": "", "MeetingID": ""}
    for m in FIELD_LINE_RE.finditer(head):
        fields[m.group("key")] = m.group("val").strip()
    return fields

def extract_transcript(head: str) -> str:
    """
    Grab everything after a line starting with 'Transcript:' until the end of head.
    This is robust for multi-line transcripts.
    """
    m = re.search(r"(?im)^Transcript:\s*(?:\r?\n)?", head)
    if not m:
        return ""
    start = m.end()
    return head[start:].strip()

def slugify(text: str, max_len: int = 50) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return s[:max_len] or "untitled"

def ensure_unique_path(base_dir: Path, name: str) -> Path:
    """
    Ensure we don't clobber an existing meeting folder.
    """
    p = base_dir / name
    if not p.exists():
        return p
    i = 2
    while True:
        cand = base_dir / f"{name}-{i}"
        if not cand.exists():
            return cand
        i += 1

def parse_meeting(block: str) -> dict:
    """
    Parse a single meeting block into {title, date, meeting_id, transcript, minutes}.
    """
    # Normalize newlines
    block = block.replace("\r\n", "\n").replace("\r", "\n")

    # Split header vs minutes
    parts = MINUTES_SPLIT_RE.split(block, maxsplit=1)
    head = parts[0].strip()
    minutes = parts[1].strip() if len(parts) > 1 else ""

    fields = extract_fields(head)
    transcript = extract_transcript(head)

    title = fields["Title"] or "Untitled"
    date = fields["Date"] or ""
    meeting_id = fields["MeetingID"].strip()

    # Fallback: build a stable, human-readable ID
    if not meeting_id:
        date_slug = date.replace("/", "-").replace(" ", "_") or "nodate"
        title_slug = slugify(title, max_len=40)
        meeting_id = f"{date_slug}_{title_slug}_{uuid.uuid4().hex[:6]}"

    return {
        "title": title,
        "date": date,
        "meeting_id": meeting_id,
        "transcript": transcript,
        "minutes": minutes,
    }

def materialize(meeting: dict) -> Path:
    """
    Write files and upsert into the index.
    """
    out_dir = ensure_unique_path(Path(MEETINGS_DIR), meeting["meeting_id"])
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "transcript.txt").write_text(meeting["transcript"], encoding="utf-8")
    (out_dir / "minutes.md").write_text(meeting["minutes"], encoding="utf-8")

    meta = {
        "title": meeting["title"],
        "date": meeting["date"],
        "meeting_id": out_dir.name,
        "path": str(out_dir),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Indexing: don't crash the whole ingest if one fails
    try:
        upsert_meeting(meeting["transcript"], meeting["minutes"], meta)
    except Exception as e:
        logging.exception("upsert_meeting failed for %s: %s", out_dir.name, e)

    return out_dir


# ---------- CLI ----------
def main():
    logfile = setup_logging()

    ap = argparse.ArgumentParser(description="Bulk-ingest minutes blocks into /storage/meetings and index them.")
    ap.add_argument("--input", required=True, help="Path to data/bulk_minutes.txt")
    ap.add_argument("--dry-run", action="store_true", help="Parse only; do not write or index")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"Input not found: {src}")

    raw = src.read_text(encoding="utf-8", errors="replace")

    count = 0
    kept = 0
    for block in parse_blocks(raw):
        count += 1
        mt = parse_meeting(block)

        if not mt["minutes"]:
            logging.warning("Skipping block %d: no minutes section found.", count)
            continue

        if args.dry_run:
            logging.info("[DRY] Parsed meeting: id=%s | title=%s | date=%s | minutes=%d chars",
                         mt["meeting_id"], mt["title"], mt["date"], len(mt["minutes"]))
            kept += 1
            continue

        out = materialize(mt)
        kept += 1
        logging.info("Ingested: %s", out)

    logging.info("Done. Parsed=%d | Ingested=%d | Log=%s", count, kept, logfile)
    print(f"✅ Done. Parsed={count} | Ingested={kept} | Log={logfile}")

if __name__ == "__main__":
    main()
