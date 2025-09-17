import os, re, json, argparse, uuid
from pathlib import Path

from config import MEETINGS_DIR
# scripts/ingest_bulk_minutes.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))  # now `import rag` works

from rag.store import upsert_meeting  # <- keep your imports after this

BLOCK_RE = re.compile(r"^### MEETING\s*$", re.M)

def parse_blocks(text: str):
    parts = BLOCK_RE.split(text)
    # parts[0] may be preamble; skip if empty
    for raw in parts:
        raw = raw.strip()
        if not raw:
            continue
        yield raw

def parse_meeting(block: str):
    def get_field(name):
        m = re.search(rf"^{name}:\s*(.*)$", block, re.M)
        return m.group(1).strip() if m else ""

    title = get_field("Title") or "Untitled"
    date  = get_field("Date") or ""
    mid   = get_field("MeetingID") or ""

    # Split Transcript / Minutes
    if "---MINUTES---" in block:
        head, minutes = block.split("---MINUTES---", 1)
    else:
        head, minutes = block, ""

    # Grab transcript text after "Transcript:" line
    transcript = ""
    m = re.search(r"^Transcript:\s*(.*)$", head, re.M | re.S)
    if m:
        transcript = m.group(1).strip()

    return {
        "title": title, "date": date, "meeting_id": mid,
        "transcript": transcript, "minutes": minutes.strip()
    }

def materialize(meeting):
    meeting_id = meeting["meeting_id"] or f"{meeting['date'].replace('/','')}_{uuid.uuid4().hex[:6]}"
    out_dir = os.path.join(MEETINGS_DIR, meeting_id)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # files
    with open(os.path.join(out_dir, "transcript.txt"), "w", encoding="utf-8") as f:
        f.write(meeting["transcript"])
    with open(os.path.join(out_dir, "minutes.md"), "w", encoding="utf-8") as f:
        f.write(meeting["minutes"])
    meta = {"title": meeting["title"], "date": meeting["date"], "meeting_id": meeting_id}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # index
    upsert_meeting(meeting["transcript"], meeting["minutes"], meta)
    return out_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to data/bulk_minutes.txt")
    args = ap.parse_args()

    text = open(args.input, "r", encoding="utf-8").read()
    n=0
    for block in parse_blocks(text):
        mt = parse_meeting(block)
        if not mt["minutes"]:
            print("Skipping a block without minutes.")
            continue
        out = materialize(mt)
        print("Ingested:", out)
        n+=1
    print(f"✅ Done. Ingested {n} meetings.")

if __name__ == "__main__":
    main()
