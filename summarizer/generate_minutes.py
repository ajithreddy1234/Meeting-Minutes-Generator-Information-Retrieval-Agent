# summarizer/generate_minutes.py
import os
import re
import sys
import json
import time
import math
import argparse
import logging
from typing import List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ----------------------------
# Config (aligned with your train_lora.py)
# ----------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "google/flan-t5-base")
LORA_DIR   = os.getenv("LORA_DIR", "artifacts/flan_t5_minutes_lora")

LOG_DIR_APP = "logs/app"
DEFAULT_CONTEXT_LEN = 512          # encoder max window (base model)
ENC_MAX_TOK = 480                  # a bit under 512 to leave room for special tokens
ENC_OVERLAP = 120                  # token overlap between windows
CHUNK_SUM_MAX_NEW = 220            # generation cap for each chunk summary
FINAL_SUM_MAX_NEW = 520            # generation cap for final minutes

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# ----------------------------
# Prompts
# ----------------------------
# Same scaffold you trained on for final synthesis:
FINAL_TEMPLATE_PREFIX = (
    "You are the official secretary producing IIT Hyderabad Gymkhana Minutes of Meeting.\n"
    "Your role is to generate structured, concise, and professional MoM from raw transcripts or section notes.\n"
    "IMPORTANT: Do NOT include any timestamps (e.g., 00:00, [00:00]) or 'Chunk' markers in the minutes.\n\n"
    "Follow this EXACT format:\n\n"
    "MINUTES OF MEETING\n\n"
    "Date: <dd/mm/yyyy>\n"
    "Agenda and Outcomes:\n"
    "1. <Topic>\n"
    "   • Issue: <short description>\n"
    "   • Discussion: <summary of points>\n"
    "   • Resolution/Decision: <final outcome>\n"
    "   • Action Item (if any): <task + responsible person>\n\n"
    "2. <Next Topic>\n"
    "   • Issue: <...>\n"
    "   • Discussion: <...>\n"
    "   • Resolution/Decision: <...>\n"
    "   • Action Item: <...>\n\n"
    "---\n\n"
)

# For per-chunk map step (short bullet summary with decisions/actions)
CHUNK_MAP_PROMPT = (
    "You are assisting the secretary. Summarize the following meeting excerpt as tight bullet points.\n"
    "Prioritize: decisions, approvals, dates, deadlines, action items (with owners if mentioned), and key issues.\n"
    "Do NOT invent details. Keep it concise and factual.\n\n"
    "Excerpt:\n"
)

# ----------------------------
# Logging
# ----------------------------
def ensure_dirs():
    os.makedirs(os.path.dirname(LORA_DIR) or ".", exist_ok=True)
    os.makedirs("storage", exist_ok=True)
    os.makedirs(LOG_DIR_APP, exist_ok=True)

def setup_logging():
    ensure_dirs()
    log_path = os.path.join(LOG_DIR_APP, "generate_minutes.log")
    # Reset handlers to avoid duplication
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    ch = logging.StreamHandler(sys.stdout)
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    fh.setLevel(logging.INFO); ch.setLevel(logging.INFO)
    root.setLevel(logging.INFO)
    root.addHandler(fh); root.addHandler(ch)
    logging.info("Logging to %s", log_path)
    return log_path

# ----------------------------
# ASR (optional)
# ----------------------------
def transcribe_audio(audio_path: str) -> str:
    """
    Use faster-whisper if installed; else raise a friendly error.
    Returns a single string transcript (keep timestamps if model outputs them).
    """
    try:
        from faster_whisper import WhisperModel
    except Exception:
        raise RuntimeError(
            "Audio transcription requires 'faster-whisper'. Install via:\n"
            "  pip install faster-whisper\n"
            "Or provide --transcript instead of --audio."
        )

    model = WhisperModel("small", device=DEVICE if DEVICE != "mps" else "cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, vad_filter=True)
    lines = []
    for seg in segments:
        # Format like: [HH:MM:SS] text
        start = time.strftime("%H:%M:%S", time.gmtime(seg.start))
        lines.append(f"[{start}] {seg.text.strip()}")
    return "\n".join(lines)

# ----------------------------
# Transcript helpers
# ----------------------------
TIMESTAMP_RE = re.compile(r"\[?\d{1,2}:\d{2}(?::\d{2})?\]?")

def read_transcript(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def clean_transcript(text: str) -> str:
    """
    Light cleanup: normalize whitespace, collapse repeated lines, remove extraneous 'Minutes:' echoes.
    Keep speaker labels and timestamps if present; we'll instruct model not to include them in final minutes.
    """
    # Remove repeated "Minutes:" spam if present
    text = re.sub(r"(?:\bMinutes:\s*){2,}", "Minutes: ", text, flags=re.IGNORECASE)
    # Normalize CRLF, trim trailing spaces
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    # De-dup consecutive identical lines
    lines = []
    prev = None
    for line in text.splitlines():
        if line != prev:
            lines.append(line)
        prev = line
    return "\n".join(lines).strip()

# ----------------------------
# Token chunking
# ----------------------------
def tokenize(tokenizer: AutoTokenizer, text: str) -> List[int]:
    enc = tokenizer(text, truncation=False, add_special_tokens=True)
    return enc["input_ids"]

def detokenize(tokenizer: AutoTokenizer, ids: List[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)

def make_windows(ids: List[int], window: int, overlap: int) -> List[Tuple[int, int]]:
    step = max(1, window - overlap)
    spans = []
    for start in range(0, len(ids), step):
        end = min(start + window, len(ids))
        spans.append((start, end))
        if end == len(ids):
            break
    return spans

# ----------------------------
# Model loader (base + LoRA)
# ----------------------------
def load_model_and_tokenizer() -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    # Prefer tokenizer from LORA_DIR (you saved it there), fallback to base
    try:
        tok = AutoTokenizer.from_pretrained(LORA_DIR, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(base, LORA_DIR)
    model.to(DEVICE)
    model.eval()
    logging.info("Model loaded | base=%s | adapter=%s | device=%s", MODEL_NAME, LORA_DIR, DEVICE)
    return model, tok

# ----------------------------
# Generation helpers
# ----------------------------
@torch.inference_mode()
def generate_text(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_beams: int = 4,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=ENC_MAX_TOK)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
        do_sample=False if num_beams > 1 else True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()

def summarize_chunks(model, tokenizer, transcript_text: str) -> List[str]:
    ids = tokenize(tokenizer, transcript_text)
    spans = make_windows(ids, window=ENC_MAX_TOK, overlap=ENC_OVERLAP)
    logging.info("Chunking: tokens=%d | windows=%d (win=%d, overlap=%d)", len(ids), len(spans), ENC_MAX_TOK, ENC_OVERLAP)

    summaries = []
    for i, (s, e) in enumerate(spans, 1):
        chunk_txt = detokenize(tokenizer, ids[s:e])
        prompt = CHUNK_MAP_PROMPT + chunk_txt
        logging.info("Chunk %d/%d | gen chunk-summary ...", i, len(spans))
        summary = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=CHUNK_SUM_MAX_NEW,
            temperature=0.3, top_p=0.9, num_beams=4
        )
        summaries.append(f"- Chunk {i} summary:\n{summary}")
    return summaries

def synthesize_minutes(model, tokenizer, section_summaries: List[str], meeting_date: Optional[str]) -> str:
    notes = "\n\n".join(section_summaries)
    date_hint = meeting_date if meeting_date else "<dd/mm/yyyy>"
    prompt = (
        FINAL_TEMPLATE_PREFIX +
        f"Use the following section notes to produce the final Minutes.\n"
        f"Only include facts present in the notes. No hallucinations.\n\n"
        f"Assume: Date: {date_hint}\n\n"
        f"Section Notes:\n{notes}\n\n"
        f"Now produce the final 'MINUTES OF MEETING' in the exact format."
    )
    logging.info("Synthesis | generating final minutes ...")
    minutes = generate_text(
        model, tokenizer, prompt,
        max_new_tokens=FINAL_SUM_MAX_NEW,
        temperature=0.3, top_p=0.9, num_beams=4
    )
    return minutes

# ----------------------------
# Date helper (optional)
# ----------------------------
DATE_RE_1 = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b")
def guess_date_from_text(text: str) -> Optional[str]:
    """
    Very light attempt: look for dd/mm/yyyy or dd-mm-yyyy patterns in transcript.
    Returns normalized dd/mm/yyyy if found.
    """
    m = DATE_RE_1.search(text)
    if not m:
        return None
    d, mth, y = m.groups()
    if len(y) == 2:
        y = "20" + y
    d = d.zfill(2); mth = mth.zfill(2)
    return f"{d}/{mth}/{y}"

# ----------------------------
# Main
# ----------------------------
def main():
    _log_file = setup_logging()

    parser = argparse.ArgumentParser(description="Generate Minutes of Meeting from audio or transcript using LoRA adapter.")
    parser.add_argument("--audio", type=str, default="", help="Path to audio file (optional). Requires faster-whisper.")
    parser.add_argument("--transcript", type=str, default="", help="Path to transcript text/markdown file.")
    parser.add_argument("--out", type=str, required=True, help="Output path for final minutes markdown.")
    parser.add_argument("--date", type=str, default="", help="Meeting date in dd/mm/yyyy (optional).")
    parser.add_argument("--scope", type=str, default="this", help="Reserved flag for app integration (this/all).")
    args = parser.parse_args()

    t0 = time.time()
    logging.info("=== generate_minutes start ===")
    logging.info("Args: %s", vars(args))

    if not args.audio and not args.transcript:
        raise SystemExit("Provide either --audio or --transcript.")

    # 1) Obtain transcript text
    if args.transcript:
        raw_text = read_transcript(args.transcript)
        logging.info("Loaded transcript: %s (%d chars)", args.transcript, len(raw_text))
    else:
        raw_text = transcribe_audio(args.audio)
        logging.info("Transcribed audio: %s (%d chars)", args.audio, len(raw_text))

    cleaned = clean_transcript(raw_text)
    # Guess date if not provided
    meeting_date = args.date.strip() or guess_date_from_text(cleaned)

    # 2) Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer()
    # Let tokenizer allow long texts for pre-chunking
    tokenizer.model_max_length = 10**9

    # 3) Map (chunk summaries)
    section_summaries = summarize_chunks(model, tokenizer, cleaned)

    # 4) Reduce (final synthesis)
    minutes = synthesize_minutes(model, tokenizer, section_summaries, meeting_date)

    # 5) Save output
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(minutes.strip() + "\n")
    logging.info("Saved minutes to: %s", out_path)

    # 6) Persist run metadata
    meta = {
        "base_model": MODEL_NAME,
        "adapter_dir": LORA_DIR,
        "device": DEVICE,
        "input_audio": args.audio,
        "input_transcript": args.transcript,
        "out_path": out_path,
        "meeting_date": meeting_date or "<unknown>",
        "tokens_window": ENC_MAX_TOK,
        "tokens_overlap": ENC_OVERLAP,
        "num_sections": len(section_summaries),
        "elapsed_sec": round(time.time() - t0, 2),
        "log_file": os.path.join(LOG_DIR_APP, "generate_minutes.log")
    }
    meta_path = re.sub(r"\.md$", ".run.json", out_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logging.info("Saved run meta: %s", meta_path)
    logging.info("=== generate_minutes done in %.2fs ===", meta["elapsed_sec"])


if __name__ == "__main__":
    main()
