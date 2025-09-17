import os, re, math, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ---- Paths (adjust if different) ----
BASE_MODEL = "google/flan-t5-base"
LORA_DIR   = "artifacts/lora_minutes"   # your trained adapter folder

# ---- Limits for flan-t5-base ----
CONTEXT_LEN = 512     # encoder limit
TARGET_LEN  = 256     # per-chunk summary length

# ---- Prompt that matches fine-tuning style ----
PROMPT_PREFIX = (
    "You are the official secretary producing IIT Hyderabad Gymkhana Minutes of Meeting.\n"
    "Your role is to generate structured, concise, and professional MoM from raw transcripts.\n\n"
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
    "Conversation:\n"
)

def load_model():
    tok = AutoTokenizer.from_pretrained(LORA_DIR)  # keep same tokenizer as training
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base, LORA_DIR)
    model.eval()

    # Device (MPS on Apple, else CPU; use CUDA if you have an NVIDIA GPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    return tok, model, device

# --- Clean up raw transcript to reduce copying noise ---
def clean_transcript(text: str) -> str:
    # Remove timestamps like [00:00] or [00:00:12]
    text = re.sub(r"\[\d{2}:\d{2}(?::\d{2})?\]\s*", "", text)
    # Normalize multiple spaces/newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

# --- Encode with truncation safeguard ---
def encode_clip(tok, device, text: str):
    # Concatenate prompt + text, but ensure encoder fits within CONTEXT_LEN
    # We let tokenizer do truncation.
    enc = tok(
        PROMPT_PREFIX + text,
        return_tensors="pt",
        truncation=True,
        max_length=CONTEXT_LEN,
    )
    return {k: v.to(device) for k, v in enc.items()}

# --- Decode settings to fight repetition and improve structure ---
GEN_KW = dict(
    num_beams=4,
    length_penalty=1.0,
    no_repeat_ngram_size=3,
    repetition_penalty=1.15,
    min_new_tokens=120,
    max_new_tokens=TARGET_LEN,
    early_stopping=True,
)

def generate_minutes(tok, model, device, transcript: str) -> str:
    transcript = clean_transcript(transcript)

    # Quick token count to decide if we must chunk
    # (rough: tokens ~= chars/4 for English; we still rely on tokenizer truncation)
    inputs = tok(PROMPT_PREFIX + transcript, truncation=False, return_tensors="pt")
    total_tokens = inputs["input_ids"].shape[1]

    if total_tokens <= CONTEXT_LEN:
        encoded = encode_clip(tok, device, transcript)
        with torch.no_grad():
            out = model.generate(**encoded, **GEN_KW)
        return tok.decode(out[0], skip_special_tokens=True)

    # --- Chunking path: sliding windows over the transcript ---
    # Split by paragraphs to keep semantic chunks, then group to fit ~CONTEXT_LEN
    paras = [p.strip() for p in re.split(r"\n\s*\n", transcript) if p.strip()]
    chunk_texts, current, cur_len = [], [], 0

    for p in paras:
        # conservative token estimate by char length; adjust grouping
        p_est = max(1, len(p) // 4)
        if cur_len + p_est > (CONTEXT_LEN - 64):  # reserve headroom for prompt
            chunk_texts.append("\n\n".join(current))
            current, cur_len = [p], p_est
        else:
            current.append(p)
            cur_len += p_est
    if current:
        chunk_texts.append("\n\n".join(current))

    partial_minutes = []
    for i, chunk in enumerate(chunk_texts, 1):
        encoded = encode_clip(tok, device, chunk)
        with torch.no_grad():
            out = model.generate(**encoded, **GEN_KW)
        piece = tok.decode(out[0], skip_special_tokens=True)
        partial_minutes.append(piece)

    # Meta-summarize the partial minutes to final minutes
    meta_input = "\n\n---\n\n".join(partial_minutes)
    meta_prompt = (
        "You are compiling the final official Minutes from several partial minutes.\n"
        "Merge them into a single clean MoM with no duplication, keep the headings, and ensure consistent tone.\n\n"
        "PARTIAL MINUTES:\n" + meta_input
    )
    enc = tok(meta_prompt, return_tensors="pt", truncation=True, max_length=CONTEXT_LEN)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        final_out = model.generate(**enc, **GEN_KW)
    return tok.decode(final_out[0], skip_special_tokens=True)

if __name__ == "__main__":
    # EXAMPLE input; replace with your real transcript
    raw_transcript = """
    [00:00] Speaker A: Mess quality concerns; request healthier breakfast without cost increase.
    [00:03] Speaker B: Caterer instructed; per-head bills expected to drop next month.
    [00:05] Speaker C: Vending machines to Library, LHC, SNCC; vendor pays electricity.
    [00:08] Speaker D: Drainage fix by 8 Nov; water tests monthly in first week.
    """
    tok, model, device = load_model()
    minutes = generate_minutes(tok, model, device, raw_transcript)
    print("=== MINUTES ===\n", minutes)
