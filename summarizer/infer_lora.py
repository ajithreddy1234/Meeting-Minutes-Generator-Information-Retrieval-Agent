import os
import argparse
import logging
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# Defaults match your training
CONTEXT_LEN = 512
CHUNK_OVERLAP = 128
TARGET_LEN = 256

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

NOTES_PREFIX = (
    "You are assisting a secretary. Read the conversation and write ultra-concise bullet NOTES "
    "(issues, decisions, actions, deadlines, amounts). No formatting beyond simple bullets.\n\nConversation:\n"
)

def setup_logging(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(path, mode="a", encoding="utf-8"), logging.StreamHandler()],
    )
    logging.info("inference logging initialized -> %s", path)

def chunk_token_ids(input_ids: List[int], max_len: int, overlap: int) -> List[List[int]]:
    if len(input_ids) <= max_len:
        return [input_ids]
    step = max(1, max_len - overlap)
    return [input_ids[i : i + max_len] for i in range(0, len(input_ids), step)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default=os.getenv("MODEL_NAME", "google/flan-t5-base"))
    ap.add_argument("--lora_dir", default=os.getenv("LORA_DIR", "artifacts/flan_t5_minutes_lora"))
    ap.add_argument("--dialogue_file", required=True, help="Path to a .txt transcript (one big string)")
    ap.add_argument("--date", default="dd/mm/yyyy")
    ap.add_argument("--out", default="artifacts/sample_minutes.txt")
    ap.add_argument("--max_new_tokens", type=int, default=TARGET_LEN)
    ap.add_argument("--two_pass", action="store_true", help="use chunk->notes->final summarization")
    ap.add_argument("--log", default="logs/training/infer.txt")
    args = ap.parse_args()

    setup_logging(args.log)
    device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    logging.info("device=%s | base=%s | lora=%s", device, args.base_model, args.lora_dir)

    # Load tokenizer from adapter dir if available, else base
    tok_src = args.lora_dir if os.path.exists(os.path.join(args.lora_dir, "tokenizer_config.json")) else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    tokenizer.model_max_length = 10**9  # we'll chunk manually

    base = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(base, args.lora_dir)
    model.eval()
    model.to(device)

    # Read transcript
    with open(args.dialogue_file, "r", encoding="utf-8") as f:
        conversation = f.read().strip()

    def generate(text, max_new_tokens=TARGET_LEN):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=CONTEXT_LEN)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                length_penalty=0.8,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        return tokenizer.decode(out_ids[0], skip_special_tokens=True)

    if not args.two_pass:
        # Single pass (will truncate to first 512 tokens if the transcript is huge)
        prompt = PROMPT_PREFIX.replace("<dd/mm/yyyy>", args.date) + conversation
        minutes = generate(prompt, args.max_new_tokens)
    else:
        # Two-pass hierarchical: chunk -> notes per chunk -> final minutes
        logging.info("two-pass mode enabled")
        # 1) chunk the raw conversation tokens
        full_ids = tokenizer(PROMPT_PREFIX + conversation, add_special_tokens=True, truncation=False)["input_ids"]
        chunks = chunk_token_ids(full_ids, CONTEXT_LEN, CHUNK_OVERLAP)
        logging.info("chunks=%d", len(chunks))

        notes_list = []
        for i, ids in enumerate(chunks, 1):
            chunk_text = tokenizer.decode(ids, skip_special_tokens=True)
            notes = generate(NOTES_PREFIX + chunk_text, max_new_tokens=min(192, args.max_new_tokens))
            notes_list.append(f"- Chunk {i} notes:\n{notes}\n")
            logging.info("chunk %d notes len=%d", i, len(notes))

        # 2) final pass using combined notes
        combined = "\n".join(notes_list)
        final_prompt = PROMPT_PREFIX.replace("<dd/mm/yyyy>", args.date) + combined
        minutes = generate(final_prompt, args.max_new_tokens)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(minutes.strip() + "\n")
    logging.info("minutes written -> %s", args.out)
    print("\n" + minutes + "\n")

if __name__ == "__main__":
    main()
