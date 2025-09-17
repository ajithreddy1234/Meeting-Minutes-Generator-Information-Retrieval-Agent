# template.py
# Creates the full structure with EMPTY files only (no code).
# Assumes you are already inside the project root folder (minutes-agent/).

import os

FILES = [
    "README.md",
    ".env.example",
    "requirements.txt",
    "config.py",
    "app.py",
    "asr/__init__.py",
    "asr/transcribe.py",
    "summarizer/__init__.py",
    "summarizer/train_lora.py",
    "summarizer/generate_minutes.py",
    "rag/__init__.py",
    "rag/store.py",
    "rag/chat.py",
    "evaluation/__init__.py",
    "evaluation/evaluate_minutes.py",
    "logs/.gitkeep",
    "logs/training/.gitkeep",
    "logs/app/.gitkeep",
    "logs/chat/.gitkeep",
    "data/lora_minutes_train.jsonl",
    "data/lora_minutes_val.jsonl",
    "storage/index/.gitkeep",
    "storage/meetings/.gitkeep",
    "artifacts/lora_minutes/.gitkeep",
    "artifacts/lora_minutes/tokenizer/.gitkeep",
]

def touch(path):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        print(f"📂 created: {folder}")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            pass
        print(f"📝 created: {path}")

def main():
    for rel in FILES:
        touch(rel)
    print("\n✅ skeleton ready. Next: paste code into files as instructed.")

if __name__ == "__main__":
    main()
