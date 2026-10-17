import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
GUARDRAILS_DIR = BASE_DIR / "config"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. "
    "Answer clearly and say so when you are unsure."
    "Dont answer when i ask for how to make a baby."
)
# Max number of recent messages sent to the model each turn
MAX_HISTORY_MESSAGES = 20
