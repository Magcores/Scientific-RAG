"""Telegram bot interface for the RAG pipeline.

This is an optional interface — the RAG logic lives entirely in rag.py.
This file just adds a Telegram layer on top so you can ask questions
directly from the Telegram app on your phone or desktop.

Setup:
    1. Create a bot via Telegram's @BotFather and get a token.
    2. Add the token to your .env file:
           TELEGRAM_BOT_TOKEN=your_token_here
    3. pip install python-telegram-bot

Run:
    python src\bot.py

Then open Telegram, find your bot, and start asking questions.
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv

try:
    from telegram import Update
    from telegram.ext import Application, ContextTypes, MessageHandler, filters
except ImportError as exc:
    raise ImportError(
        "python-telegram-bot is required. "
        "Install it with: pip install python-telegram-bot"
    ) from exc

from rag import MAX_QUERIES_PER_MINUTE, MAX_QUERY_LENGTH, RAGPipeline

load_dotenv()

pipeline = RAGPipeline()

# Per-user rate limiting
_user_timestamps: dict[int, deque[float]] = defaultdict(deque)


def check_rate_limit(user_id: int) -> bool:
    now = time.monotonic()
    timestamps = _user_timestamps[user_id]
    while timestamps and now - timestamps[0] > 60:
        timestamps.popleft()
    if len(timestamps) >= MAX_QUERIES_PER_MINUTE:
        return False
    timestamps.append(now)
    return True


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    user_id = update.message.from_user.id
    query = update.message.text.strip()

    if not check_rate_limit(user_id):
        await update.message.reply_text(
            f"You are sending questions too fast. Please wait a moment before asking again."
        )
        return

    if len(query) > MAX_QUERY_LENGTH:
        await update.message.reply_text(
            f"Your question is too long ({len(query)} characters). "
            f"Please keep it under {MAX_QUERY_LENGTH} characters."
        )
        return

    await update.message.reply_text("Searching...")
    answer = pipeline.answer(query)
    await update.message.reply_text(answer)


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hello! Ask me anything about the indexed documents and I will find the answer. "
        f"Keep your questions under {MAX_QUERY_LENGTH} characters."
    )


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is missing. "
            "Create a bot via @BotFather on Telegram and add the token to your .env file."
        )

    app = Application.builder().token(token).build()
    app.add_handler(MessageHandler(filters.TEXT & filters.Regex(r"^/start$"), handle_start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running. Open Telegram and start asking questions.")
    app.run_polling()


if __name__ == "__main__":
    main()
