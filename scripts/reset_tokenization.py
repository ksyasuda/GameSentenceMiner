#!/usr/bin/env python3

import argparse
import sys
from datetime import datetime, timedelta

from GameSentenceMiner.util.config.configuration import logger
from GameSentenceMiner.util.database.db import (
    CronTable,
    GameLinesTable,
    SQLiteDB,
    get_db_directory,
)


def reset_tokenization(db_path: str):
    """Fully remove tokenization data and schema additions.

    - Drops the words, kanji, word_occurrences, and kanji_occurrences tables
      (including all their indexes).
    - Drops the tokenized and statistics columns from game_lines.
    - Ensures daily tokenization cron exists/enabled for repopulation.
    """
    db = SQLiteDB(db_path)
    try:
        for table_cls in [GameLinesTable, CronTable]:
            table_cls.set_db(db)

        # Drop tokenization tables first (indexes are dropped automatically with each table).
        for table in ["word_occurrences", "kanji_occurrences", "words", "kanji"]:
            db.execute(f"DROP TABLE IF EXISTS {table}", commit=True)

        # Drop optional legacy tokenization index if present (older schema variants).
        db.execute("DROP INDEX IF EXISTS idx_game_lines_tokenized;", commit=True)

        # Drop tokenization / stats columns from game_lines.
        for col in [
            "tokenized",
            "total_length",
            "filtered_length",
            "word_count",
            "kanji_count",
        ]:
            if not GameLinesTable.has_column(col):
                continue
            try:
                GameLinesTable.drop_column(col)
            except Exception as exc:
                print(f"Warning: could not drop column '{col}': {exc}")

        now = datetime.now()

        daily_cron = CronTable.get_by_name("daily_tokenization")
        if daily_cron is None:
            next_run = now.replace(hour=1, minute=0, second=0, microsecond=0)
            if next_run < now:
                next_run += timedelta(days=1)
            CronTable.create_cron_entry(
                name="daily_tokenization",
                description="Daily catchup to tokenize lines that remain untokenized",
                next_run=next_run.timestamp(),
                schedule="daily",
                enabled=True,
            )
        else:
            # Nudge to run on next startup if it was already pending but completed.
            daily_cron.enabled = True
            if daily_cron.next_run <= now.timestamp():
                daily_cron.next_run = (now - timedelta(minutes=1)).timestamp()
            daily_cron.save()

        logger.info(f"Tokenization reset complete for db={db_path}")
    finally:
        db.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reset the tokenization tables")
    parser.add_argument(
        "--db-path",
        type=str,
        help="Path to the GSM database",
        default=get_db_directory(),
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    reset_tokenization(args.db_path)
    print("Tokenization tables and columns reset successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
