#!/usr/bin/env python3

import argparse
from GameSentenceMiner.util.db import get_db_directory
from GameSentenceMiner.util.cron_table import CronTable
from datetime import datetime, timedelta


def reset_tokenization(db_path: str):
    """Fully remove tokenization data and schema additions.

    - Drops the words, kanji, word_occurrences, and kanji_occurrences tables
      (including all their indexes).
    - Drops the tokenized and statistics columns from game_lines.
    - Re-enables the backfill cron job so it runs on next startup.
    """
    from GameSentenceMiner.util.db import GameLinesTable, gsm_db

    for table_cls in [GameLinesTable, CronTable]:
        if table_cls._db is None:
            table_cls.set_db(gsm_db)

    db = GameLinesTable._db

    # Drop tokenization tables (indexes are dropped automatically with the table)
    for table in ['word_occurrences', 'kanji_occurrences', 'words', 'kanji']:
        db.execute(f"DROP TABLE IF EXISTS {table}", commit=True)
    db.execute(f"DROP INDEX IF EXISTS idx_game_lines_tokenized;", commit=True)

    # Drop tokenization / stats columns from game_lines
    for col in ['tokenized', 'total_length', 'filtered_length',
                'word_count', 'kanji_count']:
        try:
            GameLinesTable.drop_column(col)
        except Exception:
            pass  # column may not exist yet

    # Re-enable backfill cron so it recreates everything on next startup
    cron = CronTable.get_by_name('backfill_tokenization')
    if cron and not cron.enabled:
        cron.next_run = (datetime.now() - timedelta(minutes=1)).timestamp()
        cron.enable()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reset the tokenization tables")
    parser.add_argument("--db-path", type=str, help="Path to the GSM database", default=get_db_directory())
    args = parser.parse_args()
    reset_tokenization(args.db_path)
    print("Tokenization tables and columns reset successfully")
