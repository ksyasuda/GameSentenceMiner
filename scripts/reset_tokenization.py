#!/usr/bin/env python3

import argparse
import sys

from GameSentenceMiner.util.config.configuration import logger
from GameSentenceMiner.util.database.db import (
    GameLinesTable,
    SQLiteDB,
    get_db_directory,
)


def reset_tokenization(db_path: str):
    """Fully remove tokenization data and schema additions.

    - Drops the words, kanji, word_occurrences, and kanji_occurrences tables
      (including all their indexes).
    - Drops the tokenized and statistics columns from game_lines.
    """
    db = SQLiteDB(db_path)
    try:
        for table_cls in [GameLinesTable]:
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
