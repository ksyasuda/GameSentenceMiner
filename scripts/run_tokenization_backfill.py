#!/usr/bin/env python3
"""Run tokenization catchup directly."""

import argparse
import sys
import time

from GameSentenceMiner.util.cron.backfill_tokenization import backfill_tokenization


def _configure_db_override(db_path: str) -> None:
    if not db_path:
        return

    from GameSentenceMiner.util.database.db import (
        CronTable,
        GameLinesTable,
        KanjiOccurrencesTable,
        KanjiTable,
        SQLiteDB,
        WordOccurrencesTable,
        WordsTable,
    )

    db = SQLiteDB(db_path)
    for table_cls in [
        GameLinesTable,
        WordsTable,
        KanjiTable,
        WordOccurrencesTable,
        KanjiOccurrencesTable,
        CronTable,
    ]:
        table_cls.set_db(db)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run tokenization catchup directly"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Optional path to a GSM database file (defaults to app DB path)",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _configure_db_override(args.db_path)

    start = time.perf_counter()
    result = backfill_tokenization()
    if not isinstance(result, dict):
        result = {"success": False, "total": 0, "processed": 0, "failed": 0, "completed": 0}
    elapsed = time.perf_counter() - start

    total = int(result.get("total", 0))
    processed = int(result.get("processed", 0))
    failed = int(result.get("failed", 0))
    completed = int(result.get("completed", processed + failed))
    skipped = bool(result.get("skipped", False))
    success = bool(result.get("success", False))

    print(f"\n{'=' * 50}")
    print(f"Tokenization catchup complete in {elapsed:.1f}s")
    print(f"  total:     {total}")
    print(f"  processed: {processed}")
    print(f"  failed:    {failed}")
    print(f"  completed: {completed}")
    print(f"  skipped:   {skipped}")
    print(f"  success:   {success}")
    if elapsed > 0 and total > 0:
        print(f"  rate:      {total / elapsed:.1f} lines/sec")
    print(f"{'=' * 50}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
