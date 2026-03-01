#!/usr/bin/env python3
"""Dry-run a single line tokenization and print the would-be DB changes."""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


GAME_LINES_COLUMNS = [
    "id",
    "line_text",
    "timestamp",
    "game_id",
    "tokenized",
    "total_length",
    "filtered_length",
    "word_count",
    "kanji_count",
]
WORDS_COLUMNS = ["id", "headword", "word", "reading", "first_seen", "last_seen", "frequency"]
KANJI_COLUMNS = ["id", "kanji", "first_seen", "last_seen", "frequency"]
WORD_OCC_COLUMNS = ["id", "word_id", "line_id", "game_id", "timestamp", "count"]
KANJI_OCC_COLUMNS = ["id", "kanji_id", "line_id", "game_id", "timestamp", "count"]


def _get_default_db_path() -> str:
    from GameSentenceMiner.util.database.db import get_db_directory

    return get_db_directory()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run tokenization for one line. "
            "No writes to your real DB: work happens on a temporary DB clone."
        )
    )
    parser.add_argument(
        "text",
        nargs="?",
        default=None,
        help="Optional line text. If omitted, a random row from game_lines is used.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=_get_default_db_path(),
        help="Path to GSM database (default: app DB path).",
    )
    parser.add_argument(
        "--source",
        choices=["realtime", "backfill"],
        default="realtime",
        help="Tokenization source mode (affects backend/chunk behavior).",
    )
    return parser


def _as_dict(columns: Sequence[str], row: Optional[Sequence[Any]]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return {column: row[idx] for idx, column in enumerate(columns)}


def _rows_as_dicts(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[Dict[str, Any]]:
    return [{column: row[idx] for idx, column in enumerate(columns)} for row in rows]


def _make_key(row: Dict[str, Any], key_fields: Sequence[str]) -> Tuple[Any, ...]:
    return tuple(row.get(field) for field in key_fields)


def compute_table_changes(
    before_rows: Sequence[Dict[str, Any]],
    after_rows: Sequence[Dict[str, Any]],
    key_fields: Sequence[str] = ("id",),
) -> Dict[str, List[Dict[str, Any]]]:
    before_map = {_make_key(row, key_fields): row for row in before_rows}
    after_map = {_make_key(row, key_fields): row for row in after_rows}

    inserted: List[Dict[str, Any]] = []
    updated: List[Dict[str, Any]] = []
    deleted: List[Dict[str, Any]] = []

    for key in sorted(after_map.keys() - before_map.keys(), key=str):
        inserted.append(after_map[key])
    for key in sorted(before_map.keys() - after_map.keys(), key=str):
        deleted.append(before_map[key])
    for key in sorted(before_map.keys() & after_map.keys(), key=str):
        before_row = before_map[key]
        after_row = after_map[key]
        changed_fields: Dict[str, Dict[str, Any]] = {}
        fields = sorted(set(before_row.keys()) | set(after_row.keys()))
        for field in fields:
            before_val = before_row.get(field)
            after_val = after_row.get(field)
            if before_val != after_val:
                changed_fields[field] = {"before": before_val, "after": after_val}
        if changed_fields:
            updated.append(
                {
                    "key": list(key),
                    "fields": changed_fields,
                    "before": before_row,
                    "after": after_row,
                }
            )

    return {"inserted": inserted, "updated": updated, "deleted": deleted}


def select_target_line(
    db: Any,
    provided_text: Optional[str],
    now_ts: Optional[float] = None,
) -> Dict[str, Any]:
    if provided_text is not None:
        return {
            "line_id": f"dry_run_{uuid.uuid4().hex[:12]}",
            "line_text": str(provided_text),
            "timestamp": float(now_ts if now_ts is not None else time.time()),
            "game_id": "",
            "created_temp_line": True,
        }

    row = db.fetchone(
        "SELECT id, line_text, timestamp, COALESCE(game_id, '') FROM game_lines ORDER BY RANDOM() LIMIT 1"
    )
    if row is None:
        raise RuntimeError("No rows found in game_lines; provide text explicitly.")

    return {
        "line_id": str(row[0]),
        "line_text": str(row[1] or ""),
        "timestamp": float(row[2] or (now_ts if now_ts is not None else time.time())),
        "game_id": str(row[3] or ""),
        "created_temp_line": False,
    }


def _clone_database(db_path: str) -> tempfile.TemporaryDirectory:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database path not found: {db_path}")

    tmp_dir = tempfile.TemporaryDirectory(prefix="gsm-tokenization-dry-run-")
    cloned_db_path = os.path.join(tmp_dir.name, "gsm.db")
    shutil.copy2(db_path, cloned_db_path)
    for suffix in ("-wal", "-shm"):
        source_sidecar = f"{db_path}{suffix}"
        cloned_sidecar = f"{cloned_db_path}{suffix}"
        if os.path.exists(source_sidecar):
            shutil.copy2(source_sidecar, cloned_sidecar)
    return tmp_dir


def _configure_tables_for_db(cloned_db_path: str) -> Any:
    from GameSentenceMiner.util.database.db import (
        GameLinesTable,
        KanjiOccurrencesTable,
        KanjiTable,
        SQLiteDB,
        WordOccurrencesTable,
        WordsTable,
    )

    db = SQLiteDB(cloned_db_path)
    for table_cls in [GameLinesTable, WordsTable, KanjiTable, WordOccurrencesTable, KanjiOccurrencesTable]:
        table_cls.set_db(db)
    return db


def _insert_temp_line(target_line: Dict[str, Any]) -> None:
    from GameSentenceMiner.util.database.db import GameLinesTable, TOKENIZED_PENDING

    GameLinesTable(
        id=target_line["line_id"],
        game_name="dry_run",
        line_text=target_line["line_text"],
        timestamp=target_line["timestamp"],
        game_id=target_line["game_id"],
        language="ja",
        tokenized=TOKENIZED_PENDING,
    ).add()


def _fetch_rows_by_ids(
    db: Any,
    table_name: str,
    columns: Sequence[str],
    id_column: str,
    ids: Iterable[Any],
) -> List[Dict[str, Any]]:
    unique_ids = [item for item in dict.fromkeys(ids) if item is not None]
    if not unique_ids:
        return []
    placeholders = ", ".join("?" for _ in unique_ids)
    rows = db.fetchall(
        f"SELECT {', '.join(columns)} FROM {table_name} WHERE {id_column} IN ({placeholders})",
        tuple(unique_ids),
    )
    return _rows_as_dicts(columns, rows)


def _capture_snapshot(
    db: Any,
    line_id: str,
    extra_word_ids: Optional[Iterable[int]] = None,
    extra_kanji_ids: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    line_row = db.fetchone(
        f"SELECT {', '.join(GAME_LINES_COLUMNS)} FROM game_lines WHERE id=?",
        (line_id,),
    )
    word_occ_rows_raw = db.fetchall(
        f"SELECT {', '.join(WORD_OCC_COLUMNS)} FROM word_occurrences WHERE line_id=?",
        (line_id,),
    )
    kanji_occ_rows_raw = db.fetchall(
        f"SELECT {', '.join(KANJI_OCC_COLUMNS)} FROM kanji_occurrences WHERE line_id=?",
        (line_id,),
    )

    word_occ_rows = _rows_as_dicts(WORD_OCC_COLUMNS, word_occ_rows_raw)
    kanji_occ_rows = _rows_as_dicts(KANJI_OCC_COLUMNS, kanji_occ_rows_raw)

    word_ids = {int(row["word_id"]) for row in word_occ_rows if row.get("word_id") is not None}
    kanji_ids = {int(row["kanji_id"]) for row in kanji_occ_rows if row.get("kanji_id") is not None}
    if extra_word_ids:
        word_ids.update(int(x) for x in extra_word_ids if x is not None)
    if extra_kanji_ids:
        kanji_ids.update(int(x) for x in extra_kanji_ids if x is not None)

    words_rows = _fetch_rows_by_ids(
        db=db,
        table_name="words",
        columns=WORDS_COLUMNS,
        id_column="id",
        ids=word_ids,
    )
    kanji_rows = _fetch_rows_by_ids(
        db=db,
        table_name="kanji",
        columns=KANJI_COLUMNS,
        id_column="id",
        ids=kanji_ids,
    )

    game_lines_rows = [_as_dict(GAME_LINES_COLUMNS, line_row)] if line_row else []
    return {
        "game_lines": game_lines_rows,
        "word_occurrences": word_occ_rows,
        "kanji_occurrences": kanji_occ_rows,
        "words": words_rows,
        "kanji": kanji_rows,
        "word_ids": sorted(word_ids),
        "kanji_ids": sorted(kanji_ids),
    }


def run_dry_run(db_path: str, text: Optional[str], source: str = "realtime") -> Dict[str, Any]:
    from GameSentenceMiner.util.tokenization_service import TokenizationService

    tmp_dir = _clone_database(db_path)
    cloned_db_path = os.path.join(tmp_dir.name, "gsm.db")
    db = _configure_tables_for_db(cloned_db_path)
    try:
        target_line = select_target_line(db, provided_text=text, now_ts=time.time())
        if target_line["created_temp_line"]:
            _insert_temp_line(target_line)

        before = _capture_snapshot(db, target_line["line_id"])
        service = TokenizationService()
        tokenization_result = service.tokenize_lines_batch(
            [
                (
                    target_line["line_id"],
                    target_line["line_text"],
                    float(target_line["timestamp"]),
                    target_line["game_id"],
                )
            ],
            source=source,
        )
        after = _capture_snapshot(
            db,
            target_line["line_id"],
            extra_word_ids=before["word_ids"],
            extra_kanji_ids=before["kanji_ids"],
        )
    finally:
        db.close()
        tmp_dir.cleanup()

    changes = {
        "game_lines": compute_table_changes(before["game_lines"], after["game_lines"], key_fields=("id",)),
        "words": compute_table_changes(before["words"], after["words"], key_fields=("id",)),
        "kanji": compute_table_changes(before["kanji"], after["kanji"], key_fields=("id",)),
        "word_occurrences": compute_table_changes(
            before["word_occurrences"], after["word_occurrences"], key_fields=("id",)
        ),
        "kanji_occurrences": compute_table_changes(
            before["kanji_occurrences"], after["kanji_occurrences"], key_fields=("id",)
        ),
    }

    return {
        "db_path": db_path,
        "source": source,
        "line": target_line,
        "tokenization_result": tokenization_result,
        "changes": changes,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_dry_run(db_path=args.db_path, text=args.text, source=args.source)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
