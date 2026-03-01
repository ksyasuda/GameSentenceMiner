"""
Tokenization backfill job.
"""

import os
import time

from GameSentenceMiner.util.config.configuration import logger

TOKENIZED_PENDING = 0
TOKENIZED_RETRYABLE_FAILED = 2


def _is_database_locked_error(exc: Exception) -> bool:
    return "database is locked" in str(exc).lower()


def _run_with_db_lock_retry(fn, operation_name: str):
    max_attempts = 12
    delay_seconds = 0.08
    max_delay_seconds = 0.6
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if not _is_database_locked_error(exc) or attempt >= max_attempts:
                raise
            logger.warning(
                f"Backfill DB lock during {operation_name}, retrying attempt {attempt + 1}/{max_attempts}."
            )
            time.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 2, max_delay_seconds)


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, sec = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def backfill_tokenization():
    from GameSentenceMiner.util.database.db import GameLinesTable
    from GameSentenceMiner.util.tokenization_service import \
        get_tokenization_service, has_pending_realtime_work

    # Retryable failures are deferred to later runs; requeue them at run start.
    _run_with_db_lock_retry(
        lambda: GameLinesTable._db.execute(
            f"""
            UPDATE {GameLinesTable._table}
            SET tokenized = {TOKENIZED_PENDING}
            WHERE COALESCE(tokenized, {TOKENIZED_PENDING}) = {TOKENIZED_RETRYABLE_FAILED}
                AND line_text IS NOT NULL
                AND TRIM(line_text) != ''
            """,
            commit=True,
        ),
        "retryable-failure reset",
    )

    total_row = GameLinesTable._db.fetchone(f"""
        SELECT COUNT(*)
        FROM {GameLinesTable._table}
        WHERE line_text IS NOT NULL
            AND TRIM(line_text) != ''
            AND COALESCE(tokenized, {TOKENIZED_PENDING}) = {TOKENIZED_PENDING}
        """)

    total = int(total_row[0]) if total_row and total_row[0] is not None else 0
    if total == 0:
        return {"success": True, "total": 0, "processed": 0, "failed": 0, "completed": 0}

    logger.info(f"Tokenization backfill starting: total={total}")

    service = get_tokenization_service()
    service.begin_backfill_session()
    try:
        wait_timeout_seconds = 90
        wait_poll_seconds = 1
        wait_start = time.perf_counter()
        if not service.is_tokenizer_available(timeout=0.5, source="backfill"):
            logger.info("Tokenization backfill waiting for tokenizer availability...")
            while (time.perf_counter() - wait_start) < wait_timeout_seconds:
                time.sleep(wait_poll_seconds)
                if service.is_tokenizer_available(timeout=0.5, source="backfill"):
                    logger.info("Tokenizer became available; starting backfill processing.")
                    break
            else:
                logger.warning(
                    f"Tokenization backfill skipped: tokenizer unavailable after {wait_timeout_seconds}s wait."
                )
                return {
                    "success": False,
                    "total": total,
                    "processed": 0,
                    "failed": 0,
                    "skipped": True,
                }

        processed = 0
        failed = 0
        completed = 0
        batch_number = 0
        batch_size = max(25, int(os.environ.get("GSM_TOKENIZER_BACKFILL_BATCH_SIZE", "1000")))
        next_progress_milestone = batch_size
        start_time = time.perf_counter()
        realtime_yield_sleep_seconds = 0.05

        while True:
            while has_pending_realtime_work():
                time.sleep(realtime_yield_sleep_seconds)

            batch_number += 1
            rows = GameLinesTable._db.fetchall(f"""
                SELECT id, line_text, timestamp, game_id
                FROM {GameLinesTable._table}
                WHERE line_text IS NOT NULL
                    AND TRIM(line_text) != ''
                    AND COALESCE(tokenized, {TOKENIZED_PENDING}) = {TOKENIZED_PENDING}
                ORDER BY timestamp ASC
                LIMIT {batch_size}
                """)

            if not rows:
                break

            result = service.tokenize_lines_batch(rows, source="backfill")
            batch_processed = int(result.get("processed", 0))
            batch_failed = int(result.get("failed", 0))
            processed += batch_processed
            failed += batch_failed
            completed += batch_processed + batch_failed

            if completed >= next_progress_milestone or completed == total:
                elapsed = time.perf_counter() - start_time
                rate = completed / elapsed if elapsed > 0 else 0.0
                remaining = max(0, total - completed)
                eta = remaining / rate if rate > 0 else 0.0
                logger.info(
                    "Tokenization backfill progress: "
                    f"batch={batch_number} completed={completed}/{total} "
                    f"processed={processed} failed={failed} "
                    f"rate={rate:.2f} lines/sec eta={_format_duration(eta)}"
                )
                while next_progress_milestone <= completed:
                    next_progress_milestone += batch_size

        completed = processed + failed

        logger.info(
            f"Tokenization backfill complete: total={total}, processed={processed}, failed={failed}, completed={completed}"
        )
        return {
            "success": True,
            "total": total,
            "processed": processed,
            "failed": failed,
            "completed": completed,
        }
    finally:
        service.end_backfill_session()
