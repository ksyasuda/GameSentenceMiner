"""
Tokenization backfill job.
"""

import time

from GameSentenceMiner.util.config.configuration import logger

TOKENIZED_PENDING = 0
TOKENIZED_RETRYABLE_FAILED = 2


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
        get_tokenization_service

    # Retryable failures are deferred to later runs; requeue them at run start.
    GameLinesTable._db.execute(f"""
        UPDATE {GameLinesTable._table}
        SET tokenized = {TOKENIZED_PENDING}
        WHERE COALESCE(tokenized, {TOKENIZED_PENDING}) = {TOKENIZED_RETRYABLE_FAILED}
            AND line_text IS NOT NULL
            AND TRIM(line_text) != ''
        """, commit=True)

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
    wait_timeout_seconds = 90
    wait_poll_seconds = 1
    wait_start = time.perf_counter()
    if not service.is_tokenizer_available(timeout=0.5):
        logger.info("Tokenization backfill waiting for tokenizer availability...")
        while (time.perf_counter() - wait_start) < wait_timeout_seconds:
            time.sleep(wait_poll_seconds)
            if service.is_tokenizer_available(timeout=0.5):
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
    batch_size = 2000
    next_progress_milestone = batch_size
    start_time = time.perf_counter()

    while True:
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

        result = service.tokenize_lines_batch(rows)
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
