"""
One-time backfill cron job to tokenize all existing game lines.
This runs on first startup when tokenization tables are empty but game lines exist.

Uses optimized batch tokenization with:
- Persistent MeCab process (avoids subprocess spawn overhead)
- Bulk database operations (reduces from ~15,000+ commits to a few per batch)
"""

import time

from GameSentenceMiner.util.configuration import logger


def format_eta(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def backfill_tokenization():
    """
    One-time backfill job to tokenize all existing game lines.
    Uses batch tokenization with persistent MeCab and bulk DB operations.

    Returns:
        Dictionary with success status and statistics
    """
    from GameSentenceMiner.util.db import GameLinesTable
    from GameSentenceMiner.util.tokenization_service import tokenization_service

    logger.info("Starting tokenization backfill...")

    batch_size = 500

    count_query = f"""
        SELECT COUNT(*) FROM {GameLinesTable._table}
        WHERE line_text IS NOT NULL AND line_text != '' AND tokenized=0
    """

    try:
        total = GameLinesTable._db.fetchone(count_query)[0]
    except Exception as e:
        logger.error(f"Failed to count game lines: {e}")
        return {
            'success': False,
            'error': str(e),
            'total': 0,
            'processed': 0,
            'failed': 0
        }

    if total == 0:
        logger.info("No lines to tokenize, backfill complete")
        return {
            'success': True,
            'total': 0,
            'processed': 0,
            'failed': 0
        }

    logger.info(f"Found {total} game lines to process (batch size: {batch_size})")

    processed = 0
    failed = 0
    start_time = time.time()
    batches_processed = 0
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 3

    while True:
        query = f"""
            SELECT id, line_text, timestamp, game_id
            FROM {GameLinesTable._table}
            WHERE line_text IS NOT NULL AND line_text != '' AND tokenized=0
            ORDER BY id
            LIMIT {batch_size}
        """

        try:
            batch = GameLinesTable._db.fetchall(query)
            consecutive_failures = 0  # Reset on successful DB fetch
        except Exception as e:
            consecutive_failures += 1
            logger.error(f"Failed to fetch batch (attempt {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}): {e}")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.error("Too many consecutive DB fetch failures, aborting backfill")
                return {
                    'success': False,
                    'error': f"Database fetch failed {MAX_CONSECUTIVE_FAILURES} times: {e}",
                    'total': total,
                    'processed': processed,
                    'failed': failed + (total - processed - failed)
                }
            continue

        if not batch:
            break

        result = tokenization_service.tokenize_batch(batch, chunk_size=batch_size)
        batch_processed = result['processed']
        batch_failed = result['failed']
        processed += batch_processed
        failed += batch_failed
        batches_processed += 1

        # Mark individually failed lines as unprocessable (tokenized=-1)
        # so they won't be re-fetched, guaranteeing progress every iteration.
        batch_failed_ids = result.get('failed_ids', [])
        if batch_failed_ids:
            logger.warning(f"{len(batch_failed_ids)} lines failed tokenization, marking as unprocessable")
            placeholders = ','.join('?' * len(batch_failed_ids))
            try:
                GameLinesTable._db.execute(
                    f"UPDATE {GameLinesTable._table} SET tokenized=-1 WHERE id IN ({placeholders})",
                    batch_failed_ids, commit=True
                )
            except Exception as e:
                logger.error(f"Failed to mark unprocessable lines: {e}")

        if result.get('interrupted'):
            logger.warning(f"Backfill interrupted by shutdown after processing {processed} lines")
            return {
                'success': False,
                'interrupted': True,
                'total': total,
                'processed': processed,
                'failed': failed
            }

        if result.get('error'):
            logger.error(f"Batch error: {result['error']}")

        progress = processed + failed
        if batches_processed % 2 == 0 or progress >= total:
            elapsed = time.time() - start_time

            if progress > 0:
                rate = progress / elapsed  # lines per second
                remaining = total - progress
                eta_seconds = remaining / rate if rate > 0 else 0
                eta_str = format_eta(eta_seconds)
                logger.info(f"Tokenization progress: {progress}/{total} lines ({processed} OK, {failed} failed) - {rate:.1f} lines/s - ETA: {eta_str}")
            else:
                logger.info(f"Tokenization progress: {progress}/{total} lines ({processed} OK, {failed} failed)")

    total_elapsed = time.time() - start_time
    elapsed_str = format_eta(total_elapsed)
    rate = processed / total_elapsed if total_elapsed > 0 else 0
    logger.info(f"Backfill complete: {processed} processed, {failed} failed out of {total} total")
    logger.info(f"Elapsed: {elapsed_str}, Average rate: {rate:.1f} lines/s")


    return {
        'success': True,
        'total': total,
        'processed': processed,
        'failed': failed
    }
