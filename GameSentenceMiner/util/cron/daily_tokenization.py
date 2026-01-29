"""
Daily cron job to catch up on any missed tokenizations.
This runs daily to tokenize any lines that failed or were missed during real-time processing.
"""

import logging

logger = logging.getLogger(__name__)


def daily_tokenization_catchup():
    """
    Daily cron to tokenize any lines that were missed.

    This reuses the backfill logic since the process is the same:
    find all untokenized lines and process them in batches.

    Typically the volume will be much smaller than the initial backfill,
    only catching stragglers that failed or were missed during real-time processing.

    Returns:
        Dictionary with success status and statistics
    """
    from GameSentenceMiner.util.cron.backfill_tokenization import backfill_tokenization

    logger.info("Starting daily tokenization catchup...")
    result = backfill_tokenization()
    logger.info(f"Daily tokenization catchup complete: {result}")

    return result
