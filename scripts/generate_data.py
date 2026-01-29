#!/usr/bin/env python
"""
Script to insert duplicated game lines data into the database for testing purposes.
Duplicates existing rows with new primary keys until the target row count is reached.
"""

import argparse
import random
import sys
import uuid
import time
from pathlib import Path

# Add parent directory to path to import from GameSentenceMiner
sys.path.insert(0, str(Path(__file__).parent.parent))

from GameSentenceMiner.util.db import GameLinesTable, gsm_db, backup_db, db_path


def get_current_row_count() -> int:
    """Get the current number of rows in the game_lines table."""
    result = gsm_db.fetchone(f"SELECT COUNT(*) FROM {GameLinesTable._table}")
    return result[0] if result else 0


def get_all_lines() -> list:
    """Fetch all existing game lines as raw rows."""
    return gsm_db.fetchall(
        f"SELECT game_name, line_text, screenshot_in_anki, audio_in_anki, "
        f"screenshot_path, audio_path, replay_path, translation, timestamp, "
        f"original_game_name, game_id, note_ids FROM {GameLinesTable._table}"
    )


def generate_modified_line_text(templates: list, min_len: int = 3, max_len: int = 100) -> str:
    """
    Generate text by truncating or extending an existing line to a random length.

    Args:
        templates: List of template rows to sample from
        min_len: Minimum line length
        max_len: Maximum line length

    Returns:
        Modified line text of random length
    """
    target_len = random.randint(min_len, max_len)

    # Pick a random starting line
    template = random.choice(templates)
    line_text = template[1] or ""  # line_text is at index 1

    # If empty, try to find a non-empty one
    if not line_text:
        for t in templates:
            if t[1]:
                line_text = t[1]
                break
        if not line_text:
            return ""

    # Truncate if longer than target
    if len(line_text) >= target_len:
        return line_text[:target_len]

    # Extend if shorter than target
    result = line_text
    while len(result) < target_len:
        # Pick another random line to extend with
        other = random.choice(templates)
        other_text = other[1] or ""
        if other_text:
            needed = target_len - len(result)
            result += other_text[:needed]

    return result[:target_len]


def insert_duplicates(target_count: int, batch_size: int = 1000, exact_copy: bool = False) -> None:
    """
    Insert duplicated rows until the target count is reached.

    Args:
        target_count: The desired total number of rows in the table
        batch_size: Number of rows to insert per batch
        exact_copy: If True, copy rows exactly. If False (default), generate random line_text
    """
    current_count = get_current_row_count()

    if current_count == 0:
        print("Error: No existing data to duplicate. Please add some game lines first.")
        sys.exit(1)

    if current_count >= target_count:
        print(f"Table already has {current_count} rows, which meets or exceeds target of {target_count}.")
        return

    rows_needed = target_count - current_count
    print(f"Current row count: {current_count}")
    print(f"Target row count: {target_count}")
    print(f"Rows to insert: {rows_needed}")

    # Fetch existing data to duplicate
    existing_rows = get_all_lines()
    if not existing_rows:
        print("Error: Could not fetch existing rows.")
        sys.exit(1)

    print(f"Found {len(existing_rows)} existing rows to use as templates")

    # Prepare insert statement
    insert_sql = f"""
        INSERT INTO {GameLinesTable._table}
        (id, game_name, line_text, screenshot_in_anki, audio_in_anki,
         screenshot_path, audio_path, replay_path, translation, timestamp,
         original_game_name, game_id, note_ids)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    rows_inserted = 0
    start_time = time.time()

    template_index = 0
    num_templates = len(existing_rows)

    while rows_inserted < rows_needed:
        # Determine batch size for this iteration
        current_batch_size = min(batch_size, rows_needed - rows_inserted)

        # Gather templates for this batch
        batch_templates = []
        for _ in range(current_batch_size):
            batch_templates.append(existing_rows[template_index % num_templates])
            template_index += 1

        # Create rows for this batch
        batch = []
        for template in batch_templates:
            new_id = str(uuid.uuid4())
            if exact_copy:
                new_row = (new_id,) + template
            else:
                random_text = generate_modified_line_text(batch_templates, 3, 100)
                new_row = (
                    new_id,           # id
                    template[0],      # game_name
                    random_text,      # line_text (RANDOMIZED)
                    template[2],      # screenshot_in_anki
                    template[3],      # audio_in_anki
                    template[4],      # screenshot_path
                    template[5],      # audio_path
                    template[6],      # replay_path
                    template[7],      # translation
                    template[8],      # timestamp
                    template[9],      # original_game_name
                    template[10],     # game_id
                    template[11],     # note_ids
                )
            batch.append(new_row)

        # Insert the batch
        gsm_db.executemany(insert_sql, batch, commit=True)
        rows_inserted += len(batch)

        # Progress update
        elapsed = time.time() - start_time
        rate = rows_inserted / elapsed if elapsed > 0 else 0
        print(f"Inserted {rows_inserted}/{rows_needed} rows ({rate:.0f} rows/sec)")

    elapsed = time.time() - start_time
    final_count = get_current_row_count()
    print(f"\nDone! Inserted {rows_inserted} rows in {elapsed:.2f} seconds")
    print(f"Final row count: {final_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate duplicate game lines data for testing purposes."
    )
    parser.add_argument(
        "target_rows",
        type=int,
        help="Target number of total rows in the game_lines table"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of rows to insert per batch (default: 1000)"
    )
    parser.add_argument(
        "--exact-copy",
        action="store_true",
        help="Copy rows exactly (only change ID). Default: generate random line_text"
    )

    args = parser.parse_args()

    if args.target_rows <= 0:
        print("Error: target_rows must be a positive integer")
        sys.exit(1)

    # Prompt for backup (default Y)
    response = input("Create a backup of the database before proceeding? [Y/n]: ").strip().lower()
    if response in ('', 'y', 'yes'):
        print(f"Backing up database: {db_path}")
        backup_db(db_path)
        print("Backup complete.")
    else:
        print("Skipping backup.")

    insert_duplicates(args.target_rows, args.batch_size, args.exact_copy)


if __name__ == "__main__":
    main()
