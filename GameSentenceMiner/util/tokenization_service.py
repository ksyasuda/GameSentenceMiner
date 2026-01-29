"""
Tokenization service for parsing Japanese text and storing word/kanji data.
"""

import atexit
import threading
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from GameSentenceMiner.util.configuration import logger
from GameSentenceMiner.web.stats import is_kanji
from GameSentenceMiner.mecab.basic_types import PartOfSpeech
from GameSentenceMiner.util.db import (
    WordsTable,
    KanjiTable,
    WordOccurrencesTable,
    KanjiOccurrencesTable,
    GameLinesTable,
    punctuation_regex,
)

_FILTERABLE_POS = frozenset(
    (
        PartOfSpeech.particle,  # 助詞 - は, から, を, て
        PartOfSpeech.symbol,  # 記号 - punctuation, special chars
        PartOfSpeech.bound_auxiliary,  # 助動詞 - だ, です, ない, ます, た
        PartOfSpeech.filler,  # フィラー - なんか, あのー, えーと
        PartOfSpeech.other,  # その他 - miscellaneous
    )
)


@dataclass
class WordMetadata:
    """Aggregated metadata for a unique word across a batch."""

    first_seen: float
    last_seen: float
    pos: str
    frequency: int = 1


@dataclass
class KanjiMetadata:
    """Aggregated metadata for a unique kanji across a batch."""

    first_seen: float
    last_seen: float
    frequency: int = 1


@dataclass
class BatchMetadata:
    """Pre-computed metadata from a single pass through tokenized data."""

    word_meta: Dict[tuple, WordMetadata] = field(
        default_factory=dict
    )  # (headword, word, reading) -> WordMetadata
    kanji_meta: Dict[str, KanjiMetadata] = field(
        default_factory=dict
    )  # kanji_char -> KanjiMetadata


@dataclass
class LineStatistics:
    """Statistics calculated for a single game line during tokenization."""

    total_length: int = 0
    filtered_length: int = 0
    word_count: int = 0
    kanji_count: int = 0


class TokenizationService:
    """
    Service for tokenizing Japanese text using Mecab and storing results in the database.
    Filters particles, symbols, auxiliaries, and common stopwords to focus on learnable vocabulary.
    Supports parallel processing for batch operations.
    """

    def __init__(self):
        """Initialize the tokenization service with Mecab controller."""
        from GameSentenceMiner.mecab.mecab_controller import MecabController

        self.mecab = MecabController(verbose=False, cache_max_size=2048, persistent=True)
        self._lock = threading.Lock()
        self._shutdown = False
        self._executor: Optional[ThreadPoolExecutor] = None
        self._executor_threads: int = 0
        self._executor_lock = threading.Lock()

        atexit.register(self._on_shutdown)

    def get_executor(self, num_threads: Optional[int] = None) -> ThreadPoolExecutor:
        """Get or create the shared thread pool executor."""
        if num_threads is None:
            import os

            num_threads = min(3, os.cpu_count() or 2)

        with self._executor_lock:
            if self._executor is None or self._executor_threads != num_threads:
                if self._executor is not None:
                    self._executor.shutdown(wait=True)
                self._executor = ThreadPoolExecutor(
                    max_workers=num_threads, thread_name_prefix="tokenize"
                )
                self._executor_threads = num_threads
        return self._executor

    def _on_shutdown(self):
        """Mark service as shutting down to prevent new work."""
        self._shutdown = True
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    @staticmethod
    def is_filterable_pos(pos) -> bool:
        """Return True if the part of speech should be filtered out of vocabulary."""
        return pos in _FILTERABLE_POS

    @staticmethod
    def _normalize_reading(reading: str, headword: str) -> str:
        """Normalize reading value. Returns '-' if reading is None/empty or equals the headword."""
        if not reading or reading == headword or reading == "*":
            return "-"
        return reading

    @staticmethod
    def _extract_tokens_and_kanji(line_text: str, tokens) -> tuple:
        """
        Extract merged word data and kanji data from MeCab tokens.

        Uses Yomitan-style merging to combine verbs with their auxiliaries/particles
        for more natural word boundaries (e.g., 食べて instead of 食べ + て).

        Returns:
            (word_data, kanji_data) where:
            - word_data: list of dicts with keys headword, word, reading, pos, position
            - kanji_data: list of dicts with keys kanji, position
        """
        from GameSentenceMiner.mecab.token_merger import merge_tokens

        merged_tokens = merge_tokens(list(tokens))

        word_data = []
        for merged in merged_tokens:
            # Skip tokens that are purely filterable (standalone particles, etc.)
            # but keep merged tokens even if they end with a filterable POS
            if (
                TokenizationService.is_filterable_pos(merged.part_of_speech)
                and not merged.is_merged
            ):
                continue

            pos_str = merged.part_of_speech.name if merged.part_of_speech else ""
            reading = TokenizationService._normalize_reading(
                merged.reading, merged.headword
            )
            word_data.append(
                {
                    "headword": merged.headword,
                    "word": merged.surface,  # Merged surface form
                    "reading": reading,  # Merged reading
                    "pos": pos_str,
                    "position": merged.start_pos,  # Character offset
                }
            )

        kanji_data = []
        for pos, char in enumerate(line_text):
            if is_kanji(char):
                kanji_data.append(
                    {
                        "kanji": char,
                        "position": pos,
                    }
                )

        return word_data, kanji_data

    @staticmethod
    def _compute_line_stats(
        line_text: str, words: list, kanji: list
    ) -> "LineStatistics":
        """Compute statistics for a single tokenized line."""
        total_length = len(line_text) if line_text else 0
        filtered_length = len(punctuation_regex.sub("", line_text)) if line_text else 0
        return LineStatistics(
            total_length=total_length,
            filtered_length=filtered_length,
            word_count=len(words),
            kanji_count=len(kanji),
        )

    def tokenize_line(
        self, game_line_id: str, line_text: str, timestamp: float, game_id: str = ""
    ) -> bool:
        """
        Tokenize a game line and store results in the database using a single transaction.

        This method:
        1. Parses the line with Mecab to extract tokens
        2. Filters out particles and symbols
        3. Extracts kanji characters
        4. Performs bulk lookups for existing words/kanji
        5. Executes all inserts/updates in a single transaction
        6. Marks the game line as tokenized

        Args:
            game_line_id: UUID of the game line
            line_text: The Japanese text to tokenize
            timestamp: Unix timestamp of when the line was created
            game_id: Optional game ID for the occurrence mapping

        Returns:
            True if tokenization succeeded, False otherwise
        """
        try:
            tokens = self.mecab.translate(line_text)
            word_items, kanji_items = self._extract_tokens_and_kanji(line_text, tokens)

            word_data = [
                (w["headword"], w["word"], w["reading"], w["pos"], w["position"])
                for w in word_items
            ]
            kanji_data = [(k["kanji"], k["position"]) for k in kanji_items]

            if not game_id:
                line = GameLinesTable.get(game_line_id)
                if line and line.game_id:
                    game_id = line.game_id

            stats = self._compute_line_stats(line_text, word_items, kanji_items)

            with self._lock:
                self._save_line_data_transactional(
                    game_line_id, word_data, kanji_data, timestamp, game_id, stats
                )

            logger.debug(
                f"Tokenized line {game_line_id}: {len(word_data)} words, {len(kanji_data)} kanji"
            )
            return True

        except Exception as e:
            logger.error(f"Tokenization failed for line {game_line_id}: {e}")
            return False

    def _save_line_data_transactional(
        self,
        game_line_id: str,
        word_data: list,
        kanji_data: list,
        timestamp: float,
        game_id: str,
        stats: Optional[LineStatistics] = None,
    ):
        """Save line data in a single transaction."""
        word_occurrence_counts = self._count_word_occurrences(word_data)
        kanji_occurrence_counts = self._count_kanji_occurrences(kanji_data)

        existing_words = (
            WordsTable.bulk_get_by_keys([(w[0], w[1], w[2]) for w in word_data])
            if word_data
            else {}
        )
        existing_kanji = (
            KanjiTable.bulk_get_by_characters(list(set(k[0] for k in kanji_data)))
            if kanji_data
            else {}
        )

        new_word_keys, new_words = self._find_new_words(
            word_data, existing_words, word_occurrence_counts, timestamp
        )
        new_kanji_chars, new_kanji = self._find_new_kanji(
            kanji_data, existing_kanji, kanji_occurrence_counts, timestamp
        )

        operations = []

        if new_words:
            operations.append(
                (
                    "INSERT OR IGNORE INTO words (headword, word, reading, first_seen, last_seen, frequency, pos) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    new_words,
                )
            )
        if new_kanji:
            operations.append(
                (
                    "INSERT OR IGNORE INTO kanji (kanji, first_seen, last_seen, frequency) VALUES (?, ?, ?, ?)",
                    new_kanji,
                )
            )

        # Execute inserts first so we can look up IDs
        if operations:
            WordsTable._db.execute_in_transaction(operations)
            operations = []

        if new_words:
            existing_words.update(WordsTable.bulk_get_by_keys(list(new_word_keys)))
        if new_kanji:
            existing_kanji.update(
                KanjiTable.bulk_get_by_characters(list(new_kanji_chars))
            )

        # Frequency updates + occurrences + mark tokenized in one transaction
        self._update_frequencies_and_occurrences(
            word_data,
            kanji_data,
            existing_words,
            existing_kanji,
            game_line_id,
            game_id,
            timestamp,
            stats,
        )

    @staticmethod
    def _count_word_occurrences(word_data: list) -> dict:
        counts = {}
        for headword, word, reading, _, _ in word_data:
            key = (headword, word, reading)
            counts[key] = counts.get(key, 0) + 1
        return counts

    @staticmethod
    def _count_kanji_occurrences(kanji_data: list) -> dict:
        counts = {}
        for character, _ in kanji_data:
            counts[character] = counts.get(character, 0) + 1
        return counts

    @staticmethod
    def _find_new_words(
        word_data: list, existing_words: dict, occurrence_counts: dict, timestamp: float
    ) -> tuple:
        new_word_keys = set()
        new_words = []
        for headword, word, reading, pos_str, _ in word_data:
            key = (headword, word, reading)
            if key not in existing_words and key not in new_word_keys:
                new_words.append(
                    (
                        headword,
                        word,
                        reading,
                        timestamp,
                        timestamp,
                        occurrence_counts[key],
                        pos_str,
                    )
                )
                new_word_keys.add(key)
        return new_word_keys, new_words

    @staticmethod
    def _find_new_kanji(
        kanji_data: list,
        existing_kanji: dict,
        occurrence_counts: dict,
        timestamp: float,
    ) -> tuple:
        new_kanji_chars = set()
        new_kanji = []
        for character, _ in kanji_data:
            if character not in existing_kanji and character not in new_kanji_chars:
                new_kanji.append(
                    (character, timestamp, timestamp, occurrence_counts[character])
                )
                new_kanji_chars.add(character)
        return new_kanji_chars, new_kanji

    @staticmethod
    def _update_frequencies_and_occurrences(
        word_data: list,
        kanji_data: list,
        existing_words: dict,
        existing_kanji: dict,
        game_line_id: str,
        game_id: str,
        timestamp: float,
        stats: Optional[LineStatistics] = None,
    ):
        word_timestamp_updates = []
        word_occurrences = []
        seen_words = set()
        for headword, word, reading, _, position in word_data:
            key = (headword, word, reading)
            word_obj = existing_words[key]
            if key not in seen_words:
                word_timestamp_updates.append((timestamp, timestamp, word_obj.id))
                seen_words.add(key)
            word_occurrences.append(
                (word_obj.id, game_line_id, game_id or None, timestamp, position)
            )

        kanji_timestamp_updates = []
        kanji_occurrences = []
        seen_kanji = set()
        for character, position in kanji_data:
            kanji_obj = existing_kanji[character]
            if character not in seen_kanji:
                kanji_timestamp_updates.append((timestamp, timestamp, kanji_obj.id))
                seen_kanji.add(character)
            kanji_occurrences.append(
                (kanji_obj.id, game_line_id, game_id or None, timestamp, position)
            )

        operations = []
        if word_timestamp_updates:
            operations.append(
                (
                    f"UPDATE {WordsTable._table} SET first_seen=MIN(first_seen, ?), last_seen=MAX(last_seen, ?) WHERE id=?",
                    word_timestamp_updates,
                )
            )
        if word_occurrences:
            operations.append(
                (
                    f"INSERT OR IGNORE INTO {WordOccurrencesTable._table} (word_id, line_id, game_id, timestamp, position) VALUES (?, ?, ?, ?, ?)",
                    word_occurrences,
                )
            )
        if kanji_timestamp_updates:
            operations.append(
                (
                    f"UPDATE {KanjiTable._table} SET first_seen=MIN(first_seen, ?), last_seen=MAX(last_seen, ?) WHERE id=?",
                    kanji_timestamp_updates,
                )
            )
        if kanji_occurrences:
            operations.append(
                (
                    f"INSERT OR IGNORE INTO {KanjiOccurrencesTable._table} (kanji_id, line_id, game_id, timestamp, position) VALUES (?, ?, ?, ?, ?)",
                    kanji_occurrences,
                )
            )

        # Keep frequency consistent with actual stored occurrences.
        # This avoids drift when a line is retokenized and occurrences are
        # ignored due to uniqueness constraints.
        if word_occurrences:
            word_ids = sorted({wid for (wid, _, _, _, _) in word_occurrences})
            operations.append(
                (
                    f"UPDATE {WordsTable._table} SET frequency=(SELECT COUNT(*) FROM {WordOccurrencesTable._table} WHERE {WordOccurrencesTable._table}.word_id = {WordsTable._table}.id) WHERE id=?",
                    [(wid,) for wid in word_ids],
                )
            )
        if kanji_occurrences:
            kanji_ids = sorted({kid for (kid, _, _, _, _) in kanji_occurrences})
            operations.append(
                (
                    f"UPDATE {KanjiTable._table} SET frequency=(SELECT COUNT(*) FROM {KanjiOccurrencesTable._table} WHERE {KanjiOccurrencesTable._table}.kanji_id = {KanjiTable._table}.id) WHERE id=?",
                    [(kid,) for kid in kanji_ids],
                )
            )

        if stats:
            operations.append(
                (
                    f"UPDATE {GameLinesTable._table} SET tokenized=1, "
                    f"total_length=?, filtered_length=?, word_count=?, kanji_count=? "
                    f"WHERE id=? AND tokenized=0",
                    [
                        (
                            stats.total_length,
                            stats.filtered_length,
                            stats.word_count,
                            stats.kanji_count,
                            game_line_id,
                        )
                    ],
                )
            )
        else:
            operations.append(
                (
                    f"UPDATE {GameLinesTable._table} SET tokenized=1 WHERE id=? AND tokenized=0",
                    [(game_line_id,)],
                )
            )

        WordsTable._db.execute_in_transaction(operations)

    def _collect_batch_metadata(self, tokenized_data: List[dict]) -> BatchMetadata:
        """
        Single pass through tokenized data to collect all metadata.

        Computes first_seen, last_seen, POS, and frequency for each unique word/kanji.

        Args:
            tokenized_data: List of tokenization result dictionaries

        Returns:
            BatchMetadata with aggregated word and kanji information
        """
        metadata = BatchMetadata()

        for data in tokenized_data:
            timestamp = data["timestamp"]

            for w in data["words"]:
                key = (w["headword"], w["word"], w["reading"])
                if key in metadata.word_meta:
                    meta = metadata.word_meta[key]
                    meta.frequency += 1
                    if timestamp < meta.first_seen:
                        meta.first_seen = timestamp
                    if timestamp > meta.last_seen:
                        meta.last_seen = timestamp
                else:
                    metadata.word_meta[key] = WordMetadata(
                        first_seen=timestamp,
                        last_seen=timestamp,
                        pos=w["pos"],
                        frequency=1,
                    )

            for k in data["kanji"]:
                char = k["kanji"]
                if char in metadata.kanji_meta:
                    meta = metadata.kanji_meta[char]
                    meta.frequency += 1
                    if timestamp < meta.first_seen:
                        meta.first_seen = timestamp
                    if timestamp > meta.last_seen:
                        meta.last_seen = timestamp
                else:
                    metadata.kanji_meta[char] = KanjiMetadata(
                        first_seen=timestamp, last_seen=timestamp, frequency=1
                    )

        return metadata

    def _prepare_word_data(
        self, metadata: BatchMetadata, existing_words: dict
    ) -> tuple:
        """
        Prepare word insert and update data.

        Args:
            metadata: Pre-computed batch metadata
            existing_words: Dict of existing words keyed by (headword, word, reading)

        Returns:
            Tuple of (new_words_data, word_updates, new_word_keys)
        """
        new_words_data = []
        word_updates = []
        new_word_keys = set()

        for key, meta in metadata.word_meta.items():
            if key in existing_words:
                word_obj = existing_words[key]
                earliest_ts = min(word_obj.first_seen, meta.first_seen)
                latest_ts = max(word_obj.last_seen, meta.last_seen)
                word_updates.append((earliest_ts, latest_ts, meta.frequency, word_obj.id))
            else:
                new_word_keys.add(key)
                new_words_data.append(
                    (
                        key[0],  # headword
                        key[1],  # word
                        key[2],  # reading
                        meta.first_seen,
                        meta.last_seen,
                        meta.frequency,
                        meta.pos,
                    )
                )

        return new_words_data, word_updates, new_word_keys

    def _prepare_kanji_data(
        self, metadata: BatchMetadata, existing_kanji: dict
    ) -> tuple:
        """
        Prepare kanji insert and update data.

        Args:
            metadata: Pre-computed batch metadata
            existing_kanji: Dict of existing kanji keyed by character

        Returns:
            Tuple of (new_kanji_data, kanji_updates, new_kanji_chars)
        """
        new_kanji_data = []
        kanji_updates = []
        new_kanji_chars = set()

        for char, meta in metadata.kanji_meta.items():
            if char in existing_kanji:
                kanji_obj = existing_kanji[char]
                earliest_ts = min(kanji_obj.first_seen, meta.first_seen)
                latest_ts = max(kanji_obj.last_seen, meta.last_seen)
                kanji_updates.append((earliest_ts, latest_ts, meta.frequency, kanji_obj.id))
            else:
                new_kanji_chars.add(char)
                new_kanji_data.append(
                    (char, meta.first_seen, meta.last_seen, meta.frequency)
                )

        return new_kanji_data, kanji_updates, new_kanji_chars

    def _prepare_occurrences(
        self, tokenized_data: List[dict], existing_words: dict, existing_kanji: dict
    ) -> tuple:
        """
        Prepare occurrence insert data for words and kanji (normalized schema).

        Args:
            tokenized_data: List of tokenization result dictionaries
            existing_words: Dict mapping word keys to WordsTable objects (with IDs)
            existing_kanji: Dict mapping kanji chars to KanjiTable objects (with IDs)

        Returns:
            Tuple of (word_occurrences_data, kanji_occurrences_data)
        """
        word_occurrences_data = []
        kanji_occurrences_data = []

        for data in tokenized_data:
            line_id = data["line_id"]
            game_id = data["game_id"]
            timestamp = data["timestamp"]

            for w in data["words"]:
                key = (w["headword"], w["word"], w["reading"])
                word_obj = existing_words[key]
                word_occurrences_data.append(
                    (
                        word_obj.id,  # INTEGER FK
                        line_id,
                        game_id,
                        timestamp,
                        w["position"],
                    )
                )

            for k in data["kanji"]:
                kanji_obj = existing_kanji[k["kanji"]]
                kanji_occurrences_data.append(
                    (
                        kanji_obj.id,  # INTEGER FK
                        line_id,
                        game_id,
                        timestamp,
                        k["position"],
                    )
                )

        return word_occurrences_data, kanji_occurrences_data

    def tokenize_batch(self, lines: list, chunk_size: int = 100) -> dict:
        """
        Optimized batch tokenization using bulk database operations with memory-bounded chunks.

        This method processes lines in smaller sub-batches to bound memory usage.
        Each chunk is committed independently, preserving progress on crashes.

        Args:
            lines: List of tuples (line_id, line_text, timestamp, game_id) or (line_id, line_text, timestamp)
            chunk_size: Number of lines to process per sub-batch (default 100)

        Returns:
            Dictionary with 'processed' and 'failed' counts, 'interrupted' if shutdown
        """
        if not lines:
            return {"processed": 0, "failed": 0}

        if self._shutdown:
            logger.warning("Tokenization service is shutting down, skipping batch")
            return {"processed": 0, "failed": 0, "interrupted": True}

        total_processed = 0
        total_failed = 0
        all_failed_ids = []

        # Process in memory-bounded chunks
        for i in range(0, len(lines), chunk_size):
            if self._shutdown:
                return {
                    "processed": total_processed,
                    "failed": total_failed + len(lines) - i,
                    "interrupted": True,
                    "failed_ids": all_failed_ids,
                }

            chunk = lines[i : i + chunk_size]

            try:
                tokenized_data, chunk_failed_ids = self._tokenize_lines_parallel(chunk)
                all_failed_ids.extend(chunk_failed_ids)

                if self._shutdown:
                    return {
                        "processed": total_processed,
                        "failed": total_failed + len(chunk),
                        "interrupted": True,
                        "failed_ids": all_failed_ids,
                    }

                with self._lock:
                    processed = self._bulk_save_all(tokenized_data)

                total_processed += processed
                total_failed += len(chunk_failed_ids)
                total_skipped = len(chunk) - processed - len(chunk_failed_ids)

            except Exception as e:
                logger.error(f"Chunk {i // chunk_size + 1} failed: {e}")
                total_failed += len(chunk)
                all_failed_ids.extend(row[0] for row in chunk)

        return {
            "processed": total_processed,
            "failed": total_failed,
            "failed_ids": list(dict.fromkeys(all_failed_ids)),
        }

    def _tokenize_lines_parallel(self, lines: list) -> tuple:
        """
        Tokenize multiple lines in parallel using MeCab (no database operations).

        All threads share the singleton MeCab controller, which serializes
        access through its internal lock.

        Args:
            lines: List of line tuples

        Returns:
            Tuple of (results: list[dict], failed_ids: list[str]) where failed_ids
            are the line IDs that failed tokenization.
        """
        executor = self.get_executor()
        results = []
        failed_ids = []

        def tokenize_single(line_data) -> Optional[dict]:
            """Tokenize a single line and return extracted data."""
            try:
                if len(line_data) >= 4:
                    line_id, line_text, timestamp, game_id = (
                        line_data[0],
                        line_data[1],
                        line_data[2],
                        line_data[3],
                    )
                else:
                    line_id, line_text, timestamp = (
                        line_data[0],
                        line_data[1],
                        line_data[2],
                    )
                    game_id = ""

                if not line_text:
                    return None

                tokens = self.mecab.translate(line_text)
                word_data, kanji_data = self._extract_tokens_and_kanji(
                    line_text, tokens
                )

                return {
                    "line_id": line_id,
                    "line_text": line_text,
                    "timestamp": timestamp,
                    "game_id": game_id or None,
                    "words": word_data,
                    "kanji": kanji_data,
                }

            except Exception as e:
                logger.error(f"Error tokenizing line: {e}")
                return None

        futures = {executor.submit(tokenize_single, line): line for line in lines}

        for future in as_completed(futures):
            if self._shutdown:
                for f in futures:
                    f.cancel()
                break
            try:
                result = future.result(timeout=30)
                if result:
                    results.append(result)
                else:
                    line_data = futures[future]
                    failed_ids.append(line_data[0])
            except Exception as e:
                logger.error(f"Tokenization future failed: {e}")
                line_data = futures[future]
                failed_ids.append(line_data[0])

        return results, failed_ids

    def _bulk_save_all(self, tokenized_data: List[dict]) -> int:
        """
        Save all tokenized data using bulk database operations with normalized schema.

        Uses a single-pass metadata collection, inserts new words/kanji,
        queries back IDs, then bulk inserts occurrences with integer FKs.

        Args:
            tokenized_data: List of tokenization result dictionaries

        Returns:
            Number of successfully processed lines
        """
        from GameSentenceMiner.util.db import (
            WordsTable,
            KanjiTable,
            WordOccurrencesTable,
            KanjiOccurrencesTable,
            GameLinesTable,
        )

        if not tokenized_data:
            return 0

        try:
            line_ids = [d["line_id"] for d in tokenized_data]
            placeholders = ",".join("?" * len(line_ids))
            rows = GameLinesTable._db.fetchall(
                f"SELECT id FROM {GameLinesTable._table} WHERE id IN ({placeholders}) AND tokenized=0",
                tuple(line_ids),
            )
            untokenized_ids = {r[0] for r in rows} if rows else set()
            tokenized_data = [
                d for d in tokenized_data if d["line_id"] in untokenized_ids
            ]
            if not tokenized_data:
                return 0
            metadata = self._collect_batch_metadata(tokenized_data)

            all_word_keys = list(metadata.word_meta.keys())
            all_kanji_chars = list(metadata.kanji_meta.keys())

            existing_words = (
                WordsTable.bulk_get_by_keys(all_word_keys) if all_word_keys else {}
            )
            existing_kanji = (
                KanjiTable.bulk_get_by_characters(all_kanji_chars)
                if all_kanji_chars
                else {}
            )

            new_words_data, word_updates, new_word_keys = self._prepare_word_data(
                metadata, existing_words
            )
            new_kanji_data, kanji_updates, new_kanji_chars = self._prepare_kanji_data(
                metadata, existing_kanji
            )

            # First transaction: insert new words/kanji
            insert_ops = []
            if new_words_data:
                insert_ops.append(
                    (
                        f"INSERT OR IGNORE INTO {WordsTable._table} (headword, word, reading, first_seen, last_seen, frequency, pos) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        new_words_data,
                    )
                )
            if new_kanji_data:
                insert_ops.append(
                    (
                        f"INSERT OR IGNORE INTO {KanjiTable._table} (kanji, first_seen, last_seen, frequency) VALUES (?, ?, ?, ?)",
                        new_kanji_data,
                    )
                )

            if insert_ops:
                WordsTable._db.execute_in_transaction(insert_ops)

            if new_words_data:
                newly_inserted = WordsTable.bulk_get_by_keys(list(new_word_keys))
                existing_words.update(newly_inserted)

            if new_kanji_data:
                newly_inserted = KanjiTable.bulk_get_by_characters(
                    list(new_kanji_chars)
                )
                existing_kanji.update(newly_inserted)

            word_occurrences_data, kanji_occurrences_data = self._prepare_occurrences(
                tokenized_data, existing_words, existing_kanji
            )

            mark_tokenized_params = []
            for data in tokenized_data:
                line_text = data.get("line_text", "")
                stats = self._compute_line_stats(
                    line_text, data["words"], data["kanji"]
                )
                mark_tokenized_params.append(
                    (
                        stats.total_length,
                        stats.filtered_length,
                        stats.word_count,
                        stats.kanji_count,
                        data["line_id"],
                    )
                )

            # Second transaction: frequency updates + occurrences + mark tokenized
            operations = []
            if word_updates:
                operations.append(
                    (
                        f"UPDATE {WordsTable._table} SET first_seen = MIN(first_seen, ?), last_seen = MAX(last_seen, ?), frequency = frequency + ? WHERE id = ?",
                        word_updates,
                    )
                )
            if kanji_updates:
                operations.append(
                    (
                        f"UPDATE {KanjiTable._table} SET first_seen = MIN(first_seen, ?), last_seen = MAX(last_seen, ?), frequency = frequency + ? WHERE id = ?",
                        kanji_updates,
                    )
                )
            if word_occurrences_data:
                operations.append(
                    (
                        f"INSERT OR IGNORE INTO {WordOccurrencesTable._table} (word_id, line_id, game_id, timestamp, position) VALUES (?, ?, ?, ?, ?)",
                        word_occurrences_data,
                    )
                )
            if kanji_occurrences_data:
                operations.append(
                    (
                        f"INSERT OR IGNORE INTO {KanjiOccurrencesTable._table} (kanji_id, line_id, game_id, timestamp, position) VALUES (?, ?, ?, ?, ?)",
                        kanji_occurrences_data,
                    )
                )
            if mark_tokenized_params:
                operations.append(
                    (
                        f"UPDATE {GameLinesTable._table} SET tokenized=1, "
                        f"total_length=?, filtered_length=?, word_count=?, kanji_count=? WHERE id=? AND tokenized=0",
                        mark_tokenized_params,
                    )
                )

            if operations:
                WordsTable._db.execute_in_transaction(operations)

            logger.debug(
                f"Batch complete: {len(new_words_data)} new words, {len(word_updates)} updated words, "
                f"{len(new_kanji_data)} new kanji, {len(kanji_updates)} updated kanji, "
                f"{len(word_occurrences_data)} word occurrences, {len(kanji_occurrences_data)} kanji occurrences"
            )

            return len(tokenized_data)

        except Exception as e:
            logger.error(f"Bulk save failed: {e}")
            raise


_tokenization_service: Optional[TokenizationService] = None
_tokenization_service_lock = threading.Lock()


def get_tokenization_service() -> TokenizationService:
    """Get the singleton tokenization service instance (lazy initialization, thread-safe)."""
    global _tokenization_service

    if _tokenization_service is None:
        with _tokenization_service_lock:
            if _tokenization_service is None:
                _tokenization_service = TokenizationService()

    return _tokenization_service


# Backwards compatibility - property-like access
class _TokenizationServiceProxy:
    """Proxy to allow lazy initialization while maintaining backwards compatibility."""

    def __getattr__(self, name):
        return getattr(get_tokenization_service(), name)


tokenization_service = _TokenizationServiceProxy()
