import os
import queue
import threading
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import regex
import requests

from GameSentenceMiner.util.config.configuration import get_stats_config, logger
from GameSentenceMiner.util.database.db import (
    GameLinesTable,
    KanjiOccurrencesTable,
    KanjiTable,
    TOKENIZED_DONE,
    TOKENIZED_PENDING,
    TOKENIZED_RETRYABLE_FAILED,
    WordOccurrencesTable,
    WordsTable,
    punctuation_regex,
    repeating_chars_regex,
)


_JAPANESE_TEXT_REGEX = regex.compile(r"[\p{Script=Hiragana}\p{Script=Katakana}\p{Han}]")
_KANJI_REGEX = regex.compile(r"\p{Han}")
_HIRAGANA_REGEX = regex.compile(r"\p{Script=Hiragana}")
_SINGLE_UNPARSED_TOKEN_FALLBACK_MAX_CHARS = 12


def _is_database_locked_error(exc: Exception) -> bool:
    return "database is locked" in str(exc).lower()


def _normalize_reading(value: Any) -> str:
    reading = str(value or "").strip()
    return reading if reading else "-"


def _normalize_word_key(token: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
    word = str(token.get("word") or token.get("text") or "").strip()
    if not word:
        return None
    headword = str(token.get("headword") or word).strip() or word
    reading = _normalize_reading(token.get("reading"))
    return (headword, word, reading)


def _extract_headword_from_group(group: Sequence[Dict[str, Any]]) -> Optional[str]:
    if not group:
        return None
    first = group[0] if isinstance(group[0], dict) else {}
    headwords = first.get("headwords")
    if not isinstance(headwords, list):
        return None
    for entry in headwords:
        if isinstance(entry, list):
            for item in entry:
                if isinstance(item, dict):
                    term = str(item.get("term") or "").strip()
                    if term:
                        return term
        elif isinstance(entry, dict):
            term = str(entry.get("term") or "").strip()
            if term:
                return term
    return None


def _is_unparsed_content_group(group: Sequence[Any]) -> bool:
    if not group or not isinstance(group, list):
        return False
    for idx, segment in enumerate(group):
        if not isinstance(segment, dict):
            return False
        text = str(segment.get("text") or "").strip()
        if not text and idx == 0:
            return False
        if text and str(segment.get("reading") or "").strip():
            return False
        if segment.get("headwords"):
            return False
    return True


def _is_token_like_unparsed_word(word: str) -> bool:
    if not word:
        return False
    if len(word) > _SINGLE_UNPARSED_TOKEN_FALLBACK_MAX_CHARS:
        return False
    if any(ch.isspace() for ch in word):
        return False
    if _HIRAGANA_REGEX.search(word):
        return False
    return any(ch.isalnum() for ch in word)


def _extract_tokens_from_yomitan_payload(payload: Any, target_index: Optional[int] = None) -> List[Dict[str, str]]:
    if isinstance(payload, dict) and isinstance(payload.get("tokens"), list):
        tokens: List[Dict[str, str]] = []
        for entry in payload.get("tokens") or []:
            if isinstance(entry, dict):
                key = _normalize_word_key(entry)
                if key:
                    tokens.append({"headword": key[0], "word": key[1], "reading": key[2]})
        return tokens

    if not isinstance(payload, list) or not payload:
        return []

    if target_index is None:
        target_index = 0

    indexed = [
        entry for entry in payload
        if isinstance(entry, dict)
        and int(entry.get("index", -1)) == target_index
        and isinstance(entry.get("content"), list)
    ]
    candidates = indexed if indexed else [
        entry
        for entry in payload
        if isinstance(entry, dict)
        and "index" not in entry
        and isinstance(entry.get("content"), list)
    ]
    if not candidates:
        return []

    selected = max(candidates, key=lambda item: len(item.get("content") or []))
    content = selected.get("content") or []

    if content and all(_is_unparsed_content_group(group) for group in content):
        if len(content) > 1:
            tokens: List[Dict[str, str]] = []
            for group in content:
                if not isinstance(group, list):
                    continue
                word = "".join(str(seg.get("text") or "") for seg in group if isinstance(seg, dict)).strip()
                if not word:
                    continue
                tokens.append({"headword": word, "word": word, "reading": "-"})
            if tokens:
                return tokens
        if len(content) == 1:
            group = content[0]
            if isinstance(group, list):
                word = "".join(str(seg.get("text") or "") for seg in group if isinstance(seg, dict)).strip()
                if _is_token_like_unparsed_word(word):
                    return [{"headword": word, "word": word, "reading": "-"}]
        return []

    tokens: List[Dict[str, str]] = []
    for group in content:
        if not isinstance(group, list):
            if isinstance(group, dict):
                group = [group]
            else:
                continue
        if not isinstance(group, list):
            continue
        word = "".join(str(seg.get("text") or "") for seg in group if isinstance(seg, dict)).strip()
        if not word:
            continue
        reading = "".join(
            str(seg.get("reading") or "") for seg in group if isinstance(seg, dict)
        ).strip()
        tokens.append(
            {
                "headword": _extract_headword_from_group(group) or word,
                "word": word,
                "reading": _normalize_reading(reading),
            }
        )
    return tokens


def _prepare_text_for_tokenization(text: str) -> str:
    cleaned = punctuation_regex.sub("", text or "").strip()
    if get_stats_config().regex_out_repetitions:
        cleaned = repeating_chars_regex.sub(r"\1\1\1", cleaned)
    return cleaned


class TokenizationService:
    def __init__(self) -> None:
        self.api_base_url = str(
            os.environ.get("GSM_TOKENIZER_API_URL", "http://127.0.0.1:19633")
        ).strip().rstrip("/")
        self.request_timeout = float(os.environ.get("GSM_TOKENIZER_TIMEOUT_SEC", "30"))
        self.scan_length = int(os.environ.get("GSM_TOKENIZER_SCAN_LENGTH", "10"))
        self.chunk_size = max(1, int(os.environ.get("GSM_TOKENIZER_CHUNK_SIZE", "64")))
        self.backfill_chunk_size = max(
            1,
            int(os.environ.get("GSM_TOKENIZER_BACKFILL_CHUNK_SIZE", "8")),
        )
        self.backfill_persist_chunk_size = max(
            1,
            int(os.environ.get("GSM_TOKENIZER_BACKFILL_PERSIST_CHUNK_SIZE", "32")),
        )
        self.backfill_priority_persist_chunk_size = max(
            1,
            int(os.environ.get("GSM_TOKENIZER_BACKFILL_PRIORITY_PERSIST_CHUNK_SIZE", "16")),
        )
        self.enabled = os.environ.get("GSM_TOKENIZER_ENABLED", "1") != "0"
        self.backend = str(os.environ.get("GSM_TOKENIZER_BACKEND", "auto")).strip().lower()
        self._backfill_session_active = False
        self._backfill_backend_choice: Optional[str] = None

    def _use_http_backend(self) -> bool:
        return self.backend in {"http", "yomitan-api"}

    def _use_auto_backend(self) -> bool:
        return self.backend in {"auto", "hybrid"}

    def begin_backfill_session(self) -> None:
        self._backfill_session_active = True
        self._backfill_backend_choice = None

    def end_backfill_session(self) -> None:
        self._backfill_session_active = False
        self._backfill_backend_choice = None

    def is_backfill_session_active(self) -> bool:
        return self._backfill_session_active

    def _select_backend(self, source: str, timeout: float = 0.35) -> str:
        source_kind = "backfill" if source == "backfill" else "realtime"
        if self._use_http_backend():
            return "http"
        if not self._use_auto_backend():
            return "none"

        if source_kind == "backfill":
            if self._backfill_backend_choice == "http":
                return self._backfill_backend_choice
            if self._is_http_tokenizer_available(timeout):
                self._backfill_backend_choice = "http"
                return "http"
            return "none"

        if self._is_http_tokenizer_available(timeout):
            return "http"
        return "none"

    def _is_http_tokenizer_available(self, timeout: float = 0.5) -> bool:
        health_url = f"{self.api_base_url}/health"
        server_version_url = f"{self.api_base_url}/serverVersion"
        tokenize_url = f"{self.api_base_url}/tokenize"
        try:
            response = requests.get(health_url, timeout=max(0.1, timeout))
            if response.ok:
                return True
        except Exception:
            pass
        try:
            response = requests.post(
                server_version_url,
                json={},
                timeout=max(0.1, timeout),
            )
            if response.ok:
                return True
        except Exception:
            pass
        try:
            response = requests.post(
                tokenize_url,
                json={"text": "。", "scanLength": 1},
                timeout=max(0.1, timeout),
            )
            return response.ok
        except Exception:
            return False

    def is_tokenizer_available(self, timeout: float = 0.5, source: str = "realtime") -> bool:
        if not self.enabled:
            return False
        backend = self._select_backend(source=source, timeout=timeout)
        if backend == "http":
            return self._is_http_tokenizer_available(timeout)
        return False

    def _tokenize_one(self, text: str) -> List[Dict[str, str]]:
        response = requests.post(
            f"{self.api_base_url}/tokenize",
            json={"text": text, "scanLength": self.scan_length},
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return _extract_tokens_from_yomitan_payload(payload)

    def _tokenize_many(self, texts: Sequence[str]) -> List[List[Dict[str, str]]]:
        response = requests.post(
            f"{self.api_base_url}/tokenize",
            json={"text": list(texts), "scanLength": self.scan_length},
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return [
            _extract_tokens_from_yomitan_payload(payload, target_index=index)
            for index in range(len(texts))
        ]

    def _tokenize_texts(self, texts: Sequence[str], source: str = "realtime") -> List[Optional[List[Dict[str, str]]]]:
        backend = self._select_backend(source=source, timeout=0.35)

        if backend == "none":
            return [None for _ in texts]

        if not texts:
            return []

        try:
            return self._tokenize_many(texts)
        except Exception as e:
            logger.warning(f"Tokenizer HTTP batch request failed: {e}. Falling back to per-text requests.")

        results: List[Optional[List[Dict[str, str]]]] = []
        for text in texts:
            try:
                tokenized = self._tokenize_one(text)
                results.append(tokenized)
            except Exception as e:
                logger.warning(f"Tokenizer HTTP request failed: {e}")
                results.append(None)
        return results

    @staticmethod
    def _contains_japanese(text: str) -> bool:
        return bool(_JAPANESE_TEXT_REGEX.search(text or ""))

    @staticmethod
    def _count_kanji(text: str) -> Counter:
        return Counter(_KANJI_REGEX.findall(text or ""))

    @staticmethod
    def _filtered_length(text: str) -> int:
        text = text or ""
        return len(punctuation_regex.sub("", text))

    def _set_line_tokenized_status(
        self,
        line_id: str,
        tokenized_status: int,
        total_length: Optional[int] = None,
        filtered_length: Optional[int] = None,
        word_count: Optional[int] = None,
        kanji_count: Optional[int] = None,
    ) -> None:
        assignments = ["tokenized=?"]
        params: List[Any] = [tokenized_status]
        if total_length is not None:
            assignments.append("total_length=?")
            params.append(total_length)
        if filtered_length is not None:
            assignments.append("filtered_length=?")
            params.append(filtered_length)
        if word_count is not None:
            assignments.append("word_count=?")
            params.append(word_count)
        if kanji_count is not None:
            assignments.append("kanji_count=?")
            params.append(kanji_count)
        params.append(line_id)
        max_attempts = 8
        retry_delay_seconds = 0.05
        max_retry_delay_seconds = 0.5
        for attempt in range(1, max_attempts + 1):
            try:
                GameLinesTable._db.execute(
                    f"UPDATE {GameLinesTable._table} SET {', '.join(assignments)} WHERE id=?",
                    tuple(params),
                    commit=True,
                )
                return
            except Exception as exc:
                if not _is_database_locked_error(exc) or attempt >= max_attempts:
                    raise
                time.sleep(retry_delay_seconds)
                retry_delay_seconds = min(retry_delay_seconds * 2, max_retry_delay_seconds)

    def _recompute_word_cache_rows(self, conn: Any, word_ids: Iterable[int]) -> None:
        for word_id in {int(x) for x in word_ids if x is not None}:
            conn.execute(
                f"""
                UPDATE {WordsTable._table}
                SET
                    frequency=COALESCE((SELECT SUM(COALESCE(count, 1)) FROM {WordOccurrencesTable._table} WHERE word_id=?), 0),
                    first_seen=COALESCE((SELECT MIN(timestamp) FROM {WordOccurrencesTable._table} WHERE word_id=?), first_seen),
                    last_seen=COALESCE((SELECT MAX(timestamp) FROM {WordOccurrencesTable._table} WHERE word_id=?), last_seen)
                WHERE id=?
                """,
                (word_id, word_id, word_id, word_id),
            )
            conn.execute(
                f"DELETE FROM {WordsTable._table} WHERE id=? AND COALESCE(frequency, 0) <= 0",
                (word_id,),
            )

    def _recompute_kanji_cache_rows(self, conn: Any, kanji_ids: Iterable[int]) -> None:
        for kanji_id in {int(x) for x in kanji_ids if x is not None}:
            conn.execute(
                f"""
                UPDATE {KanjiTable._table}
                SET
                    frequency=COALESCE((SELECT SUM(COALESCE(count, 1)) FROM {KanjiOccurrencesTable._table} WHERE kanji_id=?), 0),
                    first_seen=COALESCE((SELECT MIN(timestamp) FROM {KanjiOccurrencesTable._table} WHERE kanji_id=?), first_seen),
                    last_seen=COALESCE((SELECT MAX(timestamp) FROM {KanjiOccurrencesTable._table} WHERE kanji_id=?), last_seen)
                WHERE id=?
                """,
                (kanji_id, kanji_id, kanji_id, kanji_id),
            )
            conn.execute(
                f"DELETE FROM {KanjiTable._table} WHERE id=? AND COALESCE(frequency, 0) <= 0",
                (kanji_id,),
            )

    @staticmethod
    def _chunked(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
        for i in range(0, len(seq), size):
            yield seq[i:i + size]

    def _persist_lines_tokenization_batch(
        self,
        entries: Sequence[Tuple[str, str, float, str, Sequence[Dict[str, Any]]]],
    ) -> None:
        if not entries:
            return

        normalized: List[Tuple[str, str, float, str, Counter, Counter]] = []
        for line_id, line_text, timestamp, game_id, tokens in entries:
            word_counter: Counter = Counter()
            for token in tokens:
                key = _normalize_word_key(token if isinstance(token, dict) else {})
                if key:
                    word_counter[key] += 1
            kanji_counter = self._count_kanji(line_text)
            normalized.append((line_id, line_text, timestamp, game_id, word_counter, kanji_counter))

        line_ids = [line_id for line_id, _t, _ts, _g, _wc, _kc in normalized if line_id]
        if not line_ids:
            return

        placeholders = ", ".join("?" for _ in line_ids)
        with GameLinesTable._db.transaction() as conn:
            old_word_rows = conn.execute(
                f"SELECT DISTINCT word_id FROM {WordOccurrencesTable._table} WHERE line_id IN ({placeholders})",
                tuple(line_ids),
            ).fetchall()
            old_kanji_rows = conn.execute(
                f"SELECT DISTINCT kanji_id FROM {KanjiOccurrencesTable._table} WHERE line_id IN ({placeholders})",
                tuple(line_ids),
            ).fetchall()

            conn.execute(
                f"DELETE FROM {WordOccurrencesTable._table} WHERE line_id IN ({placeholders})",
                tuple(line_ids),
            )
            conn.execute(
                f"DELETE FROM {KanjiOccurrencesTable._table} WHERE line_id IN ({placeholders})",
                tuple(line_ids),
            )

            impacted_word_ids = {int(row[0]) for row in old_word_rows}
            impacted_kanji_ids = {int(row[0]) for row in old_kanji_rows}

            word_bounds: Dict[Tuple[str, str, str], Tuple[float, float]] = {}
            word_occurrence_rows: List[Tuple[Tuple[str, str, str], str, str, float, int]] = []
            for line_id, _line_text, timestamp, game_id, word_counter, _kanji_counter in normalized:
                for key, count in word_counter.items():
                    first_ts, last_ts = word_bounds.get(key, (timestamp, timestamp))
                    word_bounds[key] = (min(first_ts, timestamp), max(last_ts, timestamp))
                    word_occurrence_rows.append((key, line_id, game_id or None, timestamp, int(count)))

            word_id_by_key: Dict[Tuple[str, str, str], int] = {}
            if word_bounds:
                upsert_rows = [
                    (headword, word, reading, first_seen, last_seen)
                    for (headword, word, reading), (first_seen, last_seen) in word_bounds.items()
                ]
                conn.executemany(
                    f"""
                    INSERT INTO {WordsTable._table} (headword, word, reading, first_seen, last_seen, frequency)
                    VALUES (?, ?, ?, ?, ?, 0)
                    ON CONFLICT(headword, word, reading) DO UPDATE SET
                        first_seen=MIN(first_seen, excluded.first_seen),
                        last_seen=MAX(last_seen, excluded.last_seen)
                    """,
                    upsert_rows,
                )

                word_keys = list(word_bounds.keys())
                for key_chunk in self._chunked(word_keys, 250):
                    key_placeholders = ", ".join("(?, ?, ?)" for _ in key_chunk)
                    key_params: List[Any] = []
                    for key in key_chunk:
                        key_params.extend(key)
                    rows = conn.execute(
                        f"""
                        SELECT id, headword, word, reading
                        FROM {WordsTable._table}
                        WHERE (headword, word, reading) IN ({key_placeholders})
                        """,
                        tuple(key_params),
                    ).fetchall()
                    for row in rows:
                        word_id_by_key[(str(row[1]), str(row[2]), str(row[3]))] = int(row[0])

                word_insert_rows: List[Tuple[int, str, str, float, int]] = []
                for key, line_id, game_id, timestamp, count in word_occurrence_rows:
                    word_id = word_id_by_key.get(key)
                    if word_id is None:
                        continue
                    impacted_word_ids.add(word_id)
                    word_insert_rows.append((word_id, line_id, game_id, timestamp, count))
                if word_insert_rows:
                    conn.executemany(
                        f"""
                        INSERT INTO {WordOccurrencesTable._table} (word_id, line_id, game_id, timestamp, count)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        word_insert_rows,
                    )

            kanji_bounds: Dict[str, Tuple[float, float]] = {}
            kanji_occurrence_rows: List[Tuple[str, str, str, float, int]] = []
            for line_id, _line_text, timestamp, game_id, _word_counter, kanji_counter in normalized:
                for kanji_char, count in kanji_counter.items():
                    first_ts, last_ts = kanji_bounds.get(kanji_char, (timestamp, timestamp))
                    kanji_bounds[kanji_char] = (min(first_ts, timestamp), max(last_ts, timestamp))
                    kanji_occurrence_rows.append((kanji_char, line_id, game_id or None, timestamp, int(count)))

            kanji_id_by_char: Dict[str, int] = {}
            if kanji_bounds:
                upsert_rows = [
                    (kanji_char, first_seen, last_seen)
                    for kanji_char, (first_seen, last_seen) in kanji_bounds.items()
                ]
                conn.executemany(
                    f"""
                    INSERT INTO {KanjiTable._table} (kanji, first_seen, last_seen, frequency)
                    VALUES (?, ?, ?, 0)
                    ON CONFLICT(kanji) DO UPDATE SET
                        first_seen=MIN(first_seen, excluded.first_seen),
                        last_seen=MAX(last_seen, excluded.last_seen)
                    """,
                    upsert_rows,
                )

                kanji_chars = list(kanji_bounds.keys())
                for char_chunk in self._chunked(kanji_chars, 900):
                    char_placeholders = ", ".join("?" for _ in char_chunk)
                    rows = conn.execute(
                        f"SELECT id, kanji FROM {KanjiTable._table} WHERE kanji IN ({char_placeholders})",
                        tuple(char_chunk),
                    ).fetchall()
                    for row in rows:
                        kanji_id_by_char[str(row[1])] = int(row[0])

                kanji_insert_rows: List[Tuple[int, str, str, float, int]] = []
                for kanji_char, line_id, game_id, timestamp, count in kanji_occurrence_rows:
                    kanji_id = kanji_id_by_char.get(kanji_char)
                    if kanji_id is None:
                        continue
                    impacted_kanji_ids.add(kanji_id)
                    kanji_insert_rows.append((kanji_id, line_id, game_id, timestamp, count))
                if kanji_insert_rows:
                    conn.executemany(
                        f"""
                        INSERT INTO {KanjiOccurrencesTable._table} (kanji_id, line_id, game_id, timestamp, count)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        kanji_insert_rows,
                    )

            self._recompute_word_cache_rows(conn, impacted_word_ids)
            self._recompute_kanji_cache_rows(conn, impacted_kanji_ids)

            game_line_update_rows: List[Tuple[int, int, int, int, str]] = []
            for line_id, line_text, _timestamp, _game_id, word_counter, kanji_counter in normalized:
                game_line_update_rows.append(
                    (
                        TOKENIZED_DONE,
                        len(line_text or ""),
                        self._filtered_length(line_text),
                        int(sum(word_counter.values())),
                        int(sum(kanji_counter.values())),
                        line_id,
                    )
                )
            conn.executemany(
                f"""
                UPDATE {GameLinesTable._table}
                SET tokenized=?, total_length=?, filtered_length=?, word_count=?, kanji_count=?
                WHERE id=?
                """,
                game_line_update_rows,
            )

    def _persist_line_tokenization(
        self,
        line_id: str,
        line_text: str,
        timestamp: float,
        game_id: str,
        tokens: Sequence[Dict[str, Any]],
    ) -> None:
        self._persist_lines_tokenization_batch(
            [(line_id, line_text, timestamp, game_id, tokens)]
        )

    def _persist_entries_with_fallback(
        self,
        entries: Sequence[Tuple[str, str, float, str, Sequence[Dict[str, Any]]]],
        source: str = "realtime",
    ) -> Tuple[int, int]:
        if not entries:
            return (0, 0)
        source_kind = "backfill" if source == "backfill" else "realtime"
        batch_attempts = 8 if source_kind == "realtime" else 12
        retry_delay_seconds = 0.03 if source_kind == "realtime" else 0.08
        max_retry_delay_seconds = 0.5

        last_exc: Optional[Exception] = None
        for attempt in range(1, batch_attempts + 1):
            try:
                self._persist_lines_tokenization_batch(entries)
                return (len(entries), 0)
            except Exception as exc:
                last_exc = exc
                if not _is_database_locked_error(exc) or attempt >= batch_attempts:
                    break
                if source_kind == "backfill" and has_pending_realtime_work():
                    time.sleep(max(retry_delay_seconds, 0.1))
                else:
                    time.sleep(retry_delay_seconds)
                retry_delay_seconds = min(retry_delay_seconds * 2, max_retry_delay_seconds)

        logger.warning(
            f"Batch tokenization persistence failed for {len(entries)} lines: {last_exc}. Falling back to per-line persistence."
        )
        processed = 0
        failed = 0
        for line_id, line_text, timestamp, game_id, tokens in entries:
            per_line_attempts = 6 if source_kind == "realtime" else 10
            line_delay_seconds = 0.03 if source_kind == "realtime" else 0.08
            line_max_delay = 0.5
            line_persisted = False
            line_last_exc: Optional[Exception] = None
            for attempt in range(1, per_line_attempts + 1):
                try:
                    self._persist_line_tokenization(line_id, line_text, timestamp, game_id, tokens)
                    line_persisted = True
                    processed += 1
                    break
                except Exception as inner_exc:
                    line_last_exc = inner_exc
                    if not _is_database_locked_error(inner_exc) or attempt >= per_line_attempts:
                        break
                    if source_kind == "backfill" and has_pending_realtime_work():
                        time.sleep(max(line_delay_seconds, 0.1))
                    else:
                        time.sleep(line_delay_seconds)
                    line_delay_seconds = min(line_delay_seconds * 2, line_max_delay)
            if line_persisted:
                continue
            logger.warning(f"Tokenization persistence failed for line {line_id}: {line_last_exc}")
            try:
                self._set_line_tokenized_status(line_id, TOKENIZED_RETRYABLE_FAILED)
            except Exception as status_exc:
                logger.warning(
                    f"Failed to mark line {line_id} as retryable after persistence failure: {status_exc}"
                )
            failed += 1
        return (processed, failed)

    def _persist_entries_chunked(
        self,
        entries: Sequence[Tuple[str, str, float, str, Sequence[Dict[str, Any]]]],
        source: str = "realtime",
    ) -> Tuple[int, int]:
        if not entries:
            return (0, 0)
        source_kind = "backfill" if source == "backfill" else "realtime"
        total_processed = 0
        total_failed = 0
        if source_kind != "backfill":
            chunk_processed, chunk_failed = self._persist_entries_with_fallback(entries, source=source)
            total_processed += chunk_processed
            total_failed += chunk_failed
            return (total_processed, total_failed)

        entry_list = list(entries)
        index = 0
        while index < len(entry_list):
            pending_realtime = has_pending_realtime_work()
            chunk_size = (
                self.backfill_priority_persist_chunk_size
                if pending_realtime
                else self.backfill_persist_chunk_size
            )
            chunk_size = max(1, int(chunk_size))
            chunk = entry_list[index:index + chunk_size]
            chunk_processed, chunk_failed = self._persist_entries_with_fallback(chunk, source=source)
            total_processed += chunk_processed
            total_failed += chunk_failed
            index += len(chunk)

            if pending_realtime:
                time.sleep(0.06)

        return (total_processed, total_failed)

    def remove_lines_occurrences(self, line_ids: Sequence[str]) -> None:
        line_ids = [line_id for line_id in dict.fromkeys(line_ids) if line_id]
        if not line_ids:
            return
        placeholders = ", ".join("?" for _ in line_ids)
        with GameLinesTable._db.transaction() as conn:
            old_word_rows = conn.execute(
                f"SELECT DISTINCT word_id FROM {WordOccurrencesTable._table} WHERE line_id IN ({placeholders})",
                tuple(line_ids),
            ).fetchall()
            old_kanji_rows = conn.execute(
                f"SELECT DISTINCT kanji_id FROM {KanjiOccurrencesTable._table} WHERE line_id IN ({placeholders})",
                tuple(line_ids),
            ).fetchall()

            conn.execute(
                f"DELETE FROM {WordOccurrencesTable._table} WHERE line_id IN ({placeholders})",
                tuple(line_ids),
            )
            conn.execute(
                f"DELETE FROM {KanjiOccurrencesTable._table} WHERE line_id IN ({placeholders})",
                tuple(line_ids),
            )

            self._recompute_word_cache_rows(conn, [row[0] for row in old_word_rows])
            self._recompute_kanji_cache_rows(conn, [row[0] for row in old_kanji_rows])

    def remove_line_occurrences(self, line_id: str) -> None:
        if not line_id:
            return
        self.remove_lines_occurrences([line_id])

    def tokenize_lines_batch(self, rows: Sequence[Tuple[str, str, float, str]], source: str = "realtime") -> Dict[str, int]:
        if not rows:
            return {"processed": 0, "failed": 0}

        normalized_rows: List[Tuple[str, str, str, float, str]] = []
        for row in rows:
            line_id = str(row[0])
            line_text_original = str(row[1] or "")
            line_text_tokenize = _prepare_text_for_tokenization(line_text_original)
            timestamp = float(row[2] or time.time())
            game_id = str(row[3] or "")
            normalized_rows.append((line_id, line_text_original, line_text_tokenize, timestamp, game_id))

        text_to_indices: Dict[str, List[int]] = defaultdict(list)
        prefiltered_success_indices: List[int] = []
        for idx, (_line_id, _line_text_original, line_text_tokenize, _ts, _game_id) in enumerate(normalized_rows):
            if not self._contains_japanese(line_text_tokenize):
                prefiltered_success_indices.append(idx)
            else:
                text_to_indices[line_text_tokenize].append(idx)

        processed = 0
        failed = 0
        source_kind = "backfill" if source == "backfill" else "realtime"

        prefiltered_entries: List[Tuple[str, str, float, str, Sequence[Dict[str, Any]]]] = []
        for idx in prefiltered_success_indices:
            line_id, line_text_original, _line_text_tokenize, timestamp, game_id = normalized_rows[idx]
            prefiltered_entries.append((line_id, line_text_original, timestamp, game_id, []))
        batch_processed, batch_failed = self._persist_entries_chunked(prefiltered_entries, source=source_kind)
        processed += batch_processed
        failed += batch_failed

        unique_texts = list(text_to_indices.keys())
        tokenize_chunk_size = self.backfill_chunk_size if source_kind == "backfill" else self.chunk_size
        for start in range(0, len(unique_texts), tokenize_chunk_size):
            text_chunk = unique_texts[start:start + tokenize_chunk_size]
            tokenized_chunk = self._tokenize_texts(text_chunk, source=source)
            chunk_entries: List[Tuple[str, str, float, str, Sequence[Dict[str, Any]]]] = []
            for text, tokenized in zip(text_chunk, tokenized_chunk):
                line_indices = text_to_indices.get(text, [])
                if tokenized is None:
                    for idx in line_indices:
                        line_id = normalized_rows[idx][0]
                        self._set_line_tokenized_status(line_id, TOKENIZED_RETRYABLE_FAILED)
                        failed += 1
                    continue
                for idx in line_indices:
                    line_id, line_text_original, _line_text_tokenize, timestamp, game_id = normalized_rows[idx]
                    chunk_entries.append((line_id, line_text_original, timestamp, game_id, tokenized))
            batch_processed, batch_failed = self._persist_entries_chunked(chunk_entries, source=source_kind)
            processed += batch_processed
            failed += batch_failed

        return {"processed": processed, "failed": failed}


_service_lock = threading.Lock()
_service_singleton: Optional[TokenizationService] = None

_realtime_queue: "queue.Queue[Tuple[str, str, float, str]]" = queue.Queue()
_realtime_thread: Optional[threading.Thread] = None


def get_tokenization_service() -> TokenizationService:
    global _service_singleton
    with _service_lock:
        if _service_singleton is None:
            _service_singleton = TokenizationService()
        return _service_singleton


def _realtime_worker() -> None:
    while True:
        item = _realtime_queue.get()
        if item is None:
            break
        line_id, line_text, timestamp, game_id = item
        try:
            service = get_tokenization_service()
            service.tokenize_lines_batch(
                [(line_id, line_text, timestamp, game_id)]
            )
        except Exception as e:
            logger.warning(f"Realtime tokenization worker failed for line {line_id}: {e}")
        finally:
            _realtime_queue.task_done()


def _ensure_realtime_worker_started() -> None:
    global _realtime_thread
    with _service_lock:
        if _realtime_thread and _realtime_thread.is_alive():
            return
        _realtime_thread = threading.Thread(
            target=_realtime_worker,
            name="gsm-tokenization-realtime",
            daemon=True,
        )
        _realtime_thread.start()


def has_pending_realtime_work() -> bool:
    try:
        return _realtime_queue.qsize() > 0
    except Exception:
        return not _realtime_queue.empty()


def _mark_line_pending_with_retry(game_line_id: str) -> None:
    max_attempts = 5
    retry_delay_seconds = 0.05
    max_retry_delay_seconds = 0.5
    for attempt in range(1, max_attempts + 1):
        try:
            GameLinesTable._db.execute(
                f"UPDATE {GameLinesTable._table} SET tokenized=? WHERE id=?",
                (TOKENIZED_PENDING, game_line_id),
                commit=True,
            )
            return
        except Exception as exc:
            if "database is locked" not in str(exc).lower() or attempt >= max_attempts:
                raise
            time.sleep(retry_delay_seconds)
            retry_delay_seconds = min(retry_delay_seconds * 2, max_retry_delay_seconds)


def enqueue_realtime_tokenization(
    game_line_id: str,
    line_text: str,
    timestamp: float,
    game_id: str = "",
) -> None:
    if not game_line_id:
        return
    _mark_line_pending_with_retry(game_line_id)
    _ensure_realtime_worker_started()
    _realtime_queue.put((game_line_id, str(line_text or ""), float(timestamp or time.time()), game_id or ""))
