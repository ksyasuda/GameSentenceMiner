import json
import os
import platform
import queue
import shlex
import shutil
import subprocess
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
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
DEFAULT_BRIDGE_IDLE_TIMEOUT_SECONDS = 5 * 60


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


def _extract_tokens_from_yomitan_payload(payload: Any) -> List[Dict[str, str]]:
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

    indexed = [
        entry for entry in payload
        if isinstance(entry, dict)
        and int(entry.get("index", -1)) == 0
        and isinstance(entry.get("content"), list)
    ]
    candidates = indexed if indexed else [
        entry for entry in payload
        if isinstance(entry, dict) and isinstance(entry.get("content"), list)
    ]
    if not candidates:
        return []

    selected = max(candidates, key=lambda item: len(item.get("content") or []))
    content = selected.get("content") or []

    if content and all(_is_unparsed_content_group(group) for group in content):
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


class OverlayTokenizerBridgeClient:
    def __init__(self) -> None:
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._response_lock = threading.Lock()
        self._pending_responses: Dict[int, "queue.Queue[Dict[str, Any]]"] = {}
        self._request_id = 0
        self._idle_timer: Optional[threading.Timer] = None
        self._stdout_pump_thread: Optional[threading.Thread] = None
        self._stderr_pump_thread: Optional[threading.Thread] = None
        self.idle_timeout_seconds = int(
            os.environ.get(
                "GSM_TOKENIZER_BRIDGE_IDLE_TIMEOUT_SEC",
                str(DEFAULT_BRIDGE_IDLE_TIMEOUT_SECONDS),
            )
        )

    @staticmethod
    def _resolve_bridge_user_data_dir() -> str:
        bridge_data_dir = str(os.environ.get("GSM_TOKENIZER_BRIDGE_USER_DATA_DIR", "")).strip()
        if bridge_data_dir:
            return bridge_data_dir

        if platform.system().lower().startswith("win") and os.environ.get("APPDATA"):
            base_dir = Path(os.environ["APPDATA"])
        else:
            base_dir = Path.home() / ".config"
        return str(base_dir / "GameSentenceMiner" / "gsm_overlay_tokenizer_bridge")

    @staticmethod
    def _build_bridge_env() -> Dict[str, str]:
        env = dict(os.environ)
        env["GSM_TOKENIZER_BRIDGE_USER_DATA_DIR"] = OverlayTokenizerBridgeClient._resolve_bridge_user_data_dir()
        env["GSM_TOKENIZER_BRIDGE_MODE"] = "1"
        return env

    @staticmethod
    def _resolve_overlay_exec_name() -> str:
        return "gsm_overlay.exe" if platform.system().lower().startswith("win") else "gsm_overlay"

    @staticmethod
    def _resolve_overlay_platform_tag() -> str:
        sys_platform = platform.system().lower()
        if sys_platform.startswith("win"):
            return "win32"
        if sys_platform.startswith("darwin"):
            return "darwin"
        return "linux"

    @staticmethod
    def _resolve_overlay_arch_tag() -> str:
        machine = platform.machine().lower()
        if machine in {"x86_64", "amd64"}:
            return "x64"
        if machine in {"arm64", "aarch64"}:
            return "arm64"
        return machine or "x64"

    def _resolve_packaged_overlay_bridge_command(
        self,
        repo_root: Path,
    ) -> Optional[Tuple[List[str], str]]:
        overlay_exec_path = str(os.environ.get("GSM_OVERLAY_EXEC_PATH", "")).strip()
        if overlay_exec_path and Path(overlay_exec_path).exists():
            return [overlay_exec_path, "--bridge"], str(Path(overlay_exec_path).parent)

        platform_tag = self._resolve_overlay_platform_tag()
        arch_tag = self._resolve_overlay_arch_tag()
        exec_name = self._resolve_overlay_exec_name()
        candidate = (
            repo_root
            / "GSM_Overlay"
            / "out"
            / f"gsm_overlay-{platform_tag}-{arch_tag}"
            / exec_name
        )
        if candidate.exists():
            return [str(candidate), "--bridge"], str(candidate.parent)
        return None

    def _resolve_command(self) -> Tuple[List[str], str]:
        cmd_env = str(os.environ.get("GSM_TOKENIZER_BRIDGE_CMD", "")).strip()
        if cmd_env:
            cmd = shlex.split(cmd_env)
            if not cmd:
                raise RuntimeError("GSM_TOKENIZER_BRIDGE_CMD is set but empty")
            resolved_cmd = cmd[0]
            resolved_cmd_path = shutil.which(resolved_cmd) or str(Path(resolved_cmd).resolve()) if Path(resolved_cmd).exists() else None
            if not resolved_cmd_path:
                raise RuntimeError(
                    f"GSM_TOKENIZER_BRIDGE_CMD points to a missing executable: {resolved_cmd}"
                )
            return cmd, str(Path.cwd())

        bridge_bin_env = str(os.environ.get("GSM_TOKENIZER_BRIDGE_BIN", "")).strip()
        if bridge_bin_env:
            bridge_bin_path = Path(bridge_bin_env)
            if bridge_bin_path.exists():
                return [str(bridge_bin_path), "--bridge"], str(
                    bridge_bin_path.parent if bridge_bin_path.parent.exists() else Path.cwd()
                )
            resolved_bridge_bin = shutil.which(str(bridge_bin_env))
            if not resolved_bridge_bin:
                raise RuntimeError(
                    f"GSM_TOKENIZER_BRIDGE_BIN points to a missing executable: {bridge_bin_env}"
                )
            return [resolved_bridge_bin, "--bridge"], str(Path.cwd())

        electron_bin = str(os.environ.get("GSM_OVERLAY_ELECTRON_BIN", "electron")).strip()
        repo_root = Path(__file__).resolve().parents[2]
        overlay_dir = repo_root / "GSM_Overlay"
        resolved_electron_bin = shutil.which(electron_bin) or (
            str(Path(electron_bin).resolve()) if Path(electron_bin).exists() else None
        )
        if resolved_electron_bin:
            return [resolved_electron_bin, ".", "--bridge"], str(overlay_dir)

        packaged_command = self._resolve_packaged_overlay_bridge_command(repo_root)
        if packaged_command:
            return packaged_command
        raise RuntimeError(
            f"Could not resolve tokenizer bridge command. Set GSM_TOKENIZER_BRIDGE_CMD or GSM_TOKENIZER_BRIDGE_BIN. "
            f"Checked electron candidates: {electron_bin}"
        )

    def _start_stderr_pump(self) -> None:
        if self._process is None or self._process.stderr is None:
            return
        if self._stderr_pump_thread is not None and self._stderr_pump_thread.is_alive():
            return

        proc = self._process
        stderr = proc.stderr

        def _pump() -> None:
            try:
                for raw_line in iter(stderr.readline, ""):
                    if not raw_line:
                        break
                    line = raw_line.rstrip()
                    if line:
                        logger.info(f"[TokenizerBridge] {line}")
            except Exception:
                pass

        self._stderr_pump_thread = threading.Thread(
            target=_pump,
            name="gsm-tokenizer-bridge-stderr",
            daemon=True,
        )
        self._stderr_pump_thread.start()

    def _resolve_waiter(self, request_id: int) -> Optional["queue.Queue[Dict[str, Any]]"]:
        with self._response_lock:
            return self._pending_responses.get(request_id)

    def _register_waiter(self, request_id: int) -> "queue.Queue[Dict[str, Any]]":
        waiter: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=1)
        with self._response_lock:
            self._pending_responses[request_id] = waiter
        return waiter

    def _unregister_waiter(self, request_id: int) -> None:
        with self._response_lock:
            self._pending_responses.pop(request_id, None)

    def _fail_all_pending(self, error_message: str) -> None:
        with self._response_lock:
            pending = list(self._pending_responses.items())
            self._pending_responses.clear()
        for _request_id, waiter in pending:
            try:
                waiter.put_nowait({"ok": False, "error": error_message})
            except Exception:
                pass

    def _start_stdout_pump(self) -> None:
        if self._process is None or self._process.stdout is None:
            return
        if self._stdout_pump_thread is not None and self._stdout_pump_thread.is_alive():
            return

        proc = self._process
        stdout = proc.stdout

        def _pump() -> None:
            error_message = "Bridge process closed stdout unexpectedly"
            try:
                for raw_line in iter(stdout.readline, ""):
                    if not raw_line:
                        break
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line)
                    except Exception:
                        logger.warning(f"Tokenizer bridge returned invalid JSON: {line[:200]}")
                        continue
                    try:
                        request_id = int(parsed.get("id", -1))
                    except Exception:
                        continue
                    waiter = self._resolve_waiter(request_id)
                    if waiter is None:
                        continue
                    try:
                        waiter.put_nowait(parsed)
                    except Exception:
                        pass
            except Exception as exc:
                error_message = f"Bridge stdout reader failed: {exc}"
            finally:
                self._fail_all_pending(error_message)

        self._stdout_pump_thread = threading.Thread(
            target=_pump,
            name="gsm-tokenizer-bridge-stdout",
            daemon=True,
        )
        self._stdout_pump_thread.start()

    def _cancel_idle_timer(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _schedule_idle_shutdown(self) -> None:
        self._cancel_idle_timer()
        if self.idle_timeout_seconds <= 0:
            return

        self._idle_timer = threading.Timer(self.idle_timeout_seconds, self.shutdown)
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def _ensure_started(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return

        cmd, cwd = self._resolve_command()
        try:
            self._process = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=self._build_bridge_env(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            logger.warning(f"Failed to launch tokenizer bridge command={cmd} cwd={cwd}: {exc}")
            raise
        self._start_stdout_pump()
        self._start_stderr_pump()

    def _invoke(self, method: str, params: Dict[str, Any], timeout: float) -> Any:
        self._ensure_started()
        assert self._process is not None and self._process.stdin is not None

        self._request_id += 1
        request_id = self._request_id
        waiter = self._register_waiter(request_id)
        payload = {"id": request_id, "method": method, "params": params}
        try:
            self._process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._process.stdin.flush()
            try:
                parsed = waiter.get(timeout=max(1.0, timeout))
            except queue.Empty as exc:
                raise TimeoutError("Timed out waiting for bridge response") from exc
            if not parsed.get("ok", False):
                raise RuntimeError(str(parsed.get("error", "Unknown bridge error")))
            return parsed.get("result")
        finally:
            self._unregister_waiter(request_id)

    def is_available(self) -> bool:
        with self._lock:
            try:
                result = self._invoke("ping", {}, timeout=5.0)
                return bool(result and result.get("ready", False))
            except Exception:
                return False

    def tokenize_batch_raw(
        self,
        texts: Sequence[str],
        scan_length: int,
        timeout: float,
    ) -> List[Optional[Any]]:
        with self._lock:
            result = self._invoke(
                "tokenize_batch",
                {
                    "texts": [str(x or "") for x in texts],
                    "scanLength": int(scan_length),
                },
                timeout=max(10.0, float(timeout)),
            )
            self._schedule_idle_shutdown()
            if not isinstance(result, list):
                raise RuntimeError("Bridge returned invalid tokenize_batch result")
            return [item if item is not None else None for item in result]

    def shutdown(self) -> None:
        with self._lock:
            self._cancel_idle_timer()
            proc = self._process
            self._process = None
            self._stdout_pump_thread = None
            self._stderr_pump_thread = None
            if proc is None:
                return
            self._fail_all_pending("Bridge shutdown")
            try:
                if proc.stdin is not None and proc.poll() is None:
                    proc.stdin.write(json.dumps({"id": -1, "method": "shutdown", "params": {}}) + "\n")
                    proc.stdin.flush()
            except Exception:
                pass
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=3.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def restart(self) -> None:
        self.shutdown()


class TokenizationService:
    def __init__(self) -> None:
        self.api_base_url = str(
            os.environ.get("GSM_TOKENIZER_API_URL", "http://127.0.0.1:19633")
        ).strip().rstrip("/")
        self.request_timeout = float(os.environ.get("GSM_TOKENIZER_TIMEOUT_SEC", "30"))
        self.scan_length = int(os.environ.get("GSM_TOKENIZER_SCAN_LENGTH", "10"))
        self.chunk_size = max(1, int(os.environ.get("GSM_TOKENIZER_CHUNK_SIZE", "64")))
        self.enabled = os.environ.get("GSM_TOKENIZER_ENABLED", "1") != "0"
        self.backend = str(os.environ.get("GSM_TOKENIZER_BACKEND", "auto")).strip().lower()
        self.allow_bridge_fallback = os.environ.get(
            "GSM_TOKENIZER_ALLOW_BRIDGE_FALLBACK",
            "1",
        ) == "1"
        self._bridge_client: Optional[OverlayTokenizerBridgeClient] = None
        self._backfill_session_active = False
        self._backfill_backend_choice: Optional[str] = None

    def _use_bridge_backend(self) -> bool:
        return self.backend in {"bridge", "overlay-bridge"}

    def _use_http_backend(self) -> bool:
        return self.backend in {"http", "yomitan-api"}

    def _use_auto_backend(self) -> bool:
        return self.backend in {"auto", "hybrid"}

    def _get_bridge_client(self) -> OverlayTokenizerBridgeClient:
        if self._bridge_client is None:
            self._bridge_client = OverlayTokenizerBridgeClient()
        return self._bridge_client

    def begin_backfill_session(self) -> None:
        self._backfill_session_active = True
        self._backfill_backend_choice = None
        if self._bridge_client is not None:
            self._get_bridge_client()._cancel_idle_timer()

    def end_backfill_session(self) -> None:
        self._backfill_session_active = False
        self._backfill_backend_choice = None
        if self._bridge_client is not None:
            self._bridge_client.shutdown()

    def is_backfill_session_active(self) -> bool:
        return self._backfill_session_active

    def _select_backend(self, source: str, timeout: float = 0.35) -> str:
        source_kind = "backfill" if source == "backfill" else "realtime"
        if self._use_bridge_backend():
            return "bridge"
        if self._use_http_backend():
            return "http"
        if not self._use_auto_backend():
            return "none"

        if source_kind == "backfill":
            if self._backfill_backend_choice in {"http", "bridge"}:
                return self._backfill_backend_choice
            if self._is_http_tokenizer_available(timeout):
                self._backfill_backend_choice = "http"
                return "http"
            if self.allow_bridge_fallback:
                self._backfill_backend_choice = "bridge"
                return "bridge"
            self._backfill_backend_choice = "none"
            return "none"

        if self._is_http_tokenizer_available(timeout):
            return "http"
        if self.allow_bridge_fallback:
            return "bridge"
        return "none"

    def _is_http_tokenizer_available(self, timeout: float = 0.5) -> bool:
        health_url = f"{self.api_base_url}/health"
        tokenize_url = f"{self.api_base_url}/tokenize"
        try:
            response = requests.get(health_url, timeout=max(0.1, timeout))
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
        if backend == "bridge":
            return self._get_bridge_client().is_available()
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

    def _tokenize_texts(self, texts: Sequence[str], source: str = "realtime") -> List[Optional[List[Dict[str, str]]]]:
        backend = self._select_backend(source=source, timeout=0.35)

        if backend == "bridge":
            bridge_client = self._get_bridge_client()
            bridge_timeout = max(10.0, self.request_timeout * max(1, len(texts)))
            for attempt in range(2):
                try:
                    raw_payloads = bridge_client.tokenize_batch_raw(
                        texts=texts,
                        scan_length=self.scan_length,
                        timeout=bridge_timeout,
                    )
                    results: List[Optional[List[Dict[str, str]]]] = []
                    for text, payload in zip(texts, raw_payloads):
                        if payload is None:
                            results.append(None)
                        else:
                            tokenized = _extract_tokens_from_yomitan_payload(payload)
                            results.append(tokenized)
                    while len(results) < len(texts):
                        results.append(None)
                    return results[: len(texts)]
                except Exception as e:
                    logger.warning(f"Tokenizer bridge request failed: {e}")
                    if attempt == 0:
                        bridge_client.restart()
                        continue
                    return [None for _ in texts]
        elif backend == "none":
            return [None for _ in texts]

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
        GameLinesTable._db.execute(
            f"UPDATE {GameLinesTable._table} SET {', '.join(assignments)} WHERE id=?",
            tuple(params),
            commit=True,
        )

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
    ) -> Tuple[int, int]:
        if not entries:
            return (0, 0)
        try:
            self._persist_lines_tokenization_batch(entries)
            return (len(entries), 0)
        except Exception as e:
            logger.warning(
                f"Batch tokenization persistence failed for {len(entries)} lines: {e}. Falling back to per-line persistence."
            )
            processed = 0
            failed = 0
            for line_id, line_text, timestamp, game_id, tokens in entries:
                try:
                    self._persist_line_tokenization(line_id, line_text, timestamp, game_id, tokens)
                    processed += 1
                except Exception as inner_exc:
                    logger.warning(f"Tokenization persistence failed for line {line_id}: {inner_exc}")
                    self._set_line_tokenized_status(line_id, TOKENIZED_RETRYABLE_FAILED)
                    failed += 1
            return (processed, failed)

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

        prefiltered_entries: List[Tuple[str, str, float, str, Sequence[Dict[str, Any]]]] = []
        for idx in prefiltered_success_indices:
            line_id, line_text_original, _line_text_tokenize, timestamp, game_id = normalized_rows[idx]
            prefiltered_entries.append((line_id, line_text_original, timestamp, game_id, []))
        batch_processed, batch_failed = self._persist_entries_with_fallback(prefiltered_entries)
        processed += batch_processed
        failed += batch_failed

        unique_texts = list(text_to_indices.keys())
        for start in range(0, len(unique_texts), self.chunk_size):
            text_chunk = unique_texts[start:start + self.chunk_size]
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
            batch_processed, batch_failed = self._persist_entries_with_fallback(chunk_entries)
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


def enqueue_realtime_tokenization(
    game_line_id: str,
    line_text: str,
    timestamp: float,
    game_id: str = "",
) -> None:
    if not game_line_id:
        return
    GameLinesTable._db.execute(
        f"UPDATE {GameLinesTable._table} SET tokenized=? WHERE id=?",
        (TOKENIZED_PENDING, game_line_id),
        commit=True,
    )
    _ensure_realtime_worker_started()
    _realtime_queue.put((game_line_id, str(line_text or ""), float(timestamp or time.time()), game_id or ""))
