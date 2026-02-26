import importlib.util
import queue
import re
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _stub_module(monkeypatch, name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_tokenization_module(monkeypatch):
    class _StatsConfig:
        regex_out_repetitions = False

    class _Logger:
        def warning(self, *_args, **_kwargs):
            return None

        def info(self, *_args, **_kwargs):
            return None

    class _Table:
        _table = "dummy"
        _db = None

    class _FakePattern:
        def __init__(self, pattern: str):
            self._pattern = pattern
            self._fallback = None
            try:
                self._fallback = re.compile(pattern)
            except Exception:
                self._fallback = None

        @staticmethod
        def _is_japanese_char(ch: str) -> bool:
            code = ord(ch)
            return (
                0x3040 <= code <= 0x309F  # Hiragana
                or 0x30A0 <= code <= 0x30FF  # Katakana
                or 0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs (Han)
            )

        @staticmethod
        def _is_han(ch: str) -> bool:
            code = ord(ch)
            return 0x4E00 <= code <= 0x9FFF

        def search(self, text: str):
            text = str(text or "")
            if "Script=Hiragana" in self._pattern or "Script=Katakana" in self._pattern:
                return True if any(self._is_japanese_char(ch) for ch in text) else None
            if self._fallback is not None:
                return self._fallback.search(text)
            return None

        def findall(self, text: str):
            text = str(text or "")
            if self._pattern == r"\p{Han}":
                return [ch for ch in text if self._is_han(ch)]
            if self._fallback is not None:
                return self._fallback.findall(text)
            return []

        def sub(self, repl: str, text: str):
            text = str(text or "")
            if self._fallback is not None:
                return self._fallback.sub(repl, text)
            return text

    class _FakeRegexModule:
        @staticmethod
        def compile(pattern: str):
            return _FakePattern(pattern)

    _stub_module(monkeypatch, "GameSentenceMiner")
    _stub_module(monkeypatch, "GameSentenceMiner.util")
    _stub_module(monkeypatch, "GameSentenceMiner.util.config")
    _stub_module(monkeypatch, "GameSentenceMiner.util.database")
    _stub_module(monkeypatch, "regex", compile=_FakeRegexModule.compile)
    _stub_module(
        monkeypatch,
        "GameSentenceMiner.util.config.configuration",
        logger=_Logger(),
        get_stats_config=lambda: _StatsConfig(),
    )
    _stub_module(
        monkeypatch,
        "GameSentenceMiner.util.database.db",
        GameLinesTable=_Table,
        KanjiOccurrencesTable=_Table,
        KanjiTable=_Table,
        TOKENIZED_DONE=1,
        TOKENIZED_PENDING=0,
        TOKENIZED_RETRYABLE_FAILED=2,
        WordOccurrencesTable=_Table,
        WordsTable=_Table,
        punctuation_regex=re.compile(r"[、！\s]"),
        repeating_chars_regex=re.compile(r"(.+?)\1{2,}"),
    )

    module_path = REPO_ROOT / "GameSentenceMiner/util/tokenization_service.py"
    spec = importlib.util.spec_from_file_location("tokenization_service_bridge_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sample_parse_payload(text: str):
    return [
        {
            "index": 0,
            "content": [
                [{"text": text, "reading": "ガッコウ", "headwords": [[{"term": text}]]}],
            ],
        }
    ]


def test_bridge_tokenize_texts_converts_parse_payload(monkeypatch):
    module = _load_tokenization_module(monkeypatch)
    TokenizationService = module.TokenizationService

    class _Client:
        def tokenize_batch_raw(self, texts, scan_length, timeout):
            assert texts == ["学校"]
            assert scan_length > 0
            assert timeout > 0
            return [_sample_parse_payload("学校")]

        def restart(self):
            raise AssertionError("restart should not be called")

    service = TokenizationService()
    service.backend = "bridge"
    monkeypatch.setattr(service, "_get_bridge_client", lambda: _Client())

    result = service._tokenize_texts(["学校"])
    assert len(result) == 1
    assert result[0] == [{"headword": "学校", "word": "学校", "reading": "ガッコウ"}]


def test_bridge_tokenize_texts_retries_once_after_restart(monkeypatch):
    module = _load_tokenization_module(monkeypatch)
    TokenizationService = module.TokenizationService

    class _Client:
        def __init__(self):
            self.calls = 0
            self.restarts = 0

        def tokenize_batch_raw(self, texts, scan_length, timeout):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("bridge disconnected")
            return [_sample_parse_payload("学校")]

        def restart(self):
            self.restarts += 1

    client = _Client()
    service = TokenizationService()
    service.backend = "bridge"
    monkeypatch.setattr(service, "_get_bridge_client", lambda: client)

    result = service._tokenize_texts(["学校"])
    assert len(result) == 1
    assert result[0] == [{"headword": "学校", "word": "学校", "reading": "ガッコウ"}]
    assert client.calls == 2
    assert client.restarts == 1


def test_prepare_text_for_tokenization_matches_stats_cleanup(monkeypatch):
    module = _load_tokenization_module(monkeypatch)

    class _StatsConfig:
        regex_out_repetitions = True

    monkeypatch.setattr(module, "get_stats_config", lambda: _StatsConfig())
    cleaned = module._prepare_text_for_tokenization("あああああ、、、！！！")
    assert cleaned == "あああ"


def test_tokenize_lines_batch_uses_batch_persistence(monkeypatch):
    module = _load_tokenization_module(monkeypatch)
    TokenizationService = module.TokenizationService

    service = TokenizationService()
    service.chunk_size = 64
    batch_calls = []

    def _persist_lines_tokenization_batch(entries):
        batch_calls.append(entries)

    monkeypatch.setattr(service, "_persist_lines_tokenization_batch", _persist_lines_tokenization_batch)
    monkeypatch.setattr(
        service,
        "_persist_line_tokenization",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("per-line persistence should not run")),
    )
    monkeypatch.setattr(
        service,
        "_tokenize_texts",
        lambda texts, source="realtime": [
            [{"word": "学校", "headword": "学校", "reading": "ガッコウ"}] if text == "学校" else None
            for text in texts
        ],
    )

    rows = [
        ("line_jp_1", "学校", 1700000000.0, "game_1"),
        ("line_jp_2", "学校", 1700000001.0, "game_1"),
        ("line_ascii", "hello world", 1700000002.0, "game_1"),
    ]

    result = service.tokenize_lines_batch(rows)
    assert result["processed"] == 3
    assert result["failed"] == 0
    assert len(batch_calls) >= 1
    assert sum(len(call) for call in batch_calls) == 3


def test_bridge_command_does_not_append_user_data_dir_flag(monkeypatch):
    module = _load_tokenization_module(monkeypatch)
    OverlayTokenizerBridgeClient = module.OverlayTokenizerBridgeClient

    monkeypatch.delenv("GSM_TOKENIZER_BRIDGE_CMD", raising=False)
    monkeypatch.setenv("GSM_OVERLAY_ELECTRON_BIN", "electron-custom")
    monkeypatch.setenv("GSM_TOKENIZER_BRIDGE_USER_DATA_DIR", "/tmp/gsm-bridge-profile")
    monkeypatch.setattr(module.shutil, "which", lambda command: "/usr/bin/electron-custom" if command == "electron-custom" else None)

    client = OverlayTokenizerBridgeClient()
    cmd, _cwd = client._resolve_command()

    assert "--bridge" in cmd
    assert not any(arg.startswith("--user-data-dir=") for arg in cmd)


def test_bridge_command_prefers_bridge_binary_env(monkeypatch):
    module = _load_tokenization_module(monkeypatch)
    OverlayTokenizerBridgeClient = module.OverlayTokenizerBridgeClient

    monkeypatch.delenv("GSM_TOKENIZER_BRIDGE_CMD", raising=False)
    monkeypatch.setenv("GSM_TOKENIZER_BRIDGE_BIN", "/tmp/gsm_overlay_bin")
    monkeypatch.setenv("GSM_OVERLAY_ELECTRON_BIN", "electron-custom")
    monkeypatch.setattr(module.shutil, "which", lambda command: "/tmp/gsm_overlay_bin" if command == "/tmp/gsm_overlay_bin" else None)

    client = OverlayTokenizerBridgeClient()
    cmd, _cwd = client._resolve_command()

    assert cmd == [
        "/tmp/gsm_overlay_bin",
        "--bridge",
    ]


def test_realtime_worker_processes_lines_even_during_backfill(monkeypatch):
    module = _load_tokenization_module(monkeypatch)

    class _Service:
        def __init__(self):
            self.calls = []

        def is_backfill_session_active(self):
            return True

        def tokenize_lines_batch(self, rows):
            self.calls.append(rows)
            return {"processed": len(rows), "failed": 0}

    service = _Service()
    monkeypatch.setattr(module, "get_tokenization_service", lambda: service)
    monkeypatch.setattr(module, "_realtime_queue", queue.Queue())

    module._realtime_queue.put(("line_1", "学校", 1700000000.0, "game_1"))
    module._realtime_queue.put(None)
    module._realtime_worker()

    assert len(service.calls) == 1
    assert service.calls[0] == [("line_1", "学校", 1700000000.0, "game_1")]
