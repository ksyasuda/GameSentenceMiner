import importlib.util
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


def _load_script(monkeypatch, relative_path: str, module_name: str):
    script_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_reset_tokenization_parser_uses_get_db_directory(monkeypatch):
    class _Logger:
        def info(self, *_args, **_kwargs):
            return None

    class _SQLiteDB:
        def __init__(self, _db_path):
            pass

    class _GameLinesTable:
        @classmethod
        def set_db(cls, _db):
            return None

        @classmethod
        def has_column(cls, _name):
            return False

    class _CronTable:
        @classmethod
        def set_db(cls, _db):
            return None

    _stub_module(monkeypatch, "GameSentenceMiner")
    _stub_module(monkeypatch, "GameSentenceMiner.util")
    _stub_module(monkeypatch, "GameSentenceMiner.util.config")
    _stub_module(monkeypatch, "GameSentenceMiner.util.database")
    _stub_module(monkeypatch, "GameSentenceMiner.util.database.cron_table", CronTable=_CronTable)
    _stub_module(
        monkeypatch,
        "GameSentenceMiner.util.config.configuration",
        logger=_Logger(),
    )
    _stub_module(
        monkeypatch,
        "GameSentenceMiner.util.database.db",
        CronTable=_CronTable,
        GameLinesTable=_GameLinesTable,
        SQLiteDB=_SQLiteDB,
        get_db_directory=lambda: "/tmp/gsm-test.db",
    )

    module = _load_script(monkeypatch, "scripts/reset_tokenization.py", "reset_tokenization_test")
    args = module.build_parser().parse_args([])
    assert args.db_path == "/tmp/gsm-test.db"


def test_run_tokenization_backfill_has_no_import_side_effects(monkeypatch):
    class _Logger:
        def info(self, *_args, **_kwargs):
            return None

    call_count = {"n": 0}

    def _backfill():
        call_count["n"] += 1
        return {"success": True, "total": 2, "processed": 2, "failed": 0, "completed": 2}

    _stub_module(monkeypatch, "GameSentenceMiner")
    _stub_module(monkeypatch, "GameSentenceMiner.util")
    _stub_module(monkeypatch, "GameSentenceMiner.util.cron")
    _stub_module(monkeypatch, "GameSentenceMiner.util.config")
    _stub_module(
        monkeypatch,
        "GameSentenceMiner.util.config.configuration",
        logger=_Logger(),
    )
    _stub_module(
        monkeypatch,
        "GameSentenceMiner.util.cron.backfill_tokenization",
        backfill_tokenization=_backfill,
    )

    module = _load_script(
        monkeypatch,
        "scripts/run_tokenization_backfill.py",
        "run_tokenization_backfill_test",
    )
    assert call_count["n"] == 0
    exit_code = module.main([])
    assert exit_code == 0
    assert call_count["n"] == 1
