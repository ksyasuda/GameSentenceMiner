"""
Pytest configuration and fixtures for GameSentenceMiner tests.

Provides a test database that is completely separate from the production gsm.db.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def test_db_dir():
    """Create a temporary directory for test databases that persists for the session."""
    temp_dir = tempfile.mkdtemp(prefix="gsm_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def test_db(test_db_dir, monkeypatch):
    """
    Create a fresh test database for each test function.

    This fixture:
    1. Creates a new SQLite database in a temp directory
    2. Patches the database path so all table classes use the test DB
    3. Initializes all tables
    4. Yields the database instance
    5. Cleans up after the test

    Usage:
        def test_something(test_db):
            # test_db is the SQLiteDB instance
            # All table classes now point to the test database
            game = GamesTable(title_original="Test Game")
            game.save()
    """
    from GameSentenceMiner.util.db import (
        SQLiteDB,
        AIModelsTable,
        GameLinesTable,
        GoalsTable,
        WordsTable,
        KanjiTable,
        WordOccurrencesTable,
        KanjiOccurrencesTable,
    )
    from GameSentenceMiner.util.games_table import GamesTable
    from GameSentenceMiner.util.cron_table import CronTable
    from GameSentenceMiner.util.stats_rollup_table import StatsRollupTable

    import uuid
    db_path = os.path.join(test_db_dir, f"test_{uuid.uuid4().hex[:8]}.db")

    test_database = SQLiteDB(db_path, read_only=False)

    table_classes = [
        AIModelsTable,
        GameLinesTable,
        GoalsTable,
        GamesTable,
        CronTable,
        StatsRollupTable,
        WordsTable,
        KanjiTable,
        WordOccurrencesTable,
        KanjiOccurrencesTable,
    ]

    for cls in table_classes:
        cls.set_db(test_database)

    yield test_database

    test_database.close()


@pytest.fixture(scope="function")
def test_db_with_sample_data(test_db):
    """
    Test database pre-populated with sample data.

    Creates:
    - 2 games
    - 5 game lines per game
    - Words and kanji are NOT pre-tokenized (tokenized=0)

    Returns a dict with references to created objects.
    """
    from tests.fixtures import (
        create_game,
        create_game_line,
        SAMPLE_JAPANESE_LINES,
    )

    import time
    base_time = time.time() - 86400  # Start from yesterday

    game1 = create_game(
        title_original="テストゲーム1",
        title_english="Test Game 1",
    )
    game2 = create_game(
        title_original="テストゲーム2",
        title_english="Test Game 2",
    )

    lines_game1 = []
    lines_game2 = []

    for i, text in enumerate(SAMPLE_JAPANESE_LINES[:5]):
        line = create_game_line(
            game_id=game1.id,
            game_name=game1.title_original,
            line_text=text,
            timestamp=base_time + (i * 60),
        )
        lines_game1.append(line)

    for i, text in enumerate(SAMPLE_JAPANESE_LINES[5:10]):
        line = create_game_line(
            game_id=game2.id,
            game_name=game2.title_original,
            line_text=text,
            timestamp=base_time + (i * 60),
        )
        lines_game2.append(line)

    return {
        "db": test_db,
        "games": [game1, game2],
        "game1": game1,
        "game2": game2,
        "lines_game1": lines_game1,
        "lines_game2": lines_game2,
        "all_lines": lines_game1 + lines_game2,
    }


@pytest.fixture(scope="function")
def mecab_controller():
    """
    Provides a MeCab controller for tokenization tests.

    Uses the persistent controller by default for better performance.
    """
    from GameSentenceMiner.mecab.mecab_controller import MecabController

    controller = MecabController(verbose=False, cache_max_size=512, persistent=True)
    yield controller


@pytest.fixture(scope="function")
def tokenization_service(test_db):
    """
    Provides a TokenizationService instance connected to the test database.

    The service is initialized fresh for each test and uses the test database.
    """
    from GameSentenceMiner.util.tokenization_service import TokenizationService

    service = TokenizationService()
    yield service

    service._on_shutdown()
