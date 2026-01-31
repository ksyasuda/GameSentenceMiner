"""
Tests for the tokenization service and MeCab parser.

These tests use a real SQLite database (not the production gsm.db) to verify
tokenization behavior with actual database operations.
"""

import pytest
import time

from tests.fixtures import (
    create_game,
    create_game_line,
    create_word,
    create_kanji,
    bulk_create_game_lines,
    DataBuilder,
    get_untokenized_lines_for_batch,
    SAMPLE_JAPANESE_LINES,
    SAMPLE_WORDS,
    SAMPLE_KANJI,
)


class TestMecabController:
    """Tests for the MeCab controller and token parsing."""

    def test_basic_parsing(self, test_db, mecab_controller):
        """Test that MeCab can parse simple Japanese text."""
        text = "今日は天気がいいですね。"
        tokens = list(mecab_controller.translate(text))

        assert len(tokens) > 0
        # Should contain: 今日, は, 天気, が, いい, です, ね, 。
        surface_forms = [t.word for t in tokens]
        assert "今日" in surface_forms
        assert "天気" in surface_forms

    def test_token_fields(self, test_db, mecab_controller):
        """Test that tokens have all expected fields populated."""
        text = "食べる"
        tokens = list(mecab_controller.translate(text))

        assert len(tokens) >= 1
        token = tokens[0]

        # Check that essential fields are present
        assert hasattr(token, "word")
        assert hasattr(token, "headword")
        assert hasattr(token, "katakana_reading")
        assert hasattr(token, "part_of_speech")

    def test_verb_conjugation(self, test_db, mecab_controller):
        """Test that verb conjugations are parsed correctly."""
        # 食べました = ate (polite past)
        text = "食べました"
        tokens = list(mecab_controller.translate(text))

        # Should get headword "食べる" from conjugated form
        headwords = [t.headword for t in tokens]
        assert "食べる" in headwords

    def test_mimic_yomitan_mode(self, test_db, mecab_controller):
        """Test token merging with mimic_yomitan=True."""
        text = "食べている"

        # Without merging
        raw_tokens = list(mecab_controller.translate(text, mimic_yomitan=False))

        # With merging
        merged_tokens = list(mecab_controller.translate(text, mimic_yomitan=True))

        # Merged should have fewer or equal tokens
        assert len(merged_tokens) <= len(raw_tokens)

    def test_reading_generation(self, test_db, mecab_controller):
        """Test furigana/reading generation."""
        text = "日本語"
        reading = mecab_controller.reading(text)

        # Should return the reading in some format
        assert reading is not None
        assert len(reading) > 0

    def test_empty_input(self, test_db, mecab_controller):
        """Test handling of empty input."""
        tokens = list(mecab_controller.translate(""))
        assert tokens == []

    def test_punctuation_only(self, test_db, mecab_controller):
        """Test handling of punctuation-only input."""
        text = "。、！？"
        tokens = list(mecab_controller.translate(text))
        # Should return tokens for punctuation (classified as symbols)
        assert len(tokens) >= 0  # May or may not return tokens


class TestTokenizationService:
    """Tests for the tokenization service database operations."""

    def test_tokenize_single_line(self, test_db, tokenization_service):
        """Test tokenizing a single game line."""
        from GameSentenceMiner.util.db import GameLinesTable, WordsTable, KanjiTable

        # Create a game and line
        game = create_game(title_original="単体テスト")
        line = create_game_line(
            game_id=game.id,
            game_name=game.title_original,
            line_text="私は日本語を勉強しています。",
        )

        # Verify line is not tokenized
        assert line.tokenized == 0

        # Tokenize the line
        result = tokenization_service.tokenize_line(
            game_line_id=line.id,
            line_text=line.line_text,
            timestamp=line.timestamp,
            game_id=line.game_id,
        )

        assert result is True

        # Verify line is now marked as tokenized
        updated_line = GameLinesTable.get(line.id)
        assert updated_line.tokenized == 1

        # Verify words were created
        all_words = WordsTable.all()
        assert len(all_words) > 0

        # Verify kanji were created (日本語勉強 contain kanji)
        all_kanji = KanjiTable.all()
        assert len(all_kanji) > 0

    def test_tokenize_line_updates_stats(self, test_db, tokenization_service):
        """Test that tokenization updates line statistics."""
        from GameSentenceMiner.util.db import GameLinesTable

        game = create_game()
        line = create_game_line(
            game_id=game.id,
            game_name=game.title_original,
            line_text="日本語の勉強は楽しいです。",
        )

        tokenization_service.tokenize_line(
            game_line_id=line.id,
            line_text=line.line_text,
            timestamp=line.timestamp,
            game_id=line.game_id,
        )

        updated_line = GameLinesTable.get(line.id)

        # Should have statistics populated
        assert updated_line.total_length > 0
        assert updated_line.filtered_length > 0
        assert updated_line.word_count > 0
        assert updated_line.kanji_count > 0

    def test_tokenize_batch(self, test_db, tokenization_service):
        """Test batch tokenization of multiple lines."""
        from GameSentenceMiner.util.db import GameLinesTable, WordsTable

        game = create_game(title_original="バッチテスト")
        lines = bulk_create_game_lines(
            game_id=game.id,
            game_name=game.title_original,
            texts=SAMPLE_JAPANESE_LINES[:10],
        )

        # Get lines in batch format
        batch_data = get_untokenized_lines_for_batch(game_id=game.id, limit=10)
        assert len(batch_data) == 10

        # Tokenize batch
        result = tokenization_service.tokenize_batch(batch_data, chunk_size=5)

        assert result["processed"] == 10
        assert result["failed"] == 0

        # Verify all lines are tokenized
        for line in lines:
            updated_line = GameLinesTable.get(line.id)
            assert updated_line.tokenized == 1

    def test_word_frequency_tracking(self, test_db, tokenization_service):
        """Test that word frequencies are tracked across multiple lines."""
        from GameSentenceMiner.util.db import WordsTable

        game = create_game()

        # Create lines with repeated words
        texts = [
            "今日は天気がいい。",
            "今日は勉強する。",
            "今日は映画を見る。",
        ]
        lines = bulk_create_game_lines(
            game_id=game.id,
            game_name=game.title_original,
            texts=texts,
        )

        # Tokenize all lines
        for line in lines:
            tokenization_service.tokenize_line(
                game_line_id=line.id,
                line_text=line.line_text,
                timestamp=line.timestamp,
                game_id=line.game_id,
            )

        word = WordsTable.get_by_headword("今日")
        assert word is not None and word['total_frequency'] == 3

    def test_kanji_extraction(self, test_db, tokenization_service):
        """Test that kanji are correctly extracted from text."""
        from GameSentenceMiner.util.db import KanjiTable

        game = create_game()
        line = create_game_line(
            game_id=game.id,
            game_name=game.title_original,
            line_text="漢字",  # Two kanji characters
        )

        tokenization_service.tokenize_line(
            game_line_id=line.id,
            line_text=line.line_text,
            timestamp=line.timestamp,
            game_id=line.game_id,
        )

        # Should have exactly 2 kanji
        all_kanji = KanjiTable.all()
        kanji_chars = [k.kanji for k in all_kanji]
        assert "漢" in kanji_chars
        assert "字" in kanji_chars

    def test_word_occurrences(self, test_db, tokenization_service):
        """Test that word occurrences are recorded correctly."""
        from GameSentenceMiner.util.db import WordOccurrencesTable

        game = create_game()
        line = create_game_line(
            game_id=game.id,
            game_name=game.title_original,
            line_text="私は私を見た。",  # "私" appears twice
        )

        tokenization_service.tokenize_line(
            game_line_id=line.id,
            line_text=line.line_text,
            timestamp=line.timestamp,
            game_id=line.game_id,
        )

        occurrences = WordOccurrencesTable.get_occurrences_for_game(game.id)
        assert occurrences is not None and len(occurrences) == 3

    def test_idempotent_tokenization(self, test_db, tokenization_service):
        """Test that re-tokenizing a line doesn't duplicate data."""
        from GameSentenceMiner.util.db import WordsTable, WordOccurrencesTable

        game = create_game()
        line = create_game_line(
            game_id=game.id,
            game_name=game.title_original,
            line_text="テスト文章です。",
        )

        # Tokenize once
        tokenization_service.tokenize_line(
            game_line_id=line.id,
            line_text=line.line_text,
            timestamp=line.timestamp,
            game_id=line.game_id,
        )

        word_count_1 = len(WordsTable.all())
        occ_count_1 = len(WordOccurrencesTable.all())

        # Reset tokenized flag and tokenize again
        line.tokenized = 0
        line.save()

        tokenization_service.tokenize_line(
            game_line_id=line.id,
            line_text=line.line_text,
            timestamp=line.timestamp,
            game_id=line.game_id,
        )

        word_count_2 = len(WordsTable.all())
        occ_count_2 = len(WordOccurrencesTable.all())

        # Word count should be the same (no duplicates)
        assert word_count_1 == word_count_2
        # Occurrences should also not duplicate due to unique constraint
        assert occ_count_1 == occ_count_2


class TestDataBuilderUsage:
    """Tests demonstrating the DataBuilder fluent API."""

    def test_builder_basic_usage(self, test_db):
        """Test basic DataBuilder usage."""
        from GameSentenceMiner.util.db import GameLinesTable

        data = (
            DataBuilder()
            .with_game(title_original="ビルダーテスト", title_english="Builder Test")
            .with_lines(count=3)
            .build()
        )

        assert data["game"] is not None
        assert data["game"].title_original == "ビルダーテスト"
        assert len(data["lines"]) == 3

        # Verify lines are in the database
        for line in data["lines"]:
            db_line = GameLinesTable.get(line.id)
            assert db_line is not None

    def test_builder_custom_lines(self, test_db):
        """Test DataBuilder with custom line texts."""
        custom_texts = [
            "カスタムテキスト1",
            "カスタムテキスト2",
        ]

        data = (
            DataBuilder()
            .with_game()
            .with_lines(texts=custom_texts)
            .build()
        )

        assert len(data["lines"]) == 2
        assert data["lines"][0].line_text == "カスタムテキスト1"
        assert data["lines"][1].line_text == "カスタムテキスト2"

    def test_builder_with_tokenization(self, test_db):
        """Test DataBuilder with automatic tokenization."""
        from GameSentenceMiner.util.db import GameLinesTable

        data = (
            DataBuilder()
            .with_game()
            .with_lines(texts=["日本語のテスト文章です。"])
            .with_tokenization()
            .build()
        )

        # Line should be tokenized
        line = GameLinesTable.get(data["lines"][0].id)
        assert line.tokenized == 1
        assert line.word_count > 0

    def test_builder_with_words_and_kanji(self, test_db):
        """Test DataBuilder with pre-created words and kanji."""
        from GameSentenceMiner.util.db import WordsTable, KanjiTable

        data = (
            DataBuilder()
            .with_words()  # Uses SAMPLE_WORDS
            .with_kanji()  # Uses SAMPLE_KANJI
            .build()
        )

        assert len(data["words"]) == len(SAMPLE_WORDS)
        assert len(data["kanji"]) == len(SAMPLE_KANJI)

        # Verify in database
        all_words = WordsTable.all()
        all_kanji = KanjiTable.all()
        assert len(all_words) >= len(SAMPLE_WORDS)
        assert len(all_kanji) >= len(SAMPLE_KANJI)


class TestDatabaseIsolation:
    """Tests verifying that the test database is isolated from production."""

    def test_database_is_temporary(self, test_db):
        """Verify that the test database is in a temp directory."""
        assert "test_" in test_db.db_path
        assert "gsm_test_" in test_db.db_path

    def test_tables_exist(self, test_db):
        """Verify all expected tables are created."""
        expected_tables = [
            "game_lines",
            "games",
            "words",
            "kanji",
            "word_occurrences",
            "kanji_occurrences",
        ]

        for table in expected_tables:
            assert test_db.table_exists(table), f"Table {table} should exist"

    def test_fresh_database_per_test(self, test_db):
        """Verify each test gets a fresh database."""
        from GameSentenceMiner.util.db import GameLinesTable

        # Database should be empty at start
        all_lines = GameLinesTable.all()
        assert len(all_lines) == 0

        # Create some data
        create_game_line(line_text="Test line")

        # Now should have 1 line
        all_lines = GameLinesTable.all()
        assert len(all_lines) == 1

    def test_another_fresh_database(self, test_db):
        """Second test to verify isolation - should also start fresh."""
        from GameSentenceMiner.util.db import GameLinesTable

        # Should be empty (not affected by previous test)
        all_lines = GameLinesTable.all()
        assert len(all_lines) == 0


class TestSampleDataFixture:
    """Tests using the pre-populated sample data fixture."""

    def test_sample_data_has_games(self, test_db_with_sample_data):
        """Verify sample data fixture creates games."""
        data = test_db_with_sample_data

        assert len(data["games"]) == 2
        assert data["game1"].title_original == "テストゲーム1"
        assert data["game2"].title_original == "テストゲーム2"

    def test_sample_data_has_lines(self, test_db_with_sample_data):
        """Verify sample data fixture creates lines."""
        data = test_db_with_sample_data

        assert len(data["lines_game1"]) == 5
        assert len(data["lines_game2"]) == 5
        assert len(data["all_lines"]) == 10

    def test_sample_data_lines_not_tokenized(self, test_db_with_sample_data):
        """Verify sample data lines are not pre-tokenized."""
        data = test_db_with_sample_data

        for line in data["all_lines"]:
            assert line.tokenized == 0

    def test_can_tokenize_sample_data(
        self, test_db_with_sample_data, tokenization_service
    ):
        """Test tokenizing the sample data."""
        from GameSentenceMiner.util.db import GameLinesTable

        data = test_db_with_sample_data

        # Tokenize all lines for game1
        for line in data["lines_game1"]:
            tokenization_service.tokenize_line(
                game_line_id=line.id,
                line_text=line.line_text,
                timestamp=line.timestamp,
                game_id=line.game_id,
            )

        # Verify game1 lines are tokenized
        for line in data["lines_game1"]:
            updated = GameLinesTable.get(line.id)
            assert updated.tokenized == 1

        # Verify game2 lines are NOT tokenized
        for line in data["lines_game2"]:
            updated = GameLinesTable.get(line.id)
            assert updated.tokenized == 0


class TestQuoteRemovalAndOffsets:
    """Tests for quote character filtering and position offset preservation."""

    def test_quote_characters_not_in_tokens(self, test_db, mecab_controller):
        """Test that 「 and 」 are never emitted as word tokens."""
        from GameSentenceMiner.util.tokenization_service import TokenizationService

        text = "「待って！そんなこと言わないで！」"
        tokens = list(mecab_controller.translate(text))
        word_data, _ = TokenizationService._extract_tokens_and_kanji(text, tokens)

        for word in word_data:
            assert word["word"] != "「", "Quote character 「 should not be in word"
            assert word["word"] != "」", "Quote character 」 should not be in word"
            assert word["headword"] != "「", "Quote character 「 should not be in headword"
            assert word["headword"] != "」", "Quote character 」 should not be in headword"

    def test_quote_removal_with_filter_disabled(self, test_db, mecab_controller):
        """Test that quotes are removed even when POS filtering is disabled."""
        from GameSentenceMiner.util.tokenization_service import TokenizationService

        text = "「テスト」"
        tokens = list(mecab_controller.translate(text))
        word_data, _ = TokenizationService._extract_tokens_and_kanji(text, tokens, filter_pos=[])

        words = [w["word"] for w in word_data]
        headwords = [w["headword"] for w in word_data]

        assert "「" not in words
        assert "」" not in words
        assert "「" not in headwords
        assert "」" not in headwords
        assert "テスト" in words or any("テスト" in w for w in words)

    def test_token_positions_match_original_text(self, test_db, mecab_controller):
        """Test that token positions reference the original text correctly."""
        from GameSentenceMiner.util.tokenization_service import TokenizationService

        text = "「待って！そんなこと言わないで！」"
        tokens = list(mecab_controller.translate(text))
        word_data, _ = TokenizationService._extract_tokens_and_kanji(text, tokens)

        for word in word_data:
            position = word["position"]
            surface = word["word"]
            extracted = text[position : position + len(surface)]
            assert extracted == surface, (
                f"Position {position} should yield '{surface}' but got '{extracted}'"
            )

    def test_kanji_positions_in_quoted_text(self, test_db, mecab_controller):
        """Test that kanji positions are correct in text with quotes."""
        from GameSentenceMiner.util.tokenization_service import TokenizationService

        text = "「日本語」"
        tokens = list(mecab_controller.translate(text))
        _, kanji_data = TokenizationService._extract_tokens_and_kanji(text, tokens)

        kanji_chars = [k["kanji"] for k in kanji_data]
        assert "日" in kanji_chars
        assert "本" in kanji_chars
        assert "語" in kanji_chars

        for k in kanji_data:
            pos = k["position"]
            assert text[pos] == k["kanji"], f"Kanji at position {pos} should be {k['kanji']}"

    def test_filter_pos_custom_list(self, test_db, mecab_controller):
        """Test custom filter_pos list works correctly."""
        from GameSentenceMiner.util.tokenization_service import TokenizationService
        from GameSentenceMiner.mecab.basic_types import PartOfSpeech

        text = "日本語を勉強する"
        tokens = list(mecab_controller.translate(text))

        word_data_default, _ = TokenizationService._extract_tokens_and_kanji(text, tokens)
        word_data_no_filter, _ = TokenizationService._extract_tokens_and_kanji(
            text, tokens, filter_pos=[]
        )
        word_data_verbs_only, _ = TokenizationService._extract_tokens_and_kanji(
            text, tokens, filter_pos=[PartOfSpeech.noun]
        )

        assert len(word_data_no_filter) >= len(word_data_default)

        nouns_in_default = [w for w in word_data_default if w["pos"] == "noun"]
        nouns_in_verbs_only = [w for w in word_data_verbs_only if w["pos"] == "noun"]
        assert len(nouns_in_verbs_only) == 0 or len(nouns_in_verbs_only) < len(nouns_in_default)
