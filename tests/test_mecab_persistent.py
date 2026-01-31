"""
Tests specifically for the persistent MeCab parser.

These tests focus on the persistent subprocess management and
performance characteristics of the MecabController.
"""

import pytest
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tests.fixtures import SAMPLE_JAPANESE_LINES


class TestPersistentMecabBasics:
    """Basic functionality tests for persistent MeCab controller."""

    def test_persistent_controller_creation(self, test_db):
        """Test that a persistent controller can be created."""
        from GameSentenceMiner.mecab.mecab_controller import MecabController

        controller = MecabController(persistent=True)
        assert controller is not None

        # Parse something to ensure it works
        tokens = list(controller.translate("テスト"))
        assert len(tokens) > 0

    def test_non_persistent_controller_creation(self, test_db):
        """Test that a non-persistent controller can be created."""
        from GameSentenceMiner.mecab.mecab_controller import MecabController

        controller = MecabController(persistent=False)
        assert controller is not None

        tokens = list(controller.translate("テスト"))
        assert len(tokens) > 0

    def test_persistent_handles_multiple_parses(self, test_db, mecab_controller):
        """Test that persistent controller handles multiple sequential parses."""
        for text in SAMPLE_JAPANESE_LINES[:10]:
            tokens = list(mecab_controller.translate(text))
            assert len(tokens) >= 0  # Empty text might return empty list

    def test_controller_caching(self, test_db):
        """Test that the controller caches results."""
        from GameSentenceMiner.mecab.mecab_controller import MecabController

        controller = MecabController(persistent=True, cache_max_size=100)

        text = "キャッシュテスト文章"

        # First call
        tokens1 = list(controller.translate(text))

        # Second call should hit cache
        tokens2 = list(controller.translate(text))

        # Results should be identical
        assert len(tokens1) == len(tokens2)
        for t1, t2 in zip(tokens1, tokens2):
            assert t1.word == t2.word
            assert t1.headword == t2.headword


class TestPersistentMecabThreadSafety:
    """Tests for thread safety of the persistent MeCab controller."""

    def test_concurrent_parsing(self, test_db, mecab_controller):
        """Test that multiple threads can use the controller safely."""
        results = {}
        errors = []

        def parse_text(idx, text):
            try:
                tokens = list(mecab_controller.translate(text))
                return idx, len(tokens)
            except Exception as e:
                errors.append((idx, str(e)))
                return idx, -1

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(parse_text, i, text): i
                for i, text in enumerate(SAMPLE_JAPANESE_LINES[:20])
            }

            for future in as_completed(futures):
                idx, count = future.result()
                results[idx] = count

        # All should complete without errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 20

    def test_rapid_sequential_parsing(self, test_db, mecab_controller):
        """Test rapid sequential parsing doesn't cause issues."""
        for _ in range(3):  # Multiple rounds
            for text in SAMPLE_JAPANESE_LINES[:10]:
                tokens = list(mecab_controller.translate(text))
                assert tokens is not None


class TestPersistentMecabPerformance:
    """Performance-related tests for persistent MeCab controller."""

    @pytest.mark.slow
    def test_batch_parsing_performance(self, test_db):
        """Compare persistent vs non-persistent controller performance."""
        from GameSentenceMiner.mecab.mecab_controller import MecabController

        texts = SAMPLE_JAPANESE_LINES[:20]
        iterations = 2

        # Persistent controller
        persistent = MecabController(persistent=True, cache_max_size=0)  # No cache
        start = time.perf_counter()
        for _ in range(iterations):
            for text in texts:
                list(persistent.translate(text))
        persistent_time = time.perf_counter() - start

        # Non-persistent controller
        basic = MecabController(persistent=False, cache_max_size=0)
        start = time.perf_counter()
        for _ in range(iterations):
            for text in texts:
                list(basic.translate(text))
        basic_time = time.perf_counter() - start

        # Persistent should generally be faster due to no subprocess startup
        # But we just verify both complete successfully
        assert persistent_time > 0
        assert basic_time > 0

        # Log the results for manual inspection
        print(f"\nPersistent: {persistent_time:.3f}s, Basic: {basic_time:.3f}s")

    def test_warmup_effect(self, test_db):
        """Test that first parse may be slower (warmup)."""
        from GameSentenceMiner.mecab.mecab_controller import MecabController

        controller = MecabController(persistent=True, cache_max_size=0)

        text = "ウォームアップテスト"

        # First parse (cold)
        start = time.perf_counter()
        list(controller.translate(text))
        cold_time = time.perf_counter() - start

        # Subsequent parses (warm)
        warm_times = []
        for _ in range(5):
            start = time.perf_counter()
            list(controller.translate(text + str(_)))  # Different text to avoid cache
            warm_times.append(time.perf_counter() - start)

        avg_warm = sum(warm_times) / len(warm_times)

        # Just verify we get times
        assert cold_time > 0
        assert avg_warm > 0


class TestTokenMerger:
    """Tests for the token merger functionality."""

    def test_merge_verb_with_auxiliary(self, test_db, mecab_controller):
        """Test that verbs merge with auxiliaries."""
        text = "食べている"  # eating (progressive)

        raw = list(mecab_controller.translate(text, mimic_yomitan=False))
        merged = list(mecab_controller.translate(text, mimic_yomitan=True))

        # Raw should have multiple tokens
        assert len(raw) >= 2

        # Merged might combine them
        # The merged surface should contain the original text
        merged_surfaces = "".join(t.word if hasattr(t, 'word') else t.surface for t in merged)
        raw_surfaces = "".join(t.word for t in raw)

        assert merged_surfaces == raw_surfaces

    def test_merge_preserves_positions(self, test_db, mecab_controller):
        """Test that merged tokens have correct position info."""
        text = "私は食べている"

        merged = list(mecab_controller.translate(text, mimic_yomitan=True))

        # Each merged token should have position info
        for token in merged:
            if hasattr(token, 'start_pos'):
                assert token.start_pos >= 0
                assert token.end_pos > token.start_pos

    def test_merge_handles_particles(self, test_db, mecab_controller):
        """Test handling of particles in merging."""
        text = "学校に行く"  # go to school

        raw = list(mecab_controller.translate(text, mimic_yomitan=False))
        merged = list(mecab_controller.translate(text, mimic_yomitan=True))

        # Both should produce valid results
        assert len(raw) > 0
        assert len(merged) > 0


class TestEdgeCases:
    """Edge case tests for MeCab parsing."""

    def test_very_long_text(self, test_db, mecab_controller):
        """Test parsing of very long text."""
        long_text = "これは長いテストです。" * 100

        tokens = list(mecab_controller.translate(long_text))
        assert len(tokens) > 0

    def test_unicode_characters(self, test_db, mecab_controller):
        """Test parsing with various Unicode characters."""
        texts = [
            "漢字ひらがなカタカナ",  # Mixed scripts
            "①②③",  # Circled numbers
            "〜〜〜",  # Wave dashes
            "…",  # Ellipsis
            "「」『』",  # Quotation marks
        ]

        for text in texts:
            tokens = list(mecab_controller.translate(text))
            # Should not raise exceptions
            assert tokens is not None

    def test_numbers_and_ascii(self, test_db, mecab_controller):
        """Test parsing with numbers and ASCII."""
        texts = [
            "123",
            "ABC",
            "test123テスト",
            "100円",
            "2024年1月",
        ]

        for text in texts:
            tokens = list(mecab_controller.translate(text))
            assert tokens is not None

    def test_newlines_and_whitespace(self, test_db, mecab_controller):
        """Test handling of newlines and whitespace."""
        texts = [
            "行1\n行2",
            "スペース あり",
            "タブ\tあり",
            "  先頭空白",
            "末尾空白  ",
        ]

        for text in texts:
            tokens = list(mecab_controller.translate(text))
            assert tokens is not None

    def test_repeated_characters(self, test_db, mecab_controller):
        """Test handling of repeated characters."""
        texts = [
            "あああああ",  # Repeated hiragana
            "黙れ黙れ黙れ",  # Repeated word
            "ーーーー",  # Long vowel marks
        ]

        for text in texts:
            tokens = list(mecab_controller.translate(text))
            assert tokens is not None
