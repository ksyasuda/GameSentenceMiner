"""
Tests for token merger functionality, including bug fix verification.

Tests that symbols don't incorrectly merge with auxiliaries, while verbs,
auxiliaries, and adjectives correctly merge with their grammatical suffixes.
"""

import pytest

from GameSentenceMiner.mecab.mecab_controller import MecabController
from GameSentenceMiner.mecab.token_merger import merge_tokens, _can_receive_auxiliary
from GameSentenceMiner.mecab.basic_types import PartOfSpeech


class TestCanReceiveAuxiliary:
    """Tests for the _can_receive_auxiliary helper function."""

    def test_verb_can_receive_auxiliary(self, mecab_controller):
        """Verbs should be able to receive auxiliaries."""
        tokens = list(mecab_controller.translate("食べる"))
        verb_token = tokens[0]
        assert verb_token.part_of_speech == PartOfSpeech.verb
        assert _can_receive_auxiliary(verb_token) is True

    def test_bound_auxiliary_can_receive_auxiliary(self, mecab_controller):
        """Bound auxiliaries should be able to receive other auxiliaries."""
        # Parse "食べない" to get ない as a bound auxiliary (attached to verb)
        tokens = list(mecab_controller.translate("食べない"))
        # Second token should be ない as bound auxiliary
        aux_token = tokens[1]
        assert aux_token.word == "ない"
        assert aux_token.part_of_speech == PartOfSpeech.bound_auxiliary
        assert _can_receive_auxiliary(aux_token) is True

    def test_i_adjective_can_receive_auxiliary(self, mecab_controller):
        """I-adjectives should be able to receive auxiliaries."""
        tokens = list(mecab_controller.translate("高い"))
        adj_token = tokens[0]
        assert adj_token.part_of_speech == PartOfSpeech.i_adjective
        assert _can_receive_auxiliary(adj_token) is True

    def test_symbol_cannot_receive_auxiliary(self, mecab_controller):
        """Symbols should NOT be able to receive auxiliaries."""
        tokens = list(mecab_controller.translate("…"))
        symbol_token = tokens[0]
        assert symbol_token.part_of_speech == PartOfSpeech.symbol
        assert _can_receive_auxiliary(symbol_token) is False

    def test_noun_cannot_receive_auxiliary(self, mecab_controller):
        """Nouns should NOT be able to receive auxiliaries."""
        tokens = list(mecab_controller.translate("本"))
        noun_token = tokens[0]
        assert noun_token.part_of_speech == PartOfSpeech.noun
        assert _can_receive_auxiliary(noun_token) is False


class TestSymbolAuxiliaryNoMerge:
    """Bug fix verification: symbols should NOT merge with auxiliaries."""

    @pytest.mark.parametrize(
        "text,expected_surfaces",
        [
            ("…た", ["…", "た"]),
            ("~」た", ["~」", "た"]),
        ],
        ids=["ellipsis-ta", "symbols-ta"],
    )
    def test_symbol_does_not_merge_with_ta(self, mecab_controller, text, expected_surfaces):
        """Ellipsis and other symbols should not merge with た auxiliary."""
        tokens = list(mecab_controller.translate(text))
        merged = merge_tokens(tokens)

        surfaces = [m.surface for m in merged]
        assert surfaces == expected_surfaces

        # Verify no merging occurred
        assert all(not m.is_merged for m in merged)


class TestVerbAuxiliaryMerge:
    """Bug fix verification: verbs should merge with auxiliaries."""

    @pytest.mark.parametrize(
        "text,expected_surface,expected_headword",
        [
            ("食べた", "食べた", "食べる"),
            ("走った", "走った", "走る"),
            ("書いた", "書いた", "書く"),
        ],
        ids=["tabeta", "hashitta", "kaita"],
    )
    def test_verb_merges_with_ta(self, mecab_controller, text, expected_surface, expected_headword):
        """Verbs should merge with た auxiliary."""
        tokens = list(mecab_controller.translate(text))
        merged = merge_tokens(tokens)

        assert len(merged) == 1
        assert merged[0].surface == expected_surface
        assert merged[0].headword == expected_headword
        assert merged[0].is_merged is True

    def test_verb_merges_with_ba_particle(self, mecab_controller):
        """Verbs should merge with ば conditional particle."""
        tokens = list(mecab_controller.translate("走れば"))
        merged = merge_tokens(tokens)

        assert len(merged) == 1
        assert merged[0].surface == "走れば"
        assert merged[0].headword == "走る"
        assert merged[0].is_merged is True

    def test_verb_merges_with_tatte_particle(self, mecab_controller):
        """Verbs should merge with たって particle."""
        tokens = list(mecab_controller.translate("言ったって"))
        merged = merge_tokens(tokens)

        assert len(merged) == 1
        assert merged[0].surface == "言ったって"
        assert merged[0].headword == "言う"
        assert merged[0].is_merged is True


class TestAuxiliaryAuxiliaryMerge:
    """Bug fix verification: auxiliaries should merge with other auxiliaries."""

    def test_nai_merges_with_ta(self, mecab_controller):
        """なかった (ない + た) should merge."""
        tokens = list(mecab_controller.translate("なかった"))
        merged = merge_tokens(tokens)

        assert len(merged) == 1
        assert merged[0].surface == "なかった"
        assert merged[0].headword == "ない"
        assert merged[0].is_merged is True


class TestAdjectiveAuxiliaryMerge:
    """Bug fix verification: i-adjectives should merge with auxiliaries."""

    @pytest.mark.parametrize(
        "text,expected_surface,expected_headword",
        [
            ("高かった", "高かった", "高い"),
            ("寒かった", "寒かった", "寒い"),
            ("美味しかった", "美味しかった", "美味しい"),
        ],
        ids=["takakatta", "samukatta", "oishikatta"],
    )
    def test_i_adjective_merges_with_ta(self, mecab_controller, text, expected_surface, expected_headword):
        """I-adjectives should merge with た auxiliary."""
        tokens = list(mecab_controller.translate(text))
        merged = merge_tokens(tokens)

        assert len(merged) == 1
        assert merged[0].surface == expected_surface
        assert merged[0].headword == expected_headword
        assert merged[0].is_merged is True


class TestMergerExpectedOutput:
    """Test cases with specific expected merged output."""

    @pytest.mark.parametrize(
        "text,expected_merged",
        [
            ("この世界の片隅に", "この|世界|の|片隅|に"),
            ("ぐらい上目遣いで言った方がやる気出るぜ？", "ぐらい|上目遣い|で|言った|方|が|やる気|出る|ぜ|？"),
            ("奇襲でもされたときに君が真っ先にやられると全滅確定", "奇襲|で|も|された|とき|に|君|が|真っ先|に|やられる|と|全滅|確定"),
            ("でも、頑張って", "でも|、|頑張って"),
            ("そっかそっか。ならま、いいんじゃねーかな", "そっ|か|そっ|か|。|なら|ま|、|いい|ん|じゃ|ねー|か|な"),
            ("立ち止まった少女に人混みをかき分けて歩み寄り", "立ち止まった|少女|に|人混み|を|かき分けて|歩み寄り"),
            ("Jonathanです", "Jonathan|です"),
            ("結局最後まで付き合わせてしまいましたね", "結局|最後|まで|付き合わせてしまいました|ね"),
        ],
        ids=[
            "kono-sekai",
            "uemezukai",
            "kishuu",
            "demo-ganbatte",
            "sokka",
            "tachitomatta",
            "jonathan-desu",
            "te-shimau",
        ],
    )
    def test_expected_merged_output(self, mecab_controller, text, expected_merged):
        """Verify merged output matches expected tokenization."""
        tokens = list(mecab_controller.translate(text))
        merged = merge_tokens(tokens)

        actual_merged = "|".join(m.surface for m in merged)
        assert actual_merged == expected_merged
