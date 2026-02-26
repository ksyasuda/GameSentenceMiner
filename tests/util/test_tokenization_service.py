import time

from GameSentenceMiner.util.database.db import (
    GameLinesTable,
    KanjiOccurrencesTable,
    KanjiTable,
    WordOccurrencesTable,
    WordsTable,
)


def _reset_tokenization_tables() -> None:
    GameLinesTable._db.execute(f"DELETE FROM {WordOccurrencesTable._table}", commit=True)
    GameLinesTable._db.execute(f"DELETE FROM {KanjiOccurrencesTable._table}", commit=True)
    GameLinesTable._db.execute(f"DELETE FROM {WordsTable._table}", commit=True)
    GameLinesTable._db.execute(f"DELETE FROM {KanjiTable._table}", commit=True)
    GameLinesTable._db.execute(f"DELETE FROM {GameLinesTable._table}", commit=True)


def _insert_line(line_id: str, text: str, ts: float = 1700000000.0) -> None:
    GameLinesTable(
        id=line_id,
        game_name="Test",
        line_text=text,
        timestamp=ts,
        game_id="game_1",
    ).add()


def test_duplicate_tokens_in_same_line_count_toward_frequency() -> None:
    from GameSentenceMiner.util.tokenization_service import get_tokenization_service

    _reset_tokenization_tables()
    _insert_line("line_dup", "学校学校")
    service = get_tokenization_service()

    def _mock_tokenize_texts(texts):
        assert texts == ["学校学校"]
        return [[
            {"word": "学校", "headword": "学校", "reading": "ガッコウ"},
            {"word": "学校", "headword": "学校", "reading": "ガッコウ"},
        ]]

    service._tokenize_texts = _mock_tokenize_texts
    result = service.tokenize_lines_batch(
        [("line_dup", "学校学校", time.time(), "game_1")]
    )
    assert result["processed"] == 1
    assert result["failed"] == 0

    line = GameLinesTable.get("line_dup")
    assert line is not None
    assert int(line.tokenized) == 1
    assert int(line.word_count) == 2
    assert int(line.kanji_count) == 4

    word_row = GameLinesTable._db.fetchone(
        f"SELECT id, frequency FROM {WordsTable._table} WHERE headword=? AND word=? AND reading=?",
        ("学校", "学校", "ガッコウ"),
    )
    assert word_row is not None
    word_id, frequency = int(word_row[0]), int(word_row[1])
    assert frequency == 2

    occ_row = GameLinesTable._db.fetchone(
        f"SELECT count FROM {WordOccurrencesTable._table} WHERE word_id=? AND line_id=?",
        (word_id, "line_dup"),
    )
    assert occ_row is not None
    assert int(occ_row[0]) == 2

    kanji_freq = dict(
        GameLinesTable._db.fetchall(
            f"SELECT kanji, frequency FROM {KanjiTable._table}"
        )
    )
    assert kanji_freq["学"] == 2
    assert kanji_freq["校"] == 2


def test_updating_existing_line_reconciles_occurrences_and_cache() -> None:
    from GameSentenceMiner.util.tokenization_service import get_tokenization_service

    _reset_tokenization_tables()
    _insert_line("line_update", "学校学校", ts=1700000001.0)
    service = get_tokenization_service()

    service._tokenize_texts = lambda _texts: [[
        {"word": "学校", "headword": "学校", "reading": "ガッコウ"},
        {"word": "学校", "headword": "学校", "reading": "ガッコウ"},
    ]]
    service.tokenize_lines_batch([("line_update", "学校学校", 1700000001.0, "game_1")])

    GameLinesTable._db.execute(
        f"UPDATE {GameLinesTable._table} SET line_text=?, timestamp=? WHERE id=?",
        ("先生", 1700000002.0, "line_update"),
        commit=True,
    )
    service._tokenize_texts = lambda _texts: [[
        {"word": "先生", "headword": "先生", "reading": "センセイ"},
    ]]
    result = service.tokenize_lines_batch([("line_update", "先生", 1700000002.0, "game_1")])
    assert result["processed"] == 1
    assert result["failed"] == 0

    old_word = GameLinesTable._db.fetchone(
        f"SELECT COUNT(*) FROM {WordsTable._table} WHERE headword=?",
        ("学校",),
    )
    assert int(old_word[0]) == 0

    new_word = GameLinesTable._db.fetchone(
        f"SELECT frequency FROM {WordsTable._table} WHERE headword=? AND word=? AND reading=?",
        ("先生", "先生", "センセイ"),
    )
    assert new_word is not None
    assert int(new_word[0]) == 1

    line = GameLinesTable.get("line_update")
    assert line is not None
    assert int(line.word_count) == 1
    assert int(line.kanji_count) == 2


def test_delete_line_decrements_frequency_cache() -> None:
    from GameSentenceMiner.util.tokenization_service import get_tokenization_service

    _reset_tokenization_tables()
    _insert_line("line_delete", "学校学校", ts=1700000010.0)
    service = get_tokenization_service()
    service._tokenize_texts = lambda _texts: [[
        {"word": "学校", "headword": "学校", "reading": "ガッコウ"},
        {"word": "学校", "headword": "学校", "reading": "ガッコウ"},
    ]]
    service.tokenize_lines_batch([("line_delete", "学校学校", 1700000010.0, "game_1")])

    GameLinesTable.delete_line("line_delete")

    assert GameLinesTable.get("line_delete") is None
    assert GameLinesTable._db.fetchone(f"SELECT COUNT(*) FROM {WordOccurrencesTable._table}")[0] == 0
    assert GameLinesTable._db.fetchone(f"SELECT COUNT(*) FROM {KanjiOccurrencesTable._table}")[0] == 0
    assert GameLinesTable._db.fetchone(f"SELECT COUNT(*) FROM {WordsTable._table}")[0] == 0
    assert GameLinesTable._db.fetchone(f"SELECT COUNT(*) FROM {KanjiTable._table}")[0] == 0


def test_non_japanese_line_marks_tokenized_with_zero_counts() -> None:
    from GameSentenceMiner.util.tokenization_service import get_tokenization_service

    _reset_tokenization_tables()
    _insert_line("line_ascii", "hello world", ts=1700000020.0)
    service = get_tokenization_service()

    result = service.tokenize_lines_batch([("line_ascii", "hello world", 1700000020.0, "game_1")])
    assert result["processed"] == 1
    assert result["failed"] == 0

    line = GameLinesTable.get("line_ascii")
    assert line is not None
    assert int(line.tokenized) == 1
    assert int(line.word_count) == 0
    assert int(line.kanji_count) == 0
    assert int(line.total_length) == len("hello world")
    assert int(line.filtered_length) == 0


def test_extract_tokens_parses_yomitan_scan_segments() -> None:
    from GameSentenceMiner.util.tokenization_service import _extract_tokens_from_yomitan_payload

    payload = [
        {
            "id": "scan",
            "source": "scanning-parser",
            "index": 0,
            "content": [
                [{"text": "そうかあ", "reading": ""}],
                [
                    {"text": "レイプ", "reading": "レイプ", "headwords": [[{"term": "レイプ", "reading": "レイプ", "sources": []}]]},
                ],
                [{"text": "目", "reading": "め"}],
            ],
        }
    ]

    tokens = _extract_tokens_from_yomitan_payload(payload)
    assert [token["word"] for token in tokens] == ["そうかあ", "レイプ", "目"]
    assert [token["reading"] for token in tokens] == ["-", "レイプ", "め"]
    assert [token["headword"] for token in tokens] == ["そうかあ", "レイプ", "目"]


def test_extract_tokens_ignores_fully_unparsed_scan_result() -> None:
    from GameSentenceMiner.util.tokenization_service import _extract_tokens_from_yomitan_payload

    payload = [
        {
            "id": "scan",
            "source": "scanning-parser",
            "index": 0,
            "content": [[{"text": "じさあたま時差のせいで頭がくらくらしている", "reading": ""}]],
        }
    ]

    assert _extract_tokens_from_yomitan_payload(payload) == []
