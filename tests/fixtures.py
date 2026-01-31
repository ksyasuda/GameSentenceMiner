"""
Test fixtures and factory functions for GameSentenceMiner tests.

Provides easy-to-use functions for creating test data:
- Games
- Game lines
- Words
- Kanji
- Occurrences

Also includes sample Japanese text for tokenization testing.
"""

import uuid
import time
from typing import Optional, List, Dict, Any


SAMPLE_JAPANESE_LINES = [
    # Basic sentences
    "今日は天気がいいですね。",                      # Weather, polite form
    "私は日本語を勉強しています。",                   # Studying Japanese, progressive
    "この本はとても面白いです。",                     # Adjective, polite
    "昨日、友達と映画を見ました。",                   # Past tense, with friend
    "明日は学校に行きません。",                       # Negative future

    # More complex sentences
    "彼女は毎朝六時に起きて、朝ご飯を食べてから会社に行きます。",
    "日本の文化について興味があるので、もっと勉強したいと思います。",
    "この問題は難しすぎて、私には解けないかもしれません。",
    "先生に質問したら、親切に教えてくれました。",
    "雨が降っているから、傘を持って行った方がいいよ。",

    # Game/VN-style dialogue
    "「待って！そんなこと言わないで！」",
    "「ふふ、やっと会えたね。ずっと待っていたのよ」",
    "「この世界の秘密を、君だけに教えてあげよう」",
    "選択肢が表示された。どうする？",
    "主人公は静かに部屋を出た。",

    # Technical/modern vocabulary
    "スマートフォンのアプリをダウンロードしてください。",
    "オンラインゲームで友達と遊ぶのが好きです。",
    "プログラミングの勉強を始めました。",
    "インターネットで情報を検索する。",
    "コンピューターが動かなくなった。",

    # Sentences with many kanji
    "経済状況の改善について議論する必要がある。",
    "環境問題に関する国際会議が開催された。",
    "科学技術の発展により生活が便利になった。",
    "政治的決定は国民の生活に影響を与える。",
    "歴史的建造物の保存修復工事が完了した。",

    # Short sentences
    "はい。",
    "分かった。",
    "そうですか。",
    "ありがとう！",
    "えっ？",
]


SAMPLE_WORDS = [
    {"headword": "食べる", "word": "食べ", "reading": "タベ", "pos": "verb"},
    {"headword": "見る", "word": "見", "reading": "ミ", "pos": "verb"},
    {"headword": "日本語", "word": "日本語", "reading": "ニホンゴ", "pos": "noun"},
    {"headword": "勉強", "word": "勉強", "reading": "ベンキョウ", "pos": "noun"},
    {"headword": "面白い", "word": "面白い", "reading": "オモシロイ", "pos": "i_adjective"},
    {"headword": "綺麗", "word": "綺麗", "reading": "キレイ", "pos": "na_adjective"},
    {"headword": "行く", "word": "行っ", "reading": "イッ", "pos": "verb"},
    {"headword": "来る", "word": "来", "reading": "キ", "pos": "verb"},
    {"headword": "する", "word": "し", "reading": "-", "pos": "verb"},
    {"headword": "言う", "word": "言わ", "reading": "イワ", "pos": "verb"},
]


SAMPLE_KANJI = [
    "日", "本", "語", "私", "今", "天", "気", "友", "達", "映",
    "画", "学", "校", "先", "生", "質", "問", "親", "切", "雨",
    "傘", "世", "界", "秘", "密", "経", "済", "環", "境", "科",
]


def create_game(
    id: Optional[str] = None,
    title_original: str = "テストゲーム",
    title_english: str = "Test Game",
    title_romaji: str = "Tesuto Geemu",
    game_type: str = "visual_novel",
    deck_id: Optional[int] = None,
    completed: bool = False,
    **kwargs
) -> "GamesTable":
    """
    Create and save a game record to the test database.

    Args:
        id: Optional UUID. Generated if not provided.
        title_original: Japanese title
        title_english: English title
        title_romaji: Romanized title
        game_type: Type of game
        deck_id: jiten.moe deck ID
        completed: Whether game is completed
        **kwargs: Additional fields to set

    Returns:
        The created GamesTable instance
    """
    from GameSentenceMiner.util.games_table import GamesTable

    game = GamesTable(
        id=id or str(uuid.uuid4()),
        title_original=title_original,
        title_english=title_english,
        title_romaji=title_romaji,
        game_type=game_type,
        deck_id=deck_id,
        completed=completed,
        obs_scene_name=title_original,
        **kwargs
    )
    game.add()
    return game


def create_game_line(
    id: Optional[str] = None,
    game_id: str = "",
    game_name: str = "テストゲーム",
    line_text: str = "テストテキスト",
    timestamp: Optional[float] = None,
    tokenized: int = 0,
    **kwargs
) -> "GameLinesTable":
    """
    Create and save a game line record to the test database.

    Args:
        id: Optional UUID. Generated if not provided.
        game_id: Foreign key to games table
        game_name: Name of the game (OBS scene name)
        line_text: The Japanese text content
        timestamp: Unix timestamp. Defaults to current time.
        tokenized: Whether line has been tokenized (0 or 1)
        **kwargs: Additional fields to set

    Returns:
        The created GameLinesTable instance
    """
    from GameSentenceMiner.util.db import GameLinesTable

    line = GameLinesTable()
    line.id = id or str(uuid.uuid4())
    line.game_id = game_id
    line.game_name = game_name
    line.line_text = line_text
    line.timestamp = timestamp if timestamp is not None else time.time()
    line.tokenized = tokenized
    line.screenshot_in_anki = ""
    line.audio_in_anki = ""
    line.screenshot_path = ""
    line.audio_path = ""
    line.replay_path = ""
    line.translation = ""
    line.original_game_name = game_name
    line.note_ids = []
    line.last_modified = time.time()
    line.total_length = 0
    line.filtered_length = 0
    line.word_count = 0
    line.kanji_count = 0

    for key, value in kwargs.items():
        if hasattr(line, key):
            setattr(line, key, value)

    line.add()
    return line


def create_word(
    headword: str,
    word: str,
    reading: str = "-",
    pos: str = "noun",
    first_seen: Optional[float] = None,
    last_seen: Optional[float] = None,
    frequency: int = 1,
) -> "WordsTable":
    """
    Create and save a word record to the test database.

    Args:
        headword: Dictionary form of the word
        word: Surface form as it appeared
        reading: Katakana reading ("-" if same as headword)
        pos: Part of speech
        first_seen: First seen timestamp
        last_seen: Last seen timestamp
        frequency: Occurrence count

    Returns:
        The created WordsTable instance
    """
    from GameSentenceMiner.util.db import WordsTable

    now = time.time()
    word_obj = WordsTable()
    word_obj.headword = headword
    word_obj.word = word
    word_obj.reading = reading
    word_obj.pos = pos
    word_obj.first_seen = first_seen if first_seen is not None else now
    word_obj.last_seen = last_seen if last_seen is not None else now
    word_obj.frequency = frequency
    word_obj.save()
    return word_obj


def create_kanji(
    kanji: str,
    first_seen: Optional[float] = None,
    last_seen: Optional[float] = None,
    frequency: int = 1,
) -> "KanjiTable":
    """
    Create and save a kanji record to the test database.

    Args:
        kanji: Single kanji character
        first_seen: First seen timestamp
        last_seen: Last seen timestamp
        frequency: Occurrence count

    Returns:
        The created KanjiTable instance
    """
    from GameSentenceMiner.util.db import KanjiTable

    now = time.time()
    kanji_obj = KanjiTable()
    kanji_obj.kanji = kanji
    kanji_obj.first_seen = first_seen if first_seen is not None else now
    kanji_obj.last_seen = last_seen if last_seen is not None else now
    kanji_obj.frequency = frequency
    kanji_obj.save()
    return kanji_obj


def create_word_occurrence(
    word_id: int,
    line_id: str,
    game_id: str = "",
    timestamp: Optional[float] = None,
    position: int = 0,
) -> "WordOccurrencesTable":
    """
    Create and save a word occurrence record.

    Args:
        word_id: Foreign key to words table
        line_id: Foreign key to game_lines table
        game_id: Foreign key to games table
        timestamp: Unix timestamp
        position: Position in the sentence

    Returns:
        The created WordOccurrencesTable instance
    """
    from GameSentenceMiner.util.db import WordOccurrencesTable

    occ = WordOccurrencesTable()
    occ.word_id = word_id
    occ.line_id = line_id
    occ.game_id = game_id or None
    occ.timestamp = timestamp if timestamp is not None else time.time()
    occ.position = position
    occ.save()
    return occ


def create_kanji_occurrence(
    kanji_id: int,
    line_id: str,
    game_id: str = "",
    timestamp: Optional[float] = None,
    position: int = 0,
) -> "KanjiOccurrencesTable":
    """
    Create and save a kanji occurrence record.

    Args:
        kanji_id: Foreign key to kanji table
        line_id: Foreign key to game_lines table
        game_id: Foreign key to games table
        timestamp: Unix timestamp
        position: Position in the sentence

    Returns:
        The created KanjiOccurrencesTable instance
    """
    from GameSentenceMiner.util.db import KanjiOccurrencesTable

    occ = KanjiOccurrencesTable()
    occ.kanji_id = kanji_id
    occ.line_id = line_id
    occ.game_id = game_id or None
    occ.timestamp = timestamp if timestamp is not None else time.time()
    occ.position = position
    occ.save()
    return occ


def bulk_create_game_lines(
    game_id: str,
    game_name: str,
    texts: List[str],
    base_timestamp: Optional[float] = None,
    interval: float = 60.0,
) -> List["GameLinesTable"]:
    """
    Create multiple game lines for a game.

    Args:
        game_id: Foreign key to games table
        game_name: Name of the game
        texts: List of Japanese text strings
        base_timestamp: Starting timestamp (defaults to yesterday)
        interval: Seconds between each line

    Returns:
        List of created GameLinesTable instances
    """
    if base_timestamp is None:
        base_timestamp = time.time() - 86400

    lines = []
    for i, text in enumerate(texts):
        line = create_game_line(
            game_id=game_id,
            game_name=game_name,
            line_text=text,
            timestamp=base_timestamp + (i * interval),
        )
        lines.append(line)
    return lines


def bulk_create_words(word_dicts: List[Dict[str, Any]]) -> List["WordsTable"]:
    """
    Create multiple words from a list of dictionaries.

    Args:
        word_dicts: List of dicts with keys: headword, word, reading, pos

    Returns:
        List of created WordsTable instances
    """
    words = []
    for wd in word_dicts:
        word = create_word(
            headword=wd["headword"],
            word=wd["word"],
            reading=wd.get("reading", "-"),
            pos=wd.get("pos", "noun"),
        )
        words.append(word)
    return words


def bulk_create_kanji(kanji_chars: List[str]) -> List["KanjiTable"]:
    """
    Create multiple kanji records from a list of characters.

    Args:
        kanji_chars: List of single kanji characters

    Returns:
        List of created KanjiTable instances
    """
    kanji_records = []
    for char in kanji_chars:
        kanji = create_kanji(kanji=char)
        kanji_records.append(kanji)
    return kanji_records


class DataBuilder:
    """
    Fluent builder for creating complex test data scenarios.

    Usage:
        builder = TestDataBuilder()
        data = (builder
            .with_game("Test Game")
            .with_lines(["Line 1", "Line 2", "Line 3"])
            .with_tokenization()
            .build())
    """

    def __init__(self):
        self._game: Optional["GamesTable"] = None
        self._lines: List["GameLinesTable"] = []
        self._words: List["WordsTable"] = []
        self._kanji: List["KanjiTable"] = []
        self._should_tokenize = False
        self._base_timestamp = time.time() - 86400

    def with_game(
        self,
        title_original: str = "テストゲーム",
        title_english: str = "Test Game",
        **kwargs
    ) -> "DataBuilder":
        """Add a game to the test data."""
        self._game = create_game(
            title_original=title_original,
            title_english=title_english,
            **kwargs
        )
        return self

    def with_lines(
        self,
        texts: Optional[List[str]] = None,
        count: int = 5,
    ) -> "DataBuilder":
        """
        Add game lines to the test data.

        Args:
            texts: Custom line texts. If None, uses SAMPLE_JAPANESE_LINES
            count: Number of lines to create (used if texts is None)
        """
        if self._game is None:
            self.with_game()

        if texts is None:
            texts = SAMPLE_JAPANESE_LINES[:count]

        self._lines = bulk_create_game_lines(
            game_id=self._game.id,
            game_name=self._game.title_original,
            texts=texts,
            base_timestamp=self._base_timestamp,
        )
        return self

    def with_words(
        self,
        words: Optional[List[Dict[str, Any]]] = None,
    ) -> "DataBuilder":
        """
        Add words to the test data.

        Args:
            words: Custom word dicts. If None, uses SAMPLE_WORDS
        """
        if words is None:
            words = SAMPLE_WORDS
        self._words = bulk_create_words(words)
        return self

    def with_kanji(
        self,
        kanji_chars: Optional[List[str]] = None,
    ) -> "DataBuilder":
        """
        Add kanji to the test data.

        Args:
            kanji_chars: Custom kanji list. If None, uses SAMPLE_KANJI
        """
        if kanji_chars is None:
            kanji_chars = SAMPLE_KANJI
        self._kanji = bulk_create_kanji(kanji_chars)
        return self

    def with_tokenization(self) -> "DataBuilder":
        """Mark that lines should be tokenized after creation."""
        self._should_tokenize = True
        return self

    def at_timestamp(self, timestamp: float) -> "DataBuilder":
        """Set the base timestamp for line creation."""
        self._base_timestamp = timestamp
        return self

    def build(self) -> Dict[str, Any]:
        """
        Build and return all created test data.

        If with_tokenization() was called, tokenizes all lines using
        the TokenizationService.

        Returns:
            Dict with keys: game, lines, words, kanji
        """
        if self._should_tokenize and self._lines:
            from GameSentenceMiner.util.tokenization_service import TokenizationService

            service = TokenizationService()
            for line in self._lines:
                service.tokenize_line(
                    game_line_id=line.id,
                    line_text=line.line_text,
                    timestamp=line.timestamp,
                    game_id=line.game_id,
                )

        return {
            "game": self._game,
            "lines": self._lines,
            "words": self._words,
            "kanji": self._kanji,
        }


def get_untokenized_lines_for_batch(
    game_id: Optional[str] = None,
    limit: int = 100,
) -> List[tuple]:
    """
    Get untokenized lines in the format expected by tokenize_batch().

    Args:
        game_id: Optional game ID to filter by
        limit: Maximum number of lines to return

    Returns:
        List of tuples: (line_id, line_text, timestamp, game_id)
    """
    from GameSentenceMiner.util.db import GameLinesTable

    if game_id:
        query = f"SELECT id, line_text, timestamp, game_id FROM {GameLinesTable._table} WHERE tokenized=0 AND game_id=? LIMIT ?"
        rows = GameLinesTable._db.fetchall(query, (game_id, limit))
    else:
        query = f"SELECT id, line_text, timestamp, game_id FROM {GameLinesTable._table} WHERE tokenized=0 LIMIT ?"
        rows = GameLinesTable._db.fetchall(query, (limit,))

    return [(row[0], row[1], row[2], row[3]) for row in rows]
