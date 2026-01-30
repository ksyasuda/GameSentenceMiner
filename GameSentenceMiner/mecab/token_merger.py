"""
Token merger for combining MeCab tokens for display/highlighting purposes.

Implements Yomitan-style logic to group verbs with their auxiliaries
and particles for more natural word boundaries.
"""

from dataclasses import dataclass
from typing import List, Optional

try:
    from .basic_types import MecabParsedToken, PartOfSpeech
except ImportError:
    from basic_types import MecabParsedToken, PartOfSpeech


@dataclass
class MergedToken:
    """A token that may represent multiple merged MeCab tokens."""

    surface: str  # Combined surface form (e.g., "食べて")
    reading: str  # Combined reading
    headword: str  # Dictionary form of the main word
    start_pos: int  # Character offset in original text
    end_pos: int  # End character offset
    part_of_speech: PartOfSpeech
    is_merged: bool = False  # True if this token was merged from multiple


def _is_noun(tok: MecabParsedToken) -> bool:
    """isNoun = (tok) => tok.pos1 === '名詞'"""
    return tok.part_of_speech == PartOfSpeech.noun


def _is_proper_noun(tok: MecabParsedToken) -> bool:
    """isProperNoun = (tok) => tok.pos1 === '名詞' && tok.pos2 === '固有名詞'"""
    return tok.part_of_speech == PartOfSpeech.noun and tok.pos2 == "固有名詞"


def _ignore_reading(tok: MecabParsedToken) -> bool:
    """
    ignoreReading = (tok) => tok.pos1 === '記号' && tok.pos2 === '文字'

    Tokens matching this should have their reading cleared (not contribute to merged reading).
    """
    return tok.part_of_speech == PartOfSpeech.symbol and tok.pos2 == "文字"


def _is_copula(tok: MecabParsedToken) -> bool:
    """
    isCopula = (tok) => tok.inflection_type === '特殊|だ' || tok.inflection_type === '特殊|デス'
    """
    raw = tok.inflection_type_raw
    if not raw:
        return False
    if raw in ("特殊・ダ", "特殊・デス", "特殊|だ", "特殊|デス"):
        return True
    return False


def _is_aux_verb(tok: MecabParsedToken) -> bool:
    """isAuxVerb = (tok) => tok.pos1 === '助動詞' && !isCopula(tok)"""
    return tok.part_of_speech == PartOfSpeech.bound_auxiliary and not _is_copula(tok)


def _is_continuative_form(tok: MecabParsedToken) -> bool:
    """
    isContinuativeForm = (tok) => (tok.inflection_form === '連用デ接続' ||
                                    tok.inflection_form === '連用タ接続' ||
                                    tok.inflection_form.startsWith('連用形')) &&
                                   (tok.reading !== 'ない')
    """
    if not tok.inflection_form:
        return False
    inflection_form = tok.inflection_form
    is_continuative = (
        inflection_form == "連用デ接続"
        or inflection_form == "連用タ接続"
        or inflection_form.startswith("連用形")
    )
    if not is_continuative:
        return False
    # Exclude ない - use headword (lemma) check since we store katakana reading
    return tok.headword != "ない"


def _is_verb_suffix(tok: MecabParsedToken) -> bool:
    """
    isVerbSuffix (ipadic) = (tok) => tok.pos1 === '動詞' && (tok.pos2 === '非自立' || tok.pos2 === '接尾')

    Examples:
    - 待ってるじゃないです : てる is 動詞,非自立,*,*,一段,基本形,てる,テル,テル
    - やられる : れる is 動詞,接尾,*,*,一段,基本形,れる,レル,レル
    """
    return tok.part_of_speech == PartOfSpeech.verb and tok.pos2 in ("非自立", "接尾")


def _is_tatte_particle(tok: MecabParsedToken) -> bool:
    """isTatteParticle = (tok) => tok.pos1 === '助詞' && tok.pos2 === '接続助詞' && (tok.lemma === 'たって')"""
    return (
        tok.part_of_speech == PartOfSpeech.particle
        and tok.pos2 == "接続助詞"
        and tok.headword == "たって"  # headword is lemma in our implementation
    )


def _is_ba_particle(tok: MecabParsedToken) -> bool:
    """isBaParticle = (tok) => tok.pos1 === '助詞' && tok.pos2 === '接続助詞' && (tok.term === 'ば')"""
    return (
        tok.part_of_speech == PartOfSpeech.particle
        and tok.pos2 == "接続助詞"
        and tok.word == "ば"
    )


def _is_te_de_particle(tok: MecabParsedToken) -> bool:
    """isTeDeParticle = (tok) => tok.pos1 === '助詞' && tok.pos2 === '接続助詞' &&
    (tok.term === 'て' || tok.term === 'で' || tok.term === 'ちゃ')"""
    return (
        tok.part_of_speech == PartOfSpeech.particle
        and tok.pos2 == "接続助詞"
        and tok.word in ("て", "で", "ちゃ")
    )


def _is_ta_da_particle(tok: MecabParsedToken) -> bool:
    """isTaDaParticle = (tok) => isAuxVerb(tok) && (tok.term === 'た' || tok.term === 'だ')"""
    return _is_aux_verb(tok) and tok.word in ("た", "だ")


def _is_verb(tok: MecabParsedToken) -> bool:
    """isVerb = (tok) => tok.pos1 === '動詞' || tok.pos1 === '助動詞'"""
    return tok.part_of_speech in (PartOfSpeech.verb, PartOfSpeech.bound_auxiliary)


def _is_verb_non_independent(tok: MecabParsedToken) -> bool:
    """isVerbNonIndependent = (_) => true  (for ipadic, always true)"""
    return True


def _can_receive_auxiliary(tok: MecabParsedToken) -> bool:
    """Check if token can grammatically receive auxiliaries like た/だ/ば/たって."""
    return tok.part_of_speech in (
        PartOfSpeech.verb,
        PartOfSpeech.bound_auxiliary,
        PartOfSpeech.i_adjective,
    )


def _is_noun_suffix(tok: MecabParsedToken) -> bool:
    """
    isNounSuffix (ipadic) = (tok) => tok.pos1 === '動詞' && tok.pos2 === '接尾'
    """
    return tok.part_of_speech == PartOfSpeech.verb and tok.pos2 == "接尾"


def _is_counter(tok: MecabParsedToken) -> bool:
    """isCounter = (tok) => tok.pos1 === '名詞' && tok.pos3.startsWith('助数詞')"""
    return (
        tok.part_of_speech == PartOfSpeech.noun
        and tok.pos3 is not None
        and tok.pos3.startswith("助数詞")
    )


def _is_numeral(tok: MecabParsedToken) -> bool:
    """isNumeral = (tok) => tok.pos1 === '名詞' && tok.pos2.startsWith('数')"""
    return (
        tok.part_of_speech == PartOfSpeech.noun
        and tok.pos2 is not None
        and tok.pos2.startswith("数")
    )


def _should_merge(
    last_standalone_token: MecabParsedToken, token: MecabParsedToken
) -> bool:
    """
    Determine if current token should merge with the last standalone token.
    should_merge = (isVerb(last_standalone_token) && (isAuxVerb(token) ||
                    (isContinuativeForm(last_standalone_token) && isVerbSuffix(token)) ||
                    (isVerbSuffix(token) && isVerbNonIndependent(last_standalone_token)))) ||
        (isNoun(last_standalone_token) && !isProperNoun(last_standalone_token) && isNounSuffix(token)) ||
        (isCounter(token) && isNumeral(last_standalone_token)) ||
        isBaParticle(token) || isTatteParticle(token) ||
        (isTeDeParticle(token) && isContinuativeForm(last_standalone_token)) ||
        isTaDaParticle(token);
    """
    # Verb + auxiliary verb / verb suffix combinations
    if _is_verb(last_standalone_token):
        if _is_aux_verb(token):
            return True
        if _is_continuative_form(last_standalone_token) and _is_verb_suffix(token):
            return True
        if _is_verb_suffix(token) and _is_verb_non_independent(last_standalone_token):
            return True

    # Noun (non-proper) + noun suffix
    if (
        _is_noun(last_standalone_token)
        and not _is_proper_noun(last_standalone_token)
        and _is_noun_suffix(token)
    ):
        return True

    # Numeral + counter
    if _is_counter(token) and _is_numeral(last_standalone_token):
        return True

    # ば particle (conditional) - only merges with verb/auxiliary/adjective
    if _is_ba_particle(token) and _can_receive_auxiliary(last_standalone_token):
        return True

    # たって particle - only merges with verb/auxiliary/adjective
    if _is_tatte_particle(token) and _can_receive_auxiliary(last_standalone_token):
        return True

    # て/で particle after continuative form
    if _is_te_de_particle(token) and _is_continuative_form(last_standalone_token):
        return True

    # た/だ auxiliary - only merges with verb/auxiliary/adjective (e.g., なかった)
    if _is_ta_da_particle(token) and _can_receive_auxiliary(last_standalone_token):
        return True

    # After て/で particle, merge with auxiliary verbs (しまう, いる, ある, おく, みる, etc.)
    if _is_te_de_particle(last_standalone_token) and _is_verb_suffix(token):
        return True

    return False


def merge_tokens(tokens: List[MecabParsedToken]) -> List[MergedToken]:
    """
    Merge MeCab tokens using Yomitan-style grammatical rules.
    Args:
        tokens: List of MeCab parsed tokens

    Returns:
        List of merged tokens with character positions
    """
    if not tokens:
        return []

    result: List[MergedToken] = []
    char_offset = 0
    tokens_list = list(tokens)

    last_standalone_token: Optional[MecabParsedToken] = None

    for token in tokens_list:
        # Calculate character positions
        start = char_offset
        end = char_offset + len(token.word)
        char_offset = end

        should_merge = False

        if result and last_standalone_token is not None:
            should_merge = _should_merge(last_standalone_token, token)

        # Compute reading (apply ignoreReading rule)
        token_reading = "" if _ignore_reading(token) else (token.katakana_reading or token.word)

        if should_merge and result:
            # Merge with previous result token
            prev = result.pop()
            merged = MergedToken(
                surface=prev.surface + token.word,
                reading=prev.reading + token_reading,
                headword=prev.headword,  # Keep original headword
                start_pos=prev.start_pos,
                end_pos=end,
                part_of_speech=prev.part_of_speech,
                is_merged=True,
            )
            result.append(merged)
        else:
            result.append(
                MergedToken(
                    surface=token.word,
                    reading=token_reading,
                    headword=token.headword,
                    start_pos=start,
                    end_pos=end,
                    part_of_speech=token.part_of_speech,
                    is_merged=False,
                )
            )

        # Always update last_standalone_token to current raw token
        # This is the key difference from the previous implementation
        last_standalone_token = token

    return result
