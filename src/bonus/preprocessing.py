"""Text preprocessing for the AraBERT v0.2 sentiment reproduction.

The steps follow Wadhawan (2021), Section 3.1, for the
``bert-base-arabertv02`` model. Farasa segmentation is intentionally not
applied because this model was not pre-segmented (paper Table 1).
"""

from __future__ import annotations

import html
import re
import unicodedata

import pandas as pd

URL_PLACEHOLDER = "[رابط]"
EMAIL_PLACEHOLDER = "[بريد]"
MENTION_PLACEHOLDER = "[مستخدم]"

# Private-use sentinels keep placeholders intact while whitespace is inserted
# around brackets and Latin/digit runs. They are restored before returning.
_URL_SENTINEL = "\uE000"
_EMAIL_SENTINEL = "\uE001"
_MENTION_SENTINEL = "\uE002"

# A URL match stops at whitespace, "#", or any non-ASCII character, so glued
# hashtags and Arabic text survive as separate tokens. The class [!-"$-~] is
# every printable ASCII character except "#".
_URL_PATTERN = re.compile(r'(?:https?://|www\.)[!-"$-~]*', flags=re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_MENTION_PATTERN = re.compile(r"(?<!\w)@\w+")
_HTML_BREAK_PATTERN = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
# Only match real tags: a letter must follow "<" or "</". This keeps text such
# as "<3 ... >_<" from being consumed as a single HTML tag.
_HTML_TAG_PATTERN = re.compile(r"</?[A-Za-z][^>]*>")
_EMOTICON_PATTERN = re.compile(
    r"(?:"
    # Colon, semicolon, and equals emoticons, e.g. :) :D ;-) :'( :/ :|
    r"[:;=][\-^']?[)(\]\[dDpP3oOsSxXzZbBvV/\\|*<>]+"
    # Symbol emoticons, e.g. <3 ^_^ -_- T_T >_<
    r"|<3|</3|\^_?\^|\^\^|T_T|T\.T|-_-|o\.O|O\.o|>_<"
    # Letter/digit-starting emoticons only from this explicit whitelist.
    r"|(?<!\w)(?:xD|XD|D:|D=|8\)|8-\)\))(?!\w)"
    r")(?!\w)"
)
_REPEATED_NON_DIGIT_PATTERN = re.compile(r"(\D)\1{2,}", flags=re.DOTALL)
_LATIN_RUN_PATTERN = re.compile(r"([A-Za-z]+)")
_DIGIT_RUN_PATTERN = re.compile(r"([0-9]+)")
_BRACKET_PATTERN = re.compile(r"([\[\]])")
_INVISIBLE_CHARACTERS = {"\u200B", "\u200C", "\u200D", "\uFE0E", "\uFE0F", "\u20E3"}
_SENTINELS = {_URL_SENTINEL, _EMAIL_SENTINEL, _MENTION_SENTINEL}

# Tashkeel ranges and tatweel follow the official AraBERT preprocessor.
_TASHKEEL_AND_TATWEEL = (
    set(range(0x0617, 0x061B))
    | set(range(0x064B, 0x0653))
    | {0x0640}
)
_FITZPATRICK_MODIFIERS = set(range(0x1F3FB, 0x1F400))
_PRESENTATION_FORM_RANGES = ((0xFB50, 0xFDFF), (0xFE70, 0xFEFF))


def _normalize_presentation_forms(text: str) -> str:
    """Apply NFKC only to Arabic Presentation Forms characters."""
    normalized = []
    for character in text:
        codepoint = ord(character)
        if any(start <= codepoint <= end for start, end in _PRESENTATION_FORM_RANGES):
            normalized.append(unicodedata.normalize("NFKC", character))
        else:
            normalized.append(character)
    return "".join(normalized)


def _remove_emojis_and_symbols(text: str) -> str:
    """Remove emoji/symbol, invisible, private-use, tashkeel, and tatweel chars."""
    kept = []
    for character in text:
        if character in _SENTINELS:
            kept.append(character)
            continue
        if ord(character) in _TASHKEEL_AND_TATWEEL:
            continue
        if ord(character) in _FITZPATRICK_MODIFIERS:
            continue
        if character in _INVISIBLE_CHARACTERS:
            continue
        if unicodedata.category(character) in {"So", "Co", "Cn", "Cs", "Cf", "Cc"}:
            continue
        kept.append(character)
    return "".join(kept)


def preprocess_text(text: str) -> str:
    """Apply the paper's cleaning steps to one tweet.

    URLs, emails, and mentions are replaced before whitespace insertion so that
    the inserted spaces cannot break them. Private-use sentinels protect the
    placeholders until the final restoration step.
    """
    if pd.isna(text):
        return ""
    text = html.unescape(str(text))

    # Remove HTML line breaks and markup, then normalise line breaks to spaces.
    text = _HTML_BREAK_PATTERN.sub(" ", text)
    text = _HTML_TAG_PATTERN.sub(" ", text)
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").replace("\t", " ")

    # Replace URLs, emails, and mentions with placeholders.
    text = _URL_PATTERN.sub(f" {_URL_SENTINEL} ", text)
    text = _EMAIL_PATTERN.sub(f" {_EMAIL_SENTINEL} ", text)
    text = _MENTION_PATTERN.sub(f" {_MENTION_SENTINEL} ", text)

    # Normalize Arabic presentation forms to their canonical letters. NFKC is
    # applied per character so the rest of the tweet is left untouched.
    text = _normalize_presentation_forms(text)

    # Remove text emoticons and emoji/symbol characters.
    text = _EMOTICON_PATTERN.sub(" ", text)
    text = _remove_emojis_and_symbols(text)

    # Collapse runs of more than two identical non-digit characters to two.
    text = _REPEATED_NON_DIGIT_PATTERN.sub(r"\1\1", text)

    # Insert whitespace around Latin letters, Western digits, and square
    # brackets. This also separates Arabic words from adjacent numbers.
    text = _LATIN_RUN_PATTERN.sub(r" \1 ", text)
    text = _DIGIT_RUN_PATTERN.sub(r" \1 ", text)
    text = _BRACKET_PATTERN.sub(r" \1 ", text)

    # Restore placeholders after whitespace insertion so they remain intact.
    text = text.replace(_URL_SENTINEL, URL_PLACEHOLDER)
    text = text.replace(_EMAIL_SENTINEL, EMAIL_PLACEHOLDER)
    text = text.replace(_MENTION_SENTINEL, MENTION_PLACEHOLDER)

    # Replace slashes with a spaced dash, as in the official AraBERT v02
    # preprocessor. URLs are already placeholders at this point.
    text = text.replace("/", " - ")

    # Collapse extra spaces.
    return " ".join(text.split())


SANITY_EXAMPLES = (
    ("URL", "شاهد الفيديو https://t.co/abc123 الآن", "شاهد الفيديو [رابط] الآن"),
    ("Mention", "مرحبا @user_1 كيف حالك", "مرحبا [مستخدم] كيف حالك"),
    ("Hashtag", "هذا #فلول مهم", "هذا #فلول مهم"),
    ("Hashtag with Latin/digits", "#COVID19 منتشر", "# COVID 19 منتشر"),
    ("Repeated characters", "هههههههه والله", "هه والله"),
    ("Digits", "عام2021 ثم 2022عام", "عام 2021 ثم 2022 عام"),
    ("English letters", "هذاiPhoneجديد", "هذا iPhone جديد"),
    ("Brackets", "نص[مهم]جدا", "نص [ مهم ] جدا"),
    ("Emoticons", "أحبك :) 😍", "أحبك"),
    ("Extra spaces", "  هذا    نص   ", "هذا نص"),
    ("HTML", "<br>مرحبا</b>", "مرحبا"),
    ("Email", "راسلني test@example.com", "راسلني [بريد]"),
    (
        "URL with glued Arabic hashtag",
        "شاهد https://t.co/abc123#مصر الآن",
        "شاهد [رابط] #مصر الآن",
    ),
    (
        "URL with glued Latin hashtags",
        "https://t.co/cN7dLaIhnJ#tag1#tag2",
        "[رابط] # tag 1 # tag 2",
    ),
    ("www URL with glued hashtag", "www.example.com/path#قسم", "[رابط] #قسم"),
    ("Preserved bracket text", "[الزخرف:18]", "[ الزخرف: 18 ]"),
    ("Preserved word", "Naxos", "Naxos"),
    ("Preserved product name", "Dell XPS 13", "Dell XPS 13"),
    ("Preserved alphanumeric token", "SA8*", "SA 8 *"),
    ("Preserved alphanumeric token 2", "m8S", "m 8 S"),
    ("Emoticons removed", "أ :) :D xD ^_^ <3 ;-) :'(", "أ"),
    ("HTML does not swallow emoticon text", "<3 أنت رائع >_<", "أنت رائع"),
    ("Tashkeel removed", "مُحَمَّد", "محمد"),
    ("Tatweel removed", "مرحبـــا", "مرحبا"),
    (
        "Tashkeel with placeholder",
        "مُشَاهَدَة https://t.co/abc123#مصر الآن",
        "مشاهدة [رابط] #مصر الآن",
    ),
    ("Fitzpatrick modifier removed", "أحبك 👍🏻", "أحبك"),
    ("Presentation forms normalized", "ﺍﻟﺴﻼم", "السلام"),
    ("Slash replaced", "نعم/لا", "نعم - لا"),
    (
        "URL with slash keeps placeholder",
        "شاهد https://t.co/a/b الآن",
        "شاهد [رابط] الآن",
    ),
)


def sanity_check_table() -> pd.DataFrame:
    """Run the preprocessing examples and raise if any output is unexpected."""
    rows = []
    for case, source, expected in SANITY_EXAMPLES:
        actual = preprocess_text(source)
        rows.append(
            {
                "case": case,
                "input": source,
                "expected": expected,
                "actual": actual,
                "passed": actual == expected,
            }
        )

    table = pd.DataFrame(rows)
    failures = table[~table["passed"]]
    if not failures.empty:
        raise AssertionError(
            "Preprocessing sanity checks failed:\n" + failures.to_string(index=False)
        )
    return table
