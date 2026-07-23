"""Local, zero-dependency keyphrase extractor — the TID-free fallback for the
dual-pass tree half (concept decomposition).

Context: ng_embed._extract_concepts() decomposes a turn into concept strings by
calling a TID LLM (127.0.0.1:7437). Where no TID runs (e.g. CC on the laptop),
that call always fails and the dual-pass degrades to forest-only — no concept
trees, ever. This module produces concept phrases from text with no model and no
third-party deps (none are installed: verified no yake/rake/nltk/spacy/sklearn),
so the tree half can still form. Lower fidelity than an LLM's semantic
extraction, but real sub-turn structure instead of none.

Algorithm: RAKE (Rapid Automatic Keyword Extraction, Rose et al. 2010). Split
text into candidate phrases at stopwords/punctuation; score each content word by
deg(word)/freq(word) (co-occurrence degree over frequency — rewards words that
recur inside longer phrases); phrase score = sum of member word scores. Rank,
dedup, filter, return the top N. Pure Python, deterministic, ~microseconds/turn.

[2026-07-16 DudeMan CC] Task #69.
"""
from __future__ import annotations

import re
from typing import List

# A compact English stopword set (phrase delimiters). Kept inline so this module
# has no data-file or download dependency. Not exhaustive by design — RAKE only
# needs the high-frequency function words to carve sensible phrase boundaries.
_STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can can't cannot could couldn't did
didn't do does doesn't doing don't down during each few for from further had
hadn't has hasn't have haven't having he he'd he'll he's her here here's hers
herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's
its itself let's me more most mustn't my myself no nor not of off on once only or
other ought our ours ourselves out over own same shan't she she'd she'll she's
should shouldn't so some such than that that's the their theirs them themselves
then there there's these they they'd they'll they're they've this those through to
too under until up very was wasn't we we'd we'll we're we've were weren't what
what's when when's where where's which while who who's whom why why's with won't
would wouldn't you you'd you'll you're you've your yours yourself yourselves
just get got also really now then well like one two use using used via per etc
""".split())

# Word = letters/digits with internal - _ / . (keeps things like MAX_RETRIES,
# /v1/ingest, pH-levels intact as single tokens).
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-/.]*")
# Phrase-breaking punctuation (anything not word-ish or whitespace also breaks).
_SPLIT = re.compile(r"[^A-Za-z0-9_\-/.\s]+")

_MIN_WORD_LEN = 2          # drop 1-char content words
_MAX_PHRASE_WORDS = 4      # concepts are phrases, not sentences
_MIN_PHRASE_CHARS = 3


def _candidate_phrases(text: str) -> List[List[str]]:
    """Carve text into content-word runs delimited by stopwords/punctuation."""
    phrases: List[List[str]] = []
    for chunk in _SPLIT.split(text):
        cur: List[str] = []
        for tok in _WORD.findall(chunk):
            low = tok.lower()
            if low in _STOPWORDS or len(tok) < _MIN_WORD_LEN or tok.isdigit():
                if cur:
                    phrases.append(cur)
                    cur = []
            else:
                cur.append(tok)
                if len(cur) >= _MAX_PHRASE_WORDS:   # cap phrase length
                    phrases.append(cur)
                    cur = []
        if cur:
            phrases.append(cur)
    return phrases


def extract_keyphrases(text: str, max_n: int = 8) -> List[str]:
    """Return up to `max_n` concept phrases from `text`, best first.

    Deterministic, zero-dependency. Empty list for empty/structureless input
    (the caller treats that as "no concepts", same as an LLM's legitimate []).
    """
    if not text or not text.strip():
        return []
    phrases = _candidate_phrases(text)
    if not phrases:
        return []

    # RAKE word scores: degree(word)/frequency(word), word keyed case-insensitively.
    freq: dict = {}
    degree: dict = {}
    for words in phrases:
        deg = len(words) - 1  # co-occurrence degree contribution within the phrase
        for w in words:
            k = w.lower()
            freq[k] = freq.get(k, 0) + 1
            degree[k] = degree.get(k, 0) + deg
    score = {k: (degree[k] + freq[k]) / freq[k] for k in freq}  # deg/freq (deg incl. self)

    # Phrase score = sum of member word scores; keep first surface form seen.
    ranked: dict = {}
    for words in phrases:
        phrase = " ".join(words)
        key = phrase.lower()
        if len(phrase) < _MIN_PHRASE_CHARS:
            continue
        s = sum(score[w.lower()] for w in words)
        # Prefer the higher-scoring occurrence; stable on ties.
        if key not in ranked or s > ranked[key][0]:
            ranked[key] = (s, phrase)

    top = sorted(ranked.values(), key=lambda t: t[0], reverse=True)[:max_n]
    return [phrase for _s, phrase in top]
