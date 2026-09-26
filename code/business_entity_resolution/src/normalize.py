"""
Owner: Person A (Blocking / Candidate Generation)

Normalization utilities for business_name and business_address fields.
Used by Person A's blocking stage AND reused by Person B's feature stage,
so keep this dependency-light and well-tested — it's a shared interface.

Do NOT call any external API, geocoding service, or lookup here — normalization
must be self-contained (see challenge fair-play rules).
"""

from __future__ import annotations
import re

# TODO(Person A): expand from real EDA on train_source1/2/3 — this is a starter list only.
LEGAL_SUFFIXES = [
    "private limited", "pvt ltd", "pvt. ltd.", "pvt", "private",
    "limited", "ltd", "ltd.",
    "corporation", "corp", "corp.",
    "incorporated", "inc", "inc.",
    "llp", "llc", "co", "co.", "company",
]

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def normalize_name(name: str) -> str:
    """
    Normalize a business_name string for blocking/matching.

    TODO(Person A):
      - Handle '&' vs 'and'
      - Strip legal suffixes (see LEGAL_SUFFIXES) as whole tokens, not substrings
      - Handle transliteration / non-Latin scripts (e.g. Devanagari names seen
        in Source 2 India records) — decide: transliterate, or keep script-aware
        blocking keys separately
      - Handle DBA/trade-name patterns
      - Consider returning both a "strict" and "loose" normalized form for
        multi-stage blocking
    """
    if not name:
        return ""
    s = name.lower().strip()
    s = s.replace("&", " and ")
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    # TODO: strip LEGAL_SUFFIXES as trailing/leading whole-word tokens
    return s


def normalize_address(address: str, country: str) -> dict:
    """
    Normalize a business_address string into a structured dict.

    TODO(Person A):
      - Handle missing/empty address (seen in real data, e.g. some Source 3
        rows have blank business_address)
      - Handle abbreviations (Rd/Road, St/Street) — country-aware, since
        conventions differ between US and India
      - Handle landmark references ("Near SBI ATM") — decide whether to keep
        as a separate weak signal or drop
      - Handle component reordering — don't assume a fixed field order
      - MUST NOT hardcode logic to only {US, India} — France appears in test
        only, with no training examples, so keep this open-ended (e.g. generic
        tokenization fallback for unseen country labels)

    Returns:
      dict with keys like {"raw": str, "tokens": list[str], "locality": str|None,
      "postal_code": str|None} — finalize this schema with Person B since they
      consume it directly.
    """
    if not address:
        return {"raw": "", "tokens": [], "locality": None, "postal_code": None}
    s = address.lower().strip()
    tokens = _WS_RE.sub(" ", _PUNCT_RE.sub(" ", s)).split()
    return {"raw": address, "tokens": tokens, "locality": None, "postal_code": None}
