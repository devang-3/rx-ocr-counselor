"""Brand and OCR shorthand → generic name mapping."""

from __future__ import annotations

import re

# Indian Rx brand / OCR typo → generic for KB lookup
BRAND_TO_GENERIC: dict[str, str] = {
    "allegra": "fexofenadine",
    "allegrai": "fexofenadine",
    "fexofenadin": "fexofenadine",
    "fexofenadine": "fexofenadine",
    "ambrolite": "ambroxol",
    "ambrolites": "ambroxol",
    "ambroxol": "ambroxol",
    "azithral": "azithromycin",
    "azithrail": "azithromycin",
    "azithromycin": "azithromycin",
    "pegura": "polyethylene glycol",
    "lactulose": "lactulose",
    "zimcovit": "multivitamin",
    "syrupzimcovit": "multivitamin",
    "salbutamol": "salbutamol",
    "terbutaline": "terbutaline",
    "tabutaline": "terbutaline",
    "terbutaline": "terbutaline",
    "levosalbutamol": "levosalbutamol",
    "levosalbutamoil": "levosalbutamol",
    "menthol": "menthol",
    "paracetamol": "paracetamol",
    "pcm": "paracetamol",
    "amox": "amoxycillin",
    "amor": "amoxycillin",
    "amoxicillin": "amoxycillin",
    "amoxycillin": "amoxycillin",
    "augmentin": "amoxycillin",
    "moxikind": "amoxycillin",
    "clavam": "amoxycillin",
}

# Regex patterns for OCR-mangled generic names inside noisy text
FUZZY_GENERIC_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)fexofenadin\w*"), "fexofenadine"),
    (re.compile(r"(?i)ambroxol\w*"), "ambroxol"),
    (re.compile(r"(?i)azithro\w*"), "azithromycin"),
    (re.compile(r"(?i)azithromycin\w*"), "azithromycin"),
    (re.compile(r"(?i)tabutalin\w*|terbutalin\w*"), "terbutaline"),
    (re.compile(r"(?i)salbutamol\w*"), "salbutamol"),
    (re.compile(r"(?i)levosalbut\w*"), "levosalbutamol"),
    (re.compile(r"(?i)lactulose\w*"), "lactulose"),
    (re.compile(r"(?i)polyethylene\s*glycol\w*"), "polyethylene glycol"),
    (re.compile(r"(?i)folic\s*acid\w*"), "folic acid"),
    (re.compile(r"(?i)elemental\s*iron\w*"), "elemental iron"),
]


def normalize_token(token: str) -> str:
    return re.sub(r"[^a-z0-9]", "", token.lower())


def resolve_generic_name(name: str) -> str:
    """Map brand/OCR token to generic name when possible."""
    cleaned = name.strip().lower()
    token = normalize_token(cleaned)
    if not token:
        return cleaned
    if token in BRAND_TO_GENERIC:
        return BRAND_TO_GENERIC[token]

    # Avoid false positives like Tal->terbutaline or 1->fexofenadine.
    if len(token) >= 4:
        for key, generic in BRAND_TO_GENERIC.items():
            if len(key) >= 4 and (key == token or key in token or token in key):
                return generic

    return cleaned


def fuzzy_generic_in_text(text: str) -> str | None:
    for pattern, generic in FUZZY_GENERIC_PATTERNS:
        if pattern.search(text):
            return generic
    return None
