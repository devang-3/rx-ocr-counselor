from __future__ import annotations

import re

from prescription_pipeline.ner.aliases import (
    BRAND_TO_GENERIC,
    fuzzy_generic_in_text,
    normalize_token,
    resolve_generic_name,
)
from prescription_pipeline.schemas import BBoxPrediction, ExtractedMedication

FREQUENCY_MAP = {
    "od": "once daily",
    "bd": "twice daily",
    "bid": "twice daily",
    "tds": "three times daily",
    "tid": "three times daily",
    "qid": "four times daily",
    "qds": "four times daily",
    "hs": "at bedtime",
    "sos": "as needed",
    "stat": "immediately",
    "ac": "before meals",
    "pc": "after meals",
    "prn": "as needed",
}

BLOCKLIST = frozenset(
    {
        "allergies", "findings", "examination", "diagnosis", "medicines", "advice",
        "investigations", "instructions", "history", "complaints", "screening",
        "provisional", "performance", "significant", "nutritional", "lifestyle",
        "rehab", "diet", "normal", "clear", "pallor", "constipation", "infection",
        "leading", "days", "months", "vegetables", "fruits", "water", "juice",
        "letter", "sign", "eat", "mixed", "socket", "mantoux", "abdomen", "fecolith",
        "palpable", "suggestive", "vitb", "polic", "chestclear", "chest-normal",
        "throat", "order", "look", "till", "acute", "respiratory", "tract", "final",
        "xray", "ray", "negatiy", "negativ", "grapers", "guaphe", "papaya",
        "mangoes", "melons", "oranges", "grapes", "kishmish", "spinach", "jaggery",
        "dates", "rich", "food", "green",
    }
)

BLOCKLIST_PHRASES = (
    r"(?i)\b(history and complaints|significant history|nutritional screening|"
    r"provisional final diagnosis|acute respiratory tract|performance in the|"
    r"chest\s*clear|throat\s*normal|mixed in \d+|lifestyle|x[\s-]?ray|"
    r"iron rich food|eat papaya|a letter from|abdomen.fecolith)\b"
)

MEDICATION_FORMS = re.compile(
    r"(?i)\b(syrup|suspension|tablet|tab|capsule|cap|injection|inj|sachet|"
    r"drops|nebul(?:isation|ization)?|cream|ointment|gel|susp)\b"
)

DOSAGE_RE = re.compile(
    r"(?i)\b(\d+(?:\.\d+)?)\s*"
    r"(mg|mcg|g|gm|ml|iu|iu/ml|%|maj|meq|units?|tab|tabs|tablet|tablets|cap|caps)\b"
)
INLINE_DOSAGE_RE = re.compile(
    r"(?i)(?<=[a-z])(\d+(?:\.\d+)?)\s*(mg|mcg|g|gm|ml|maj)\b|"
    r"\b(\d+(?:\.\d+)?)(mg|mcg|g|gm|ml|maj)\b"
)
EMBEDDED_DRUG_STRENGTH_RE = re.compile(
    r"(?i)\b([a-z][a-z0-9-]{2,}?)[x\s]*(\d+(?:\.\d+)?)\s*(mg|mcg|g|gm|ml|maj)\b"
)
SALT_ONLY = frozenset(
    {"hydrochloride", "sulphate", "sulfate", "acetate", "citrate", "menthol", "vitb", "vitb2"}
)
CAPS_DRUG_RE = re.compile(
    r"(?i)\b([A-Z][A-Z0-9-]{2,}(?:\s+(?:HYDROCHLORIDE|SULPHATE|SULFATE|ACETATE|CITRATE))?)"
    r"\s*\(\s*(\d+(?:\.\d+)?)\s*(mg|mcg|g|gm|ml|maj)"
)
PAREN_STRENGTH_RE = re.compile(
    r"(?i)\(([a-z0-9 .+-]+?)\s*(\d+(?:\.\d+)?)\s*(mg|mcg|g|gm|ml|maj)\s*\)"
)
FREQUENCY_TOKEN_RE = re.compile(
    r"(?i)\b(od|bd|bid|tds|tid|qid|qds|hs|sos|stat|ac|pc|prn)\b"
)
FREQUENCY_PHRASE_RE = re.compile(
    r"(?i)\b(once|twice|thrice)\s+(?:a\s+)?day\b|"
    r"\b(?:every|per)\s+\d+\s+hours?\b|"
    r"\b\d+\s*times?\s+(?:a\s+)?day\b"
)
DURATION_RE = re.compile(
    r"(?i)\bx\s*(\d+)\s*(day|days|week|weeks|month|months)\b|"
    r"\bfor\s+(\d+)\s*(day|days|week|weeks|month|months)\b|"
    r"\b(\d+)\s*(day|days|week|weeks|month|months)\b"
)
NUMBERED_RX_RE = re.compile(
    r"(?i)^\s*\d+\s*[\.\)]\s*(?:syrup|suspension|tab|tablet|sachet|cap|capsule|inj)"
)
SECTION_LABEL_RE = re.compile(r"(?i)^[\w\s-]{2,40}:\s*$")
FORM_PREFIX_RE = re.compile(
    r"(?i)^[\d\s\.\)#\"']*(?:tab|tal|cap|capsule|syrup|susp|inj|sachet)\.?\s+"
)
DOSE_SCHEDULE_RE = re.compile(
    r"(?i)^\s*\d+\s*[-–—]\s*\d+(?:\s*[-–—]\s*\d+)?(?:\s*[x×]\s*\d+\s*(?:day|days))?\s*\.?\s*$|"
    r"^\s*\d+\s*[-–—]\s*\d+(?:\d|[x×])+\s*(?:day|days)\s*\.?\s*$"
)
STANDALONE_STRENGTH_RE = re.compile(r"(?i)\b(\d{2,4})(?:mg|gm|mcg|maj)?\b")
MIN_CONFIDENCE_FOR_KB = 0.50


def _strip_form_prefix(text: str) -> str:
    stripped = FORM_PREFIX_RE.sub("", text.strip())
    return stripped.strip(" .,;:-#\"'")


def _is_schedule_only_line(text: str) -> bool:
    compact = re.sub(r"\s+", "", text.strip().lower())
    if DOSE_SCHEDULE_RE.match(text.strip()):
        return True
    if re.fullmatch(r"[\d\.\-–—xday]+", compact):
        return True
    words = re.findall(r"[A-Za-z]+", text)
    drugish = [word for word in words if len(word) >= 4 and not re.fullmatch(r"(?i)x?\d*days?", word)]
    if re.search(r"\d+\s*[-–—]\s*\d+", text) and not drugish:
        return True
    return False


def _pick_best_drug_token(text: str) -> str:
    working = _strip_form_prefix(text)
    best_name = ""
    best_score = -1.0

    for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", working):
        if re.fullmatch(r"(?i)(tab|tal|cap|syrup|susp|inj|sachet|tablet|capsule)", token):
            continue
        if _first_token(token) in BLOCKLIST:
            continue

        resolved = resolve_generic_name(token)
        score = float(len(resolved))
        if resolved in BRAND_TO_GENERIC.values() or normalize_token(token) in BRAND_TO_GENERIC:
            score += 5.0
        if len(resolved) >= 5:
            score += 1.0
        if score > best_score:
            best_score = score
            best_name = resolved

    return best_name


def normalize_frequency(token: str) -> str:
    lowered = token.lower().strip()
    if lowered in FREQUENCY_MAP:
        return FREQUENCY_MAP[lowered]
    phrase = lowered.replace("_", " ")
    if "once" in phrase and "day" in phrase:
        return "once daily"
    if "twice" in phrase and "day" in phrase:
        return "twice daily"
    if "thrice" in phrase and "day" in phrase:
        return "three times daily"
    return phrase


def _clean_drug_name(name: str) -> str:
    name = re.sub(r"^[\d\s\.\)#]+", "", name)
    name = re.sub(r"\s+", " ", name).strip(" .,;:-#\"'")
    name = re.sub(r"(?i)^(syrup|suspension|sachet|tablet|tab|cap|capsule|inj|injection)\s+", "", name)
    return name.strip()


def _first_token(name: str) -> str:
    if not name.split():
        return ""
    return re.sub(r"[^a-z0-9]", "", name.lower().split()[0])


def _is_blocklisted(text: str, drug_name: str) -> bool:
    if re.search(BLOCKLIST_PHRASES, text):
        return True
    if SECTION_LABEL_RE.match(text.strip()):
        return True

    first = _first_token(drug_name)
    if first in BLOCKLIST:
        return True
    if len(first) <= 2 and not DOSAGE_RE.search(text):
        return True

    if re.fullmatch(r"(?i)(syrup|suspension|tablet|tab|sachet|cap|capsule|inj)", drug_name.strip()):
        return True

    clinical = re.search(
        r"(?i)\b(normal|clear|palpable|suggestive|screening|findings|examination)\b",
        drug_name,
    )
    if clinical and not DOSAGE_RE.search(text) and not MEDICATION_FORMS.search(drug_name):
        return True

    return False


def _extract_dosage(text: str) -> str:
    for pattern in (DOSAGE_RE, INLINE_DOSAGE_RE):
        match = pattern.search(text)
        if match:
            groups = [g for g in match.groups() if g]
            if len(groups) >= 2:
                return _normalize_dosage_unit(f"{groups[-2]}{groups[-1]}")

    embedded = EMBEDDED_DRUG_STRENGTH_RE.search(text)
    if embedded:
        return _normalize_dosage_unit(f"{embedded.group(2)}{embedded.group(3)}")

    caps = CAPS_DRUG_RE.search(text)
    if caps:
        return _normalize_dosage_unit(f"{caps.group(2)}{caps.group(3)}")

    paren = PAREN_STRENGTH_RE.search(text)
    if paren:
        return _normalize_dosage_unit(f"{paren.group(2)}{paren.group(3)}")

    standalone = STANDALONE_STRENGTH_RE.search(text)
    if standalone:
        amount = int(standalone.group(1))
        if 50 <= amount <= 2000:
            return _normalize_dosage_unit(f"{standalone.group(1)}mg")

    return ""


def _extract_frequency(text: str) -> str:
    token = FREQUENCY_TOKEN_RE.search(text)
    if token:
        return token.group(0)
    phrase = FREQUENCY_PHRASE_RE.search(text)
    if phrase:
        return phrase.group(0)
    return ""


def _extract_duration(text: str) -> str:
    match = DURATION_RE.search(text)
    if not match:
        return ""
    groups = [g for g in match.groups() if g]
    if len(groups) >= 2:
        return f"{groups[0]} {groups[1]}".lower()
    return match.group(0)


def _normalize_dosage_unit(dosage: str) -> str:
    return dosage.lower().replace("maj", "mg").replace("gm", "g")


def _extract_drug_name(text: str, dosage: str, frequency: str) -> str:
    fuzzy = fuzzy_generic_in_text(text)
    if fuzzy:
        return fuzzy

    best_token = _pick_best_drug_token(text)
    if best_token:
        return best_token

    caps = CAPS_DRUG_RE.search(text)
    if caps:
        return resolve_generic_name(_clean_drug_name(caps.group(1)))

    embedded = EMBEDDED_DRUG_STRENGTH_RE.search(text)
    if embedded:
        return resolve_generic_name(_clean_drug_name(embedded.group(1)))

    paren = PAREN_STRENGTH_RE.search(text)
    if paren:
        inner = paren.group(1).strip()
        if not re.fullmatch(r"\d+(?:\.\d+)?", inner):
            return resolve_generic_name(_clean_drug_name(inner))

    lead = re.match(r"(?i)^[\d\s\.\)#]*([A-Za-z][A-Za-z0-9-]{2,})", text)
    if lead:
        candidate = resolve_generic_name(_clean_drug_name(lead.group(1)))
        if candidate and _first_token(candidate) not in BLOCKLIST:
            if not re.fullmatch(r"(?i)(syrup|suspension|tablet|tab|sachet|cap|capsule)", candidate):
                return candidate

    working = text
    if dosage:
        working = re.sub(re.escape(dosage), " ", working, flags=re.IGNORECASE)
    if frequency:
        working = re.sub(re.escape(frequency), " ", working, flags=re.IGNORECASE)

    working = FREQUENCY_PHRASE_RE.sub(" ", working)
    working = DURATION_RE.sub(" ", working)
    working = re.sub(r"(?i)\b\d+\s*[\.\)]\s*", " ", working)
    working = re.sub(r"(?i)[#\"']+", " ", working)

    parts = re.split(r"\s*[-–—]\s*", working, maxsplit=1)
    candidate = resolve_generic_name(_clean_drug_name(parts[0]))

    tokens = [
        resolve_generic_name(t)
        for t in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", candidate)
        if _first_token(t) not in BLOCKLIST
        and not re.fullmatch(r"(?i)(syrup|suspension|tablet|tab|sachet|cap|capsule|susp)", t)
    ]
    if tokens:
        return max(tokens, key=len)

    return candidate


def _has_medication_signal(text: str, dosage: str, frequency: str, duration: str, drug_name: str) -> bool:
    if dosage or frequency or duration:
        return True
    if fuzzy_generic_in_text(text):
        return True
    if NUMBERED_RX_RE.search(text) and len(drug_name) > 3:
        return True
    if MEDICATION_FORMS.search(text) and len(drug_name) > 4:
        return True
    if EMBEDDED_DRUG_STRENGTH_RE.search(text) or PAREN_STRENGTH_RE.search(text) or CAPS_DRUG_RE.search(text):
        return True
    if drug_name in {
        "fexofenadine", "ambroxol", "azithromycin", "salbutamol", "terbutaline",
        "levosalbutamol", "lactulose", "paracetamol", "amoxycillin", "polyethylene glycol",
        "folic acid", "elemental iron", "menthol",
    }:
        return True
    return False


def _strip_section_prefix(text: str) -> str:
    return re.sub(
        r"(?i)^(?:advice|investigations|diagnosis|medicines|instructions)\s*:\s*",
        "",
        text,
    ).strip()


def _candidate_substrings(text: str) -> list[str]:
    """Generate candidate medication substrings from a noisy OCR row."""
    text = _strip_section_prefix(text)
    if not text:
        return []

    candidates: list[str] = []

    numbered = re.split(
        r"(?=(?<!\d)\d+\s*[\.\)]\s*(?:syrup|suspension|sachet|tab|tablet|cap|capsule|inj|susp)\b)",
        text,
        flags=re.IGNORECASE,
    )
    for seg in numbered:
        seg = seg.strip()
        if seg:
            candidates.append(seg)

    for match in EMBEDDED_DRUG_STRENGTH_RE.finditer(text):
        start = max(0, match.start() - 20)
        end = min(len(text), match.end() + 40)
        snippet = text[start:end].strip()
        if snippet and snippet not in candidates:
            candidates.append(snippet)

    for match in CAPS_DRUG_RE.finditer(text):
        start = max(0, match.start())
        end = min(len(text), match.end() + 50)
        snippet = text[start:end].strip()
        if snippet and snippet not in candidates:
            candidates.append(snippet)

    for match in PAREN_STRENGTH_RE.finditer(text):
        start = max(0, match.start() - 30)
        end = min(len(text), match.end() + 30)
        snippet = text[start:end].strip()
        if snippet and snippet not in candidates:
            candidates.append(snippet)

    if not candidates:
        candidates.append(text)

    return candidates


def _merge_row_fragments(predictions: list[BBoxPrediction], y_tolerance: int = 15) -> list[tuple[str, list[int]]]:
    if not predictions:
        return []

    sorted_preds = sorted(predictions, key=lambda p: (p.bbox[1], p.bbox[0]))
    rows: list[list[BBoxPrediction]] = []
    current_row: list[BBoxPrediction] = [sorted_preds[0]]
    anchor_y = sorted_preds[0].bbox[1]

    for pred in sorted_preds[1:]:
        if abs(pred.bbox[1] - anchor_y) <= y_tolerance:
            current_row.append(pred)
        else:
            rows.append(current_row)
            current_row = [pred]
            anchor_y = pred.bbox[1]
    rows.append(current_row)

    merged: list[tuple[str, list[int]]] = []
    for row in rows:
        row = sorted(row, key=lambda p: p.bbox[0])
        text = " ".join(p.text.strip() for p in row if p.text.strip())
        if not text:
            continue
        merged.append(
            (
                text,
                [
                    min(p.bbox[0] for p in row),
                    min(p.bbox[1] for p in row),
                    max(p.bbox[2] for p in row),
                    max(p.bbox[3] for p in row),
                ],
            )
        )
    return merged


class PrescriptionParser:
    """Rule-based entity extraction from OCR lines (LayoutLMv3 placeholder)."""

    def parse_line(self, line_text: str, bbox: list[int] | None = None) -> ExtractedMedication | None:
        text = re.sub(r"\s+", " ", line_text.strip())
        if not text or len(text) < 3:
            return None
        if _is_schedule_only_line(text):
            return None

        dosage = _extract_dosage(text)
        frequency = _extract_frequency(text)
        duration = _extract_duration(text)
        drug_name = _extract_drug_name(text, dosage, frequency)

        if not drug_name or len(drug_name) < 3 or drug_name.isdigit():
            return None
        if drug_name.lower() in SALT_ONLY or _first_token(drug_name) in SALT_ONLY:
            return None
        if _is_blocklisted(text, drug_name):
            return None
        if not _has_medication_signal(text, dosage, frequency, duration, drug_name):
            return None

        confidence = 0.35
        if dosage:
            confidence += 0.3
        if frequency:
            confidence += 0.2
        if duration:
            confidence += 0.1
        if MEDICATION_FORMS.search(text):
            confidence += 0.1
        if NUMBERED_RX_RE.search(text):
            confidence += 0.05
        if fuzzy_generic_in_text(text):
            confidence += 0.1
        if len(drug_name) >= 5:
            confidence += 0.05

        return ExtractedMedication(
            drug_name=drug_name,
            dosage=dosage,
            frequency=frequency,
            duration=duration,
            source_line=text,
            bbox=bbox or [],
            confidence=min(confidence, 1.0),
        )

    def parse_predictions(self, predictions: list[BBoxPrediction]) -> list[ExtractedMedication]:
        medications: list[ExtractedMedication] = []
        seen: set[str] = set()

        for text, bbox in _merge_row_fragments(predictions):
            for segment in _candidate_substrings(text):
                med = self.parse_line(segment, bbox)
                if med is None:
                    continue
                key = f"{med.drug_name.lower()}|{med.dosage}|{med.frequency}"
                if key in seen:
                    continue
                seen.add(key)
                medications.append(med)

        return medications


def should_match_knowledge_base(medication: ExtractedMedication) -> bool:
    if medication.confidence < MIN_CONFIDENCE_FOR_KB:
        return False
    if _is_blocklisted(medication.source_line, medication.drug_name):
        return False
    known_generics = {
        "fexofenadine", "ambroxol", "azithromycin", "salbutamol", "terbutaline",
        "levosalbutamol", "lactulose", "paracetamol", "amoxycillin", "polyethylene glycol",
        "folic acid", "elemental iron", "menthol", "multivitamin",
    }
    if medication.drug_name in known_generics:
        if medication.drug_name == "multivitamin":
            return False  # KB lacks reliable multivitamin entries
        return True
    if medication.dosage or medication.frequency:
        return True
    return len(medication.drug_name) >= 8
