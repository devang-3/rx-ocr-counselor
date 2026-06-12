"""
Clean MID.xlsx into a deduplicated knowledge-base table.

Keeps fields needed for drug matching and patient-facing facts:
generic name, strength, form, side effects, usage, safety, tips, food instruction.
"""

from __future__ import annotations

import html
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

COLUMNS = [
    "Name",
    "Contains",
    "SideEffect",
    "HowToUse",
    "SafetyAdvice",
    "QuickTips",
]

OUTPUT_COLUMNS = [
    "drug_id",
    "generic_name",
    "strength",
    "form",
    "is_combo",
    "side_effects",
    "how_to_use",
    "safety_advice",
    "quick_tips",
    "food_instruction",
    "brand_example",
    "sku_count",
]


def col_to_idx(col: str) -> int:
    n = 0
    for c in col:
        n = n * 26 + (ord(c) - 64)
    return n - 1


def get_cell_val(cell: ET.Element) -> str:
    if cell.attrib.get("t") == "inlineStr":
        return "".join(t.text or "" for t in cell.iter(f"{NS}t"))
    value = cell.find(f"{NS}v")
    if value is not None and value.text is not None:
        return value.text
    inline = cell.find(f"{NS}is")
    if inline is not None:
        return "".join(t.text or "" for t in inline.iter(f"{NS}t"))
    return ""


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\\n", " ").replace('\\"', '"').replace("\\/", "/")
    if text.strip().lower() in {":null", "null", "none"}:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_contains(contains: str) -> tuple[str, str, bool]:
    contains = clean_text(contains)
    if not contains:
        return "", "", False

    is_combo = "+" in contains
    strength_match = re.search(r"\(([^)]+)\)", contains)
    strength = strength_match.group(1).strip() if strength_match else ""
    generic = re.sub(r"\s*\(.*", "", contains).strip().lower()
    generic = re.sub(r"\s+", " ", generic)
    return generic, strength, is_combo


def parse_form(name: str) -> str:
    name_lower = name.lower()
    rules = [
        (r"\b(injection|inj\.?)\b", "injection"),
        (r"\b(capsule|cap\.?)\b", "capsule"),
        (r"\b(tablet|tab\.?)\b", "tablet"),
        (r"\b(syrup|suspension|solution)\b", "liquid"),
        (r"\b(cream|ointment|gel|lotion)\b", "topical"),
        (r"\b(inhaler|respules|nebul)\b", "inhalation"),
        (r"\b(drops)\b", "drops"),
        (r"\b(powder|sachet)\b", "powder"),
    ]
    for pattern, form in rules:
        if re.search(pattern, name_lower):
            return form
    return "other"


def food_instruction(how_to_use: str) -> str:
    text = how_to_use.lower()
    if "with or without food" in text:
        return "either"
    without = any(
        phrase in text
        for phrase in (
            "without food",
            "empty stomach",
            "before food",
            "before meals",
            "on an empty stomach",
        )
    )
    with_food = any(
        phrase in text
        for phrase in ("with food", "after food", "after meals", "take with meals")
    )
    if without and not with_food:
        return "without_food"
    if with_food and not without:
        return "with_food"
    if with_food and without:
        return "either"
    return "unknown"


def make_drug_id(generic_name: str, strength: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", generic_name.lower()).strip("_")
    strength_slug = re.sub(r"[^a-z0-9]+", "_", strength.lower()).strip("_")
    if strength_slug:
        return f"{slug}_{strength_slug}"
    return slug


def text_quality_score(row: dict[str, str]) -> int:
    return sum(len(row[field]) for field in ("side_effects", "how_to_use", "safety_advice", "quick_tips"))


def iter_rows(xlsx_path: Path):
    with zipfile.ZipFile(xlsx_path) as archive:
        with archive.open("xl/worksheets/sheet1.xml") as sheet:
            for _, elem in ET.iterparse(sheet, events=("end",)):
                if elem.tag != f"{NS}row":
                    continue

                cells: dict[int, str] = {}
                for cell in elem.findall(f"{NS}c"):
                    ref = cell.attrib.get("r", "")
                    match = re.match(r"([A-Z]+)", ref)
                    if not match:
                        continue
                    cells[col_to_idx(match.group(1))] = get_cell_val(cell)

                elem.clear()
                if not cells:
                    continue

                yield {
                    "Name": clean_text(cells.get(0, "")),
                    "Contains": clean_text(cells.get(1, "")),
                    "SideEffect": clean_text(cells.get(5, "")),
                    "HowToUse": clean_text(cells.get(6, "")),
                    "SafetyAdvice": clean_text(cells.get(9, "")),
                    "QuickTips": clean_text(cells.get(8, "")),
                }


def build_clean_records(xlsx_path: Path) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    skipped = 0

    for idx, raw in enumerate(iter_rows(xlsx_path)):
        if idx == 0:
            if raw["Name"] == "Name":
                continue

        generic_name, strength, is_combo = parse_contains(raw["Contains"])
        if not generic_name:
            skipped += 1
            continue

        record = {
            "generic_name": generic_name,
            "strength": strength,
            "form": parse_form(raw["Name"]),
            "is_combo": is_combo,
            "side_effects": raw["SideEffect"],
            "how_to_use": raw["HowToUse"],
            "safety_advice": raw["SafetyAdvice"],
            "quick_tips": raw["QuickTips"],
            "food_instruction": food_instruction(raw["HowToUse"]),
            "brand_example": raw["Name"],
        }

        key = f"{generic_name}|{strength.lower()}"
        grouped.setdefault(key, []).append(record)

    deduped: list[dict] = []
    for key, records in grouped.items():
        best = max(records, key=text_quality_score)
        forms = Counter(record["form"] for record in records)
        brands = [record["brand_example"] for record in records if record["brand_example"]]

        deduped.append(
            {
                "drug_id": make_drug_id(best["generic_name"], best["strength"]),
                "generic_name": best["generic_name"],
                "strength": best["strength"],
                "form": forms.most_common(1)[0][0],
                "is_combo": best["is_combo"],
                "side_effects": best["side_effects"],
                "how_to_use": best["how_to_use"],
                "safety_advice": best["safety_advice"],
                "quick_tips": best["quick_tips"],
                "food_instruction": best["food_instruction"],
                "brand_example": brands[0] if brands else "",
                "sku_count": len(records),
            }
        )

    deduped.sort(key=lambda row: (row["generic_name"], row["strength"]))
    print(f"Skipped rows without generic name: {skipped}")
    print(f"Deduplicated records: {len(deduped)}")
    return deduped


def save_outputs(records: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "mid_clean.json"
    csv_path = output_dir / "mid_clean.csv"

    json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    import csv

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(records)

    try:
        import pandas as pd

        parquet_path = output_dir / "mid_clean.parquet"
        pd.DataFrame(records).to_parquet(parquet_path, index=False)
        print(f"Wrote {parquet_path}")
    except ImportError:
        print("pyarrow not installed; skipped parquet output")

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    xlsx_path = base_dir / "MID.xlsx"
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Missing source file: {xlsx_path}")

    print(f"Cleaning {xlsx_path} ...")
    records = build_clean_records(xlsx_path)
    save_outputs(records, base_dir)

    combo_count = sum(1 for row in records if row["is_combo"])
    print(f"Combo drugs: {combo_count}")
    print(f"Unique generics: {len({row['generic_name'] for row in records})}")


if __name__ == "__main__":
    main()
