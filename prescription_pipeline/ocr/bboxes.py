from __future__ import annotations

from typing import Literal, Sequence

BBoxLevel = Literal["line", "word", "full"]


def geom_to_xyxy(
    geometry: Sequence[Sequence[float]],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    points = [(float(x), float(y)) for x, y in geometry]
    if len(points) < 2:
        raise ValueError(f"Invalid geometry with fewer than 2 points: {geometry!r}")

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (
        max(0, int(round(min(xs) * width))),
        max(0, int(round(min(ys) * height))),
        min(width, int(round(max(xs) * width))),
        min(height, int(round(max(ys) * height))),
    )


def pad_bbox(
    bbox: list[int],
    width: int,
    height: int,
    padding_frac: float,
    min_pad_px: int = 2,
) -> list[int]:
    x0, y0, x1, y1 = bbox
    box_w = x1 - x0
    box_h = y1 - y0
    pad_x = max(min_pad_px, int(round(box_w * padding_frac)))
    pad_y = max(min_pad_px, int(round(box_h * padding_frac)))
    return [
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(width, x1 + pad_x),
        min(height, y1 + pad_y),
    ]


def union_word_bbox(words: list[dict], width: int, height: int) -> list[int] | None:
    boxes = []
    for word in words:
        geometry = word.get("geometry")
        if not geometry:
            continue
        x0, y0, x1, y1 = geom_to_xyxy(geometry, width, height)
        if x1 > x0 and y1 > y0:
            boxes.append((x0, y0, x1, y1))
    if not boxes:
        return None
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def extract_bboxes(
    exported: dict,
    width: int,
    height: int,
    bbox_level: BBoxLevel,
    crop_padding: float,
    min_box_height: int,
) -> list[dict]:
    if bbox_level == "full":
        bbox = pad_bbox([0, 0, width, height], width, height, crop_padding)
        return [{"level": "full", "index": 0, "bbox": bbox}]

    boxes: list[dict] = []
    for page in exported.get("pages", []):
        for block_idx, block in enumerate(page.get("blocks", [])):
            lines = block.get("lines", [])
            if bbox_level == "line":
                for line_idx, line in enumerate(lines):
                    geometry = line.get("geometry")
                    if geometry:
                        x0, y0, x1, y1 = geom_to_xyxy(geometry, width, height)
                        bbox = [x0, y0, x1, y1]
                    else:
                        union = union_word_bbox(line.get("words", []), width, height)
                        if union is None:
                            continue
                        bbox = union
                    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                        continue
                    if (bbox[3] - bbox[1]) < min_box_height:
                        continue
                    boxes.append(
                        {
                            "level": "line",
                            "block_index": block_idx,
                            "line_index": line_idx,
                            "index": len(boxes),
                            "bbox": pad_bbox(bbox, width, height, crop_padding),
                        }
                    )
                continue

            for line_idx, line in enumerate(lines):
                for word_idx, word in enumerate(line.get("words", [])):
                    geometry = word.get("geometry")
                    if not geometry:
                        continue
                    x0, y0, x1, y1 = geom_to_xyxy(geometry, width, height)
                    if x1 <= x0 or y1 <= y0:
                        continue
                    if (y1 - y0) < min_box_height:
                        continue
                    boxes.append(
                        {
                            "level": "word",
                            "block_index": block_idx,
                            "line_index": line_idx,
                            "word_index": word_idx,
                            "index": len(boxes),
                            "bbox": pad_bbox([x0, y0, x1, y1], width, height, crop_padding),
                        }
                    )
    return boxes
