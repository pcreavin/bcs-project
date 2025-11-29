import csv, pathlib, re, xml.etree.ElementTree as ET

ROOT = pathlib.Path("data/raw/dataset")
OUT = pathlib.Path("data/manifest.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Map the 5 BCS folders to a fixed class order (0..4)
ORDER = [3.25, 3.5, 3.75, 4.0, 4.25]
IDX = {v: i for i, v in enumerate(ORDER)}

def parse_bbox(xml_path: pathlib.Path):
    """Parse Pascal VOC-style bbox from XML, returns (xmin, ymin, xmax, ymax) or None."""
    if not xml_path.exists():
        return None
    try:
        root = ET.parse(xml_path).getroot()
        # look for the first <object><bndbox>...</bndbox>
        obj = root.find(".//object")
        if obj is None:
            obj = root  # sometimes bndbox is directly under root (rare)
        bnd = obj.find(".//bndbox") if obj is not None else None
        if bnd is None:
            return None
        def _get(tag):
            el = bnd.find(tag)
            return int(float(el.text)) if el is not None and el.text is not None else None
        xmin, ymin, xmax, ymax = _get("xmin"), _get("ymin"), _get("xmax"), _get("ymax")
        if None in (xmin, ymin, xmax, ymax) or xmin >= xmax or ymin >= ymax:
            return None
        return xmin, ymin, xmax, ymax
    except Exception:
        return None

rows = []
for jpg in ROOT.rglob("*.jpg"):
    # label from parent folder name (e.g., "3.25")
    try:
        bcs_float = float(jpg.parent.name)
    except ValueError:
        # skip unexpected folders
        continue
    if bcs_float not in IDX:
        continue
    label_5 = IDX[bcs_float]  # 0..4

    # paired XML (same stem)
    xml = jpg.with_suffix(".xml")
    bbox = parse_bbox(xml)

    row = {
        "image_path": str(jpg),
        "bcs_float": bcs_float,
        "bcs_5class": label_5,
        "xmin": bbox[0] if bbox else "",
        "ymin": bbox[1] if bbox else "",
        "xmax": bbox[2] if bbox else "",
        "ymax": bbox[3] if bbox else "",
    }
    rows.append(row)

# Sort rows by image_path for deterministic ordering
# This ensures reproducibility across different systems/filesystems
rows.sort(key=lambda x: x["image_path"])

# write CSV
with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["image_path","bcs_float","bcs_5class","xmin","ymin","xmax","ymax"])
    w.writeheader()
    w.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUT}")
