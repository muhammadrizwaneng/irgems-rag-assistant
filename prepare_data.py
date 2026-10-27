"""
prepare_data.py
---------------
Clean the WooCommerce products.csv export and save a RAG-friendly
JSON file at data/cleaned_products.json.

Each product becomes a structured record plus a natural-language
`text` field used later for embedding / retrieval.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INPUT_CSV = DATA_DIR / "products.csv"
OUTPUT_JSON = DATA_DIR / "cleaned_products.json"

# WooCommerce columns we care about (CSV headers)
KEEP_FIELDS = [
    "Name",
    "SKU",
    "Sale price",
    "Regular price",
    "Categories",
    "Short description",
    "Description",
    "Tags",
]

# Extra columns used only while cleaning (not kept as top-level output keys)
HELPER_FIELDS = ["Type", "Parent", "Attribute 1 name", "Attribute 1 value(s)",
                 "Attribute 2 name", "Attribute 2 value(s)"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def strip_html(raw: str | None) -> str:
    """Remove HTML tags / entities and collapse whitespace."""
    if not raw:
        return ""
    soup = BeautifulSoup(raw, "lxml")
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_price(value: str | None) -> str | None:
    """Return a numeric price string, or None if blank / invalid."""
    if value is None:
        return None
    value = str(value).strip().replace(",", "")
    if not value:
        return None
    try:
        number = float(value)
        # Prefer integer display when there are no decimals
        if number.is_integer():
            return str(int(number))
        return f"{number:.2f}"
    except ValueError:
        return None


def format_pkr(price: str | None) -> str:
    """Human-readable PKR label."""
    if not price:
        return "Not listed"
    return f"PKR {price}"


def normalize_header(name: str) -> str:
    """Strip BOM / whitespace from CSV headers."""
    return name.replace("\ufeff", "").strip()


def build_product_text(product: dict) -> str:
    """
    Build a rich plain-text block for each product.
    This is what gets embedded into ChromaDB.
    """
    lines = [
        f"Product Name: {product['name']}",
        f"SKU: {product['sku'] or 'N/A'}",
        f"Category: {product['category'] or 'Uncategorized'}",
        f"Regular Price: {format_pkr(product['regular_price'])}",
        f"Sale Price: {format_pkr(product['sale_price'])}",
    ]

    # Effective / customer-facing price
    effective = product["sale_price"] or product["regular_price"]
    lines.append(f"Current Price: {format_pkr(effective)}")

    if product.get("attributes"):
        lines.append(f"Options: {product['attributes']}")

    if product.get("short_description"):
        lines.append(f"Short Description: {product['short_description']}")

    if product.get("description"):
        lines.append(f"Full Description: {product['description']}")

    if product.get("tags"):
        lines.append(f"Tags: {product['tags']}")

    # Store policy reminders useful for customer questions
    lines.append(
        "Availability: Sold as individual piece(s). IRGEMS does not sell wholesale. "
        "Shop: https://irgems.pk"
    )

    return "\n".join(lines)


def collect_attributes(row: dict) -> str:
    """Combine WooCommerce attribute columns into a short string."""
    parts: list[str] = []
    for i in (1, 2):
        name = (row.get(f"Attribute {i} name") or "").strip().strip('"')
        value = (row.get(f"Attribute {i} value(s)") or "").strip().strip('"')
        if name and value:
            parts.append(f"{name}: {value}")
    return "; ".join(parts)


def load_csv_rows(path: Path) -> list[dict]:
    """Load the WooCommerce CSV with normalized headers."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        # Normalize fieldnames in case of BOM leftovers
        if reader.fieldnames:
            reader.fieldnames = [normalize_header(h) for h in reader.fieldnames]
        rows = []
        for row in reader:
            cleaned = {normalize_header(k): (v or "").strip() for k, v in row.items()}
            rows.append(cleaned)
        return rows


# ---------------------------------------------------------------------------
# Main cleaning logic
# ---------------------------------------------------------------------------
def clean_products(rows: list[dict]) -> list[dict]:
    """
    Convert WooCommerce rows into cleaned product records.

    Strategy:
    - Keep simple products with a name.
    - Skip variable *parent* rows (they usually have no price).
    - Keep variations; inherit description / category / tags from the parent SKU.
    """
    # Index parents by SKU so variations can inherit content
    parents_by_sku: dict[str, dict] = {}
    for row in rows:
        row_type = (row.get("Type") or "").lower()
        sku = row.get("SKU") or ""
        if row_type == "variable" and sku:
            parents_by_sku[sku] = row

    products: list[dict] = []

    for row in rows:
        row_type = (row.get("Type") or "").lower()
        name = strip_html(row.get("Name"))
        if not name:
            continue

        # Skip variable parents — their sellable children are the variations
        if row_type == "variable":
            continue

        parent_sku = row.get("Parent") or ""
        parent = parents_by_sku.get(parent_sku, {})

        # Prefer row values; fall back to parent for empty variation fields
        short_desc = strip_html(row.get("Short description")) or strip_html(
            parent.get("Short description")
        )
        description = strip_html(row.get("Description")) or strip_html(
            parent.get("Description")
        )
        category = row.get("Categories") or parent.get("Categories") or ""
        tags = row.get("Tags") or parent.get("Tags") or ""
        sku = row.get("SKU") or parent_sku or ""

        sale_price = clean_price(row.get("Sale price"))
        regular_price = clean_price(row.get("Regular price"))

        # Skip rows that still have no usable price and no description
        if not sale_price and not regular_price and not description and not short_desc:
            continue

        attributes = collect_attributes(row)
        # If variation name already encodes options, attributes still help retrieval

        product = {
            "name": name,
            "sku": sku,
            "sale_price": sale_price,
            "regular_price": regular_price,
            "category": category,
            "short_description": short_desc,
            "description": description,
            "tags": tags,
            "attributes": attributes,
            "product_type": row_type or "simple",
        }
        product["text"] = build_product_text(product)
        products.append(product)

    return products


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing products CSV: {INPUT_CSV}")

    print(f"Reading: {INPUT_CSV}")
    rows = load_csv_rows(INPUT_CSV)
    print(f"Raw WooCommerce rows: {len(rows)}")

    products = clean_products(rows)
    print(f"Cleaned products: {len(products)}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(products, handle, ensure_ascii=False, indent=2)

    print(f"Saved: {OUTPUT_JSON}")
    for p in products:
        price = p["sale_price"] or p["regular_price"] or "?"
        print(f"  - {p['name'][:70]}  |  PKR {price}")


if __name__ == "__main__":
    main()
