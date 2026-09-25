import gc
import re
import unicodedata
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS — edit BASE if your dataset is elsewhere
# ============================================================
BASE = Path("/Users/bharadwaj/Downloads/student_resource/dataset")
OUT  = BASE / "clean"
OUT.mkdir(exist_ok=True)

FILES = [
    ("train", "train_source1.tsv", "train_source1.parquet"),
    ("train", "train_source2.tsv", "train_source2.parquet"),
    ("train", "train_source3.tsv", "train_source3.parquet"),
    ("test",  "test_source1.tsv",  "test_source1.parquet"),
    ("test",  "test_source2.tsv",  "test_source2.parquet"),
    ("test",  "test_source3.tsv",  "test_source3.parquet"),
]


# ============================================================
# ABBREVIATION DICTIONARY
# Only maps tokens you actually saw in the data or that are
# universally standard. Conservative on purpose.
# ============================================================
ABBREV = {
    # Legal suffixes
    "pvt": "private",
    "ltd": "limited",
    "corp": "corporation",
    "inc": "incorporated",
    "llc": "limited liability company",
    "llp": "limited liability partnership",
    "co":  "company",
    # Street / address
    "st":   "street",
    "rd":   "road",
    "ave":  "avenue",
    "av":   "avenue",
    "blvd": "boulevard",
    "bd":   "boulevard",
    "ln":   "lane",
    "dr":   "drive",
    "ct":   "court",
    "pl":   "place",
    "hno":  "house number",
    "flr":  "floor",
    "fl":   "floor",
    "gf":   "ground floor",
    "ff":   "first floor",
    # Indian states
    "rj": "rajasthan",
    "tn": "tamil nadu",
    "mh": "maharashtra",
    "ka": "karnataka",
    "ap": "andhra pradesh",
    "ts": "telangana",
    "up": "uttar pradesh",
    "mp": "madhya pradesh",
    "wb": "west bengal",
    "dl": "delhi",
    "hr": "haryana",
    "pb": "punjab",
    "gj": "gujarat",
    # US states
    "ny": "new york",
    "ca": "california",
    "tx": "texas",
    "fl": "florida",
    "il": "illinois",
    "pa": "pennsylvania",
    "oh": "ohio",
    "ga": "georgia",
    "nc": "north carolina",
    "sc": "south carolina",
    "va": "virginia",
    # French address markers
    "rue": "street",
    "che": "chemin",
    "imp": "impasse",
    "all": "allee",
    "sq":  "square",
}

LEGAL_SUFFIXES = {
    "private", "limited", "corporation", "incorporated",
    "llc", "llp", "company", "co", "inc", "ltd",
    "limited liability company", "limited liability partnership",
}


# ============================================================
# SCRIPT DETECTION & TRANSLITERATION
# Only used when the record contains Indic script.
# ============================================================
SCRIPT_RANGES = {
    "devanagari": (0x0900, 0x097F),
    "bengali":    (0x0980, 0x09FF),
    "gurmukhi":   (0x0A00, 0x0A7F),
    "gujarati":   (0x0A80, 0x0AFF),
    "tamil":      (0x0B80, 0x0BFF),
    "telugu":     (0x0C00, 0x0C7F),
    "kannada":    (0x0C80, 0x0CFF),
    "malayalam":  (0x0D00, 0x0D7F),
}

def has_indic_script(text: str) -> bool:
    """True if any char is in an Indic Unicode block."""
    for ch in text:
        cp = ord(ch)
        for lo, hi in SCRIPT_RANGES.values():
            if lo <= cp <= hi:
                return True
    return False

# Lazy import so we only load the library if needed
_transliterate_fn = None

def _transliterate(text: str, script_name: str) -> str:
    global _transliterate_fn
    if _transliterate_fn is None:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
        _transliterate_fn = (sanscript, transliterate)

    sanscript, transliterate = _transliterate_fn
    script_const = getattr(sanscript, script_name.upper())
    return transliterate(text, script_const, sanscript.IAST)


# ============================================================
# NORMALIZATION
# ============================================================
_PUNCT_RE       = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE          = re.compile(r"\s+")
_PLACEHOLDER_RE = re.compile(r"[#@+]")

def strip_accents(text: str) -> str:
    """é → e, ç → c, ñ → n."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def normalize_text(text) -> str:
    """Full pipeline: transliterate → lowercase → strip accents → clean."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    text = str(text).strip()
    if not text:
        return ""

    # 1. Transliterate Indic script → Latin
    if has_indic_script(text):
        for script_name, (lo, hi) in SCRIPT_RANGES.items():
            if any(lo <= ord(c) <= hi for c in text):
                try:
                    text = _transliterate(text, script_name)
                except Exception:
                    pass   # fall through with original
                break

    # 2. Strip accents (handles French é/ç, IAST diacritics)
    text = strip_accents(text)

    # 3. Lowercase
    text = text.lower()

    # 4. & → and
    text = text.replace("&", " and ")

    # 5. Remove placeholder characters
    text = _PLACEHOLDER_RE.sub(" ", text)

    # 6. Remove punctuation
    text = _PUNCT_RE.sub(" ", text)

    # 7. Collapse whitespace
    text = _WS_RE.sub(" ", text).strip()

    return text

def apply_abbrev(text: str) -> str:
    """Map known abbreviations to canonical forms."""
    if not text:
        return text
    return " ".join(ABBREV.get(tok, tok) for tok in text.split())


# ============================================================
# PINCODE EXTRACTION
# ============================================================
_PIN_6_RE = re.compile(r"\b(\d{6})\b")   # India PIN
_PIN_5_RE = re.compile(r"\b(\d{5})\b")   # US ZIP, France postal

def extract_pincode(address: str) -> str:
    if not address:
        return ""
    # Prefer 6-digit (India) since 5-digit can be a house number
    m = _PIN_6_RE.search(address)
    if m:
        return m.group(1)
    m = _PIN_5_RE.search(address)
    return m.group(1) if m else ""


# ============================================================
# ROW-LEVEL CLEANING
# ============================================================
def clean_row(business_name, business_address, country) -> dict:
    raw_name = "" if pd.isna(business_name) else str(business_name)
    raw_addr = "" if pd.isna(business_address) else str(business_address)

    name_norm = apply_abbrev(normalize_text(raw_name))
    addr_norm = apply_abbrev(normalize_text(raw_addr))

    # Split name into core tokens + legal-suffix tokens
    tokens = name_norm.split()
    legal_tokens = [t for t in tokens if t in LEGAL_SUFFIXES]
    core_tokens  = [t for t in tokens if t not in LEGAL_SUFFIXES]

    return {
        "name_norm":         name_norm,
        "name_core":         " ".join(core_tokens),
        "name_legal_suffix": " ".join(legal_tokens),
        "addr_norm":         addr_norm,
        "pincode":           extract_pincode(raw_addr),
        "country_norm":      "" if pd.isna(country) else str(country).strip().lower(),
        "has_address":       int(bool(raw_addr.strip())),
        "name_len":          len(name_norm),
        "addr_len":          len(addr_norm),
    }


# ============================================================
# PROCESS A SINGLE FILE
# ============================================================
def process_file(input_path: Path, output_path: Path):
    print(f"\n>>> {input_path.name}")
    print("    Loading...")

    df = pd.read_csv(
        input_path,
        sep="\t",
        dtype={
            "entity_id":        "string",
            "business_name":    "string",
            "business_address": "string",
            "country":          "string",
        },
        keep_default_na=False,
    )
    print(f"    Loaded {len(df):,} rows")

    print("    Cleaning...")
    chunk_size = 100_000
    cleaned_chunks = []

    for start in range(0, len(df), chunk_size):
        end = min(start + chunk_size, len(df))
        sub = df.iloc[start:end]

        rows = [
            clean_row(n, a, c)
            for n, a, c in zip(
                sub["business_name"],
                sub["business_address"],
                sub["country"],
            )
        ]
        cleaned_chunks.append(pd.DataFrame(rows))
        print(f"      {end:>9,} / {len(df):,}", end="\r")
    print()

    cleaned = pd.concat(cleaned_chunks, ignore_index=True)

    out = pd.DataFrame({
        "entity_id":         df["entity_id"].astype("string"),
        "country_norm":      cleaned["country_norm"].astype("category"),
        "name_norm":         cleaned["name_norm"].astype("string"),
        "name_core":         cleaned["name_core"].astype("string"),
        "name_legal_suffix": cleaned["name_legal_suffix"].astype("string"),
        "addr_norm":         cleaned["addr_norm"].astype("string"),
        "pincode":           cleaned["pincode"].astype("string"),
        "has_address":       cleaned["has_address"].astype("int8"),
        "name_len":          cleaned["name_len"].astype("int16"),
        "addr_len":          cleaned["addr_len"].astype("int16"),
    })

    out.to_parquet(output_path, index=False, compression="snappy")
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"    Saved {output_path.name}  ({size_mb:.1f} MB)")

    # Free memory
    del df, cleaned_chunks, cleaned, out
    gc.collect()


# ============================================================
# MAIN
# ============================================================
def main():
    for split, tsv_name, parquet_name in FILES:
        in_path  = BASE / split / tsv_name
        out_path = OUT / parquet_name

        if not in_path.exists():
            print(f"[SKIP] {in_path} not found")
            continue

        if out_path.exists():
            print(f"[SKIP] {out_path.name} already exists")
            continue

        process_file(in_path, out_path)

    # Summary
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Output: {OUT}\n")
    for f in sorted(OUT.glob("*.parquet")):
        print(f"  {f.name:30s} {f.stat().st_size / 1024 / 1024:>8.1f} MB")


if __name__ == "__main__":
    main()