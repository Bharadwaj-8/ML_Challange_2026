import pandas as pd
from langdetect import detect, DetectorFactory
from collections import Counter

DetectorFactory.seed = 0   # reproducibility

# ---------- Load all 6 files ----------
s1 = pd.read_csv("/Users/bharadwaj/Downloads/student_resource/dataset/train/train_source1.tsv", sep="\t")
s2 = pd.read_csv("/Users/bharadwaj/Downloads/student_resource/dataset/train/train_source2.tsv", sep="\t")
s3 = pd.read_csv("/Users/bharadwaj/Downloads/student_resource/dataset/train/train_source3.tsv", sep="\t")

ts1 = pd.read_csv("/Users/bharadwaj/Downloads/student_resource/dataset/test/test_source1.tsv", sep="\t")
ts2 = pd.read_csv("/Users/bharadwaj/Downloads/student_resource/dataset/test/test_source2.tsv", sep="\t")
ts3 = pd.read_csv("/Users/bharadwaj/Downloads/student_resource/dataset/test/test_source3.tsv", sep="\t")

# ---------- Detector ----------
def safe_detect(text, min_len=15):
    if pd.isna(text):
        return 'null'
    text = str(text).strip()
    if len(text) < min_len:
        return 'too_short'
    try:
        return detect(text)
    except:
        return 'unknown'

# ---------- Sample and detect ----------
def detect_languages(df, col, sample_size=30_000):
    """Return a Series of detected languages for a sample of the column."""
    if len(df) > sample_size:
        sample = df[col].sample(sample_size, random_state=42)
    else:
        sample = df[col]
    return sample.apply(safe_detect)

# ---------- Report per source × field ----------
def report_source(df, name, sample_size=30_000):
    print(f"\n{'='*70}")
    print(f"  {name}  (n={len(df):,})")
    print(f"{'='*70}")
    for col in ["business_name", "business_address"]:
        langs = detect_languages(df, col, sample_size)
        counts = Counter(langs)
        total = len(langs)
        print(f"\n  [{col}]")
        for lang, c in counts.most_common(15):
            print(f"    {lang:12s}: {c:>7,}  ({100*c/total:6.2f}%)")

# ---------- Report per country × source ----------
def report_by_country(df, source_name, sample_per_country=30_000):
    print(f"\n{'#'*70}")
    print(f"  {source_name} — BY COUNTRY")
    print(f"{'#'*70}")
    for country in sorted(df['country'].dropna().unique()):
        sub = df[df['country'] == country]
        print(f"\n----- {source_name} | country = {country} (n={len(sub):,}) -----")
        for col in ["business_name", "business_address"]:
            langs = detect_languages(sub, col, sample_per_country)
            counts = Counter(langs)
            total = len(langs)
            top = ", ".join(f"{l}({100*c/total:.1f}%)" for l, c in counts.most_common(5))
            print(f"    {col:20s} → {top}")

# =========================================================
# PART A: Language distribution per source (train + test)
# =========================================================
print("\n" + "#"*70)
print("  PART A: LANGUAGE DISTRIBUTION PER SOURCE")
print("#"*70)

report_source(s1,  "TRAIN S1")
report_source(s2,  "TRAIN S2")
report_source(s3,  "TRAIN S3")
report_source(ts1, "TEST  S1")
report_source(ts2, "TEST  S2")
report_source(ts3, "TEST  S3")

# =========================================================
# PART B: Language distribution per country (the key part)
# =========================================================
print("\n" + "#"*70)
print("  PART B: LANGUAGE DISTRIBUTION PER COUNTRY")
print("#"*70)

report_by_country(s1,  "TRAIN S1")
report_by_country(ts1, "TEST  S1")
report_by_country(ts2, "TEST  S2")
report_by_country(ts3, "TEST  S3")