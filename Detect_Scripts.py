#Script-1: Detect Scripts VIA Unicode Ranges


import pandas as pd
import re
from collections import Counter

# Load your data
s1 = pd.read_csv("/Users/bharadwaj/Downloads/student_resource/dataset/train/train_source1.tsv", sep="\t")
s2 = pd.read_csv("/Users/bharadwaj/Downloads/student_resource/dataset/train/train_source2.tsv", sep="\t")
s3 = pd.read_csv("/Users/bharadwaj/Downloads/student_resource/dataset/train/train_source3.tsv", sep="\t")

ts1 = pd.read_csv("/Users/bharadwaj/Downloads/student_resource/dataset/test/test_source1.tsv", sep="\t")
ts2 = pd.read_csv("/Users/bharadwaj/Downloads/student_resource/dataset/test/test_source2.tsv", sep="\t")
ts3 = pd.read_csv("/Users/bharadwaj/Downloads/student_resource/dataset/test/test_source3.tsv", sep="\t")

# Define script ranges
SCRIPT_RANGES = {
    'Latin':       [(0x0041, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF)],
    'Devanagari':  [(0x0900, 0x097F)],
    'Tamil':       [(0x0B80, 0x0BFF)],
    'Telugu':      [(0x0C00, 0x0C7F)],
    'Kannada':     [(0x0C80, 0x0CFF)],
    'Malayalam':   [(0x0D00, 0x0D7F)],
    'Bengali':     [(0x0980, 0x09FF)],
    'Gujarati':    [(0x0A80, 0x0AFF)],
    'Gurmukhi':    [(0x0A00, 0x0A7F)],
    'Arabic':      [(0x0600, 0x06FF)],
    'CJK':         [(0x4E00, 0x9FFF)],
    'Cyrillic':    [(0x0400, 0x04FF)],
}

def detect_script(char):
    cp = ord(char)
    for script, ranges in SCRIPT_RANGES.items():
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return script
    return 'Other'

def get_scripts_in_text(text):
    """Return set of scripts used in the text (ignoring digits/punct/spaces)."""
    if pd.isna(text):
        return set()
    scripts = set()
    for ch in str(text):
        if ch.isalpha():   # only letters — skip digits, punctuation, spaces
            s = detect_script(ch)
            if s != 'Other':
                scripts.add(s)
    return scripts

def analyze_column(df, col, name):
    """Count script distribution across a column."""
    all_scripts = Counter()
    for text in df[col]:
        for s in get_scripts_in_text(text):
            all_scripts[s] += 1
    total = len(df)
    print(f"\n--- {name} :: {col} ---")
    for script, count in all_scripts.most_common():
        print(f"  {script:15s}: {count:>9,}  ({100*count/total:.2f}%)")


for name, df in [("train_S1", s1), ("train_S2", s2), ("train_S3", s3),
                 ("test_S1",  ts1), ("test_S2",  ts2), ("test_S3",  ts3)]:
    analyze_column(df, "business_name",    name)
    analyze_column(df, "business_address", name)