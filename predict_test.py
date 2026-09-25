import gc
import pickle
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

# ============================================================
# PATHS
# ============================================================
CLEAN = Path("/Users/bharadwaj/Downloads/student_resource/dataset/clean")
MODEL_DIR = Path("/Users/bharadwaj/Desktop/ML-Challange-Code/models")
OUT = Path("/Users/bharadwaj/Desktop/ML-Challange-Code/output")
OUT.mkdir(exist_ok=True)

# ============================================================
# 1. Load model bundle
# ============================================================
print("Loading model bundle...")
model = lgb.Booster(model_file=str(MODEL_DIR / "model.lgb"))
with open(MODEL_DIR / "best_threshold.txt") as f:
    THRESHOLD = float(f.read().strip())
with open(MODEL_DIR / "feature_cols.pkl", "rb") as f:
    FEATURE_COLS = pickle.load(f)

print(f"  Trees: {model.num_trees()}")
print(f"  Threshold: {THRESHOLD}")
print(f"  Features: {len(FEATURE_COLS)}")
assert model.feature_name() == FEATURE_COLS, "Feature order mismatch!"
print("  Feature order matches ✅")

# ============================================================
# 2. Load cleaned test records
# ============================================================
print("\nLoading cleaned test data...")
ts1 = pd.read_parquet(CLEAN / "test_source1.parquet")
ts2 = pd.read_parquet(CLEAN / "test_source2.parquet")
ts3 = pd.read_parquet(CLEAN / "test_source3.parquet")
print(f"  ts1: {len(ts1):,}  ts2: {len(ts2):,}  ts3: {len(ts3):,}")

# ============================================================
# 3. Build pool — extract numpy arrays
# ============================================================
print("\nBuilding pool...")
pool = pd.concat([ts2, ts3], ignore_index=True)
del ts2, ts3
gc.collect()
print(f"  Pool: {len(pool):,}")

print("  Extracting pool arrays...")
pool_ids     = pool['entity_id'].to_numpy()
pool_country = pool['country_norm'].astype(str).to_numpy()
pool_pincode = pool['pincode'].fillna("").astype(str).to_numpy()
pool_name    = pool['name_norm'].fillna("").astype(str).to_numpy()
pool_core    = pool['name_core'].fillna("").astype(str).to_numpy()
pool_legal   = pool['name_legal_suffix'].fillna("").astype(str).to_numpy()
pool_addr    = pool['addr_norm'].fillna("").astype(str).to_numpy()
pool_nlen    = pool['name_len'].fillna(0).astype(int).to_numpy()
pool_alen    = pool['addr_len'].fillna(0).astype(int).to_numpy()
del pool
gc.collect()
print("  Pool arrays ready")

# NOTE: We do NOT pre-build name_set / addr_set for the full pool.
# We build them on-the-fly only for the ~15 candidates per S1 that
# actually matter. This keeps RAM under 3 GB.

# ============================================================
# 4. Build blocking indexes
# ============================================================
print("\nBuilding pincode index (India only)...")
pincode_index = defaultdict(list)
for i in range(len(pool_ids)):
    if pool_pincode[i] and pool_country[i] == "india":
        pincode_index[(pool_country[i], pool_pincode[i])].append(i)
print(f"  {len(pincode_index):,} keys")

print("Name token df pass...")
name_df = defaultdict(int)
for name in pool_name:
    for tok in set(name.split()):
        name_df[tok] += 1
name_keep = {t for t, d in name_df.items() if 2 <= d <= 1000}
del name_df
gc.collect()
print(f"  {len(name_keep):,} tokens kept")

print("Name token index...")
name_index = defaultdict(list)
for i, name in enumerate(pool_name):
    for tok in set(name.split()):
        if tok in name_keep:
            name_index[tok].append(i)
del name_keep
gc.collect()

print("Address token df pass...")
addr_df = defaultdict(int)
for addr in pool_addr:
    for tok in set(addr.split()):
        if len(tok) >= 4:
            addr_df[tok] += 1
addr_keep = {t for t, d in addr_df.items() if 2 <= d <= 1000}
del addr_df
gc.collect()
print(f"  {len(addr_keep):,} tokens kept")

print("Address token index...")
addr_index = defaultdict(list)
for i, addr in enumerate(pool_addr):
    for tok in set(addr.split()):
        if len(tok) >= 4 and tok in addr_keep:
            addr_index[tok].append(i)
del addr_keep
gc.collect()

# ============================================================
# 5. Prepare test S1 arrays
# ============================================================
s1_ids    = ts1['entity_id'].to_numpy()
s1_cntry  = ts1['country_norm'].astype(str).to_numpy()
s1_pin    = ts1['pincode'].fillna("").astype(str).to_numpy()
s1_name   = ts1['name_norm'].fillna("").astype(str).to_numpy()
s1_core   = ts1['name_core'].fillna("").astype(str).to_numpy()
s1_legal  = ts1['name_legal_suffix'].fillna("").astype(str).to_numpy()
s1_addr   = ts1['addr_norm'].fillna("").astype(str).to_numpy()
s1_nlen   = ts1['name_len'].fillna(0).astype(int).to_numpy()
s1_alen   = ts1['addr_len'].fillna(0).astype(int).to_numpy()

MAX_CAND = 15

# ============================================================
# 6. Single-pass: blocking + feature computation
# ============================================================
print("\nGenerating candidates + computing features...")
print("  (building sets on-the-fly for candidates only)")

all_s1 = []
all_cand = []
all_feat_rows = []

for i in range(len(s1_ids)):
    s1_c = s1_cntry[i]
    cands = set()

    if not s1_name[i] and not s1_addr[i]:
        continue

    # (a) Pincode — India only
    if s1_pin[i] and s1_c == "india":
        for c in pincode_index.get((s1_c, s1_pin[i]), ()):
            cands.add(c)

    # (b) Name tokens
    for tok in set(s1_core[i].split()):
        lst = name_index.get(tok)
        if lst:
            for c in lst:
                if pool_country[c] == s1_c:
                    cands.add(c)

    # (c) Address tokens
    for tok in set(s1_addr[i].split()):
        if len(tok) >= 4:
            lst = addr_index.get(tok)
            if lst:
                for c in lst:
                    if pool_country[c] == s1_c:
                        cands.add(c)

    if not cands:
        continue

    # Cap by quick token overlap
    if len(cands) > MAX_CAND:
        s1_nt = set(s1_core[i].split())
        s1_at = set(s1_addr[i].split())
        scored = []
        for c in cands:
            # Build sets on-the-fly (only for candidates)
            no = len(s1_nt & set(pool_core[c].split()))
            ao = len(s1_at & set(pool_addr[c].split()))
            scored.append((c, no * 3 + ao))
        scored.sort(key=lambda x: -x[1])
        cands = set(c for c, _ in scored[:MAX_CAND])

    # Compute features
    for c in cands:
        sn, pn = s1_name[i], pool_name[c]
        sc, pc = s1_core[i], pool_core[c]
        sa, pa = s1_addr[i], pool_addr[c]
        s1_n = s1_nlen[i]; p_n = pool_nlen[c]
        s1_a = s1_alen[i]; p_a = pool_alen[c]

        # Build sets only here, per pair
        sn_set = set(sn.split())
        pn_set = set(pn.split())
        sa_set = set(sa.split())
        pa_set = set(pa.split())

        feat = {
            'name_ratio':       fuzz.ratio(sn, pn) / 100.0,
            'name_partial':     fuzz.partial_ratio(sn, pn) / 100.0,
            'name_token_set':   fuzz.token_set_ratio(sn, pn) / 100.0,
            'name_token_sort':  fuzz.token_sort_ratio(sn, pn) / 100.0,
            'name_core_ratio':  fuzz.ratio(sc, pc) / 100.0,
            'legal_match':      int(s1_legal[i] == pool_legal[c]),
            'name_exact':       int(sn == pn and bool(sn)),
            'name_len_ratio':   s1_n / max(s1_n, p_n) if max(s1_n, p_n) > 0 else 0.0,
            'addr_ratio':       fuzz.ratio(sa, pa) / 100.0,
            'addr_partial':     fuzz.partial_ratio(sa, pa) / 100.0,
            'addr_token_set':   fuzz.token_set_ratio(sa, pa) / 100.0,
            'addr_token_sort':  fuzz.token_sort_ratio(sa, pa) / 100.0,
            'addr_exact':       int(sa == pa and bool(sa)),
            'addr_len_ratio':   s1_a / max(s1_a, p_a) if max(s1_a, p_a) > 0 else 0.0,
            'pincode_match':    int(bool(s1_pin[i]) and s1_pin[i] == pool_pincode[c]),
            'both_have_addr':   int(bool(sa) and bool(pa)),
            'name_overlap_cnt': len(sn_set & pn_set),
            'addr_overlap_cnt': len(sa_set & pa_set),
            'combined_ratio':   fuzz.token_set_ratio(f"{sn} {sa}", f"{pn} {pa}") / 100.0,
        }

        all_s1.append(s1_ids[i])
        all_cand.append(pool_ids[c])
        all_feat_rows.append(feat)

    if (i + 1) % 100_000 == 0:
        print(f"  {i+1:>9,} / {len(s1_ids):,}  |  pairs {len(all_s1):,}", end="\r")

print(f"\n  Total candidate pairs: {len(all_s1):,}")

# Free pool arrays we no longer need
del pool_ids, pool_country, pool_pincode, pool_name, pool_core
del pool_legal, pool_addr, pool_nlen, pool_alen
del pincode_index, name_index, addr_index
gc.collect()

# ============================================================
# 7. Score
# ============================================================
print("\nScoring with model...")
X_test = pd.DataFrame(all_feat_rows)[FEATURE_COLS].values.astype(np.float32)
del all_feat_rows
gc.collect()
proba = model.predict(X_test)
print(f"  Predicted {len(proba):,} pairs")

# ============================================================
# 8. Apply threshold + group
# ============================================================
print("\nApplying threshold...")
matches_by_s1 = defaultdict(set)
cands_by_s1 = defaultdict(set)

for s1, cand, p in zip(all_s1, all_cand, proba):
    cands_by_s1[s1].add(cand)
    if p >= THRESHOLD:
        matches_by_s1[s1].add(cand)

print(f"  S1 with matches:    {len(matches_by_s1):,}")
print(f"  S1 with candidates: {len(cands_by_s1):,}")

# ============================================================
# 9. Write matching_results.tsv
# ============================================================
print("\nWriting matching_results.tsv...")
all_test_s1 = ts1['entity_id'].tolist()

with open(OUT / "matching_results.tsv", "w") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for s1 in all_test_s1:
        matches = sorted(matches_by_s1.get(s1, set()))
        f.write(f"{s1}\t{','.join(matches)}\n")

# ============================================================
# 10. Write candidate_pairs.tsv
# ============================================================
print("Writing candidate_pairs.tsv...")
with open(OUT / "candidate_pairs.tsv", "w") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for s1 in all_test_s1:
        cands = sorted(cands_by_s1.get(s1, set()))
        f.write(f"{s1}\t{','.join(cands)}\n")

# ============================================================
# 11. Summary
# ============================================================
n_matches = sum(len(v) for v in matches_by_s1.values())
n_cands = sum(len(v) for v in cands_by_s1.values())

print(f"\n{'='*60}")
print("DONE")
print(f"{'='*60}")
print(f"Output: {OUT}")
print(f"  matching_results.tsv   ({len(all_test_s1):,} rows, {n_matches:,} matches)")
print(f"  candidate_pairs.tsv    ({len(all_test_s1):,} rows, {n_cands:,} candidates)")
print(f"\nNext: run utils/validate_submission.py")