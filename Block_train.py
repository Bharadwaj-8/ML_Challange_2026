import gc
from collections import defaultdict
from pathlib import Path

import pandas as pd

# ============================================================
# PATHS
# ============================================================
CLEAN = Path("/Users/bharadwaj/Downloads/student_resource/dataset/clean")
TRAIN = Path("/Users/bharadwaj/Downloads/student_resource/dataset/train")
OUT   = Path("/Users/bharadwaj/Downloads/student_resource/dataset/candidates")
OUT.mkdir(exist_ok=True)

# ============================================================
# TUNABLE PARAMETERS
# ============================================================
DF_MIN = 2          # token must appear in at least 2 records
DF_MAX = 1_000      # token must appear in at most 1,000 records (rare enough to matter)
MAX_CANDIDATES_PER_S1 = 15   # keep top-15 by token overlap

# ============================================================
# 1. Load pool (S2 + S3)
# ============================================================
print("Loading S2...")
s2 = pd.read_parquet(CLEAN / "train_source2.parquet")
print(f"  {len(s2):,}")

print("Loading S3...")
s3 = pd.read_parquet(CLEAN / "train_source3.parquet")
print(f"  {len(s3):,}")

pool = pd.concat([s2, s3], ignore_index=True)
del s2, s3
gc.collect()
print(f"Pool total: {len(pool):,}")

# Extract as numpy arrays for speed + low memory
pool_ids     = pool['entity_id'].to_numpy()
pool_country = pool['country_norm'].astype(str).to_numpy()
pool_pincode = pool['pincode'].astype(str).to_numpy()
pool_name    = pool['name_core'].astype(str).to_numpy()
pool_addr    = pool['addr_norm'].astype(str).to_numpy()

del pool
gc.collect()

# ============================================================
# 2. Load S1
# ============================================================
print("\nLoading S1...")
s1 = pd.read_parquet(CLEAN / "train_source1.parquet")
print(f"  {len(s1):,}")

# ============================================================
# 3. Pincode index (used only for India — US pincodes unreliable)
# ============================================================
print("\nBuilding pincode index (India only)...")
pincode_index = defaultdict(list)
for i in range(len(pool_ids)):
    pin = pool_pincode[i]
    if pin and pool_country[i] == "india":
        pincode_index[(pool_country[i], pin)].append(i)
print(f"  {len(pincode_index):,} (country, pincode) keys")

# ============================================================
# 4. Name token index
# ============================================================
print("\nPass 1: name token document frequency...")
name_df = defaultdict(int)
for name in pool_name:
    for tok in set(name.split()):
        name_df[tok] += 1

name_keep = {t for t, df in name_df.items() if DF_MIN <= df <= DF_MAX}
print(f"  {len(name_df):,} unique tokens → {len(name_keep):,} kept "
      f"(DF in [{DF_MIN}, {DF_MAX}])")
del name_df
gc.collect()

print("Pass 2: name token inverted index...")
name_index = defaultdict(list)
for i, name in enumerate(pool_name):
    for tok in set(name.split()):
        if tok in name_keep:
            name_index[tok].append(i)
del name_keep
gc.collect()
print(f"  Indexed {len(name_index):,} tokens")

# ============================================================
# 5. Address token index
# ============================================================
print("\nPass 1: address token document frequency...")
addr_df = defaultdict(int)
for addr in pool_addr:
    for tok in set(addr.split()):
        if len(tok) >= 4:
            addr_df[tok] += 1

addr_keep = {t for t, df in addr_df.items() if DF_MIN <= df <= DF_MAX}
print(f"  {len(addr_df):,} unique tokens → {len(addr_keep):,} kept")
del addr_df
gc.collect()

print("Pass 2: address token inverted index...")
addr_index = defaultdict(list)
for i, addr in enumerate(pool_addr):
    for tok in set(addr.split()):
        if len(tok) >= 4 and tok in addr_keep:
            addr_index[tok].append(i)
del addr_keep
gc.collect()
print(f"  Indexed {len(addr_index):,} tokens")

# ============================================================
# 6. Generate candidates for each S1
# ============================================================
print("\nGenerating candidates for S1...")

s1_ids       = s1['entity_id'].to_numpy()
s1_countries = s1['country_norm'].astype(str).to_numpy()
s1_pins      = s1['pincode'].astype(str).to_numpy()
s1_names     = s1['name_core'].astype(str).to_numpy()
s1_addrs     = s1['addr_norm'].astype(str).to_numpy()

pairs_s1 = []
pairs_cand = []
n_with_cand = 0

for i in range(len(s1_ids)):
    s1_country = s1_countries[i]
    cands = set()

    # Skip S1 with no usable fields
    if not s1_names[i] and not s1_addrs[i]:
        continue

    # (a) Pincode — INDIA ONLY
    if s1_pins[i] and s1_country == "india":
        for c in pincode_index.get((s1_country, s1_pins[i]), ()):
            cands.add(c)

    # (b) Name tokens — all countries
    for tok in set(s1_names[i].split()):
        lst = name_index.get(tok)
        if lst:
            for c in lst:
                if pool_country[c] == s1_country:
                    cands.add(c)

    # (c) Address tokens — all countries
    for tok in set(s1_addrs[i].split()):
        if len(tok) >= 4:
            lst = addr_index.get(tok)
            if lst:
                for c in lst:
                    if pool_country[c] == s1_country:
                        cands.add(c)

    if not cands:
        continue

    n_with_cand += 1

    # Cap — keep top-K by quick token overlap score
    if len(cands) > MAX_CANDIDATES_PER_S1:
        s1_name_tokens = set(s1_names[i].split())
        s1_addr_tokens = set(s1_addrs[i].split())
        scored = []
        for c in cands:
            name_overlap = len(s1_name_tokens & set(pool_name[c].split()))
            addr_overlap = len(s1_addr_tokens & set(pool_addr[c].split()))
            score = name_overlap * 3 + addr_overlap   # weight name higher
            scored.append((c, score))
        scored.sort(key=lambda x: -x[1])
        cands = set(c for c, _ in scored[:MAX_CANDIDATES_PER_S1])

    s1_id = s1_ids[i]
    for c in cands:
        pairs_s1.append(s1_id)
        pairs_cand.append(pool_ids[c])

    if (i + 1) % 100_000 == 0:
        print(f"  {i+1:>9,} / {len(s1_ids):,}  |  pairs {len(pairs_s1):,}",
              end="\r")

print(f"\n  {n_with_cand:,} S1 have ≥1 candidate "
      f"({100*n_with_cand/len(s1_ids):.1f}%)")
print(f"  Total pairs: {len(pairs_s1):,}")

pairs_df = pd.DataFrame({
    "source1_entity_id":   pairs_s1,
    "candidate_entity_id": pairs_cand,
})
del pairs_s1, pairs_cand
gc.collect()

# ============================================================
# 7. Label pairs with ground truth
# ============================================================
print("\nAdding ground-truth labels...")
gt = pd.read_csv(TRAIN / "train_ground_truth.tsv", sep="\t")
gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")

gt_pairs = []
for s1_id, matches in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
    if not matches:
        continue
    for m in matches.split(","):
        m = m.strip()
        if m:
            gt_pairs.append((s1_id, m))

gt_df = pd.DataFrame(gt_pairs, columns=["source1_entity_id", "candidate_entity_id"])
gt_df["is_match"] = 1
print(f"  Ground truth pairs: {len(gt_df):,}")

pairs_df = pairs_df.merge(
    gt_df,
    on=["source1_entity_id", "candidate_entity_id"],
    how="left",
)
pairs_df["is_match"] = pairs_df["is_match"].fillna(0).astype("int8")
del gt_df, gt_pairs
gc.collect()

n_pos = int(pairs_df["is_match"].sum())
n_neg = len(pairs_df) - n_pos
print(f"  Positive pairs: {n_pos:,} ({100*n_pos/len(pairs_df):.2f}%)")
print(f"  Negative pairs: {n_neg:,}")

# ============================================================
# 8. Recall ceiling check
# ============================================================
# What fraction of true matches made it into the candidate set?
total_true = len(gt_df) if 'gt_df' in dir() else 0
# Recompute from pairs_df since gt_df was deleted
n_true_in_pairs = n_pos
total_gt_pairs = gt["matched_entity_ids"].fillna("").apply(
    lambda x: len([m for m in x.split(",") if m.strip()])
).sum()
print(f"\n  Recall ceiling: {100*n_true_in_pairs/total_gt_pairs:.2f}% "
      f"({n_true_in_pairs:,} / {total_gt_pairs:,} true matches captured)")

# ============================================================
# 9. Save
# ============================================================
out_path = OUT / "candidate_pairs_train.parquet"
pairs_df.to_parquet(out_path, index=False)
size_mb = out_path.stat().st_size / 1024 / 1024
print(f"\nSaved: {out_path}")
print(f"  {size_mb:.1f} MB  |  {len(pairs_df):,} pairs")