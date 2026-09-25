"""
STEP 3: Compute similarity features for each candidate pair.

Input:  candidate_pairs_train.parquet (~28M pairs)
Output: features_train.parquet (features + is_match)

Memory-safe: processes in chunks of 200k pairs.
Uses rapidfuzz (C-optimized) for speed.
"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

# ============================================================
# PATHS
# ============================================================
CLEAN = Path("/Users/bharadwaj/Downloads/student_resource/dataset/clean")
CAND  = Path("/Users/bharadwaj/Downloads/student_resource/dataset/candidates")
OUT   = CAND   # save features alongside candidates

CHUNK_SIZE = 200_000

# ============================================================
# 1. Load pairs
# ============================================================
print("Loading candidate pairs...")
pairs = pd.read_parquet(CAND / "candidate_pairs_train.parquet")
print(f"  {len(pairs):,} pairs")
print(f"  Columns: {pairs.columns.tolist()}")

# ============================================================
# 2. Load S1 + pool (S2+S3) with cleaned records
# ============================================================
print("\nLoading S1...")
s1 = pd.read_parquet(CLEAN / "train_source1.parquet")
s1 = s1.set_index("entity_id")
print(f"  {len(s1):,}")

print("Loading pool (S2 + S3)...")
s2 = pd.read_parquet(CLEAN / "train_source2.parquet")
s3 = pd.read_parquet(CLEAN / "train_source3.parquet")
pool = pd.concat([s2, s3], ignore_index=True)
del s2, s3
gc.collect()
pool = pool.set_index("entity_id")
print(f"  {len(pool):,}")

# ============================================================
# 3. Feature computation
# ============================================================
def compute_features_for_chunk(s1_chunk, pool_chunk):
    """Compute features for a batch of aligned S1/pool records."""
    n = len(s1_chunk)

    # Pre-extract as lists for fast iteration — with NaN safety
    s1_name   = s1_chunk['name_norm'].fillna("").astype(str).tolist()
    s1_core   = s1_chunk['name_core'].fillna("").astype(str).tolist()
    s1_legal  = s1_chunk['name_legal_suffix'].fillna("").astype(str).tolist()
    s1_addr   = s1_chunk['addr_norm'].fillna("").astype(str).tolist()
    s1_pin    = s1_chunk['pincode'].fillna("").astype(str).tolist()
    s1_cntry  = s1_chunk['country_norm'].fillna("").astype(str).tolist()
    s1_nlen   = s1_chunk['name_len'].fillna(0).astype(int).tolist()
    s1_alen   = s1_chunk['addr_len'].fillna(0).astype(int).tolist()

    p_name    = pool_chunk['name_norm'].fillna("").astype(str).tolist()
    p_core    = pool_chunk['name_core'].fillna("").astype(str).tolist()
    p_legal   = pool_chunk['name_legal_suffix'].fillna("").astype(str).tolist()
    p_addr    = pool_chunk['addr_norm'].fillna("").astype(str).tolist()
    p_pin     = pool_chunk['pincode'].fillna("").astype(str).tolist()
    p_cntry   = pool_chunk['country_norm'].fillna("").astype(str).tolist()
    p_nlen    = pool_chunk['name_len'].fillna(0).astype(int).tolist()
    p_alen    = pool_chunk['addr_len'].fillna(0).astype(int).tolist()

    # Output buffers
    name_ratio       = np.zeros(n, dtype=np.float32)
    name_partial     = np.zeros(n, dtype=np.float32)
    name_token_set   = np.zeros(n, dtype=np.float32)
    name_token_sort  = np.zeros(n, dtype=np.float32)
    name_core_ratio  = np.zeros(n, dtype=np.float32)
    legal_match      = np.zeros(n, dtype=np.int8)
    name_exact       = np.zeros(n, dtype=np.int8)
    name_len_ratio   = np.zeros(n, dtype=np.float32)

    addr_ratio       = np.zeros(n, dtype=np.float32)
    addr_partial     = np.zeros(n, dtype=np.float32)
    addr_token_set   = np.zeros(n, dtype=np.float32)
    addr_token_sort  = np.zeros(n, dtype=np.float32)
    addr_exact       = np.zeros(n, dtype=np.int8)
    addr_len_ratio   = np.zeros(n, dtype=np.float32)

    pincode_match    = np.zeros(n, dtype=np.int8)
    country_match    = np.zeros(n, dtype=np.int8)
    both_have_addr   = np.zeros(n, dtype=np.int8)
    name_overlap_cnt = np.zeros(n, dtype=np.int16)
    addr_overlap_cnt = np.zeros(n, dtype=np.int16)
    combined_ratio   = np.zeros(n, dtype=np.float32)

    for i in range(n):
        sn, pn = s1_name[i], p_name[i]
        sa, pa = s1_addr[i], p_addr[i]

        # --- Name features ---
        name_ratio[i]      = fuzz.ratio(sn, pn) / 100.0
        name_partial[i]    = fuzz.partial_ratio(sn, pn) / 100.0
        name_token_set[i]  = fuzz.token_set_ratio(sn, pn) / 100.0
        name_token_sort[i] = fuzz.token_sort_ratio(sn, pn) / 100.0
        name_core_ratio[i] = fuzz.ratio(s1_core[i], p_core[i]) / 100.0
        legal_match[i]     = int(s1_legal[i] == p_legal[i])
        name_exact[i]      = int(sn == pn and bool(sn))
        max_nlen = max(s1_nlen[i], p_nlen[i])
        name_len_ratio[i]  = s1_nlen[i] / max_nlen if max_nlen > 0 else 0.0

        # --- Address features ---
        addr_ratio[i]      = fuzz.ratio(sa, pa) / 100.0
        addr_partial[i]    = fuzz.partial_ratio(sa, pa) / 100.0
        addr_token_set[i]  = fuzz.token_set_ratio(sa, pa) / 100.0
        addr_token_sort[i] = fuzz.token_sort_ratio(sa, pa) / 100.0
        addr_exact[i]      = int(sa == pa and bool(sa))
        max_alen = max(s1_alen[i], p_alen[i])
        addr_len_ratio[i]  = s1_alen[i] / max_alen if max_alen > 0 else 0.0

        # --- Structural features ---
        pincode_match[i]   = int(bool(s1_pin[i]) and s1_pin[i] == p_pin[i])
        country_match[i]   = int(s1_cntry[i] == p_cntry[i] and bool(s1_cntry[i]))
        both_have_addr[i]  = int(bool(sa) and bool(pa))

        # --- Token overlap counts ---
        name_overlap_cnt[i] = len(set(sn.split()) & set(pn.split()))
        addr_overlap_cnt[i] = len(set(sa.split()) & set(pa.split()))

        # --- Combined name+address ratio ---
        combined_ratio[i] = fuzz.token_set_ratio(
            f"{sn} {sa}", f"{pn} {pa}"
        ) / 100.0

    return pd.DataFrame({
        "name_ratio":        name_ratio,
        "name_partial":      name_partial,
        "name_token_set":    name_token_set,
        "name_token_sort":   name_token_sort,
        "name_core_ratio":   name_core_ratio,
        "legal_match":       legal_match,
        "name_exact":        name_exact,
        "name_len_ratio":    name_len_ratio,
        "addr_ratio":        addr_ratio,
        "addr_partial":      addr_partial,
        "addr_token_set":    addr_token_set,
        "addr_token_sort":   addr_token_sort,
        "addr_exact":        addr_exact,
        "addr_len_ratio":    addr_len_ratio,
        "pincode_match":     pincode_match,
        "country_match":     country_match,
        "both_have_addr":    both_have_addr,
        "name_overlap_cnt":  name_overlap_cnt,
        "addr_overlap_cnt":  addr_overlap_cnt,
        "combined_ratio":    combined_ratio,
    })


# ============================================================
# 4. Process pairs in chunks
# ============================================================
print("\nComputing features...")
n_total = len(pairs)
feature_chunks = []

for start in range(0, n_total, CHUNK_SIZE):
    end = min(start + CHUNK_SIZE, n_total)
    chunk = pairs.iloc[start:end]

    # Lookup S1 records
    s1_chunk   = s1.loc[chunk['source1_entity_id'].values]
    pool_chunk = pool.loc[chunk['candidate_entity_id'].values]

    # Compute
    feat = compute_features_for_chunk(s1_chunk, pool_chunk)

    # Attach IDs + label
    feat['source1_entity_id']   = chunk['source1_entity_id'].values
    feat['candidate_entity_id'] = chunk['candidate_entity_id'].values
    feat['is_match']            = chunk['is_match'].values

    feature_chunks.append(feat)
    print(f"  {end:>9,} / {n_total:,}", end="\r")

print()

# ============================================================
# 5. Concatenate and save
# ============================================================
print("\nConcatenating...")
features = pd.concat(feature_chunks, ignore_index=True)
del feature_chunks
gc.collect()

print(f"  Features shape: {features.shape}")

out_path = OUT / "features_train.parquet"
features.to_parquet(out_path, index=False)
size_mb = out_path.stat().st_size / 1024 / 1024
print(f"\nSaved: {out_path}")
print(f"  {size_mb:.1f} MB  |  {len(features):,} rows  |  {features.shape[1]} cols")

# Quick stats
print("\nFeature distribution (match vs non-match):")
pos = features[features['is_match'] == 1]
neg = features[features['is_match'] == 0]
for col in ['name_token_set', 'addr_token_set', 'combined_ratio',
            'pincode_match', 'name_overlap_cnt', 'country_match']:
    print(f"  {col:20s}  match_mean={pos[col].mean():.3f}  "
          f"non-match_mean={neg[col].mean():.3f}")