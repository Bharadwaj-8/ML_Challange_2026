"""
STEP 3: Train LightGBM on feature matrix + tune threshold for macro F_0.5.

Input:  features_train.parquet
Output: models/model.lgb + models/best_threshold.txt + training_report.txt
"""

import gc
import pickle
import random
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================
CLEAN = Path("/Users/bharadwaj/Downloads/student_resource/dataset/clean")
CAND  = Path("/Users/bharadwaj/Downloads/student_resource/dataset/candidates")
TRAIN = Path("/Users/bharadwaj/Downloads/student_resource/dataset/train")
MODEL_DIR = Path("/Users/bharadwaj/Desktop/ML-Challange-Code/models")
MODEL_DIR.mkdir(exist_ok=True)

# ============================================================
# 1. Load S1 IDs (as plain Python list to avoid Arrow shuffle issues)
# ============================================================
print("Loading S1 IDs...")
s1_all = pd.read_parquet(CLEAN / "train_source1.parquet", columns=['entity_id'])
all_s1 = s1_all['entity_id'].astype(str).tolist()
del s1_all
gc.collect()
print(f"  {len(all_s1):,} S1 entities")

# ============================================================
# 2. Split 80/20 by S1 entity
# ============================================================
random.seed(42)
shuffled = all_s1.copy()
random.shuffle(shuffled)

n_val = int(len(shuffled) * 0.2)
val_s1_set = set(shuffled[:n_val])
train_s1_set = set(shuffled[n_val:])
del shuffled
gc.collect()

print(f"  Train S1: {len(train_s1_set):,}")
print(f"  Val S1:   {len(val_s1_set):,}")

# ============================================================
# 3. Load ground truth (for macro F_0.5 with singletons)
# ============================================================
print("\nLoading ground truth...")
gt = pd.read_csv(TRAIN / "train_ground_truth.tsv", sep="\t")
gt['matched_entity_ids'] = gt['matched_entity_ids'].fillna('')
gt_lookup = {}
for s1, m in zip(gt['source1_entity_id'], gt['matched_entity_ids']):
    gt_lookup[s1] = set(x.strip() for x in m.split(',') if x.strip())
del gt
gc.collect()
print(f"  {len(gt_lookup):,} S1 with ground truth entries")

# ============================================================
# 4. Load features
# ============================================================
print("\nLoading features...")
features = pd.read_parquet(CAND / "features_train.parquet")
print(f"  {len(features):,} rows")

# Drop useless features (all 1.0 due to country partition in blocking)
DROP_COLS = ['country_match']
feature_cols = [c for c in features.columns
                if c not in ['source1_entity_id', 'candidate_entity_id', 'is_match']
                and c not in DROP_COLS]
print(f"  Using {len(feature_cols)} features: {feature_cols}")

# Split masks
train_mask = features['source1_entity_id'].isin(train_s1_set).values
val_mask   = features['source1_entity_id'].isin(val_s1_set).values

print(f"  Train rows: {train_mask.sum():,}")
print(f"  Val rows:   {val_mask.sum():,}")

# ============================================================
# 5. Extract numpy arrays (free pandas memory early)
# ============================================================
print("\nBuilding numpy arrays...")

X_train = features.loc[train_mask, feature_cols].values.astype(np.float32)
y_train = features.loc[train_mask, 'is_match'].values.astype(np.int8)

X_val = features.loc[val_mask, feature_cols].values.astype(np.float32)
y_val = features.loc[val_mask, 'is_match'].values.astype(np.int8)

# For scoring: keep S1 IDs and candidate IDs aligned with X_val
val_s1_ids   = features.loc[val_mask, 'source1_entity_id'].values
val_cand_ids = features.loc[val_mask, 'candidate_entity_id'].values

del features
gc.collect()
print(f"  X_train: {X_train.shape}, positives: {y_train.sum():,}")
print(f"  X_val:   {X_val.shape}, positives: {y_val.sum():,}")

# ============================================================
# 6. Train LightGBM
# ============================================================
print("\nTraining LightGBM...")

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.05,
    'num_leaves': 127,
    'min_data_in_leaf': 100,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l2': 1.0,
    'verbose': -1,
    'num_threads': -1,
}

train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
val_data   = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feature_cols)

model = lgb.train(
    params,
    train_data,
    num_boost_round=2000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
)

print(f"\nBest iteration: {model.best_iteration}")
print(f"Best validation logloss: {model.best_score['valid_0']['binary_logloss']:.5f}")

# Free training data
del train_data, val_data, X_train, y_train
gc.collect()

# ============================================================
# 7. Predict on validation
# ============================================================
print("\nPredicting on validation...")
val_proba = model.predict(X_val, num_iteration=model.best_iteration)
print(f"  Predictions: {val_proba.shape}, range [{val_proba.min():.4f}, {val_proba.max():.4f}]")

# ============================================================
# 8. Macro F_0.5 threshold sweep
# ============================================================
print("\nSweeping thresholds for macro F_0.5...")

# Build per-S1 lists of (proba, cand_id)
s1_to_preds = defaultdict(list)
for s1, cand, p in zip(val_s1_ids, val_cand_ids, val_proba):
    s1_to_preds[s1].append((p, cand))

# S1 in validation with no candidates (blocking found nothing)
val_s1_no_cand = val_s1_set - set(s1_to_preds.keys())
print(f"  Val S1 with candidates: {len(s1_to_preds):,}")
print(f"  Val S1 without candidates: {len(val_s1_no_cand):,}")

def macro_f05_at_threshold(threshold):
    """Compute macro F_0.5 across all validation S1 entities."""
    scores = []

    for s1, preds in s1_to_preds.items():
        pred_set = set(c for p, c in preds if p >= threshold)
        true_set = gt_lookup.get(s1, set())

        if len(true_set) == 0:
            scores.append(1.0 if len(pred_set) == 0 else 0.0)
            continue
        if len(pred_set) == 0:
            scores.append(0.0)
            continue

        tp = len(true_set & pred_set)
        p = tp / len(pred_set)
        r = tp / len(true_set)
        if p + r == 0:
            scores.append(0.0)
        else:
            scores.append((1.25 * p * r) / (0.25 * p + r))

    for s1 in val_s1_no_cand:
        true_set = gt_lookup.get(s1, set())
        scores.append(1.0 if len(true_set) == 0 else 0.0)

    return np.mean(scores)

thresholds = np.arange(0.10, 0.96, 0.025)
results = []
for t in thresholds:
    f05 = macro_f05_at_threshold(t)
    results.append((t, f05))
    print(f"  t={t:.3f}  F_0.5={f05:.4f}")

best_t, best_f05 = max(results, key=lambda x: x[1])
print(f"\n>>> Best threshold: {best_t:.3f}")
print(f">>> Best validation macro F_0.5: {best_f05:.4f}")

# ============================================================
# 9. Breakdown by country (for diagnosis)
# ============================================================
print("\nBreakdown by country...")
s1_country = pd.read_parquet(CLEAN / "train_source1.parquet",
                              columns=['entity_id', 'country_norm'])
country_map = dict(zip(s1_country['entity_id'].astype(str),
                        s1_country['country_norm'].astype(str)))
del s1_country
gc.collect()

for country in ['us', 'india']:
    country_s1 = [s for s in val_s1_set if country_map.get(s) == country]
    if not country_s1:
        continue
    scores = []
    for s1 in country_s1:
        pred_set = set(c for p, c in s1_to_preds.get(s1, []) if p >= best_t)
        true_set = gt_lookup.get(s1, set())
        if len(true_set) == 0:
            scores.append(1.0 if len(pred_set) == 0 else 0.0)
            continue
        if len(pred_set) == 0:
            scores.append(0.0)
            continue
        tp = len(true_set & pred_set)
        p = tp / len(pred_set)
        r = tp / len(true_set)
        if p + r == 0:
            scores.append(0.0)
        else:
            scores.append((1.25 * p * r) / (0.25 * p + r))
    print(f"  {country:8s} | {len(country_s1):>7,} S1 | F_0.5 = {np.mean(scores):.4f}")

# ============================================================
# 10. Feature importance
# ============================================================
print("\nFeature importance (top 20):")
importance = model.feature_importance(importance_type='gain')
for name, imp in sorted(zip(feature_cols, importance), key=lambda x: -x[1]):
    print(f"  {name:20s}  {imp:>12,.0f}")

# ============================================================
# 11. Save model + threshold
# ============================================================
model_path = MODEL_DIR / "model.lgb"
model.save_model(str(model_path))
print(f"\nSaved model: {model_path}")

with open(MODEL_DIR / "best_threshold.txt", "w") as f:
    f.write(f"{best_t:.6f}\n")
with open(MODEL_DIR / "feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)
print(f"Saved threshold: {best_t:.3f}")
print(f"Saved feature list: {len(feature_cols)} features")

# ============================================================
# 12. Summary report
# ============================================================
with open(MODEL_DIR / "training_report.txt", "w") as f:
    f.write(f"Best iteration: {model.best_iteration}\n")
    f.write(f"Best validation logloss: {model.best_score['valid_0']['binary_logloss']:.5f}\n")
    f.write(f"Best threshold: {best_t:.4f}\n")
    f.write(f"Best validation macro F_0.5: {best_f05:.4f}\n")

print("\nDone.")