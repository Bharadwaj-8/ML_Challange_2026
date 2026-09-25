import pandas as pd 
import numpy as np 

s1 = pd.read_csv("/Users/bharadwaj/Downloads/student_resource/dataset/train/train_source1.tsv", sep="\t")
s2 = pd.read_csv("/Users/bharadwaj/Downloads/student_resource/dataset/train/train_source2.tsv", sep="\t")
s3 = pd.read_csv("/Users/bharadwaj/Downloads/student_resource/dataset/train/train_source3.tsv", sep="\t")
gt = pd.read_csv("/Users/bharadwaj/Downloads/student_resource/dataset/train/train_ground_truth.tsv", sep="\t")

print("S1 Shape:", s1.shape)
print("S2 shape:", s2.shape)
print("S3 shape:", s3.shape)
print("GT shape:", gt.shape)

print("\nS1 columns:", s1.columns.tolist())
print("\nS2 columns:", s2.columns.tolist())
print("\nS3 columns:", s3.columns.tolist())
print("\nGT columns:", gt.columns.tolist())


#Section-1: Basic Sanity Check 
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    print(f"\n===== {name} =====")
    print("Rows:", len(df))
    print("Unique entity_ids:", df['entity_id'].nunique())
    print("Null counts:\n", df.isnull().sum())
    print("ID prefixes:", df['entity_id'].str[:3].unique())


#Section-2: Country Distribution (Critical for France Problem)
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    print(f"\n{name} country distribution:")
    print(df['country'].value_counts(dropna=False))
    print("Unique country labels:", df['country'].unique())


#Section-3: Look at Raw Records (Eyeball the noise)
pd.set_option('display.max_colwidth', 200)

print("--- Sample S1 ---")
print(s1.sample(10, random_state=42).to_string())

print("\n--- Sample S2 ---")
print(s2.sample(10, random_state=42).to_string())

print("\n--- Sample S3 ---")
print(s3.sample(10, random_state=42).to_string())

print("\n--- Sample GT ---")
print(gt.sample(10, random_state=42).to_string())


#section-4: Ground Truth Analysis (Understand teh Match Distribution)
# Parse the GT
gt['matched_list'] = gt['matched_entity_ids'].fillna('').apply(
    lambda x: [m.strip() for m in x.split(',') if m.strip() != '']
)
gt['num_matches'] = gt['matched_list'].apply(len)

print("Total S1 entities in GT:", len(gt))
print("\nMatch count distribution:")
print(gt['num_matches'].value_counts().sort_index())

print("\nNumber of singletons (0 matches):", (gt['num_matches'] == 0).sum())
print("Percentage singletons: {:.2f}%".format(
    100 * (gt['num_matches'] == 0).mean()
))
print("Max matches for a single S1 entity:", gt['num_matches'].max())


#Section-5: Where Do Matches come from? (S2 vs S3 vs Both)
def source_of(matches):
    s2 = sum(1 for m in matches if m.startswith('S2-'))
    s3 = sum(1 for m in matches if m.startswith('S3-'))
    return pd.Series({'S2_matches': s2, 'S3_matches': s3})

gt = gt.join(gt['matched_list'].apply(source_of))

print("Total S2 matches across all S1:", gt['S2_matches'].sum())
print("Total S3 matches across all S1:", gt['S3_matches'].sum())

# Do some S1 entities match both S2 and S3?
gt['matches_both'] = (gt['S2_matches'] > 0) & (gt['S3_matches'] > 0)
print("S1 entities matching BOTH S2 and S3:", gt['matches_both'].sum())
print("S1 entities matching ONLY S2:",
      ((gt['S2_matches'] > 0) & (gt['S3_matches'] == 0)).sum())
print("S1 entities matching ONLY S3:",
      ((gt['S3_matches'] == 0) & (gt['S3_matches'] > 0)).sum())


#section-6: Name and Address Length Distributions
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    df['name_len'] = df['business_name'].str.len()
    df['addr_len'] = df['business_address'].str.len()
    print(f"\n{name} name length: mean={df['name_len'].mean():.1f}, "
          f"min={df['name_len'].min()}, max={df['name_len'].max()}")
    print(f"{name} addr length: mean={df['addr_len'].mean():.1f}, "
          f"min={df['addr_len'].min()}, max={df['addr_len'].max()}")


#section-7: Look at true match pairs side-by-side(it tells us what kind of noise we must handle)
# Build a lookup for S2 and S3
s2_lookup = s2.set_index('entity_id')
s3_lookup = s3.set_index('entity_id')

# Take a few S1 entities that have matches
sample_gt = gt[gt['num_matches'] > 0].sample(5, random_state=1)

for _, row in sample_gt.iterrows():
    s1_id = row['source1_entity_id']
    s1_rec = s1[s1['entity_id'] == s1_id].iloc[0]
    print("=" * 90)
    print(f"S1 [{s1_id}]")
    print(f"  NAME: {s1_rec['business_name']}")
    print(f"  ADDR: {s1_rec['business_address']}")
    print(f"  COUNTRY: {s1_rec['country']}")
    for m in row['matched_list']:
        src = s2_lookup if m.startswith('S2-') else s3_lookup
        if m in src.index:
            rec = src.loc[m]
            print(f"\n  --> MATCH [{m}]")
            print(f"      NAME: {rec['business_name']}")
            print(f"      ADDR: {rec['business_address']}")
            print(f"      COUNTRY: {rec['country']}")



#section-8: Look at singletons(Entities with no match)
singleton_ids = gt[gt['num_matches'] == 0]['source1_entity_id'].tolist()
s1_singletons = s1[s1['entity_id'].isin(singleton_ids)]
print("Sample singletons:")
print(s1_singletons.sample(min(10, len(s1_singletons)), random_state=1).to_string())



#section-9: Quick Token Overlap Sanity Check
from difflib import SequenceMatcher

def ratio(a, b):
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

# Take 50 true match pairs, 50 random pairs
true_pairs = []
for _, row in gt[gt['num_matches'] > 0].head(50).iterrows():
    s1_rec = s1[s1['entity_id'] == row['source1_entity_id']].iloc[0]
    m = row['matched_list'][0]
    src = s2_lookup if m.startswith('S2-') else s3_lookup
    if m in src.index:
        m_rec = src.loc[m]
        true_pairs.append(ratio(s1_rec['business_name'], m_rec['business_name']))

# Random pairs
rng = np.random.default_rng(0)
random_pairs = []
for _ in range(50):
    a = s1.sample(1, random_state=rng.integers(1e9)).iloc[0]
    b = s2.sample(1, random_state=rng.integers(1e9)).iloc[0]
    random_pairs.append(ratio(a['business_name'], b['business_name']))

print("Avg name similarity — TRUE matches:", np.mean(true_pairs))
print("Avg name similarity — RANDOM pairs:", np.mean(random_pairs))



#section-10: Check teh test set structure
ts1 = pd.read_csv("/Users/bharadwaj/Downloads/student_resource/dataset/test/test_source1.tsv", sep="\t")
ts2 = pd.read_csv("/Users/bharadwaj/Downloads/student_resource/dataset/test/test_source2.tsv", sep="\t")
ts3 = pd.read_csv("/Users/bharadwaj/Downloads/student_resource/dataset/test/test_source3.tsv", sep="\t")

print("Test S1:", ts1.shape)
print("Test S2:", ts2.shape)
print("Test S3:", ts3.shape)
print("\nTest country distribution (S1):")
print(ts1['country'].value_counts())
print("\nTest country distribution (S2):")
print(ts2['country'].value_counts())
print("\nTest country distribution (S3):")
print(ts3['country'].value_counts())