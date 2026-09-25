import pandas as pd

s1 = pd.read_parquet("/Users/bharadwaj/Downloads/student_resource/dataset/clean/train_source1.parquet")
ts1 = pd.read_parquet("/Users/bharadwaj/Downloads/student_resource/dataset/clean/test_source1.parquet")

print("Train S1 countries:", s1['country_norm'].value_counts().to_dict())
print("Test S1 countries: ", ts1['country_norm'].value_counts().to_dict())
print()
print("Sample cleaned record:")
print(s1[['entity_id', 'name_norm', 'addr_norm', 'pincode']].head(3).to_string())