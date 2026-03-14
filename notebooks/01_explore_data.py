import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def parse_filename(filepath, machine_type):
    """Extract label and metadata from filename"""
    name = Path(filepath).stem  # remove .wav
    parts = name.split('_')
    
    # label is either 'normal' or 'anomaly'
    label = 0 if 'normal' in parts else 1
    domain = 'source' if 'source' in parts else 'target'
    split = 'train' if 'train' in parts else 'test'
    
    return {
        'filepath': str(filepath),
        'machine_type': machine_type,
        'label': label,
        'domain': domain,
        'original_split': split
    }

def build_dataframe(data_dir):
    records = []
    
    for machine in ['bearing', 'gearbox']:
        machine_dir = Path(data_dir) / machine
        
        # walk through train and test folders
        for split_folder in ['train', 'test']:
            folder = machine_dir / split_folder
            for wav_file in folder.glob('*.wav'):
                record = parse_filename(wav_file, machine)
                records.append(record)
    
    return pd.DataFrame(records)

# Build the dataframe
df = build_dataframe('data/raw')

# Print summary
print("Dataset Summary")
print(f"Total clips: {len(df)}")
print(f"\nBy machine type:")
print(df['machine_type'].value_counts())
print(f"\nBy label (0=normal, 1=anomaly):")
print(df['label'].value_counts())
print(f"\nBy original split:")
print(df['original_split'].value_counts())
print(f"\nBy domain:")
print(df['domain'].value_counts())

# Mix everything and do a fresh 80/20 stratified split
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['label'],  # maintains 11:1 ratio in both splits
    random_state=42
)

# Add a split column
train_df = train_df.copy()
test_df = test_df.copy()
train_df['split'] = 'train'
test_df['split'] = 'test'

# Combine back into one dataframe
final_df = pd.concat([train_df, test_df])

# Print new split summary
print("\nAfter 80/20 Split")
print(f"Training clips: {len(train_df)}")
print(f"  Normal: {len(train_df[train_df['label']==0])}")
print(f"  Anomalous: {len(train_df[train_df['label']==1])}")
print(f"\nTest clips: {len(test_df)}")
print(f"  Normal: {len(test_df[test_df['label']==0])}")
print(f"  Anomalous: {len(test_df[test_df['label']==1])}")

# Save to CSV
final_df.to_csv('data/dataset_split.csv', index=False)
print("\n CSV saved to data/dataset_split.csv")