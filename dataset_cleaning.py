import pandas as pd
import numpy as np
import os
from datetime import datetime

# Step 1: Load the dataset
# Replace 'diabetes.csv' with your actual file path
df = pd.read_csv(r"C:\Users\KIIT0001\Desktop\Projects\Research Paper\diabetes_prediction_dataset.csv")

print("="*70)
print("ORIGINAL DATASET SUMMARY")
print("="*70)
print(f"Dataset Shape: {df.shape}")
print(f"\nClass Distribution:")
print(df['diabetes'].value_counts())
print(f"\nClass Distribution (%):")
print(df['diabetes'].value_counts(normalize=True) * 100)

# Step 2: Separate data by diabetes status
diabetes_positive = df[df['diabetes'] == 1]  # rows with diabetes
diabetes_negative = df[df['diabetes'] == 0]  # rows without diabetes

print(f"\nDiabetes Positive Rows: {len(diabetes_positive)}")
print(f"Diabetes Negative Rows: {len(diabetes_negative)}")

# Step 3: Randomly sample 8,000 rows from each class
sample_size = 8000
random_state = 42  # for reproducibility

# Sample from positive class
positive_sample = diabetes_positive.sample(
    n=min(sample_size, len(diabetes_positive)), 
    random_state=random_state, 
    replace=(sample_size > len(diabetes_positive))
)

# Sample from negative class
negative_sample = diabetes_negative.sample(
    n=min(sample_size, len(diabetes_negative)), 
    random_state=random_state, 
    replace=(sample_size > len(diabetes_negative))
)

# Step 4: Combine and shuffle the balanced dataset
balanced_df = pd.concat(
    [positive_sample, negative_sample], 
    ignore_index=True
)

# Shuffle the combined dataset
balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

print(f"\n{'='*70}")
print("BALANCED DATASET SUMMARY")
print(f"{'='*70}")
print(f"Balanced Dataset Shape: {balanced_df.shape}")
print(f"\nBalanced Class Distribution:")
print(balanced_df['diabetes'].value_counts())
print(f"\nBalanced Class Distribution (%):")
print(balanced_df['diabetes'].value_counts(normalize=True) * 100)

# ========================================================================
# STEP 5: SAVE BALANCED DATASET TO CSV FILE
# ========================================================================

# Create a dedicated folder for outputs
output_dir = 'diabetes_dataset_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"\n✓ Created output directory: '{output_dir}'")

# Save balanced dataset with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
balanced_file = os.path.join(output_dir, f'balanced_diabetes_dataset_{timestamp}.csv')

balanced_df.to_csv(
    balanced_file,
    index=False,           # Do NOT save row indices
    encoding='utf-8'       # Handle special characters
)
print(f"\n✓ Balanced dataset saved to: '{balanced_file}'")
print(f"  File size: {os.path.getsize(balanced_file) / 1024:.2f} KB")

# Also save with a fixed name (simpler to reference)
balanced_file_simple = os.path.join(output_dir, 'balanced_diabetes.csv')
balanced_df.to_csv(balanced_file_simple, index=False, encoding='utf-8')
print(f"✓ Balanced dataset saved to: '{balanced_file_simple}'")

# ========================================================================
# STEP 6: CREATE A SUMMARY REPORT
# ========================================================================

summary_report = f"""
DIABETES DATASET BALANCING SUMMARY REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

ORIGINAL DATASET:
- Total rows: {len(df)}
- Diabetic (1): {len(diabetes_positive)} ({len(diabetes_positive)/len(df)*100:.2f}%)
- Non-diabetic (0): {len(diabetes_negative)} ({len(diabetes_negative)/len(df)*100:.2f}%)

BALANCED DATASET (CREATED):
- Total rows: {len(balanced_df)}
- Diabetic (1): {(balanced_df['diabetes']==1).sum()} (50.00%)
- Non-diabetic (0): {(balanced_df['diabetes']==0).sum()} (50.00%)

FILES SAVED:
1. {balanced_file_simple}
2. {balanced_file}

PARAMETERS USED:
- Sample size per class: {sample_size}
- Random state (seed): {random_state}
- Encoding: UTF-8
"""

report_file = os.path.join(output_dir, 'dataset_summary_report.txt')
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(summary_report)

print(f"\n{'='*70}")
print("SUMMARY REPORT")
print(f"{'='*70}")
print(summary_report)
print(f"\n✓ Full report saved to: '{report_file}'")

# ========================================================================
# STEP 7: VERIFY FILES
# ========================================================================

print(f"\n{'='*70}")
print("FILES IN OUTPUT DIRECTORY")
print(f"{'='*70}")
for filename in os.listdir(output_dir):
    filepath = os.path.join(output_dir, filename)
    filesize = os.path.getsize(filepath)
    print(f"✓ {filename:<45} ({filesize:>10,} bytes)")

print(f"\n✓ All files saved successfully in directory: '{output_dir}/'")
