"""
ByT5-Base Preprocessing Script
Target: BLEU 25+ on Deep Past Challenge

Strategy:
- Minimal preprocessing (following high-score notebook approach)
- Keep it simple - ByT5 handles special characters well
- Filter only extreme cases
"""

import pandas as pd
import os
from pathlib import Path

print("="*80)
print("📊 ByT5-Base Data Preprocessing")
print("="*80)

# ================= Configuration =================
INPUT_FILE = "../../data/deep-past-initiative-machine-translation/train.csv"
OUTPUT_FILE = "clean.csv"
MIN_LENGTH = 1
MAX_LENGTH = 500  # Filter extreme outliers only

# ================= Load Data =================
print(f"\n📥 Loading data from: {INPUT_FILE}")

if not os.path.exists(INPUT_FILE):
    print(f"❌ Error: File not found at {INPUT_FILE}")
    print("Please ensure the dataset is in the correct location.")
    exit(1)

df = pd.read_csv(INPUT_FILE)
print(f"✅ Loaded {len(df):,} samples")
print(f"Columns: {df.columns.tolist()}")

# ================= Basic Cleaning =================
print("\n🧹 Applying minimal preprocessing...")

# Ensure columns exist
required_cols = ['transliteration', 'translation']
for col in required_cols:
    if col not in df.columns:
        print(f"❌ Missing required column: {col}")
        exit(1)

# Convert to string and strip whitespace
df['transliteration'] = df['transliteration'].astype(str).str.strip()
df['translation'] = df['translation'].astype(str).str.strip()

# Remove empty entries
df = df[(df['transliteration'].str.len() > 0) & (df['translation'].str.len() > 0)]
print(f"  After removing empty: {len(df):,} samples")

# Filter by length (remove extreme outliers only)
df = df[
    (df['transliteration'].str.len() >= MIN_LENGTH) & 
    (df['transliteration'].str.len() <= MAX_LENGTH) &
    (df['translation'].str.len() >= MIN_LENGTH) & 
    (df['translation'].str.len() <= MAX_LENGTH)
]
print(f"  After length filtering: {len(df):,} samples")

# ================= Create Training Format =================
print("\n📝 Creating training format...")

# Simple format: just add task prefix
PREFIX = "translate Akkadian to English: "

df['input_text'] = PREFIX + df['transliteration']
df['target_text'] = df['translation']

# Keep only necessary columns
clean_df = df[['input_text', 'target_text']].copy()

# ================= Statistics =================
print("\n📈 Dataset Statistics:")
print(f"  Total samples: {len(clean_df):,}")
print(f"  Avg input length: {clean_df['input_text'].str.len().mean():.1f} chars")
print(f"  Avg target length: {clean_df['target_text'].str.len().mean():.1f} chars")
print(f"  Max input length: {clean_df['input_text'].str.len().max()} chars")
print(f"  Max target length: {clean_df['target_text'].str.len().max()} chars")

# ================= Save =================
clean_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Saved preprocessed data to: {OUTPUT_FILE}")

# Show sample
print("\n📋 Sample entries:")
for i in range(min(3, len(clean_df))):
    print(f"\n[{i+1}] Input:  {clean_df.iloc[i]['input_text'][:100]}...")
    print(f"    Target: {clean_df.iloc[i]['target_text'][:100]}...")

print("\n" + "="*80)
print("✅ Preprocessing Complete!")
print("="*80)
