"""
ByT5-Base Inference Script
Compatible with Kaggle submission format

Usage:
1. Local testing: python 3_inference.py
2. Kaggle: Copy-paste this code into notebook
"""

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🔮 ByT5-Base Inference Pipeline")
print("="*80)

# ================= Configuration =================
# For local testing
LOCAL_MODEL_PATH = "./final_model"
LOCAL_TEST_DATA = "../../data/deep-past-initiative-machine-translation/test.csv"

# For Kaggle (auto-detect)
KAGGLE_PATHS = [
    "/kaggle/input/akkadian-byt5-base-model/transformers/default/1",
    "/kaggle/input/akkadian-byt5-base-model/pytorch/default/1/final_model",
    "/kaggle/input/akkadian-byt5-base-model",
]
KAGGLE_TEST_DATA = "/kaggle/input/deep-past-initiative-machine-translation/test.csv"

BATCH_SIZE = 8
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n⚙️  Configuration:")
print(f"   Device: {DEVICE.upper()}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Max Length: {MAX_LENGTH}")

# ================= Auto-detect Environment =================
def find_model_path():
    """Auto-detect model path (local or Kaggle)"""
    
    # Check local first
    if os.path.exists(LOCAL_MODEL_PATH) and "config.json" in os.listdir(LOCAL_MODEL_PATH):
        return LOCAL_MODEL_PATH, False
    
    # Check Kaggle paths
    for path in KAGGLE_PATHS:
        if os.path.exists(path) and "config.json" in os.listdir(path):
            return path, True
    
    return None, False

def find_test_data(is_kaggle):
    """Find test data file"""
    test_path = KAGGLE_TEST_DATA if is_kaggle else LOCAL_TEST_DATA
    
    if os.path.exists(test_path):
        return test_path
    
    return None

MODEL_PATH, IS_KAGGLE = find_model_path()
TEST_DATA_PATH = find_test_data(IS_KAGGLE)

if MODEL_PATH is None:
    print("\n❌ Model not found!")
    print("Please ensure model is available at one of these locations:")
    print(f"   Local: {LOCAL_MODEL_PATH}")
    for path in KAGGLE_PATHS:
        print(f"   Kaggle: {path}")
    exit(1)

if TEST_DATA_PATH is None:
    print("\n❌ Test data not found!")
    exit(1)

print(f"\n✅ Found model at: {MODEL_PATH}")
print(f"✅ Found test data at: {TEST_DATA_PATH}")
print(f"   Environment: {'Kaggle' if IS_KAGGLE else 'Local'}")

# ================= Load Model =================
print(f"\n{'='*80}")
print("🤖 Loading Model & Tokenizer")
print("="*80)

print("📥 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("📥 Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

param_count = sum(p.numel() for p in model.parameters())
print(f"✅ Model loaded: {param_count:,} parameters")

# ================= Load Test Data =================
print(f"\n{'='*80}")
print("📂 Loading Test Data")
print("="*80)

test_df = pd.read_csv(TEST_DATA_PATH)
print(f"✅ Loaded {len(test_df):,} test samples")

# ================= Dataset & DataLoader =================
PREFIX = "translate Akkadian to English: "

class InferenceDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df['transliteration'].astype(str).tolist()
        self.texts = [PREFIX + text for text in self.texts]
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0)
        }

test_dataset = InferenceDataset(test_df, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\n📊 Inference setup:")
print(f"   Dataset size: {len(test_dataset):,}")
print(f"   Number of batches: {len(test_loader)}")
print(f"   Estimated time: ~{len(test_loader) * 2:.0f} seconds")

# ================= Run Inference =================
print(f"\n{'='*80}")
print("🔮 Generating Predictions")
print("="*80)

all_predictions = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Inference"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=MAX_LENGTH,
            num_beams=4,
            early_stopping=True
        )
        
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_predictions.extend([text.strip() for text in decoded])

print(f"\n✅ Generated {len(all_predictions):,} predictions")

# ================= Create Submission =================
print(f"\n{'='*80}")
print("📝 Creating Submission File")
print("="*80)

submission = pd.DataFrame({
    "id": test_df["id"],
    "translation": all_predictions
})

# Handle empty predictions
empty_count = (submission["translation"].str.len() == 0).sum()
if empty_count > 0:
    print(f"⚠️  Warning: {empty_count} empty predictions found, filling with placeholder")
    submission["translation"] = submission["translation"].apply(
        lambda x: x if len(str(x)) > 0 else "broken text"
    )

# Statistics
trans_lengths = submission['translation'].str.len()
print(f"\n📊 Translation Statistics:")
print(f"   Total: {len(submission):,}")
print(f"   Avg length: {trans_lengths.mean():.1f} chars")
print(f"   Min length: {trans_lengths.min()} chars")
print(f"   Max length: {trans_lengths.max()} chars")

# Save
submission.to_csv("submission.csv", index=False)
print(f"\n✅ Submission saved: submission.csv")
print(f"   File size: {os.path.getsize('submission.csv') / 1024:.2f} KB")

# ================= Show Samples =================
print(f"\n{'='*80}")
print("📋 Sample Predictions")
print("="*80)

for i in range(min(5, len(submission))):
    print(f"\n[{i+1}] ID: {submission.iloc[i]['id']}")
    print(f"    Translation: {submission.iloc[i]['translation'][:100]}...")

print("\n" + "="*80)
print("🎉 Inference Complete!")
print("="*80)
print("\n💡 Next steps:")
print("   1. Review sample predictions above")
print("   2. Upload submission.csv to Kaggle")
print("   3. Check leaderboard score")
print("\n🎯 Expected score: BLEU 25+ (if model trained properly)")
print("="*80)
