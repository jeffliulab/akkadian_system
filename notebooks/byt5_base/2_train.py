"""
ByT5-Base Training Script
Target: BLEU 25+ on Deep Past Challenge

Key Optimizations:
- byt5-base (582M params) instead of small
- Optimized hyperparameters for base model
- BF16 mixed precision training
- Safety monitoring with auto-kill on anomaly
"""

import sys
import os
import math

# ================= Environment Check =================
try:
    import torch
except ImportError:
    print("❌ PyTorch not installed")
    sys.exit(1)

def check_environment():
    print(f"\n{'='*80}")
    print("🔧 Environment Check")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("❌ No GPU detected! Training will be very slow.")
        print("Consider using Kaggle/Colab with GPU enabled.")
        response = input("Continue with CPU? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    else:
        device_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU: {device_name}")
        print(f"   VRAM: {total_mem:.2f} GB")
        
        if total_mem < 15:
            print(f"⚠️  Warning: Low VRAM ({total_mem:.1f} GB)")
            print("   Recommended: 16GB+ for byt5-base with batch_size=4")
            print("   Consider reducing batch_size to 2 if OOM occurs")
    
    print("="*80 + "\n")

check_environment()

import pandas as pd
import numpy as np
from datasets import Dataset
import evaluate
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    TrainerCallback
)

# ================= Configuration =================
INPUT_FILE = "clean.csv"
MODEL_CHECKPOINT = "google/byt5-base"  # Base model for higher capacity
OUTPUT_DIR = "./byt5_base_finetuned"
FINAL_MODEL_DIR = "./final_model"

# Optimized for byt5-base
MAX_INPUT_LENGTH = 512   # Increased from 1024
MAX_TARGET_LENGTH = 512  # Increased from 256
BATCH_SIZE = 4           # Reduced from 8 (base is larger)
LEARNING_RATE = 5e-5     # Reduced from 1e-4 (larger models need smaller LR)
NUM_EPOCHS = 30          # Can train fewer epochs with base
LOGGING_STEPS = 50

print("="*80)
print("🚀 ByT5-Base Training Pipeline")
print("="*80)
print("\n📋 Configuration:")
print(f"  Model: {MODEL_CHECKPOINT}")
print(f"  Max Input Length: {MAX_INPUT_LENGTH}")
print(f"  Max Target Length: {MAX_TARGET_LENGTH}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  Epochs: {NUM_EPOCHS}")
print("="*80)

# ================= Safety Callback =================
class SafetyCallback(TrainerCallback):
    """Monitor training and auto-kill on anomalies"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        if "loss" in logs:
            loss_val = logs["loss"]
            
            # Check NaN
            if math.isnan(loss_val):
                print(f"\n❌ SAFETY KILL: Loss = NaN detected!")
                sys.exit(1)
            
            # Check Inf
            if math.isinf(loss_val):
                print(f"\n❌ SAFETY KILL: Loss = Inf detected!")
                sys.exit(1)
            
            # Check suspicious zeros
            if loss_val == 0.0 and state.global_step < 100:
                print(f"\n❌ SAFETY KILL: Loss = 0.0 (suspicious)")
                sys.exit(1)

# ================= Main Training Function =================
def train():
    print(f"\n{'='*80}")
    print("📂 Loading Data")
    print("="*80)
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ {INPUT_FILE} not found!")
        print("Please run 1_preprocess.py first")
        return
    
    df = pd.read_csv(INPUT_FILE)
    raw_dataset = Dataset.from_pandas(df)
    
    # 9:1 split
    split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
    train_size = len(split_dataset['train'])
    val_size = len(split_dataset['test'])
    
    print(f"✅ Loaded dataset:")
    print(f"   Training: {train_size:,} samples")
    print(f"   Validation: {val_size:,} samples")
    
    # ================= Load Model =================
    print(f"\n{'='*80}")
    print("🤖 Loading Model & Tokenizer")
    print("="*80)
    print(f"📥 Loading {MODEL_CHECKPOINT}...")
    print("   (This may take a few minutes for first-time download)")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded: {param_count:,} parameters")
    
    # ================= Preprocessing =================
    def preprocess_function(examples):
        inputs = examples["input_text"]
        targets = examples["target_text"]
        
        model_inputs = tokenizer(
            inputs, 
            max_length=MAX_INPUT_LENGTH, 
            truncation=True
        )
        
        labels = tokenizer(
            text_target=targets, 
            max_length=MAX_TARGET_LENGTH, 
            truncation=True
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    print("\n⚙️  Tokenizing dataset...")
    tokenized_datasets = split_dataset.map(preprocess_function, batched=True)
    print("✅ Tokenization complete")
    
    # ================= Metrics =================
    metric = evaluate.load("sacrebleu")
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        try:
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            
            # Replace -100 with pad token
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # Clean
            decoded_preds = [pred.strip() for pred in decoded_preds]
            decoded_labels = [[label.strip()] for label in decoded_labels]
            
            result = metric.compute(predictions=decoded_preds, references=decoded_labels)
            return {"bleu": result["score"]}
        
        except Exception as e:
            # Handle ByT5 early training instabilities
            if "chr()" in str(e) or "range" in str(e):
                print(f"⚠️  Evaluation skipped (early training instability)")
                return {"bleu": 0.0}
            else:
                print(f"⚠️  Evaluation error: {e}")
                return {"bleu": 0.0}
    
    # ================= Training Arguments =================
    print(f"\n{'='*80}")
    print("📝 Configuring Training")
    print("="*80)
    
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=NUM_EPOCHS,
        predict_with_generate=True,
        
        # Mixed precision (BF16 for modern GPUs)
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
        fp16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
        
        generation_max_length=MAX_TARGET_LENGTH,
        logging_steps=LOGGING_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        report_to="none",
        
        # Performance optimizations
        dataloader_num_workers=2,
        gradient_accumulation_steps=2,  # Simulate larger batch
    )
    
    print(f"✅ Training configuration:")
    print(f"   Precision: {'BF16' if args.bf16 else 'FP16' if args.fp16 else 'FP32'}")
    print(f"   Effective batch size: {BATCH_SIZE * 2} (with gradient accumulation)")
    print(f"   Total training steps: ~{(train_size // (BATCH_SIZE * 2)) * NUM_EPOCHS}")
    
    # ================= Trainer =================
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[SafetyCallback()]
    )
    
    # ================= Resume Support =================
    last_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        checkpoints = [
            os.path.join(OUTPUT_DIR, d) 
            for d in os.listdir(OUTPUT_DIR) 
            if d.startswith("checkpoint")
        ]
        if checkpoints:
            checkpoints.sort(key=os.path.getmtime)
            last_checkpoint = checkpoints[-1]
            print(f"\n♻️  Resuming from checkpoint: {last_checkpoint}")
    
    # ================= Start Training =================
    print(f"\n{'='*80}")
    print("🔥 Starting Training")
    print("="*80)
    print("⏰ This will take several hours...")
    print("💡 Tip: Monitor GPU usage with 'nvidia-smi' in another terminal")
    print("="*80 + "\n")
    
    try:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ================= Save Final Model =================
    print(f"\n{'='*80}")
    print("💾 Saving Final Model")
    print("="*80)
    
    model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    
    print(f"✅ Model saved to: {FINAL_MODEL_DIR}")
    print(f"   Total size: ~2.2 GB")
    
    # ================= Final Evaluation =================
    print(f"\n{'='*80}")
    print("📊 Final Evaluation")
    print("="*80)
    
    eval_results = trainer.evaluate()
    final_bleu = eval_results.get('eval_bleu', 0.0)
    
    print(f"\n🎯 Final BLEU Score: {final_bleu:.2f}")
    
    if final_bleu >= 25:
        print("🎉 TARGET ACHIEVED! (BLEU ≥ 25)")
    elif final_bleu >= 20:
        print("✅ Good score! Close to target.")
    else:
        print("⚠️  Below target. Consider:")
        print("   - Training longer (increase NUM_EPOCHS)")
        print("   - Adjusting learning rate")
        print("   - Using more training data")
    
    print("\n" + "="*80)
    print("✅ Training Complete!")
    print("="*80)

if __name__ == "__main__":
    train()
