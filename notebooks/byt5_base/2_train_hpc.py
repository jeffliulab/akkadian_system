"""
ByT5-Base Training Script - HPC Optimized
Target: BLEU 25+ on Deep Past Challenge

HPC Optimizations:
- Multi-GPU distributed training support
- Local model path (pre-downloaded)
- Frequent checkpoint saving (step-based)
- Optimized data loading for HPC clusters
- Auto-resume from latest checkpoint
- Enhanced monitoring and logging
"""

import sys
import os
import math

# ================= Environment Check =================
try:
    import torch
    import torch.distributed as dist
except ImportError:
    print("❌ PyTorch not installed")
    sys.exit(1)

def check_environment():
    print(f"\n{'='*80}")
    print("🖥️  HPC Environment Check")
    print("="*80)
    
    # Check distributed setup
    is_distributed = 'WORLD_SIZE' in os.environ
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if is_distributed:
        print(f"🌐 Distributed Training Detected:")
        print(f"   - World Size: {world_size} GPUs")
        print(f"   - Local Rank: {local_rank}")
    
    if not torch.cuda.is_available():
        print("❌ No GPU detected!")
        sys.exit(1)
    
    # GPU info (only print from rank 0 to avoid spam)
    if local_rank == 0 or not is_distributed:
        num_gpus = torch.cuda.device_count()
        print(f"\n✅ GPU Status:")
        print(f"   - Available GPUs: {num_gpus}")
        
        for i in range(num_gpus):
            device_name = torch.cuda.get_device_name(i)
            total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   - GPU {i}: {device_name} ({total_mem:.1f} GB)")
        
        # Check GPU architecture for mixed precision
        compute_cap = torch.cuda.get_device_capability(0)
        if compute_cap[0] >= 8:
            print(f"   - Compute Capability: {compute_cap[0]}.{compute_cap[1]} (BF16 supported)")
        else:
            print(f"   - Compute Capability: {compute_cap[0]}.{compute_cap[1]} (FP16 recommended)")
    
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

# ================= HPC Configuration =================
INPUT_FILE = "clean.csv"

# Model path - IMPORTANT: Pre-download model before submitting job
# On login node (with internet): python download_model.py
# Then use local path here:
MODEL_CHECKPOINT = os.environ.get('MODEL_PATH', "./pretrained_models/byt5-base")
# Fallback to HuggingFace if local not found
if not os.path.exists(MODEL_CHECKPOINT):
    print(f"⚠️  Local model not found at {MODEL_CHECKPOINT}")
    print("   Falling back to HuggingFace (requires internet)")
    MODEL_CHECKPOINT = "google/byt5-base"

OUTPUT_DIR = "./byt5_base_finetuned_hpc"
FINAL_MODEL_DIR = "./final_model_hpc"
LOG_DIR = "./logs"

# HPC-optimized hyperparameters
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 512
BATCH_SIZE = 2              # Per GPU (lower for multi-GPU)
LEARNING_RATE = 5e-5
NUM_EPOCHS = 30
LOGGING_STEPS = 50
SAVE_STEPS = 500            # Save every 500 steps (frequent for HPC jobs)
EVAL_STEPS = 500            # Evaluate every 500 steps

# Data loading optimization
NUM_WORKERS = int(os.environ.get('SLURM_CPUS_PER_TASK', 8)) // 2  # Half of allocated CPUs

local_rank = int(os.environ.get('LOCAL_RANK', 0))
if local_rank == 0:
    print("="*80)
    print("🚀 ByT5-Base HPC Training Pipeline")
    print("="*80)
    print("\n📋 Configuration:")
    print(f"  Model: {MODEL_CHECKPOINT}")
    print(f"  Max Input Length: {MAX_INPUT_LENGTH}")
    print(f"  Max Target Length: {MAX_TARGET_LENGTH}")
    print(f"  Batch Size (per GPU): {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Save Steps: {SAVE_STEPS}")
    print(f"  DataLoader Workers: {NUM_WORKERS}")
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
                print(f"\n❌ SAFETY KILL: Loss = NaN detected at step {state.global_step}!")
                sys.exit(1)
            
            # Check Inf
            if math.isinf(loss_val):
                print(f"\n❌ SAFETY KILL: Loss = Inf detected at step {state.global_step}!")
                sys.exit(1)
            
            # Check suspicious zeros
            if loss_val == 0.0 and state.global_step < 100:
                print(f"\n❌ SAFETY KILL: Loss = 0.0 at step {state.global_step} (suspicious)")
                sys.exit(1)

# ================= Main Training Function =================
def train():
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    if local_rank == 0:
        print(f"\n{'='*80}")
        print("📂 Loading Data")
        print("="*80)
    
    if not os.path.exists(INPUT_FILE):
        if local_rank == 0:
            print(f"❌ {INPUT_FILE} not found!")
            print("Please run 1_preprocess.py first")
        sys.exit(1)
    
    df = pd.read_csv(INPUT_FILE)
    raw_dataset = Dataset.from_pandas(df)
    
    # 9:1 split
    split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
    train_size = len(split_dataset['train'])
    val_size = len(split_dataset['test'])
    
    if local_rank == 0:
        print(f"✅ Loaded dataset:")
        print(f"   Training: {train_size:,} samples")
        print(f"   Validation: {val_size:,} samples")
    
    # ================= Load Model =================
    if local_rank == 0:
        print(f"\n{'='*80}")
        print("🤖 Loading Model & Tokenizer")
        print("="*80)
        print(f"📥 Loading {MODEL_CHECKPOINT}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
    
    if local_rank == 0:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ Model loaded: {param_count:,} parameters (~{param_count/1e6:.0f}M)")
    
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
    
    if local_rank == 0:
        print("\n⚙️  Tokenizing dataset...")
    
    tokenized_datasets = split_dataset.map(
        preprocess_function, 
        batched=True,
        num_proc=4  # Parallel processing
    )
    
    if local_rank == 0:
        print("✅ Tokenization complete")
    
    # ================= Metrics =================
    metric = evaluate.load("sacrebleu")
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        try:
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            decoded_preds = [pred.strip() for pred in decoded_preds]
            decoded_labels = [[label.strip()] for label in decoded_labels]
            
            result = metric.compute(predictions=decoded_preds, references=decoded_labels)
            return {"bleu": result["score"]}
        
        except Exception as e:
            if "chr()" in str(e) or "range" in str(e):
                if local_rank == 0:
                    print(f"⚠️  Evaluation skipped (early training instability)")
                return {"bleu": 0.0}
            else:
                if local_rank == 0:
                    print(f"⚠️  Evaluation error: {e}")
                return {"bleu": 0.0}
    
    # ================= HPC Training Arguments =================
    if local_rank == 0:
        print(f"\n{'='*80}")
        print("📝 Configuring HPC Training")
        print("="*80)
    
    # Detect compute capability for mixed precision
    compute_cap = torch.cuda.get_device_capability(0)
    use_bf16 = compute_cap[0] >= 8  # Ampere (A100, H100) or newer
    use_fp16 = compute_cap[0] < 8   # Volta (V100) or older
    
    # Create log directory
    os.makedirs(LOG_DIR, exist_ok=True)
    
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        
        # Step-based strategies (better for HPC interruptions)
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=5,  # Keep more checkpoints
        
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        num_train_epochs=NUM_EPOCHS,
        predict_with_generate=True,
        
        # Mixed precision (auto-detect based on GPU)
        bf16=use_bf16,
        fp16=use_fp16,
        
        generation_max_length=MAX_TARGET_LENGTH,
        logging_dir=LOG_DIR,
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        report_to="tensorboard",  # Enable TensorBoard logging
        
        # HPC performance optimizations
        dataloader_num_workers=NUM_WORKERS,
        dataloader_pin_memory=True,
        gradient_accumulation_steps=1,  # Adjust based on GPU count
        
        # Distributed training optimizations
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=25,
        
        # Auto-resume
        resume_from_checkpoint=True,
    )
    
    if local_rank == 0:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        effective_batch = BATCH_SIZE * world_size * args.gradient_accumulation_steps
        
        print(f"✅ HPC Training Configuration:")
        print(f"   Precision: {'BF16' if use_bf16 else 'FP16' if use_fp16 else 'FP32'}")
        print(f"   Number of GPUs: {world_size}")
        print(f"   Batch per GPU: {BATCH_SIZE}")
        print(f"   Effective batch size: {effective_batch}")
        print(f"   DataLoader workers: {NUM_WORKERS}")
        print(f"   Save frequency: Every {SAVE_STEPS} steps")
        print(f"   Log directory: {LOG_DIR}")
    
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
    
    # ================= Auto-Resume =================
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
            if local_rank == 0:
                print(f"\n♻️  Auto-resuming from: {last_checkpoint}")
    
    # ================= Start Training =================
    if local_rank == 0:
        print(f"\n{'='*80}")
        print("🔥 Starting HPC Training")
        print("="*80)
        print("⏰ Training will auto-checkpoint every 500 steps")
        print("💡 Monitor with: tensorboard --logdir ./logs")
        print("="*80 + "\n")
    
    try:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    except Exception as e:
        if local_rank == 0:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    # ================= Save Final Model =================
    if local_rank == 0:
        print(f"\n{'='*80}")
        print("💾 Saving Final Model")
        print("="*80)
        
        model.save_pretrained(FINAL_MODEL_DIR)
        tokenizer.save_pretrained(FINAL_MODEL_DIR)
        
        print(f"✅ Model saved to: {FINAL_MODEL_DIR}")
        print(f"   Size: ~2.2 GB")
    
    # ================= Final Evaluation =================
    if local_rank == 0:
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
            print("   - Training longer")
            print("   - Adjusting learning rate")
            print("   - Using more training data")
        
        print("\n" + "="*80)
        print("✅ HPC Training Complete!")
        print("="*80)

if __name__ == "__main__":
    train()
