import sys
import os
import math

# --- 1. 环境自检 ---
try:
    import torch
except ImportError:
    print("❌ 错误: 未安装 PyTorch。")
    sys.exit(1)

def check_environment():
    print(f"\n{'='*30} 环境硬件自检 (V10 Ultimate) {'='*30}")
    
    if not torch.cuda.is_available():
        print("❌ 致命错误: 未检测到 GPU！")
        sys.exit(1)
    else:
        try:
            device_name = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU 状态: 正常 (CUDA Active)")
            print(f"   - 显卡型号: {device_name}")
            print(f"   - 显存大小: {total_mem:.2f} GB")
            
            if "4090" in device_name:
                print("🚀 检测到 RTX 4090！BF16 稳健模式已激活。")
                
        except Exception as e:
            print(f"⚠️ GPU 信息获取失败: {e}")
            
    print(f"{'='*80}\n")

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

# ================= 配置区 =================
INPUT_FILE = "clean.csv"
MODEL_CHECKPOINT = "google/byt5-small"
OUTPUT_DIR = "./byt5_akkadian_finetuned"
FINAL_MODEL_DIR = "./final_akkadian_model"

MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 256
BATCH_SIZE = 8          
LEARNING_RATE = 1e-4    
NUM_EPOCHS = 50         
LOGGING_STEPS = 10      

# ================= 🛡️ 安全熔断器 (保留) =================
class SafetyCallback(TrainerCallback):
    """
    实时监控训练状态，一旦发现 Loss 异常 (NaN/Inf/0.0)，立即终止训练。
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None: return
        
        if "loss" in logs:
            loss_val = logs["loss"]
            
            # 1. 检查 NaN
            if math.isnan(loss_val):
                print(f"\n\n❌ [熔断触发] 检测到 Loss = NaN (梯度爆炸)！")
                sys.exit(1)
            
            # 2. 检查 Inf
            if math.isinf(loss_val):
                print(f"\n\n❌ [熔断触发] 检测到 Loss = Inf (数值溢出)！")
                sys.exit(1)
                
            # 3. 检查 0.0
            if loss_val == 0.0 and state.global_step < 100:
                print(f"\n\n❌ [熔断触发] 检测到 Loss = 0.0 (异常归零)！")
                sys.exit(1)

def train():
    print(f"{'='*30} 启动训练引擎 (V10) {'='*30}")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误: 找不到 {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    raw_dataset = Dataset.from_pandas(df)
    
    # 9:1 划分
    split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
    print(f"📊 数据概况:\n  - 训练集: {len(split_dataset['train'])} 条\n  - 验证集: {len(split_dataset['test'])} 条")

    print(f"🚀 加载基座模型: {MODEL_CHECKPOINT}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

    def preprocess_function(examples):
        inputs = examples["input_text"]
        targets = examples["target_text"]
        model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)
        labels = tokenizer(text_target=targets, max_length=MAX_TARGET_LENGTH, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("⚙️ 正在对齐与编码数据...")
    tokenized_datasets = split_dataset.map(preprocess_function, batched=True)

    metric = evaluate.load("sacrebleu")

    # [功能升级] 修复 chr() 报错的 compute_metrics
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
            
        try:
            # 尝试正常解码
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds = [pred.strip() for pred in decoded_preds]
            decoded_labels = [[label.strip()] for label in decoded_labels]

            result = metric.compute(predictions=decoded_preds, references=decoded_labels)
            return {"bleu": result["score"]}
        
        except ValueError as e:
            # 捕获 chr() not in range 等解码错误
            if "chr()" in str(e) or "range" in str(e):
                print(f"\n⚠️ [Warning] 评估跳过: 检测到非法字符生成 (ByT5 早期常见震荡)，不影响训练。")
                return {"bleu": 0.0}
            else:
                # 其他错误则打印详情
                print(f"\n⚠️ [Warning] 评估未知错误: {e}")
                return {"bleu": 0.0}

    print("📈 配置训练参数 (BF16 + Auto-Kill enabled)...")
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",            
        save_strategy="epoch",            
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=NUM_EPOCHS,
        predict_with_generate=True,       
        
        # 4090 核心配置: BF16
        bf16=True,                        
        fp16=False,                       
        
        generation_max_length=256,        
        logging_steps=LOGGING_STEPS,      
        load_best_model_at_end=True,      
        metric_for_best_model="bleu",     
        greater_is_better=True,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[SafetyCallback]  # <--- 保留安全熔断器
    )

    # [新增功能] 自动检测断点，支持 Resume
    last_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        # 寻找 checkpoint-XXX 文件夹
        checkpoints = [os.path.join(OUTPUT_DIR, d) for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint")]
        if checkpoints:
            # 按修改时间排序，找最新的
            checkpoints.sort(key=os.path.getmtime)
            last_checkpoint = checkpoints[-1]
            print(f"♻️ 检测到训练存档，将从断点恢复: {last_checkpoint}")

    print("🔥 点火起飞！开始训练...")
    try:
        # 如果有 checkpoint 就续训，没有就重头开始
        trainer.train(resume_from_checkpoint=last_checkpoint)
    except Exception as e:
        print(f"\n❌ 训练过程中发生未知错误: {e}")
        return

    print(f"💾 保存最佳模型至: {FINAL_MODEL_DIR}")
    model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    print("✅ 训练全流程结束！")

if __name__ == "__main__":
    train()