import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import re
import sys
import math
import evaluate # 使用 evaluate 库调用 sacrebleu

# ================= 配置区 =================
# 测试集文件路径
TEST_FILE = "../../data/deep-past-initiative-machine-translation/test.csv"
# 原始训练清洗文件 (用于计算本地验证分)
TRAIN_FILE = "clean.csv"

# 模型路径
MODEL_PATH = "./final_akkadian_model"

# 输出文件
SUBMISSION_FILE = "submission.csv"

# 推理参数
BATCH_SIZE = 32         # 4090 显存大，推理时可以开大一点
MAX_INPUT_LENGTH = 1024 
MAX_TARGET_LENGTH = 256
BEAM_SIZE = 4           # Beam Search 宽度

# ================= 清洗逻辑 (SourceNormalizer) =================
class SourceNormalizer:
    """阿卡德语输入端清洗器 (必须与训练时完全一致)"""
    def __init__(self):
        self.sub_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

    def normalize(self, text: str) -> str:
        if not isinstance(text, str): return ""
        text = text.replace('ḫ', 'h').replace('Ḫ', 'H')
        text = text.replace('(', '{').replace(')', '}')
        text = text.replace('[... ...]', '@BIGGAP@').replace('...', '@BIGGAP@').replace('[x]', '@GAP@')
        text = re.sub(r'\{(.*?)\}', r'@DET_\1@', text)
        text = text.translate(self.sub_map)
        text = re.sub(r'[!?:;\[\]\(\)˹˺/\\<>\.]', '', text)
        text = text.replace('@BIGGAP@', '<big_gap>').replace('@GAP@', '<gap>')
        text = re.sub(r'@DET_(.*?)@', r'{\1}', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

# ================= 数据集类 =================
class InferenceDataset(Dataset):
    def __init__(self, ids, texts, normalizer):
        self.ids = ids
        self.texts = texts
        self.normalizer = normalizer

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        raw_text = self.texts[idx]
        id_val = self.ids[idx]
        
        # 清洗 + 添加前缀
        clean_text = self.normalizer.normalize(raw_text)
        input_text = f"translate Akkadian to English: {clean_text}"
        
        return {"id": id_val, "input_text": input_text}

# ================= 评分工具 =================
def calculate_score(predictions, references):
    """
    计算比赛指标: Geometric Mean of BLEU and chrF++
    Score = sqrt(BLEU * chrF++)
    """
    try:
        # 加载指标
        metric_bleu = evaluate.load("sacrebleu")
        metric_chrf = evaluate.load("chrf")
        
        # 1. 计算 BLEU
        # sacrebleu 期望 references 是 list of list
        # refs_for_bleu = [['ref1'], ['ref2'], ...]
        refs_for_bleu = [[r] for r in references]
        bleu_res = metric_bleu.compute(predictions=predictions, references=refs_for_bleu)
        bleu_score = bleu_res['score']
        
        # 2. 计算 chrF++
        # sacrebleu 的 chrf 实现中，word_order=2 即为 chrF++
        chrf_res = metric_chrf.compute(predictions=predictions, references=refs_for_bleu, word_order=2)
        chrf_score = chrf_res['score']
        
        # 3. 计算几何平均
        # 避免 0 分导致数学错误
        if bleu_score < 0: bleu_score = 0
        if chrf_score < 0: chrf_score = 0
        
        final_score = math.sqrt(bleu_score * chrf_score)
        
        return {
            "geom_mean": final_score,
            "bleu": bleu_score,
            "chrf++": chrf_score
        }
    except Exception as e:
        print(f"⚠️ 评分计算出错: {e}")
        return {"geom_mean": 0.0, "bleu": 0.0, "chrf++": 0.0}

# ================= 推理核心函数 =================
def run_inference(model, tokenizer, dataloader, device):
    results = []
    print(f"🔥 开始推理 (Batch Size: {BATCH_SIZE})...")
    
    for batch in tqdm(dataloader, desc="Generating"):
        ids = batch["id"]
        input_texts = batch["input_text"]
        
        # Tokenize
        inputs = tokenizer(
            input_texts, 
            max_length=MAX_INPUT_LENGTH, 
            truncation=True, 
            padding=True, 
            return_tensors="pt"
        ).to(device)
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                inputs["input_ids"],
                max_length=MAX_TARGET_LENGTH,
                num_beams=BEAM_SIZE,
                early_stopping=True
            )
        
        # Decode
        decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Collect
        for id_val, pred in zip(ids, decoded_preds):
            # [修正] 提取 scalar value，修复 tensor(0) 问题
            clean_id = id_val.item() if isinstance(id_val, torch.Tensor) else id_val
            results.append({"id": clean_id, "translation": pred.strip()})
            
    return results

# ================= 主程序 =================
def main():
    print(f"{'='*30} Deep Past 推理与评分系统 {'='*30}")

    # 1. 资源检查
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到模型: {MODEL_PATH}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 设备: {device}")
    if "cuda" in str(device):
        print(f"   - 显卡: {torch.cuda.get_device_name(0)}")

    # 2. 加载模型
    print("📥 加载模型与分词器...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    normalizer = SourceNormalizer()

    # ================= 阶段 1: 生成 Submission (Test Set) =================
    if os.path.exists(TEST_FILE):
        print(f"\n{'='*10} 阶段 1: 生成比赛提交文件 (Test Set) {'='*10}")
        df_test = pd.read_csv(TEST_FILE)
        
        # 列名适配
        text_col = 'transliteration'
        if text_col not in df_test.columns:
            possible = [c for c in df_test.columns if 'text' in c.lower() or 'translit' in c.lower()]
            if possible: text_col = possible[0]
        
        print(f"📄 测试集: {len(df_test)} 条样本")
        
        test_ds = InferenceDataset(df_test['id'].tolist(), df_test[text_col].tolist(), normalizer)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        # 执行推理
        test_results = run_inference(model, tokenizer, test_loader, device)
        
        # 保存
        sub_df = pd.DataFrame(test_results)
        sub_df.to_csv(SUBMISSION_FILE, index=False)
        print(f"✅ Submission 已生成: {SUBMISSION_FILE}")
        print(sub_df.head(3))
    else:
        print(f"⚠️ 未找到测试集 {TEST_FILE}，跳过生成步骤。")

    # ================= 阶段 2: 本地验证评分 (Validation Set) =================
    if os.path.exists(TRAIN_FILE):
        print(f"\n{'='*10} 阶段 2: 计算本地验证分数 (Local CV) {'='*10}")
        print("ℹ️ 说明: 使用训练时划分出的验证集(10%)进行评估，作为Leaderboard分数的参考。")
        
        # 加载清洗后的训练数据
        df_full = pd.read_csv(TRAIN_FILE)
        
        # 复现训练时的切分 (必须用相同的 seed=42)
        from sklearn.model_selection import train_test_split
        _, df_val = train_test_split(df_full, test_size=0.1, random_state=42)
        
        print(f"🧪 验证集: {len(df_val)} 条样本")
        
        # 准备数据
        # 提取 clean.csv 中的原始内容（需要去掉之前可能加的前缀）
        val_texts = df_val['input_text'].apply(lambda x: x.replace("translate Akkadian to English: ", "")).tolist()
        val_refs = df_val['target_text'].tolist()
        
        # 这里不需要 id，用 range 代替
        val_ds = InferenceDataset(range(len(val_texts)), val_texts, normalizer)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        # 执行推理
        val_preds_raw = run_inference(model, tokenizer, val_loader, device)
        val_preds_text = [item['translation'] for item in val_preds_raw]
        
        # 计算分数
        print("🧮 正在计算分数 (BLEU + chrF++)...")
        scores = calculate_score(val_preds_text, val_refs)
        
        print(f"\n{'*'*40}")
        print(f"🏆 本地验证集预估分数 (Local CV Score)")
        print(f"{'*'*40}")
        print(f"   BLEU Score  : {scores['bleu']:.2f}")
        print(f"   chrF++ Score: {scores['chrf++']:.2f}")
        print(f"   ------------------------------")
        print(f"   Geometric Mean: {scores['geom_mean']:.4f}")
        print(f"{'*'*40}\n")
        
    else:
        print(f"⚠️ 未找到训练文件 {TRAIN_FILE}，无法计算本地分数。")

if __name__ == "__main__":
    main()