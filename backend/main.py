from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import T5ForConditionalGeneration, AutoTokenizer
import backend.default as default_engine # 引用你的哈希逻辑

# --- 配置部分 ---
# 你的模型仓库 ID (直接从 Hugging Face 拉取)
MODEL_REPO_ID = "jeffliulab/akkadian-translator-byt5"

app = FastAPI()

# 配置 CORS (允许前端跨域访问)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议改为你的 GitHub Pages 域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 全局加载模型 (Space 启动时执行一次) ---
print("正在从 Hugging Face 加载 ByT5 模型... (可能需要几秒钟)")
try:
    # 自动下载并加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_REPO_ID)
    print("✅ 模型加载成功！")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    # 这里不抛出致命错误，防止影响 default 模式的使用
    model = None
    tokenizer = None

# 定义请求体格式
class TranslationRequest(BaseModel):
    text: str
    model_id: str = "default"  # 默认为哈希模式

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Akkadian Translator API is running"}

@app.post("/api/v1/translate")
async def translate_text(request: TranslationRequest):
    input_text = request.text.strip()
    
    # --- 分流逻辑 ---
    
    # 模式 1: Default (哈希算法 / 规则)
    if request.model_id == "default":
        # 直接调用你原来写好的 logic
        return default_engine.predict(input_text)

    # 模式 2: ByT5 (真实 AI 模型)
    elif request.model_id == "byt5":
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Model not loaded successfully")
        
        try:
            # 1. 预处理 (ByT5 不需要分词，直接转 input_ids)
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids
            
            # 2. 推理生成
            outputs = model.generate(
                input_ids, 
                max_length=128,         # 限制生成长度
                num_beams=4,            # 束搜索，提升质量
                early_stopping=True
            )
            
            # 3. 解码结果
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                "translation": translated_text,
                "model_used": "byt5-finetuned",
                "original": input_text
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    else:
        raise HTTPException(status_code=400, detail="Unknown model_id")