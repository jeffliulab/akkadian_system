import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.schemas import TranslationRequest, TranslationResponse

# 导入具体模型模块
import server.model.default as default_engine

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/translate", response_model=TranslationResponse)
async def translate_router(request: TranslationRequest):
    input_text = request.text.strip()
    model_id = request.model_id
    output_text = ""
    
    # --- 分流 ---
    if model_id == "default":
        # 调用 server/model/model_default.py
        output_text = default_engine.predict(input_text)
        
    # elif model_id == "byt5":
    #     output_text = "[ByT5 module not implemented yet]"
        
    else:
        output_text = "[Error: Unknown Model ID]"

    return {
        "translation": output_text,
        "model_used": model_id
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)