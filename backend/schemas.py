from pydantic import BaseModel

# 前端 -> 后端 的请求格式
class TranslationRequest(BaseModel):
    text: str
    model_id: str = "default"

# 后端 -> 前端 的响应格式
class TranslationResponse(BaseModel):
    translation: str
    model_used: str