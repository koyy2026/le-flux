from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from openai import OpenAI
from enum import Enum
from typing import Annotated # 引入 Annotated

# ... (AvailableModels Enum 保持不變) ...
class AvailableModels(str, Enum):
    schnell = "flux.1-schnell"
    por = "flux.1.1-por"
    latest = "flux.latest"
    krea_dev = "flux.1-krea-dev"
    kontext_pro = "flux.1-kontext-pro"
    kontext_max = "flux.1-kontext-max"


app = FastAPI(
    title="User-Key AI Image Generation API",
    description="API that uses the user's provided API key for image generation.",
    version="2.0.0"
)

class ImageRequest(BaseModel):
    prompt: str
    model: AvailableModels = AvailableModels.schnell
    size: str = "1024x1024"

@app.get("/")
def read_root():
    return {
        "status": "ok",
        "available_models": [model.value for model in AvailableModels]
    }

# --- 主要修改點 ---
@app.post("/generate-image-with-user-key")
async def generate_image(
    request: ImageRequest,
    # 從 Authorization 標頭中讀取 Bearer Token
    authorization: Annotated[str | None, Header()] = None
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header. Please provide an API key as a Bearer token.")
    
    # 提取 API 金鑰
    user_api_key = authorization.split("Bearer ")[1]

    try:
        # 使用使用者提供的金鑰初始化客戶端
        client = OpenAI(
            api_key=user_api_key,
            base_url="https://api.apipie.ai/v1"  # 範例 URL
        )

        response = client.images.generate(
            model=request.model.value,
            prompt=request.prompt,
            size=request.size,
            n=1,
        )
        image_url = response.data[0].url
        return {"prompt": request.prompt, "model": request.model.value, "image_url": image_url}
    except Exception as e:
        # 這裡可以更細緻地處理 API 金鑰無效的錯誤
        if "authentication" in str(e).lower():
             raise HTTPException(status_code=403, detail=f"Authentication failed. Please check if your API key is correct and has enough credits. Error: {e}")
        raise HTTPException(status_code=503, detail=f"An error occurred while communicating with the image generation service: {e}")


