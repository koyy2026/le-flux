from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os

# 1. 初始化 FastAPI 應用
app = FastAPI(
    title="FLUX.1 Image Generation API",
    description="An API to generate images using FLUX.1 models, deployed on Leapcell.",
    version="1.0.0"
)

# 2. 初始化 OpenAI 客戶端以指向新的 API 提供商
#    這裡以 apipie.ai 作為範例
try:
    # 我們仍然使用 OpenAI 的 Python 函式庫，但將 base_url 指向新的服務
    client = OpenAI(
        api_key=os.environ.get("sk-navy-cNgDa6klEXzaF7EDgmHR0AKitLPJCt2IZQBzxo_SbWY"), # 將環境變數名稱改為更通用的名稱
        base_url="https://api.navy/v1"  # 更改為您的 API 提供商的 URL
    )
except Exception:
    print("Error: FLUX_API_KEY environment variable not set.")
    client = None

# 3. 定義請求的資料模型
class ImageRequest(BaseModel):
    prompt: str
    model: str = "FLUX.1-schnell"  # 預設使用快速的 schnell 模型
    size: str = "1024x1024"

# 4. 建立根端點 (/)
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Welcome to the FLUX.1 Image Generation API!"}

# 5. 建立圖像生成端點 (/generate-image)
@app.post("/generate-image")
async def generate_image(request: ImageRequest):
    if not client:
        return {"error": "API client is not configured. Check API key."}

    try:
        # API 呼叫的結構與 OpenAI 的 images.generate 完全相同
        response = client.images.generate(
            model=request.model,      # 讓使用者可以選擇 dev 或 schnell
            prompt=request.prompt,
            size=request.size,
            n=1,
        )
        image_url = response.data[0].url
        return {"prompt": request.prompt, "model": request.model, "image_url": image_url}
    except Exception as e:
        return {"error": f"An error occurred while generating the image: {e}"}
