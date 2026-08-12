import os
import json
import re
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# 1. 初始化與環境變數載入
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# 免費 Serverless Inference API 支援的模型 (建議使用 Qwen2.5-7B 或 Llama-3-8B)
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

client = InferenceClient(token=HF_TOKEN)

app = FastAPI(
    title="Study 2 AI Nudging API",
    description="Render.com API Endpoint for HF AI Nudging Recommendations"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],             # 允許所有來源（包含 localhost:8000）
    allow_credentials=True,
    allow_methods=["*"],             # 允許所有 HTTP 方法 (POST, GET 等)
    allow_headers=["*"],             # 允許所有 Headers
)
@app.post("/recommend")
@app.post("/recommend/")

# 2. 定義 Request 格式 (Pydantic Schema)
class RecommendationRequest(BaseModel):
    products: List[Dict[str, Any]]  # 例如: [{"id": 1, "name": "Red Fuji Apple", "price": 5.99, "isEco": True}, ...]
    preferences: Dict[str, Any]      # 例如: {"budget": 50, "price_sensitivity": "high", "sustainability_importance": "high", ...}


# 3. 完全由 Hugging Face LLM 做商品分析與推薦的的核心函式
def get_ai_recommendations_from_hf(products: List[Dict[str, Any]], preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    將商品清單與使用者偏好丟給 Hugging Face LLM，讓 LLM 評估排序並回傳 Top 3 推薦與理由。
    """
    prompt = f"""
You are an expert AI recommender system and behavioral marketing expert. 
Analyze the provided user preferences and available products, then select and rank the top 3 best matching products.

[User Preferences]
{json.dumps(preferences, ensure_ascii=False, indent=2)}

[Available Products]
{json.dumps(products, ensure_ascii=False, indent=2)}

[Task Guidelines]
1. Select the Top 3 products that best match the user's preferences (considering budget, price sensitivity, eco/sustainability preferences, category interests, etc.).
2. For each recommended product, write a persuasive, personalized nudge reason (1-2 sentences) explaining why it fits them without using technical jargon.
3. Output MUST be valid JSON, strictly matching the array structure format below with NO extra markdown text before or after:

[
  {{
    "rank": 1,
    "item_id": 1,
    "reason": "Personalized AI nudge explanation for item 1..."
  }},
  {{
    "rank": 2,
    "item_id": 2,
    "reason": "Personalized AI nudge explanation for item 2..."
  }},
  {{
    "rank": 3,
    "item_id": 3,
    "reason": "Personalized AI nudge explanation for item 3..."
  }}
]
"""

    try:
        response = client.chat_completion(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a JSON-only API assistant. Output strictly valid JSON arrays without markdown formatting."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=600,
            temperature=0.3  # 低溫有助於維持穩定的 JSON 輸出格式
        )

        content = response.choices[0].message.content.strip()

        # 移除 markdown ```json ... ``` 標記（若模型不小心輸出）
        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
            content = re.sub(r"\n?```$", "", content)
            content = content.strip()

        recommendations = json.loads(content)
        return recommendations

    except json.JSONDecodeError as e:
        print(f"JSON Parse Error. Raw content: {content}")
        raise HTTPException(status_code=500, detail=f"Failed to parse LLM JSON response: {str(e)}")
    except Exception as e:
        print(f"Hugging Face API Error: {e}")
        raise HTTPException(status_code=500, detail=f"Hugging Face Inference API Error: {str(e)}")


# 4. API Endpoint (提供給前端或客戶端呼叫)
@app.post("/recommend")
def recommend_products(req: RecommendationRequest):
    if not req.products or len(req.products) < 3:
        raise HTTPException(status_code=400, detail="Products list must contain at least 3 items.")

    recommendations = get_ai_recommendations_from_hf(req.products, req.preferences)
    return recommendations


# 5. 本地測試入口
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("Study2_AI:app", host="0.0.0.0", port=8000, reload=True)