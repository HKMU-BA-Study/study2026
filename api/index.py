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

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

client = InferenceClient(token=HF_TOKEN)

app = FastAPI(
    title="Study 2 AI Nudging API",
    description="Vercel API Endpoint for HF AI Nudging Recommendations",
    redirect_slashes=False
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 定義 Request 格式
class RecommendationRequest(BaseModel):
    products: List[Dict[str, Any]]
    preferences: Dict[str, Any]


# 3. 核心推薦邏輯
def get_ai_recommendations_from_hf(products: List[Dict[str, Any]], preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
    simplified_products = [
        {"id": p.get("id"), "name": p.get("name"), "price": p.get("price"), "isEco": p.get("isEco")}
        for p in products
    ]

    prompt = f"""
You are an AI recommender system.
Select and rank the top 3 best matching products based on user preferences.

[User Preferences]
{json.dumps(preferences, ensure_ascii=False)}

[Available Products]
{json.dumps(simplified_products, ensure_ascii=False)}

[Task]
Select Top 3 products. Write 1 short sentence reason for each.
Output MUST be strict JSON array with NO markdown formatting:
[
  {{"rank": 1, "item_id": 1, "reason": "Short reason..."}},
  {{"rank": 2, "item_id": 2, "reason": "Short reason..."}},
  {{"rank": 3, "item_id": 3, "reason": "Short reason..."}}
]
"""

    try:
        response = client.chat_completion(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a JSON-only API assistant. Output strictly valid JSON arrays without markdown formatting."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.2
        )

        content = response.choices[0].message.content.strip()

        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
            content = re.sub(r"\n?```$", "", content)
            content = content.strip()

        recommendations = json.loads(content)
        return recommendations

    except Exception as e:
        print(f"Hugging Face API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 4. 修正後的路由裝飾器（包含 /api 前綴，確保相容性）
@app.post("/api/recommend")
@app.post("/api/recommend/")
@app.post("/recommend")
@app.post("/recommend/")
def recommend_products(req: RecommendationRequest):
    if not req.products or len(req.products) < 3:
        raise HTTPException(status_code=400, detail="Products list must contain at least 3 items.")

    return get_ai_recommendations_from_hf(req.products, req.preferences)

@app.get("/")
@app.get("/api")
@app.get("/api/")
def read_root():
    return {"message": "AI Nudging API is Running"}