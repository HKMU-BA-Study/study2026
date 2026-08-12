import os
import json
import re
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

client = InferenceClient(token=HF_TOKEN)

app = FastAPI(title="Study 2 AI Nudging API")

# 1. CORS Middleware
app.add_middleware(
    CORSMiddleware,  # type: ignore
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 最外層攔截器：強制攔截 OPTIONS，絕不丟給 Vercel 做 308 重定向
@app.middleware("http")
def force_options_200(request: Request, call_next):
    if request.method == "OPTIONS":
        response = Response(status_code=200)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response
    return call_next(request)

class RecommendationRequest(BaseModel):
    products: List[Dict[str, Any]]
    preferences: Dict[str, Any]

def get_ai_recommendations_from_hf(products: List[Dict[str, Any]], preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
    simplified_products = [
        {"id": p.get("id"), "name": p.get("name"), "price": p.get("price"), "isEco": p.get("isEco")}
        for p in products
    ]

    prompt = f"""
You are an AI recommender system. Select top 3 products.
User preferences: {json.dumps(preferences, ensure_ascii=False)}
Products: {json.dumps(simplified_products, ensure_ascii=False)}

Output MUST be a valid JSON array:
[
  {{"rank": 1, "item_id": 1, "reason": "Reason 1"}},
  {{"rank": 2, "item_id": 2, "reason": "Reason 2"}},
  {{"rank": 3, "item_id": 3, "reason": "Reason 3"}}
]
"""

    try:
        response = client.chat_completion(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Output strictly valid JSON arrays without markdown."},
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

        return json.loads(content)

    except Exception as e:
        print(f"Hugging Face API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# API 路由綁定
@app.post("/api/recommend")
@app.post("/api/recommend/")
def recommend_products(req: RecommendationRequest):
    if not req.products or len(req.products) < 3:
        raise HTTPException(status_code=400, detail="Products list must contain at least 3 items.")

    return get_ai_recommendations_from_hf(req.products, req.preferences)

@app.get("/api/recommend")
@app.get("/api/recommend/")
def test_endpoint():
    return {"status": "ok", "message": "API endpoint is active"}