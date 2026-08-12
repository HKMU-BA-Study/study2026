import os
import json
import re
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv() # cite: 6
HF_TOKEN = os.getenv("HF_TOKEN") # cite: 6
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct" # cite: 6

client = InferenceClient(token=HF_TOKEN) # cite: 6

app = FastAPI(
    title="Study 2 AI Nudging API",
    redirect_slashes=False  # 💡 關閉 FastAPI 內部的斜線轉址
)

# 💡 完整允許跨域請求
app.add_middleware(
    CORSMiddleware, # type: ignore
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendationRequest(BaseModel): # cite: 6
    products: List[Dict[str, Any]] # cite: 6
    preferences: Dict[str, Any] # cite: 6

def get_ai_recommendations_from_hf(products: List[Dict[str, Any]], preferences: Dict[str, Any]) -> List[Dict[str, Any]]: # cite: 6
    simplified_products = [ # cite: 6
        {"id": p.get("id"), "name": p.get("name"), "price": p.get("price"), "isEco": p.get("isEco")} # cite: 6
        for p in products # cite: 6
    ] # cite: 6

    prompt = f"""
You are an AI recommender system. Select top 3 products.
User preferences: {json.dumps(preferences, ensure_ascii=False)}
Products: {json.dumps(simplified_products, ensure_ascii=False)}

Output MUST be a valid JSON array:
[
  {{"rank": 1, "item_id": 1, "reason": "Reason 1"}},
  {{"rank": 2, "item_id": 2, "reason": "Reason 2"}},
  {{"rank": 3, "item_id": 3, "reason": "Reason 2"}}
]
""" # cite: 6

    try: # cite: 6
        response = client.chat_completion( # cite: 6
            model=MODEL_NAME, # cite: 6
            messages=[ # cite: 6
                {"role": "system", "content": "Output strictly valid JSON arrays without markdown."}, # cite: 6
                {"role": "user", "content": prompt} # cite: 6
            ], # cite: 6
            max_tokens=300, # cite: 6
            temperature=0.2 # cite: 6
        ) # cite: 6

        content = response.choices[0].message.content.strip() # cite: 6

        if content.startswith("```"): # cite: 6
            content = re.sub(r"^```[a-zA-Z]*\n?", "", content) # cite: 6
            content = re.sub(r"\n?```$", "", content) # cite: 6
            content = content.strip() # cite: 6

        return json.loads(content) # cite: 6

    except Exception as e: # cite: 6
        print(f"Hugging Face API Error: {e}") # cite: 6
        raise HTTPException(status_code=500, detail=str(e)) # cite: 6

# 💡 攔截所有路徑的 OPTIONS 預檢，直接帶上 CORS 標頭回傳 200 OK
@app.options("/{full_path:path}")
def options_handler(full_path: str):
    response = Response(status_code=200)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# 💡 POST 路由 (完全不帶尾端斜線)
@app.post("/api/recommend")
def recommend_products(req: RecommendationRequest): # cite: 6
    if not req.products or len(req.products) < 3: # cite: 6
        raise HTTPException(status_code=400, detail="Products list must contain at least 3 items.") # cite: 6

    return get_ai_recommendations_from_hf(req.products, req.preferences) # cite: 6

@app.get("/api/recommend")
def test_endpoint(): # cite: 6
    return {"status": "ok", "message": "API endpoint is active"} # cite: 6