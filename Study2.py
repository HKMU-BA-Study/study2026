import os
import random
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# 初始化 Hugging Face Inference Client
load_dotenv()
client = InferenceClient(api_key=os.getenv("HF_TOKEN"))


def calculate_score(product, preferences):
    """
    根据 Study 2 规则计算推荐分数
    """
    # 1. 价格匹配度 (PriceFit)
    price_diff = abs(product['price'] - preferences['budget'])
    if preferences['price_sensitivity'] == 'high':
        price_fit = max(0, 1 - (price_diff / preferences['budget']))
    elif preferences['price_sensitivity'] == 'medium':
        price_fit = 0.7 + random.uniform(0, 0.3)
    else:  # low
        price_fit = 0.9 + random.uniform(0, 0.1)

    # 2. 可持续性匹配度 (SustainabilityMatch)
    if preferences['sustainability_importance'] == 'high':
        sustainability_fit = product['sustainability_score'] / 10.0
    elif preferences['sustainability_importance'] == 'medium':
        sustainability_fit = 0.5
    else:  # low
        sustainability_fit = 0.2

    # 3. 兴趣匹配度 (InterestMatch)
    if product['category'] == preferences['product_interest']:
        interest_fit = 1.0
    else:
        interest_fit = 0.3

    # 总分计算（权重：价格30%，可持续40%，兴趣30%）
    total_score = (price_fit * 0.3) + (sustainability_fit * 0.4) + (interest_fit * 0.3)
    return round(total_score, 2)


def rank_products(products, preferences, top_n=4):
    """
    对所有商品打分并排序，返回 top_n 个推荐商品
    """
    scored_products = []
    for product in products:
        score = calculate_score(product, preferences)
        scored_products.append({
            **product,
            'score': score
        })
    scored_products.sort(key=lambda x: x['score'], reverse=True)
    return scored_products[:top_n]


def generate_ai_nudge_message(preferences, top_product):
    """
    使用 Hugging Face Serverless Inference API 動態生成 AI 個性化 Nudge 文案
    """
    prompt = f"""
    You are an expert in behavioral economics and marketing nudges. 
    Generate a short, persuasive, personalized nudge message (1 sentence, max 20 words) for a user to buy a recommended product.

    [User Preferences]
    - Budget: ${preferences['budget']}
    - Price Sensitivity: {preferences['price_sensitivity']}
    - Sustainability Importance: {preferences['sustainability_importance']}
    - Preferred Category: {preferences['product_interest']}

    [Product Features]
    - Name: {top_product['name']}
    - Price: ${top_product['price']}
    - Sustainability Score: {top_product['sustainability_score']}/10
    - Category: {top_product['category']}
    - Match Score: {top_product['score']}

    [Guidelines]
    - If user values sustainability highly and product score is high, highlight green/eco values.
    - If user is price sensitive and product is within budget, highlight cost value.
    - Do NOT mention raw math, preferences, or score numbers.
    - Keep it subtle, appealing, and directly focused on why this product fits them.
    - Return ONLY the nudge text in English.
    """

    try:
        response = client.chat_completion(
            # 改用免費 Serverless API 支援的熱門模型
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=60,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"API Error: {e}")
        return "This product is carefully matched to your preferences."


# --- 测试代码 ---
if __name__ == "__main__":
    products_pool = [
        {'name': 'Eco Tour', 'price': 500, 'sustainability_score': 8, 'category': 'adventure'},
        {'name': 'Fast Fashion', 'price': 300, 'sustainability_score': 3, 'category': 'clothing'},
        {'name': 'Organic Clothing', 'price': 450, 'sustainability_score': 9, 'category': 'clothing'},
        {'name': 'Adventure Gear', 'price': 550, 'sustainability_score': 6, 'category': 'adventure'},
        {'name': 'Generic Product', 'price': 400, 'sustainability_score': 5, 'category': 'home'}
    ]

    my_prefs = {
        'budget': 600,
        'price_sensitivity': 'medium',
        'sustainability_importance': 'high',
        'product_interest': 'adventure'
    }

    # 1. 计算单个商品分数
    my_product = products_pool[0]
    score = calculate_score(my_product, my_prefs)
    print(f"单个商品 '{my_product['name']}' 的推荐分数为: {score}")

    # 2. 商品排序与推荐
    top_recommendations = rank_products(products_pool, my_prefs, top_n=3)
    print("\nTop 3 推荐商品 (含 AI Nudge 文案):")
    for idx, rec in enumerate(top_recommendations, 1):
        print(f"{idx}. {rec['name']} - 分数: {rec['score']}")

        # 3. 呼叫 Hugging Face 生成 AI 引导文案
        ai_nudge = generate_ai_nudge_message(my_prefs, rec)
        print(f"   🤖 AI Nudge 文案: {ai_nudge}\n")