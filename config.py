import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this")
    JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "vector_store")
    KNOWLEDGE_PATH = os.getenv("KNOWLEDGE_PATH", "knowledge")
    PROMPTS_PATH = os.getenv("PROMPTS_PATH", "prompt")
    PROMPT_FILE = os.getenv("PROMPT_FILE", "prompt.yaml")
    
    # ETL設定
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
    
    # RAG設定
    RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "6"))
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")  # 最新の推奨モデル (2025年1月時点)
    
    # サーバー設定
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))

    # MCP Server Configuration
    MCP_SERVER_CONFIG = {
        "knowledge_base_description": {
            "title": "日本料理専門知識ベース",
            "description": "伝統的な日本料理のレシピ、調理技術、文化的背景に関する包括的な知識ベース",
            "categories": [
                "基本的な米料理（親子丼、チャーハン、おにぎり等）",
                "麺料理（ラーメン、うどん、そば等）", 
                "焼き物（焼き魚、焼き鳥、お好み焼き等）",
                "煮物（肉じゃが、おでん、筑前煮等）",
                "汁物（味噌汁、すまし汁、豚汁等）",
                "副菜・前菜（漬物、サラダ、小鉢料理等）",
                "和菓子・デザート（大福、どら焼き、抹茶アイス等）",
                "基本的な調理技術（だしの取り方、切り方、調味等）",
                "飲み物（茶、日本酒、季節の飲み物等）",
                "季節の食べ物（おせち料理、季節の野菜等）"
            ],
            "languages": ["日本語", "英語"],
            "expertise_level": "家庭料理から本格的な調理技術まで",
            "example_queries": [
                "親子丼の作り方を教えて",
                "だしの取り方について",
                "季節の料理を教えて",
                "How to make sushi rice?"
            ]
        }
    } 