"""
RAG Starter Kit MCP Server
MCPプロトコル経由でRAG機能にアクセスできるシンプルなサーバー実装
"""

import os
import asyncio
import sys
from typing import Any, Dict, List
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# スクリプトのディレクトリを取得してパスを設定
SCRIPT_DIR = Path(__file__).parent.absolute()
os.chdir(SCRIPT_DIR)  # 作業ディレクトリを変更

# プロジェクトディレクトリをPythonパスに追加
sys.path.insert(0, str(SCRIPT_DIR))

# 既存のRAGServerをインポート
from server import RAGServer

# .envファイルを読み込み（絶対パス）
load_dotenv(SCRIPT_DIR / ".env")

# 環境変数の確認と警告
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
    print("🔍 Checking .env file...")
    
    env_file = SCRIPT_DIR / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            content = f.read()
            if "OPENAI_API_KEY=" in content:
                # .envファイルから直接読み込み
                for line in content.split('\n'):
                    if line.startswith('OPENAI_API_KEY='):
                        api_key = line.split('=', 1)[1].strip()
                        os.environ['OPENAI_API_KEY'] = api_key
                        print("✅ OPENAI_API_KEY loaded from .env file")
                        break
else:
    print("✅ OPENAI_API_KEY found in environment variables")

# FastMCPサーバーを初期化
mcp = FastMCP("rag-starter-kit")

# RAGサーバーインスタンスをグローバル変数として作成
rag_server = None


async def init_rag_server():
    """RAGサーバーを初期化する"""
    global rag_server
    if rag_server is None:
        print("🔄 Initializing RAG server...")
        print(f"📁 Current working directory: {os.getcwd()}")
        print(f"📁 Script directory: {SCRIPT_DIR}")
        
        # 必要なファイルの存在確認
        required_paths = [
            SCRIPT_DIR / "knowledge",
            SCRIPT_DIR / "vector_store", 
            SCRIPT_DIR / ".env",
            SCRIPT_DIR / "config.py"
        ]
        
        for path in required_paths:
            if path.exists():
                print(f"✅ Found: {path}")
            else:
                print(f"❌ Missing: {path}")
        
        try:
            rag_server = RAGServer()
            rag_server.initialize()
            print("✅ RAG server initialization completed!")
        except Exception as e:
            print(f"❌ RAG server initialization failed: {e}")
            raise


@mcp.tool()
async def query_knowledge_base(question: str) -> str:
    """
    カスタム知識ベースに質問して回答を取得する
    
    この知識ベースの詳細な内容についてはget_server_statusツールを使用して確認できます。
    
    Args:
        question: 知識ベースに関する質問（日本語または英語）
    
    Returns:
        str: 知識ベースに基づく詳細な回答
    """
    global rag_server
    
    # RAGサーバーが初期化されていない場合は初期化
    if rag_server is None:
        await init_rag_server()
    
    try:
        # 既存のRAGサーバーのprocess_queryメソッドを使用
        result = rag_server.process_query(question)
        
        # 回答と参考ソースを整形して返す
        response = result['answer']
        
        if result.get('sources'):
            response += f"\n\n📚 **参考ソース:**"
            for source in result['sources']:
                response += f"\n- {source}"
        
        return response
        
    except Exception as e:
        return f"❌ エラーが発生しました: {str(e)}"


@mcp.tool()
async def search_documents(keywords: str, max_results: int = 5) -> str:
    """
    知識ベースのドキュメントをキーワードで検索する
    
    Args:
        keywords: 検索キーワード
        max_results: 返す結果の最大数（デフォルト: 5）
    
    Returns:
        str: ドキュメントの検索結果リスト
    """
    global rag_server
    
    # RAGサーバーが初期化されていない場合は初期化
    if rag_server is None:
        await init_rag_server()
    
    try:
        # ベクターストアから類似ドキュメントを検索
        if rag_server.vector_store is None:
            return "❌ ベクターストアが初期化されていません"
        
        # 類似ドキュメントを検索
        docs = rag_server.vector_store.similarity_search(
            keywords, 
            k=max_results
        )
        
        if not docs:
            return f"'{keywords}' に関連するドキュメントが見つかりませんでした。"
        
        # 結果を整形
        response = f"🔍 **'{keywords}' の検索結果:**\n\n"
        
        for i, doc in enumerate(docs, 1):
            # ドキュメント内容の最初の200文字を表示
            content_preview = doc.page_content[:200]
            if len(doc.page_content) > 200:
                content_preview += "..."
            
            # メタデータからソース情報を取得
            source = doc.metadata.get('source', 'Unknown')
            
            response += f"**{i}. {source}**\n"
            response += f"{content_preview}\n\n"
        
        return response
        
    except Exception as e:
        return f"❌ 検索エラー: {str(e)}"


@mcp.tool()
async def get_available_documents() -> str:
    """
    利用可能な知識ベースドキュメントの一覧を取得する
    
    Returns:
        str: ドキュメント一覧
    """
    try:
        knowledge_path = SCRIPT_DIR / "knowledge"
        
        if not knowledge_path.exists():
            return "❌ knowledgeディレクトリが見つかりません"
        
        # markdownファイルを取得
        md_files = list(knowledge_path.glob("*.md"))
        
        if not md_files:
            return "❌ knowledgeディレクトリにmarkdownファイルが見つかりません"
        
        response = "📚 **利用可能なドキュメント:**\n\n"
        
        for i, file_path in enumerate(sorted(md_files), 1):
            file_name = file_path.name
            # ファイルサイズを取得
            file_size = file_path.stat().st_size
            size_kb = round(file_size / 1024, 1)
            
            response += f"{i}. **{file_name}** ({size_kb}KB)\n"
        
        return response
        
    except Exception as e:
        return f"❌ ドキュメント一覧取得エラー: {str(e)}"


@mcp.tool()
async def debug_paths() -> str:
    """
    デバッグ用：パス情報を詳細に表示
    
    Returns:
        str: パス情報
    """
    try:
        debug_info = "🔍 **Debug Path Information**\n\n"
        
        debug_info += f"📁 Current working directory: {os.getcwd()}\n"
        debug_info += f"📁 Script directory: {SCRIPT_DIR}\n"
        debug_info += f"📁 Script file: {__file__}\n\n"
        
        # 重要なパスの確認
        paths_to_check = [
            ("knowledge", SCRIPT_DIR / "knowledge"),
            ("vector_store", SCRIPT_DIR / "vector_store"),
            (".env", SCRIPT_DIR / ".env"),
            ("config.py", SCRIPT_DIR / "config.py"),
            ("server.py", SCRIPT_DIR / "server.py")
        ]
        
        debug_info += "📋 **Path Check Results:**\n"
        for name, path in paths_to_check:
            if path.exists():
                if path.is_file():
                    size = path.stat().st_size
                    debug_info += f"✅ {name}: {path} ({size} bytes)\n"
                else:
                    files = list(path.glob("*"))
                    debug_info += f"✅ {name}: {path} ({len(files)} files)\n"
            else:
                debug_info += f"❌ {name}: {path} (NOT FOUND)\n"
        
        # 環境変数の確認
        debug_info += f"\n🔑 **Environment Variables:**\n"
        debug_info += f"OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}\n"
        
        return debug_info
        
    except Exception as e:
        return f"❌ Debug path error: {str(e)}"


@mcp.tool()
async def get_server_status() -> str:
    """
    RAGサーバーの現在のステータスと知識ベースの詳細情報を取得する
    
    Returns:
        str: サーバーステータスと知識ベースの内容情報
    """
    global rag_server
    
    try:
        status_info = "🚀 **RAG Starter Kit MCP Server Status**\n\n"
        
        # 作業ディレクトリ情報を追加
        status_info += f"📁 作業ディレクトリ: {os.getcwd()}\n"
        status_info += f"📁 スクリプトディレクトリ: {SCRIPT_DIR}\n\n"
        
        if rag_server is None:
            status_info += "⚠️  RAGサーバー: 未初期化\n"
        else:
            status_info += "✅ RAGサーバー: 初期化済み\n"
            
            # ベクターストアのステータス
            if rag_server.vector_store is not None:
                # インデックスの情報を取得
                vector_count = rag_server.vector_store.index.ntotal
                status_info += f"✅ ベクターストア: 有効 ({vector_count} vectors)\n"
            else:
                status_info += "❌ ベクターストア: 無効\n"
            
            # QAチェーンのステータス
            if rag_server.qa_chain is not None:
                status_info += "✅ QAチェーン: 有効\n"
            else:
                status_info += "❌ QAチェーン: 無効\n"
        
        # 環境変数の確認
        openai_key_set = "✅" if os.getenv("OPENAI_API_KEY") else "❌"
        status_info += f"{openai_key_set} OpenAI API Key: {'設定済み' if openai_key_set == '✅' else '未設定'}\n"
        
        # 知識ベースディレクトリの確認
        knowledge_path = SCRIPT_DIR / "knowledge"
        if knowledge_path.exists():
            md_files = list(knowledge_path.glob("*.md"))
            status_info += f"✅ 知識ベース: {len(md_files)} ファイル\n"
        else:
            status_info += "❌ 知識ベース: ディレクトリなし\n"
        
        # 知識ベースの詳細情報を追加
        status_info += "\n📚 **Knowledge Base Information:**\n"
        kb_info = analyze_knowledge_base()
        
        status_info += f"📖 **{kb_info.get('title', 'Knowledge Base')}**\n"
        status_info += f"概要: {kb_info.get('description', 'カスタム知識ベース')}\n"
        
        categories = kb_info.get('categories', [])
        if categories:
            status_info += "\n📋 **含まれる内容:**\n"
            for category in categories:
                status_info += f"- {category}\n"
        
        example_queries = kb_info.get('example_queries', [])
        if example_queries:
            status_info += "\n💡 **質問例:**\n"
            for query in example_queries:
                status_info += f"- {query}\n"
        
        expertise_level = kb_info.get('expertise_level')
        if expertise_level:
            status_info += f"\n🎯 **専門レベル:** {expertise_level}\n"
        
        languages = kb_info.get('languages', [])
        if languages:
            status_info += f"🌍 **対応言語:** {', '.join(languages)}\n"
        
        return status_info
        
    except Exception as e:
        return f"❌ ステータス取得エラー: {str(e)}"


def analyze_knowledge_base():
    """
    知識ベースの内容を分析して説明を生成する
    
    Returns:
        dict: 知識ベースの概要情報
    """
    try:
        from config import MCP_SERVER_CONFIG
        # 設定ファイルから情報を取得
        return MCP_SERVER_CONFIG.get("knowledge_base_description", {})
    except ImportError:
        # 設定ファイルがない場合は動的に分析
        return analyze_knowledge_base_dynamically()


def analyze_knowledge_base_dynamically():
    """
    知識ベースのファイルを動的に分析して概要を生成
    
    Returns:
        dict: 分析結果
    """
    try:
        knowledge_path = SCRIPT_DIR / "knowledge"
        if not knowledge_path.exists():
            return {"title": "Knowledge Base", "description": "Custom knowledge base"}
        
        md_files = list(knowledge_path.glob("*.md"))
        if not md_files:
            return {"title": "Knowledge Base", "description": "Custom knowledge base"}
        
        # ファイル名から内容を推測
        categories = []
        for file_path in sorted(md_files):
            file_name = file_path.stem
            # ファイル名から日本語や英語のカテゴリを推測
            if any(keyword in file_name.lower() for keyword in ['rice', 'dish', '料理']):
                categories.append(f"ファイル: {file_name}")
        
        return {
            "title": "汎用知識ベース",
            "description": f"{len(md_files)}個のドキュメントを含む知識ベース",
            "categories": categories[:10],  # 最大10個まで
            "file_count": len(md_files),
            "languages": ["日本語", "English"],
            "expertise_level": "カスタム知識ベース"
        }
    except Exception as e:
        return {
            "title": "Knowledge Base",
            "description": f"Custom knowledge base (Analysis error: {str(e)})"
        }


def get_knowledge_base_description():
    """
    知識ベースの説明を取得してツールの説明文を生成
    
    Returns:
        str: ツール用の説明文
    """
    kb_info = analyze_knowledge_base()
    
    title = kb_info.get("title", "知識ベース")
    description = kb_info.get("description", "カスタム知識ベース")
    categories = kb_info.get("categories", [])
    
    desc_text = f"{title}に質問して回答を取得する\n\n{description}"
    
    if categories:
        desc_text += "\n\n含まれる内容:"
        for category in categories:
            desc_text += f"\n- {category}"
    
    example_queries = kb_info.get("example_queries", [])
    if example_queries:
        desc_text += "\n\n例："
        for query in example_queries[:3]:  # 最大3つまで
            desc_text += f"\n- {query}"
    
    return desc_text


if __name__ == "__main__":
    print("🎯 RAG Starter Kit MCP Server starting...")
    print("🔗 MCP Protocol: stdio transport")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"📁 Script directory: {SCRIPT_DIR}")
    print("🛠️  Available tools:")
    print("   - query_knowledge_base: 知識ベースに質問")
    print("   - search_documents: ドキュメント検索")
    print("   - get_available_documents: ドキュメント一覧")
    print("   - get_server_status: サーバーステータス")
    print("   - debug_paths: パス情報デバッグ")
    print("🚀 Server ready!")
    
    # MCPサーバーを起動（stdio transport使用）
    mcp.run(transport='stdio') 