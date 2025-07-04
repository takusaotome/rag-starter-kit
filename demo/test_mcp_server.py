"""
MCPサーバーのテストスクリプト
作成したMCPサーバーが正常に動作するかテストします
"""

import asyncio
import subprocess
import json
import sys
from pathlib import Path


async def test_mcp_server():
    """MCPサーバーをテストする"""
    print("🧪 Testing MCP Server...")
    
    # 現在のディレクトリを取得
    current_dir = Path.cwd()
    print(f"📁 Current directory: {current_dir}")
    
    # MCPサーバーのパスを確認
    server_path = current_dir / "mcp_rag_server.py"
    if not server_path.exists():
        print(f"❌ MCPサーバーファイルが見つかりません: {server_path}")
        return False
    
    print(f"✅ MCPサーバーファイル確認: {server_path}")
    
    # 必要なファイルの存在確認
    required_files = [
        "server.py",
        "config.py",
        ".env",
        "knowledge/",
        "vector_store/"
    ]
    
    print("\n📋 Required files check:")
    for file_path in required_files:
        path = current_dir / file_path
        if path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - 見つかりません")
    
    # Python環境の確認
    print(f"\n🐍 Python version: {sys.version}")
    
    # 依存関係の確認
    try:
        import mcp
        print("✅ MCP library: imported successfully")
    except ImportError:
        print("❌ MCP library not found")
        return False
    
    try:
        from server import RAGServer
        print("✅ RAGServer import successful")
    except ImportError as e:
        print(f"❌ RAGServer import failed: {e}")
        return False
    
    print("\n🚀 All checks passed!")
    return True


def create_claude_config():
    """Claude Desktop設定ファイルを作成する（セキュアなテンプレート）"""
    current_dir = Path.cwd()
    
    # セキュリティのため、実際のパスではなくプレースホルダーを使用
    config = {
        "mcpServers": {
            "rag-starter-kit": {
                "command": "/ABSOLUTE/PATH/TO/YOUR/PROJECT/rag-starter-kit/venv/bin/python",
                "args": ["/ABSOLUTE/PATH/TO/YOUR/PROJECT/rag-starter-kit/mcp_rag_server.py"],
                "env": {
                    "OPENAI_API_KEY": "${OPENAI_API_KEY}"
                }
            }
        }
    }
    
    config_file = current_dir / "claude_desktop_config_template.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Claude Desktop設定テンプレートを作成しました: {config_file}")
    print("\n🔧 **Claude Desktopで使用する手順:**")
    print("1. Claude Desktopをインストール（最新版）")
    print("2. 設定ファイルのパスを実際のプロジェクトパスに変更")
    print("   '/ABSOLUTE/PATH/TO/YOUR/PROJECT/' → 実際のパス")
    print("3. 設定ファイルを以下の場所にコピー:")
    print("   macOS: ~/Library/Application Support/Claude/claude_desktop_config.json")
    print("   Windows: %AppData%\\Claude\\claude_desktop_config.json")
    print("4. Claude Desktopを再起動")
    print("5. 「Search and tools」アイコンが表示されることを確認")
    print("\n⚠️  **セキュリティ重要**: APIキーを設定後は絶対にGitにコミットしないでください！")
    
    return config_file


async def main():
    """メイン関数"""
    print("🎯 RAG Starter Kit MCP Server Test")
    print("=" * 50)
    
    # テスト実行
    success = await test_mcp_server()
    
    if success:
        # 設定ファイル作成
        config_file = create_claude_config()
        
        print("\n✅ **テスト完了！MCPサーバーの準備ができました**")
        print("\n🛠️  **利用可能なツール:**")
        print("   - query_knowledge_base: 知識ベースに質問")
        print("   - search_documents: ドキュメント検索")
        print("   - get_available_documents: ドキュメント一覧")
        print("   - get_server_status: サーバーステータス")
        
        print(f"\n📋 **設定例ファイル:** {config_file}")
        print("\n🚀 **MCPサーバー起動コマンド:**")
        print("   python mcp_rag_server.py")
        
    else:
        print("\n❌ テストに失敗しました。依存関係や設定を確認してください。")


if __name__ == "__main__":
    asyncio.run(main()) 