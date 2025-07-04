# 🔌 MCP サーバー設定ガイド

**RAG Starter Kit を Model Context Protocol (MCP) 経由で使用する方法**

## 🎯 概要

このRAG Starter KitはMCPサーバーとして動作し、Claude Desktop からツールとして直接アクセスできます。

## 📋 必要なもの

- Python 3.11以上
- OpenAI API Key
- Claude Desktop（最新版）

## 🚀 セットアップ手順

### 1. 仮想環境の作成（Python 3.11）

```bash
# Python 3.11で仮想環境を作成
python3.11 -m venv venv

# 仮想環境をアクティベート
source venv/bin/activate  # macOS/Linux
# または
venv\Scripts\activate     # Windows
```

### 2. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 3. 環境変数の設定

`.env`ファイルにOpenAI API Keyを設定:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. MCPサーバーのテスト

```bash
python test_mcp_server.py
```

成功メッセージが表示されることを確認してください。

## 🔧 Claude Desktop 設定

### 1. 設定ファイルの場所を確認

```bash
# macOS
~/Library/Application Support/Claude/claude_desktop_config.json

# Windows
%AppData%\Claude\claude_desktop_config.json
```

### 2. 設定ファイルの作成・編集

⚠️ **セキュリティ重要**: APIキーが含まれる設定ファイルは絶対にGitにコミットしないでください！

#### セキュアな設定手順：

1. **テンプレートファイルをコピー**:
   ```bash
   cp claude_desktop_config_template.json ~/temp_config.json
   ```

2. **テンプレートファイルを編集**:
   ```json
   {
     "mcpServers": {
       "rag-starter-kit": {
         "command": "/Users/takueisaotome/PycharmProjects/rag-starter-kit/venv/bin/python",
         "args": [
           "/Users/takueisaotome/PycharmProjects/rag-starter-kit/mcp_rag_server.py"
         ],
         "env": {
           "OPENAI_API_KEY": "sk-proj-YOUR_ACTUAL_API_KEY_HERE"
         }
       }
     }
   }
   ```

3. **正しい場所に移動**:
   ```bash
   # macOS
   mv ~/temp_config.json ~/Library/Application\ Support/Claude/claude_desktop_config.json
   
   # Windows (PowerShell)
   # Move-Item ~/temp_config.json $env:AppData/Claude/claude_desktop_config.json
   ```

**🔐 セキュリティチェックリスト**:
- ✅ 実際のAPIキーをプロジェクトファイルに含めない
- ✅ .gitignoreで機密ファイルを除外済み
- ✅ 設定ファイルはClaude Desktopの指定場所のみに配置
- ❌ プロジェクトディレクトリに実APIキー入り設定ファイルを置かない

### 3. Claude Desktop の再起動

設定を変更後、Claude Desktop を完全に再起動してください。

## 🛠️ 利用可能なツール

MCPサーバーでは以下のツールが利用できます：

### 1. `query_knowledge_base`
知識ベースに質問して回答を取得

```
パラメータ:
- question (str): 質問内容（日本語または英語）

例: "親子丼の作り方を教えて"
```

### 2. `search_documents`
キーワードで関連ドキュメントを検索

```
パラメータ:
- keywords (str): 検索キーワード
- max_results (int): 最大結果数（デフォルト: 5）

例: keywords="卵料理", max_results=3
```

### 3. `get_available_documents`
利用可能な知識ベースドキュメントの一覧を取得

```
パラメータ: なし

例: 利用可能な料理レシピファイルの一覧を表示
```

### 4. `get_server_status`
RAGサーバーの現在のステータスを取得

```
パラメータ: なし

例: サーバーの初期化状態、ベクターストアの状態を確認
```

## 💡 使用例

Claude Desktop でMCPサーバーが認識されると、以下のようにツールを使用できます：

```
You: 親子丼の作り方を教えて

Claude: query_knowledge_base ツールを使って知識ベースから親子丼の作り方を検索します。
[query_knowledge_base: question="親子丼の作り方を教えて"]
```

## 🔍 トラブルシューティング

### MCPサーバーが認識されない場合

1. **設定ファイルのパスを確認**
   ```bash
   python test_mcp_server.py
   ```

2. **Claude Desktop のログを確認**
   ```bash
   # macOS
   tail -f ~/Library/Logs/Claude/mcp*.log
   ```

3. **Python パスの確認**
   ```bash
   which python
   ```

### よくあるエラー

#### "ModuleNotFoundError: No module named 'mcp'"
```bash
pip install "mcp[cli]>=1.10.0"
```

#### "OPENAI_API_KEY not found"
`.env`ファイルにAPIキーが正しく設定されているか確認

#### "Permission denied"
スクリプトファイルの実行権限を確認:
```bash
chmod +x mcp_rag_server.py
```

## 🔄 サーバーの直接起動

デバッグ用に、MCPサーバーを直接起動することもできます：

```bash
python mcp_rag_server.py
```

## 📈 パフォーマンス最適化

### 1. ベクターストアの事前作成

```bash
python run_etl.py
```

### 2. 知識ベースの更新

新しいドキュメントを追加した場合：

```bash
# 新しいドキュメントをknowledge/に追加
python run_etl.py  # ベクターストアを再構築
```

## 🔐 セキュリティ注意事項

- OpenAI API Key は環境変数で管理
- 設定ファイルに直接APIキーを記述しない
- 本番環境では適切なアクセス制御を実装

## 📖 関連ドキュメント

- [MCP 公式ドキュメント](https://modelcontextprotocol.io/)
- [Claude Desktop MCP Guide](https://docs.anthropic.com/claude/docs/mcp)
- [RAG Starter Kit README](README.md)

---

## 📞 サポート

問題が発生した場合：

1. [GitHub Issues](https://github.com/takusaotome/rag-starter-kit/issues) で報告
2. `test_mcp_server.py` でのテスト結果を含める
3. エラーメッセージとログを詳細に記載 