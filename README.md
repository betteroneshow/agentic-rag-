# Agentic RAG 旅游问答与行程规划系统

一个面向旅游知识问答和个性化行程规划的 Agentic RAG 项目。系统使用 LangGraph 编排查询理解、信息完整性检查、动态路由、混合检索、答案生成、自校验以及长短期记忆，并提供 Streamlit 对话界面和实时天气/地图路线 MCP Server。

## 核心能力

- **Agent 工作流**：使用 LangGraph 管理条件路由、检索重试、答案修正和停止条件。
- **查询改写**：结合对话摘要和最近历史消解“那里、那个、便宜的”等指代。
- **路线信息澄清**：用户询问距离、到达方式或路线但缺少起点/终点时，先追问所在地或目的地。
- **混合检索**：BM25 稀疏检索与 `text-embedding-v4` 稠密检索并行召回，经 RRF 融合后使用 `qwen3-rerank` 精排。
- **小块检索、大块返回**：350 字符子块负责召回，命中后返回对应的 1000 字符父块。
- **多数据源路由**：支持直接回答、本地直接区块检索、摘要分层检索和 Tavily 网络搜索。
- **两级质量检查**：生成前评估文档相关性，生成后评估答案相关性。
- **降级与防循环**：Rerank 失败回退 RRF；本地检索失败切换策略；状态字段限制重复检索和答案修正。
- **上下文工程**：XML 标签隔离上下文；超过阈值时进行有损压缩；保留来源编号和关键约束。
- **长短期记忆**：短期使用滚动摘要和最近对话，长期使用 SQLite + ChromaDB 持久化和语义召回。
- **增量知识库**：文件指纹、manifest、稳定 ID 和 Chroma `upsert` 防止重复处理。
- **OCR 与表格处理**：PDF/图片使用 `qwen3.5-ocr`，失败时回退 PDF 原生文本；表格保留行级结构并生成目的地聚合记录。
- **Web UI**：Streamlit 聊天、检索来源、Agent 状态、知识库统计和记忆摘要展示。
- **旅游 MCP 工具**：提供实时天气、天气预报、地点解析和距离/路线规划。

## 系统流程

```text
用户问题
  ↓
召回长期记忆
  ↓
检查路线信息完整性
  ├─ 缺少起点/终点 → 向用户追问并等待下一轮补充
  └─ 信息完整
       ↓
动态路由
  ├─ direct：直接回答
  ├─ direct_chunk_search：全库混合检索
  ├─ hierarchical_search：摘要定位 → 文档内混合检索
  └─ web_search：实时网络搜索
       ↓
上下文感知查询改写
       ↓
BM25 + Dense → RRF → qwen3-rerank
       ↓
子块命中 → 父块返回
       ↓
文档相关性评估
  ├─ 不相关 → 切换未尝试策略
  └─ 相关
       ↓
上下文压缩 + XML 格式化
       ↓
生成答案 → 答案相关性评估
  ├─ 不相关 → 修正查询并重试（最多 2 次）
  └─ 相关/达到上限
       ↓
更新短期记忆 + 提炼长期记忆
```

> 当前主图的顺序是先动态路由、再对需要检索的问题执行查询改写。路线完整性检查位于路由之前。

## 检索架构

### 父子分块

叙事文档采用递归字符分块，依次尝试段落、换行、句子、分句、词和字符边界。

| 层级 | 默认大小 | 重叠 | 用途 |
|---|---:|---:|---|
| 父块 `doc_chunks` | 1000 字符 | 200 | 保留完整语境并提供给生成模型 |
| 子块 `retrieval_chunks` | 350 字符 | 50 | BM25、向量召回和 Rerank |

表格行不会被普通字符切分。系统还会按“目的地 + 资料类型”生成聚合记录，兼顾精确字段查询和综合行程问题。

### 混合排序

```text
BM25 候选 ─┐
            ├─ RRF 融合 → qwen3-rerank → 去重/多样性 → 父块回填
向量候选 ──┘
```

BM25 会提高目的地、景点、地区和资料类型等 metadata 字段的权重。查询包含已知目的地时，还会对稀疏和稠密检索同时应用精确 metadata 过滤。

### ChromaDB 集合

| 集合 | 用途 |
|---|---|
| `doc_summaries` | 分层检索的摘要层 |
| `doc_chunks` | 上下文父块 |
| `retrieval_chunks` | 高精度检索子块 |
| `long_term_memory` | 长期记忆向量 |

## 文档入库

支持的主要格式：

- PDF、PNG、JPG、JPEG、BMP、TIFF、WEBP、HEIC
- TXT、Markdown、Word
- XLSX、XLS、CSV

处理流程：

```text
扫描文件 → 对比 manifest → OCR/结构化解析 → 摘要
→ 父子分块 → Embedding → 批量 upsert → 清理旧记录 → 更新 manifest
```

增量判断使用文件大小与纳秒级修改时间生成指纹。未变化文件会在 OCR 前跳过；只有一个文件的全部记录处理成功后，它才会写入 `chroma_db/ingestion_manifest.json`。

如果 Embedding 模型、OCR 模型或父子分块配置发生变化，流水线签名会改变，系统会要求执行全量重建，防止不兼容向量混入同一个集合。

## 记忆策略

### 短期记忆

- 保存当前会话的 Human/AI 对话历史。
- 默认对话达到 10 轮后，每 5 轮生成一次滚动摘要。
- 摘要后保留最近 5 轮原始对话。
- CLI 和 Streamlit 都通过显式会话状态跨轮传递；进程或浏览器会话结束后不会自动恢复。

### 工作记忆

- 独立于对话历史，结构化维护当前目标、任务类型、已知事实、约束、起终点、旅行日期、已完成事项、待办和缺失信息。
- 每轮开始时结合旧工作记忆、最近对话和新输入增量更新；用户纠正或切换任务时会替换或清理旧字段。
- 参与路线信息检查、查询改写和答案生成，避免多轮补充后丢失任务参数。
- 当前所在地、单次预算等临时信息只保存在工作记忆，不进入长期记忆。
- 工作记忆随当前 CLI/Streamlit 会话保存，默认不跨进程持久化。

### 长期记忆

长期记忆元数据已通过 `MemoryRepository` 统一访问，支持 SQLite 与 MySQL；ChromaDB 仍负责语义向量索引。默认仍使用 SQLite，升级代码不会自动修改现有数据。

迁移时先配置 `.env` 中的 `MEMORY_MYSQL_*`，执行只读检查：

```bash
python scripts/migrate_memory_sqlite_to_mysql.py --dry-run
```

确认后执行幂等迁移，并在验证应用后切换后端：

```bash
python scripts/migrate_memory_sqlite_to_mysql.py
```

```dotenv
MEMORY_DB_BACKEND=mysql
```

迁移保留原始记忆 ID，不调用嵌入模型，也不重建 Chroma，因此现有向量索引可以继续使用。切回 `sqlite` 即可回滚读取后端。

- SQLite 保存正文、类型、重要性、创建时间和最近访问时间。
- ChromaDB 保存记忆向量并进行语义候选召回。
- 排序综合语义相关度、重要性和新近度。
- 每轮结束后 LLM 返回结构化 JSON，只有 `should_save=true` 的稳定事实、偏好或结论才会写入。
- 写入前执行精确文本和向量近重复检查，相似记忆只更新重要性和置信度。
- 记忆支持 `user_id`、状态、有效期、置信度、访问次数和同步状态；旧数据库启动时自动迁移。
- SQLite 采用 pending 状态配合 Chroma upsert，索引失败时回滚 SQLite，并提供双库索引修复。
- 到期记忆会标记为 expired 并从向量索引移除。
- `AGENT_USER_ID` 用于划分记忆命名空间；默认 `default`。项目尚未包含用户登录认证。

CLI 支持：

```text
!show_memories
!forget 长沙旅游
```

## 天气与地图 MCP Server

MCP Server 位于 [`mcp_server/travel_tools.py`](mcp_server/travel_tools.py)，提供：

| 工具 | 作用 |
|---|---|
| `geocode_location` | 解析城市、景点、车站和地址 |
| `get_current_weather` | 当前温度、体感温度、降水、湿度和风况 |
| `get_weather_forecast` | 未来 1～14 天天气预报 |
| `plan_route` | 直线距离、驾车、步行和公交路线 |

默认数据源：

- 地点：OpenStreetMap Nominatim
- 天气：Open-Meteo
- 驾车路线：OSRM
- 直线距离：Haversine

配置高德 Web Service Key 后，国内地点解析以及驾车、步行、公交路线会优先使用高德：

```dotenv
AMAP_API_KEY="your_amap_web_service_key"
```

启动 stdio MCP Server：

```bash
python -m mcp_server.travel_tools
```

客户端配置示例：

```json
{
  "mcpServers": {
    "agentic-rag-travel-tools": {
      "command": "D:\\agentic rag\\.venv\\Scripts\\python.exe",
      "args": ["-m", "mcp_server.travel_tools"],
      "cwd": "D:\\agentic rag\\AgentiRAG-master"
    }
  }
}
```

> MCP Server 已可被外部 MCP 客户端调用。目前 LangGraph 主流程尚未将该 MCP Server 注册为内部工具；实时问题在主流程中仍默认走 Tavily `web_search`。路线信息澄清节点已经接入主图。

## 技术栈

- Python 3.11+
- LangChain、LangGraph
- ChromaDB、SQLite
- Qwen 兼容 OpenAI API
- `text-embedding-v4`
- `qwen3-rerank`
- `qwen3.5-ocr`
- Tavily Search
- Streamlit
- Model Context Protocol Python SDK
- Open-Meteo、OpenStreetMap Nominatim、OSRM、高德 Web Service

## 项目结构

```text
AgentiRAG-master/
├── agentic_rag/
│   ├── chains.py                  # LLM、Embedding、Prompt 与结构化链
│   ├── graph.py                   # LangGraph 工作流
│   ├── hierarchical_retriever.py # BM25 + Dense + RRF + Rerank
│   ├── memory.py                  # Repository + Chroma 长期记忆业务层
│   ├── memory_repository.py       # SQLite/MySQL 数据访问实现
│   ├── nodes.py                   # 工作流节点
│   ├── ocr.py                     # qwen3.5-ocr 与 PDF 回退
│   ├── retrievers.py              # 网络搜索
│   └── state.py                   # AgentState
├── mcp_server/
│   ├── travel_tools.py            # 天气与地图 MCP Server
│   └── README.md
├── evaluation/                    # 评估脚本与样例数据
├── docs/                          # 架构说明
├── ingest.py                      # 增量知识库构建
├── main.py                        # CLI 对话入口
├── streamlit_app.py               # Agent 对话 UI
├── vector_db_admin.py             # ChromaDB 可视化管理
├── config.py
├── .env.example
└── requirements.txt
```

## 安装

```bash
git clone https://github.com/betteroneshow/agentic-rag-.git
cd agentic-rag-

python -m venv .venv
```

激活虚拟环境：

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

安装依赖：

```bash
pip install -r requirements.txt
```

## 环境变量

复制配置模板：

```bash
# Windows
Copy-Item .env.example .env

# macOS / Linux
cp .env.example .env
```

最低限度配置示例：

```dotenv
# 主 LLM（OpenAI 兼容接口）
OPENAI_API_KEY="your_key"
OPENAI_API_BASE="https://your-provider.example/v1"

# Embedding，可使用独立端点和 Key
EMBEDDING_PROVIDER="openai"
EMBEDDING_MODEL_NAME="text-embedding-v4"
EMBEDDING_API_KEY="your_embedding_key"
EMBEDDING_API_BASE="https://your-provider.example/v1"

# 网络搜索
TAVILY_API_KEY="your_tavily_key"

# 可选：高德路线服务
AMAP_API_KEY=""
```

其他 OCR、Rerank、分块、上下文压缩和短期记忆参数请参考 [`.env.example`](.env.example)。不要提交包含真实密钥的 `.env`。

## 运行

### 1. 构建知识库

将文件放入 `data/`，首次执行：

```bash
python ingest.py
```

后续仍执行同一命令，系统只处理新增或修改的文件：

```bash
python ingest.py
```

只有模型、向量维度、分块结构或入库流水线变化时才执行：

```bash
python ingest.py --rebuild
```

`--rebuild` 会删除现有 `chroma_db` 后重新构建，请谨慎使用。

### 2. CLI 对话

```bash
python main.py
```

### 3. Streamlit 对话界面

```bash
python -m streamlit run streamlit_app.py
```

浏览器访问 `http://localhost:8501`。界面支持聊天记录、短期会话、检索来源、Agent 执行状态、知识库数量和最近长期记忆展示。

### 4. 向量数据库管理

```bash
python -m streamlit run vector_db_admin.py
```

管理界面支持集合浏览、分页、关键词过滤、语义搜索、metadata 过滤和选定记录删除。

### 5. 模型连通性测试

```bash
python test_custom_model.py
```

## 配置项

| 配置 | 默认值 | 说明 |
|---|---|---|
| `EMBEDDING_PROVIDER` | `openai` | `openai` 或 `local` |
| `EMBEDDING_MODEL_NAME` | `text-embedding-v4` | 稠密检索模型 |
| `RERANK_MODEL_NAME` | `qwen3-rerank` | 候选精排模型 |
| `OCR_MODEL_NAME` | `qwen3.5-ocr` | PDF/图片 OCR 模型 |
| `DOCUMENT_CHUNK_SIZE` | `1000` | 父块字符数 |
| `DOCUMENT_CHUNK_OVERLAP` | `200` | 父块重叠 |
| `RETRIEVAL_CHUNK_SIZE` | `350` | 子块字符数 |
| `RETRIEVAL_CHUNK_OVERLAP` | `50` | 子块重叠 |
| `CONTEXT_COMPRESS_THRESHOLD` | `12000` | 触发上下文压缩的字符数 |
| `CONTEXT_TARGET_CHARS` | `8000` | 压缩目标长度 |
| `CONVERSATION_SUMMARY_START_TURNS` | `10` | 开始滚动摘要的轮数 |
| `CONVERSATION_SUMMARY_INTERVAL` | `5` | 摘要更新间隔 |
| `CONVERSATION_RECENT_TURNS` | `5` | 保留的最近原始对话轮数 |

## 容错与限制

- `qwen3-rerank` 调用失败时回退到 RRF。
- OCR 单页失败时优先回退 PDF 原生文本；仍为空则跳过该页。
- 本地检索结果不足时会切换尚未尝试的检索策略。
- 所有检索策略耗尽后返回安全兜底答案。
- 答案修正最多执行两次，图递归限制为 50。
- 当前系统是单用户本地应用，长期记忆没有用户隔离。
- 短期记忆只存在于当前 CLI 进程或 Streamlit Session。
- MCP 路线的步行和公交模式需要 `AMAP_API_KEY`；未配置时支持驾车和直线距离。
- 公开 Nominatim、Open-Meteo 和 OSRM 服务适合开发验证；生产环境应遵守服务条款、限流并考虑自建或商业服务。

## 评估

`evaluation/` 中保留了路由分类和 RAGAS 评估脚本，但当前样例黄金数据仍包含旧业务示例，不能直接代表旅游系统效果。正式评估前应替换为旅游领域数据，并分别评估：

- 查询改写的意图保持和检索提升
- 路由 Accuracy / Macro-F1 / 混淆矩阵
- 检索 Recall@K / MRR / nDCG / Context Precision
- 生成 Faithfulness / Answer Relevancy / Answer Correctness
- Agent 工具选择、目标完成率和端到端延迟
- 长期记忆误存率、重复率与召回 Precision@K

## 安全提示

- `.env`、`data/`、`chroma_db/`、本地 SQLite 和入库日志已在 `.gitignore` 中排除。
- 不要将真实 API Key、用户记忆或本地知识库提交到公开仓库。
- 删除或重建向量库前应确认目标路径并保留必要备份。

## License

当前仓库尚未声明开源许可证。在添加 License 前，默认保留所有权利。
