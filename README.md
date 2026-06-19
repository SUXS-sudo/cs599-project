# SmartRecipe Multi-Agent System

SmartRecipe 是一个面向菜谱问答、饮食偏好记忆、结构化查询和图片识别的 FastAPI + LangGraph 多 Agent 项目。当前开发约定是：

> 本机 conda 环境运行 Python app，Docker 只运行 MySQL、Redis、Neo4j。

这样做的好处是开发 Python 代码时不用反复重建镜像，数据库又能保持隔离、可复现、可清理。以后默认不再依赖本机安装的 MySQL、Redis、Neo4j，也不把 app 放进 Docker 里调试。

## 功能介绍

- 文本菜谱问答：通过 `/chat` 支持菜谱推荐、做法查询、食材替换、低脂高蛋白等饮食目标推荐。
- 图片菜品识别：通过 `/chat/image` 上传菜品图片，识别图像语义并检索相似菜谱。
- 多 Agent 工作流：基于 LangGraph 串联 Safety、Preference、Router、Recipe、Nutrition、SQL、Cypher、Fusion、Vision、Rerank、Answer 等 Agent。
- RAG 菜谱检索：支持本地 JSON 菜谱、BM25 关键词检索、Chroma、FAISS HNSW 等检索后端。
- MySQL 结构化查询：保存菜谱、食材、标签、适用人群等结构化数据，SQL Agent 支持只读 Text2SQL 查询。
- Neo4j 图谱查询：构建菜谱、食材、标签、目标、约束之间的图关系，Cypher Agent 支持只读 Text2Cypher 查询。
- Redis 会话记忆：保存多轮会话、用户偏好、过敏和忌口信息。
- Fusion 多源融合：把 RAG、SQL、Cypher、GraphRAG 等结果去重、打分、合并后返回。
- Answer Guard：在证据不足、检索缺失或图片识别不确定时自动降级为谨慎回答。
- Debug 接口：提供会话、数据源、评估集、Pipeline、消融实验等调试入口。
- 浏览器调试页：通过 `/ui` 直接测试文本问答和图片上传。
- 自动化测试与评估脚本：覆盖数据集、Agent 流程、Debug API、GraphRAG、Vision Pipeline、Docker 配置等。

## 项目结构

```text
Smart Recipe/
├─ app/
│  ├─ main.py                    # FastAPI 入口，注册 /chat、/chat/image、/health、/debug、/ui
│  ├─ graph.py                   # LangGraph 多 Agent 工作流编排
│  ├─ state.py                   # Agent 之间传递的状态结构
│  ├─ models.py                  # API 请求和响应模型
│  ├─ retriever.py               # 菜谱 RAG 检索器
│  ├─ agents/
│  │  ├─ router_agent.py         # 意图路由
│  │  ├─ recipe_agent.py         # 菜谱推荐和做法查询
│  │  ├─ nutrition_agent.py      # 营养分析
│  │  ├─ sql_agent.py            # MySQL/Text2SQL 查询
│  │  ├─ cypher_agent.py         # Neo4j/Text2Cypher 查询
│  │  ├─ fusion_agent.py         # 多源候选融合
│  │  ├─ vision_agent.py         # 图片识别结果接入
│  │  ├─ rerank_agent.py         # 候选重排
│  │  ├─ answer_agent.py         # 最终回答生成
│  │  └─ support_agents.py       # Safety、General 等辅助 Agent
│  ├─ services/
│  │  ├─ mysql_store.py          # MySQL 连接和菜谱结构化查询
│  │  ├─ neo4j_store.py          # Neo4j 图谱连接和查询
│  │  ├─ redis_memory.py         # Redis 会话记忆
│  │  ├─ memory.py               # 内存版/Redis 版记忆统一接口
│  │  ├─ graph_rag.py            # 图谱增强检索
│  │  ├─ image_analyzer.py       # 图片分析和 fallback
│  │  ├─ embeddings.py           # embedding provider 和本地 fallback
│  │  ├─ llm_client.py           # OpenAI-compatible / Anthropic LLM 封装
│  │  ├─ cache_store.py          # Redis/内存缓存：检索、SQL、菜名匹配和 LLM 兜底
│  │  ├─ answer_guard.py         # 回答可信度保护
│  │  ├─ query_guard.py          # SQL/Cypher 安全校验
│  │  ├─ data_pipeline.py        # 菜谱清洗和评估种子生成
│  │  ├─ ablation.py             # 消融实验
│  │  └─ logger.py               # 日志配置
│  └─ static/
│     ├─ index.html              # /ui 调试页面
│     └─ database.html           # /ui/database.html 数据库只读浏览页面
├─ data/
│  ├─ recipes.json               # 原始菜谱数据
│  ├─ processed/recipes_clean.json
│  ├─ evals/                     # 路由、检索、SQL、Cypher、偏好记忆等评估集
│  └─ test_images/               # 图片接口测试样例
├─ scripts/
│  ├─ init_mysql_schema.py       # 初始化 MySQL schema
│  ├─ import_recipes_to_mysql.py # 导入菜谱到 MySQL
│  ├─ import_recipes_to_neo4j.py # 导入菜谱图谱到 Neo4j
│  ├─ check_mysql_store.py       # MySQL 连通性检查
│  ├─ check_neo4j_graph.py       # Neo4j 连通性检查
│  ├─ check_redis_memory.py      # Redis 连通性检查
│  ├─ check_llm.py               # 文本模型和视觉模型连通性检查
│  ├─ rebuild_pdf1_pipeline.py   # 重建 pdf\1.pdf 的索引、MySQL 和 Neo4j 数据
│  ├─ evaluate_*.py              # 各类评估脚本
│  └─ run_ablation.py            # 消融实验脚本
├─ tests/                        # pytest 自动化测试
├─ docker/
│  └─ mysql/init.sql             # Docker MySQL 首次初始化 SQL
├─ docker-compose.yml            # Docker MySQL、Redis、Neo4j 服务
├─ requirements.txt              # conda 环境安装的 Python 依赖
├─ .env                          # 本机 conda app 使用的真实配置，不提交
├─ .env_example                  # 可提交的本地开发配置模板
└─ pytest.ini
```

## 核心原理

整体请求链路如下：

```text
用户文本/图片
  -> FastAPI
  -> LangGraph
  -> Safety Agent
  -> Preference Agent
  -> Router Agent
  -> Recipe / Nutrition / SQL / Cypher / Fusion / Vision / General Agent
  -> Rerank Agent
  -> Answer Agent
  -> Answer Guard
  -> 返回结果
```

文本请求进入 `/chat` 后，系统先做安全检查和偏好提取，再由 Router 判断用户意图。普通推荐走 Recipe + RAG；营养问题走 Nutrition；结构化统计走 SQL；关系型问题走 Cypher；复杂问题可走 Fusion，把多个来源的候选融合成统一结果。

图片请求进入 `/chat/image` 后，系统先分析图片内容，得到菜品、食材或描述，再把视觉结果转换成可检索文本，交给 Vision/Fusion/RAG 找相似菜谱。这样图片能力不是孤立模块，而是复用已有的菜谱检索和回答生成能力。

## Agent 分工

| Agent | 作用 |
|---|---|
| Safety Agent | 拦截明显越界输入，对高风险健康问题给出谨慎提示。 |
| Preference Agent | 从多轮对话里提取口味、忌口、过敏、目标等偏好。 |
| Router Agent | 判断用户意图，把问题分流到对应 Agent。 |
| Recipe Agent | 处理菜谱推荐、做法查询、食材替换。 |
| Nutrition Agent | 处理热量、蛋白质、低脂、高蛋白等营养相关问题。 |
| SQL Agent | 面向 MySQL 做只读结构化查询。 |
| Cypher Agent | 面向 Neo4j 做只读图谱查询。 |
| Fusion Agent | 合并 RAG、SQL、Cypher、GraphRAG 等多源结果。 |
| Vision Agent | 接入图片识别结果并转换为菜谱检索问题。 |
| Rerank Agent | 对候选菜谱进行二次排序。 |
| Answer Agent | 统一组织最终自然语言回答。 |

## 数据与存储

当前开发模式下，数据来源和存储职责如下：

| 组件 | 运行位置 | 用途 |
|---|---|---|
| FastAPI app | 本机 conda 环境 | 开发、调试、接口服务。 |
| MySQL | Docker | 保存菜谱、食材、标签、适用人群等结构化数据。 |
| Redis | Docker | 保存 session、偏好和多轮记忆。 |
| Neo4j | Docker | 保存菜谱和食材、标签、目标之间的图关系。 |
| JSON/向量索引 | 本机项目目录 | RAG 原始数据和本地检索缓存。 |

默认端口：

| 服务 | 地址 |
|---|---|
| FastAPI | `http://127.0.0.1:8010` |
| MySQL | `127.0.0.1:3307` |
| Redis | `127.0.0.1:6379` |
| Neo4j Browser | `http://127.0.0.1:7474` |
| Neo4j Bolt | `bolt://127.0.0.1:7687` |

MySQL 使用宿主机 `3307` 是为了避开很多电脑上本机 MySQL 已占用的 `3306`。

## 环境配置

`.env` 使用本机访问 Docker 服务的配置。不要把真实 API Key 提交到仓库。

```env
SMART_RECIPE_PROVIDER=openai
BASE_URL=https://api.deepseek.com
API_KEY=<DS_KEY>
MODEL=deepseek-chat

VISION_PROVIDER=anthropic
VISION_BASE_URL=https://token-plan-sgp.xiaomimimo.com/anthropic
VISION_API_KEY=<MIMO_KEY>
VISION_MODEL=mimo-v2.5

ENABLE_DATABASE_AGENTS=true
ENABLE_LLM_QUERY_GENERATION=true
ENABLE_VISION_LLM=true

RAG_BACKEND=faiss
# RAG_BACKEND=chroma 或 faiss 时启用混合召回。
# faiss 后端使用 HNSW 内积索引，并叠加 LLM HyDE + BM25 关键词召回。
RAG_HYBRID_VECTOR_WEIGHT=0.65
RAG_HYBRID_KEYWORD_WEIGHT=0.35
HYDE_ENABLED=true
HYDE_MAX_CHARS=420
ENABLE_LLM_TOOL_PLANNER=true
TOOL_PLANNER_MAX_TOKENS=1200
TOOL_PLANNER_TIMEOUT=30
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=models/bge-small-zh-v1.5
RERANK_ENABLED=true
RERANK_CROSS_ENCODER_MODEL=models/BAAI/bge-reranker-base

CACHE_ENABLED=true
CACHE_BACKEND=redis
CACHE_DATA_VERSION=recipes-v1
CACHE_RETRIEVAL_TTL_SECONDS=86400
CACHE_SQL_TTL_SECONDS=86400
CACHE_RECIPE_MATCH_TTL_SECONDS=86400
CACHE_LLM_FALLBACK_TTL_SECONDS=86400

MEMORY_BACKEND=redis
REDIS_URL=redis://127.0.0.1:6379/0
REDIS_TTL_SECONDS=604800
REDIS_USERNAME=
REDIS_PASSWORD=

MYSQL_HOST=127.0.0.1
MYSQL_PORT=3307
MYSQL_USER=root
MYSQL_PASSWORD=123456
MYSQL_DATABASE=smart_recipe
MYSQL_SSL_DISABLED=true

NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=12345678
NEO4J_DATABASE=neo4j
```

当前推荐配置是全能力开启：数据库 Agent、LLM Text2SQL/Text2Cypher、视觉 LLM、FAISS HNSW、BGE embedding、LLM HyDE、BM25、Cross-Encoder rerank、Redis 记忆和 Redis 缓存一起使用。`.env` 里需要补充真实文本模型和视觉模型 API Key；如果 LLM 不可用，HyDE、LLM 兜底和视觉 LLM 会自然降级，但文档主线统一按全能力开启书写。

## 当前检索策略

SmartRecipe 现在把“召回”和“精排”分开处理：

```text
用户问题
  -> 快慢分流 Router
  -> 菜名解析：标准菜名精确匹配 -> 别名精确匹配 -> 模糊候选匹配 -> 向量候选匹配
  -> 简单命中：SQL / 本地菜谱 fast path 直接结构化输出
  -> 复杂条件：SQL / RAG / GraphRAG / Fusion 多源检索
  -> 规则 Query Rewrite（文档 PDF RAG）
  -> LLM HyDE 生成假设检索文本（仅用于向量召回）
  -> Chroma/FAISS 多路向量召回：原始问题 + 扩展问题 + HyDE 文本
  -> BM25 关键词召回
  -> 候选去重和分数融合
  -> Cross-Encoder 精排
  -> Answer Agent 生成最终回答
  -> 本地库空结果：不缓存空结果，转 LLM 兜底并声明不是本地菜谱库内容
```

- `RAG_BACKEND=chroma` 或 `RAG_BACKEND=faiss` 时使用混合检索：BGE 向量召回 + LLM HyDE 向量召回 + BM25 关键词召回。
- `RAG_BACKEND=faiss` 当前使用 FAISS HNSW 内积索引，不再使用 `IndexFlatIP` 暴力精确索引；默认参数为 `M=32`、`efConstruction=200`、`efSearch=64`。
- 本项目当前推荐 `RAG_BACKEND=faiss`，即默认走本地 BGE + FAISS HNSW + BM25 的混合召回。
- `EMBEDDING_MODEL` 用于向量召回，建议指向本地 BGE embedding 模型，例如 `models/bge-small-zh-v1.5`。
- `HYDE_ENABLED=true` 时会调用默认文本 `MODEL` 生成 HyDE 文本；推荐让 `SMART_RECIPE_PROVIDER` / `BASE_URL` / `API_KEY` / `MODEL` 指向 DS。LLM 不可用或返回空时会跳过 HyDE，不会使用模板文本兜底。
- HyDE 文本只参与 Chroma/FAISS 召回，不作为最终回答证据。
- `RERANK_ENABLED=true` 时启用 Cross-Encoder 精排；当前不再使用单独的 Bi-Encoder rerank。
- `CACHE_BACKEND=redis` 时缓存非空检索结果、非空 SQL 结果、菜名匹配命中和 LLM 兜底答案，默认 TTL 为 1 天；空 SQL / 空 RAG / 菜名 miss 不写缓存。

## 启动命令

以下命令默认在项目根目录执行：

```cmd
cd "F:\SU\java_study\Smart Recipe"
conda activate smart_recipe
```

### 1. 安装 conda 环境依赖

```cmd
python -m pip install -r requirements.txt -i https://pypi.org/simple
```

如果只缺图片上传依赖：

```cmd
python -m pip install "python-multipart>=0.0.9" -i https://pypi.org/simple
```

如果 MySQL 8 报错提示需要 `cryptography package is required`，执行：

```cmd
python -m pip install cryptography -i https://pypi.org/simple
```

### 2. 启动 Docker 数据库

只启动 MySQL、Redis、Neo4j：

```cmd
docker compose up -d mysql redis neo4j
```

确认容器状态：

```cmd
docker compose ps
```

### 3. 查看 Docker 数据库可视化界面

如果只想打开内置页面查看 MySQL、Redis、Neo4j 里的数据，按下面顺序执行：

```cmd
cd /d "F:\SU\java_study\Smart Recipe"
conda activate smart_recipe
docker compose up -d mysql redis neo4j
python -m uvicorn app.main:app --host 127.0.0.1 --port 8010 --reload
```

然后浏览器打开：

```cmd
http://127.0.0.1:8010/ui/database.html
```

页面会读取 `.env` 里的 `MYSQL_*`、`REDIS_URL`、`NEO4J_*` 配置，左侧选择 MySQL 表、Redis key 或 Neo4j 节点标签，右侧预览数据。这个页面是只读预览，不会修改数据库。

如果要先确认容器状态：

```cmd
docker compose ps
```

如果 `8010` 端口被占用，可以换成 `8000`：

```cmd
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

对应打开：

```cmd
http://127.0.0.1:8000/ui/database.html
```

### 4. 初始化和导入数据

MySQL 首次启动会执行 `docker/mysql/init.sql`。如果需要重新导入菜谱：

```cmd
python scripts/import_recipes_to_mysql.py
```

导入 Neo4j 图谱：

```cmd
python scripts/import_recipes_to_neo4j.py
```

检查三个 Docker 服务是否能被 conda app 访问：

```cmd
python scripts/check_mysql_store.py
python scripts/check_redis_memory.py
python scripts/check_neo4j_graph.py
```

检查文本模型和视觉模型是否可用：

```cmd
python scripts/check_llm.py --image "data/images/地三鲜.png"
```

只检查文本模型：

```cmd
python scripts/check_llm.py --text-only
```

只检查视觉模型：

```cmd
python scripts/check_llm.py --vision-only --image "data/images/地三鲜.png"
```

### 5. 启动 FastAPI app

```cmd
python -m uvicorn app.main:app --host 127.0.0.1 --port 8010 --reload
```

启动后访问：

- API 文档：`http://127.0.0.1:8010/docs`
- 调试页面：`http://127.0.0.1:8010/ui`
- 数据库浏览页面：`http://127.0.0.1:8010/ui/database.html`
- 健康检查：`http://127.0.0.1:8010/health`

## 接口测试命令

下面示例都是单行 `cmd` 命令。

### 健康检查
```cmd
curl http://127.0.0.1:8010/health
```

### Debug 状态
```cmd
curl http://127.0.0.1:8010/debug/stats
curl http://127.0.0.1:8010/debug/evaluation
curl http://127.0.0.1:8010/debug/pipeline
```

### 查看 Agent 日志
每个 LangGraph 节点执行完成后都会输出一行中文日志，包含 Agent 名称、耗时、意图、目标 Agent 和该节点的输出摘要。Windows PowerShell 查看日志时建议显式指定 UTF-8：

```powershell
Get-Content .\logs\smart_recipe.log -Encoding UTF8 -Tail 50
```

### 文本问答
```cmd
curl -X POST http://127.0.0.1:8010/chat -H "Content-Type: application/json" -d "{\"message\":\"推荐三道低脂高蛋白晚餐\",\"session_id\":\"demo\",\"top_k\":3}"
```

### 图片问答
```cmd
curl -X POST http://127.0.0.1:8010/chat/image -F "session_id=demo-image" -F "top_k=3" -F "message=这张图片里是什么菜，推荐相似菜谱" -F "image=@data/test_images/tomato_eggs_api_test.png"
```

### 查看会话调试信息
```cmd
curl http://127.0.0.1:8010/debug/session/demo
```

## 测试与评估命令

运行 pytest：

```cmd
python -m pytest
```

只跑关键测试：

```cmd
python -m pytest tests/test_debug_api.py tests/test_fusion_agent.py tests/test_graph_rag.py
```

运行 RAG 检索评估：

```cmd
python scripts/evaluate_retrieval.py
```

### 文档 PDF RAG 构建与测试

扫描版菜谱 PDF 可以走 OCR、文档清洗、菜谱版面解析、结构化整理、智能切块、metadata 保存、向量化和 FAISS HNSW 索引构建。当前本地默认处理目录里的 `pdf\1.pdf`，推荐用一键脚本重建整条 PDF 数据管线。

先启动数据库容器，并确认 conda 环境已激活：

```cmd
conda activate smart_recipe
docker compose up -d mysql redis neo4j
```

建议先试跑构建流程，不写入数据库：

```cmd
python scripts\rebuild_pdf1_pipeline.py --dry-run
```

确认 OCR、分块和索引生成正常后，执行完整重建：

```cmd
python scripts\rebuild_pdf1_pipeline.py
```

这个脚本会重新读取 `pdf\1.pdf`，强制 OCR，按菜谱结构分块，建立 FAISS HNSW 索引，并清空后重新写入：

```text
data\processed\1_recipe_chunks.jsonl
data\processed\1_recipe.index
data\processed\1_recipe_metadata.json

MySQL: document_indexes, document_chunks
MySQL: recipes, ingredients, recipe_ingredients, recipe_tags, recipe_suitable_for
Neo4j: Recipe, Ingredient, Tag, Constraint, Goal, MealTime 及关系
```

如果只想重建 PDF 文档 RAG，不想把 PDF 解析出的菜谱导入 `recipes` 表和 Neo4j：

```cmd
python scripts\rebuild_pdf1_pipeline.py --skip-structured-recipes
```

注意：`document_indexes` / `document_chunks` 保存 PDF chunk 正文、metadata 和本地 FAISS 路径；`recipes` 等结构化表保存“一道菜一行”的规范菜谱；Neo4j 保存菜谱、食材、标签、目标之间的图关系。不要把原始 chunk 文本直接当作 `recipes` 表的长期结构化数据，脚本会先从 chunk metadata 转成标准 recipe 行再导入。

如果需要手工分步执行，底层命令仍然可用：

```cmd
python scripts\build_document_index.py "pdf\1.pdf" --ocr --ocr-force --recipe-refine --ocr-engine rapidocr --ocr-dpi 160 --ocr-progress-every 5 --chunk-size 800 --chunk-overlap 120 --output-prefix 1 --preview 8
python scripts\import_document_chunks_to_mysql.py --metadata data\processed\1_recipe_metadata.json --index data\processed\1_recipe.index --reset
```

第二条命令的 `--reset` 只清空 `document_indexes` 和 `document_chunks`，不会影响 `recipes`、`ingredients` 等菜谱结构化表。

测试文档 RAG 检索效果。默认会先做菜谱 query 优化，然后执行多路召回与 Cross-Encoder 重排：`original_query + expanded_query + llm_hyde_text` 用于 FAISS 多路向量召回，`original_query + expanded_query + keywords` 用于 BM25 关键词召回，metadata 只从 `original_query` 抽取，Cross-Encoder 和最终答案生成都使用 `original_query`。

```cmd
python scripts\search_document_faiss.py --index data\processed\1_recipe.index --metadata data\processed\1_recipe_metadata.json --query "老醋花生米怎么做" --top-k 5 --preview-chars 700
```

当前默认使用完整混合检索。需要显式确认 HyDE 开启时，可以加 `--hyde`：

```cmd
python scripts\search_document_faiss.py --index data\processed\1_recipe.index --metadata data\processed\1_recipe_metadata.json --query "老醋花生米怎么做" --top-k 5 --preview-chars 700 --hyde
```

如果要显式指定 Cross-Encoder 本地模型：

```cmd
python scripts\search_document_faiss.py --index data\processed\1_recipe.index --metadata data\processed\1_recipe_metadata.json --query "老醋花生米怎么做" --top-k 5 --preview-chars 700 --cross-encoder-model "F:\SU\java_study\Smart Recipe\models\BAAI\bge-reranker-base"
```

主要输出文件：

```text
data\processed\<PDF文件名>_recipe_chunks.jsonl      # 一道菜一个结构化 chunk
data\processed\<PDF文件名>_recipe.index             # FAISS HNSW 向量索引
data\processed\<PDF文件名>_recipe_metadata.json     # chunk metadata、embedding 信息和 index_type/hnsw 配置
```

运行 Router 评估：

```cmd
python scripts/evaluate_router.py
```

运行 SQL/Cypher 评估：

```cmd
python scripts/evaluate_text2sql.py
python scripts/evaluate_text2cypher.py
```

运行消融实验：

```cmd
python scripts/run_ablation.py
```

## 常见问题

### `Form data requires "python-multipart" to be installed`

`/chat/image` 需要 `python-multipart`。执行：

```cmd
python -m pip install "python-multipart>=0.0.9" -i https://pypi.org/simple
```

### MySQL 端口 3306 被占用

项目的 Docker MySQL 已映射到宿主机 `3307`：

```yaml
ports:
  - "3307:3306"
```

所以 `.env` 里应使用：

```env
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3307
```

### 模型文件路径怎么配置

当前推荐直接使用完整混合检索，请把 embedding 和 rerank 模型放到本地 `models/` 后使用：

```env
RAG_BACKEND=faiss
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=models/bge-small-zh-v1.5
HYDE_ENABLED=true
RERANK_ENABLED=true
RERANK_CROSS_ENCODER_MODEL=models/BAAI/bge-reranker-base
```

`RAG_BACKEND=faiss` 使用 FAISS HNSW 内积索引，适合更多 chunk 的近似向量召回；`RAG_BACKEND=chroma` 也使用同一套混合召回策略。关键词召回统一使用 BM25。HyDE 会调用 LLM 生成假设答案文本，只参与 Chroma/FAISS 的向量召回；LLM 不可用或返回空时会自动跳过，不使用模板兜底。Cross-Encoder 使用 `RERANK_CROSS_ENCODER_MODEL` 指向的本地模型做最终精排。

### 我还需要 conda 吗

需要。当前约定是 conda 跑 app，Docker 跑数据库：

- Python 包安装在 conda 环境里。
- MySQL、Redis、Neo4j 的软件和数据卷在 Docker 里。
- 开发和改代码都在本机项目目录完成。
- 不需要再单独启动本机 MySQL、Redis、Neo4j。

### 如何停止环境

停止 FastAPI：在运行 uvicorn 的终端按 `Ctrl+C`。

停止 Docker 数据库：

```cmd
docker compose stop mysql redis neo4j
```

如果要删除数据库容器但保留数据卷：

```cmd
docker compose down
```

如果要彻底清空 Docker 数据库数据卷，先确认不需要里面的数据，再执行：

```cmd
docker compose down -v
```
