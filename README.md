# SmartRecipe Multi-Agent System

SmartRecipe 是一个面向菜谱问答、饮食偏好记忆、结构化查询和图片识别的 FastAPI + LangGraph 多 Agent 项目。当前开发约定是：

> 本机 conda 环境运行 Python app，Docker 只运行 MySQL、Redis、Neo4j。

这样做的好处是开发 Python 代码时不用反复重建镜像，数据库又能保持隔离、可复现、可清理。以后默认不再依赖本机安装的 MySQL、Redis、Neo4j，也不把 app 放进 Docker 里调试。

## 功能介绍

- 文本菜谱问答：通过 `/chat` 支持菜谱推荐、做法查询、食材替换、低脂高蛋白等饮食目标推荐。
- 图片菜品识别：通过 `/chat/image` 上传菜品图片，识别图像语义并检索相似菜谱。
- 多 Agent 工作流：基于 LangGraph 串联 Safety、Preference、Router、Recipe、Nutrition、SQL、Cypher、Fusion、Vision、Rerank、Answer 等 Agent。
- RAG 菜谱检索：支持本地 JSON 菜谱、BM25 关键词检索、Chroma、FAISS HNSW 等检索后端。
- MySQL 结构化查询：保存菜谱、食材关系和 PDF 文档分块/索引元数据，SQL Agent 支持按热量、时间、分类和食材进行只读 Text2SQL 查询。
- Neo4j 图谱查询：构建菜谱、食材、标签、目标、约束之间的图关系，Cypher Agent 支持只读 Text2Cypher 查询。
- 上下文预算管理：按 128K Token 总预算执行滑动窗口和增量动态摘要，预留系统提示、RAG 证据及回答空间，降低长会话冗余。
- Redis Stack 会话状态：保存多轮会话、偏好和 LangGraph Checkpoint，并以 `session_id/thread_id` 隔离不同对话。
- Fusion 多源融合：把 RAG、SQL、Cypher、GraphRAG 等结果去重、打分、合并后返回。
- Query Boundary Guard：在路由前规范化输入，识别提示词注入、危险请求、超长输入和健康敏感问题，并输出结构化边界决策。
- Query Understanding：通过"LLM 抽取意图实体 → 程序生成纠错候选 → LLM 候选裁决"恢复错别字输入的真实意图。
- 数据工程 Pipeline：RecipeParsingAgent 统一解析 JSON、JSONL、CSV、TSV、Excel、PDF、DOCX、HTML、TXT、Markdown，自动清洗、去重、记录来源并生成 Neo4j 图谱清单。
- Answer Guard：核验回答中的数字和显式实体；发现无证据声明时最多自纠错两次，仍不通过则安全降级。医疗、用药、孕期、婴幼儿和过敏问题会进一步启用健康敏感回答策略。
- Debug 接口：提供会话、数据源、评估集、Pipeline、消融实验等调试入口。
- 浏览器调试页：访问根地址 `/` 会自动进入 `/ui/`；左侧对话记录为每个对话保存独立 `session_id` 和本地消息，可新建、切换，也可点击对话右侧的"⋯"并选择"删除"。输入框旁的加号用于附加图片：有图片时自动进入 `/chat/image` 视觉流程，没有图片时自动进入 `/chat` 文本流程。

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
│  │  ├─ data_engineering_agent.py # 文件输入到结构化数据、索引和数据库的离线编排
│  │  ├─ query_understanding_agent.py # 在线输入的 LLM 意图/实体抽取与纠错裁决
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
│  │  ├─ query_boundary_guard.py # 输入规范化、边界检测和风险分类
│  │  ├─ query_understanding.py  # 领域纠错候选生成和关键约束保护
│  │  ├─ document_chunking.py    # PDF/DOCX/HTML/TXT 输入读取、OCR 和分块
│  │  ├─ recipe_chunk_refiner.py # 文档块整理为菜谱块
│  │  ├─ recipe_enrichment.py    # 菜谱字段校验、营养和类别补全
│  │  ├─ heterogeneous_recipe_pipeline.py # LLM 优化、清洗、去重、索引和数据库导入
│  │  ├─ llm_client.py           # OpenAI-compatible / Anthropic LLM 调用
│  │  ├─ embeddings.py           # 分块向量化和本地 fallback
│  │  ├─ document_faiss.py       # 文档 FAISS HNSW 索引输出
│  │  ├─ mysql_store.py          # 结构化菜谱和文档索引元数据落库
│  │  ├─ neo4j_store.py          # 菜谱图谱节点与关系落库
│  │  ├─ redis_memory.py         # Redis 会话状态输入输出
│  │  ├─ memory.py               # 内存版/Redis 版记忆统一接口
│  │  ├─ checkpoint_store.py     # LangGraph Checkpoint 持久化
│  │  ├─ context_budget.py       # 长会话上下文裁剪与摘要预算
│  │  ├─ cache_store.py          # 检索、SQL、菜名匹配和 LLM 结果缓存
│  │  ├─ graph_rag.py            # MySQL/Neo4j/RAG 图谱增强检索
│  │  ├─ image_analyzer.py       # 图片输入分析和 fallback
│  │  ├─ query_guard.py          # SQL/Cypher 安全校验
│  │  ├─ data_pipeline.py        # 菜谱清洗和评估种子生成
│  │  ├─ answer_guard.py         # 最终回答事实与健康安全校验
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
│  ├─ check.py                   # 连通性检查（mysql/redis/neo4j/llm/schema 子命令）
│  ├─ import_data.py             # 数据导入（mysql/neo4j/chunks 子命令）
│  ├─ evaluate.py                # 评估工具（chat/retrieval/router/text2sql/… 子命令）
│  ├─ build_heterogeneous_recipe_pipeline.py # 一键数据工程 Pipeline
│  ├─ build_document_index.py    # PDF/DOCX/HTML 文档索引构建
│  ├─ build_data_pipeline.py     # 菜谱清洗和评估种子生成
│  ├─ search_document_faiss.py   # 文档 FAISS 索引检索测试
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

## 启动命令

当前通过浏览器前端使用 SmartRecipe，不需要再用 `curl` 手动发送对话请求。

### 首次运行

进入项目目录、激活 conda 环境并安装依赖：

```cmd
cd /d "F:\SU\java_study\Smart Recipe"
conda activate smart_recipe
python -m pip install -r requirements.txt -i https://pypi.org/simple
```

首次准备数据库时，启动 Docker 中的 MySQL、Redis、Neo4j，然后导入菜谱和图谱数据：

```cmd
docker compose up -d mysql redis-stack neo4j
python scripts/import_data.py mysql
python scripts/import_data.py neo4j
```

MySQL 首次启动会自动执行 `docker/mysql/init.sql`。上述导入命令通常只需执行一次，已有数据时无需重复导入。

### 日常启动

以后使用前端时，只需在项目根目录执行：

```cmd
conda activate smart_recipe
docker compose up -d mysql redis-stack neo4j
python scripts/start_server.py --host 127.0.0.1 --port 8010
```

服务启动后，在浏览器打开：

- 对话前端：`http://127.0.0.1:8010/`
- 数据库预览：`http://127.0.0.1:8010/ui/database.html`
- API 文档：`http://127.0.0.1:8010/docs`

根地址会自动跳转到 `/ui/`。页面在当前浏览器的 `localStorage` 中保存对话列表和显示消息，后端按每个对话独立的 `session_id` 在 Redis 中保存会话记忆。删除前端对话时，对应的后端历史、摘要和偏好也会同步删除。

需要查看 MySQL、Redis 或 Neo4j 数据时，点击前端左侧底部的"数据库预览"。预览页面只读，不会修改数据库。

如果 `8010` 端口被占用，可以把启动命令中的端口改为 `8000`，然后访问 `http://127.0.0.1:8000/`。

## 核心原理

整体请求链路如下：

```text
用户文本/图片
  -> FastAPI
  -> LangGraph
  -> Safety Agent
  -> Query Understanding Agent
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

### 错别字识别与真实意图恢复

文本请求采用以下查询理解链路：

```text
用户输入
  -> 文本清洗（NFKC、零宽字符、空白规范化）
  -> LLM 抽取意图和实体
  -> 程序根据菜谱名、食材、标签、饮食目标和常见错字生成候选
  -> LLM 只能从候选编号中选择最合理结果
  -> 置信度与关键约束校验
  -> 再次执行查询边界检查
  -> Router / MySQL / Neo4j / RAG / Answer
```

第一次 LLM 调用只输出意图、实体及规范化实体；程序随后使用领域词典、常见错字表和字符串相似度生成最多 8 个候选。第二次 LLM 调用不能自由改写，只能返回候选编号、置信度和原因。候选达到 `QUERY_UNDERSTANDING_MIN_CONFIDENCE` 后才会写入下游 `user_input`，原始文本始终保存在 `meta.original_query`。

程序禁止修改否定、过敏、疾病、药品、数字和单位。纠错后的文本还会再次经过 Query Boundary Guard；如果错字还原后变成危险请求，工作流会立即阻断，不进入 Router、数据库或 RAG。

如果输入只是一个菜名，但附着了无意义的字母、数字或符号，例如 `###红晒肘子123abc!!!`，系统会先纠正为 `红烧肘子`，再启用 `dish_name_only` 模式。该模式的最终回答严格只输出纯菜名，不添加"识别结果"、置信度、推荐语、标点或其他说明。`推荐2道菜`、`20分钟以内` 等具有实际语义的数量不会被清除。

如果噪声字符夹在菜名内部且用户仍带有明确问题，例如 `西红柿c炒鸡蛋怎么做` 或 `红烧123肘子怎么做`，程序会先生成确定性的内嵌噪声清理候选，分别恢复为 `西红柿炒鸡蛋怎么做` 和 `红烧肘子怎么做`，再交给 Router。已知完整词不会被模糊匹配错误扩展，例如不会把其中的"鸡蛋"改成"鸡蛋羹"。

字符噪声和错别字同时出现时会串行合并处理，而不是只执行其中一步：`红晒c肘子` 会先去掉 `c`，再把 `红晒` 纠正为 `红烧`，最终只输出 `红烧肘子`；`红晒c肘子怎么做` 则恢复为 `红烧肘子怎么做` 后进入菜谱详情路由。

`/chat` 响应的 `meta.query_understanding` 提供抽取意图、实体、程序候选、选中编号、最终置信度、纠错状态和 `resolved_query`。LLM 不可用时会退回高置信度本地词典纠错。

### 安全防御与回答策略

聊天链路在路由前执行 Query Boundary Guard，对输入做 Unicode NFKC 规范化、零宽字符清理和空白压缩，再输出 `allow`、`caution` 或 `block` 结构化决策。提示词注入、明确危险请求和超长输入会被阻断并直接返回安全拒答，不再进入 Router、检索或数据库 Agent。

医疗、用药、孕期、婴幼儿和过敏问题会进入 `health_sensitive_v1` 回答策略：

- 生成前：限制疾病诊断、疗效承诺、停药改药以及使用饮食替代处方或治疗；
- 生成后：强制追加一般饮食参考、专业咨询、配料标签和交叉污染提示；
- 高风险改写：出现"建议停药""无需就医""替代处方""根治"或"保证治愈"等断言时，直接替换为安全回答；
- 幻觉检测：核验回答中的数字和显式实体，发现无证据声明时最多自纠错两次，重试耗尽后只返回安全降级结果。

`/chat` 响应的 `meta` 会暴露实际判定和处理动作：

- `query_boundary`：边界决策、风险类型、置信度和原因码；
- `normalized_query`：规范化后的查询文本；
- `response_policy`：实际回答策略，例如 `standard` 或 `health_sensitive_v1`；
- `response_policy_action`：追加健康提示或替换高风险声明等动作；
- `answer_guard_initial_status`：第一次回答校验结果；
- `answer_guard_retry_count`：实际纠错次数；
- `unsupported_claims`：未被当前证据支持的声明；
- `answer_guard`：最终校验或安全降级状态。

### RAG 检索策略

SmartRecipe 把"召回"和"精排"分开处理：

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

## 数据工程：智能菜谱解析与异构清洗 Pipeline

`RecipeParsingAgent` 提供离线数据工程能力，统一处理 `.json`、`.jsonl/.ndjson`、`.csv/.tsv`、`.xlsx/.xlsm`、`.pdf`、`.docx`、`.html/.htm`、`.txt` 和 `.md`。PDF、Word、HTML 和纯文本复用现有 Document Chunking 与 Recipe Chunk Refiner；PDF 在文本层不足时自动进入 OCR，清洗与 FAISS 构建共用首次提取的文本，不会重复 OCR。Excel 通过 Python 标准库直接读取 XLSX/XML，不要求额外安装 `openpyxl`。

处理链路：

```text
异构文件发现
  -> 格式读取与文档分块
  -> RecipeParsingAgent 字段映射和规则草稿
  -> 默认 LLM 逐块纠正 OCR 与优化结构
  -> 菜名/原料/调料/步骤抽取
  -> 用量标准化与原始值保留
  -> 规则清洗与质量拒绝
  -> 跨来源菜名去重与 provenance 合并
  -> FAISS 索引与图谱节点/关系清单
  -> 默认写入 MySQL 和批量 MERGE 到 Neo4j
```

LLM 默认处理每条分块后的菜谱，用于纠正 OCR 错字、清理噪声并优化固定 JSON 结构；LLM 输出仍必须通过确定性规则校验，不得编造输入中不存在的菜名、关键食材、用量或步骤。LLM 不可用或返回无效结构时保留规则解析结果，可通过 `--no-llm` 主动关闭。每个清洗记录都保存 `source`、`source_type`、`record_id` 和页码，能够从图谱节点反查原始文件。

图谱不是只建立 Recipe、Ingredient 等少量去重节点，还包含可追溯的 `SourceRecord`、`IngredientUse`、`SeasoningUse`、`RecipeStep`、`NutritionFact` 和 `CookingProfile`。主要关系包括 `CONTAINS`、`DESCRIBES`、`HAS_INGREDIENT_USE`、`USES_INGREDIENT`、`HAS_SEASONING_USE`、`USES_SEASONING`、`HAS_STEP`、`NEXT_STEP`、`HAS_NUTRITION` 等。

## 数据与存储

当前开发模式下，数据来源和存储职责如下：

| 组件 | 运行位置 | 用途 |
|---|---|---|
| FastAPI app | 本机 conda 环境 | 开发、调试、接口服务。 |
| MySQL | Docker | 保存菜谱、食材关系，以及 PDF 文档分块和索引元数据。 |
| Redis Stack | Docker | 按 `session_id` 保存偏好、多轮会话、动态摘要及 LangGraph Checkpoint；不可用时会回退。 |
| Neo4j | Docker | 保存菜谱和食材、标签、目标之间的图关系。 |
| JSON/向量索引 | 本机项目目录 | RAG 原始数据和本地检索缓存。 |

当前 MySQL 保留以下 5 张业务表：

| 表名 | 用途 |
|---|---|
| `recipes` | 菜名、分类、耗时分钟数、难度、每100克热量/蛋白质/脂肪和制作步骤。 |
| `ingredients` | 规范化食材名称。 |
| `recipe_ingredients` | 菜谱与食材的多对多关系及用量。 |
| `document_indexes` | PDF 文档索引文件、embedding 后端和 HNSW 配置。 |
| `document_chunks` | PDF 分块正文、页码和结构化 metadata。 |

会话偏好不写入 MySQL，而是由 Redis/内存按 `session_id` 隔离保存；标签、目标和约束关系仍保存在 Neo4j 图谱中。

默认端口：

| 服务 | 地址 |
|---|---|
| SmartRecipe 对话界面 | `http://127.0.0.1:8010/`（自动跳转到 `/ui/`） |
| MySQL | `127.0.0.1:3307` |
| Redis | `127.0.0.1:6379` |
| Neo4j Browser | `http://127.0.0.1:7474` |
| Neo4j Bolt | `bolt://127.0.0.1:7687` |

MySQL 使用宿主机 `3307` 是为了避开很多电脑上本机 MySQL 已占用的 `3306`。

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
QUERY_UNDERSTANDING_ENABLED=true
QUERY_UNDERSTANDING_MIN_CONFIDENCE=0.80

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
CHECKPOINT_BACKEND=redis
REDIS_URL=redis://127.0.0.1:6379/0
REDIS_TTL_SECONDS=604800
CHECKPOINT_TTL_SECONDS=604800
REDIS_USERNAME=
REDIS_PASSWORD=<REDIS_PASSWORD>

CONTEXT_TOKEN_BUDGET=131072
CONTEXT_SYSTEM_RESERVE_TOKENS=12288
CONTEXT_RETRIEVAL_RESERVE_TOKENS=24576
CONTEXT_OUTPUT_RESERVE_TOKENS=4096
CONTEXT_SAFETY_RESERVE_TOKENS=4096
CONTEXT_SUMMARY_MAX_TOKENS=8192
CONTEXT_COMPACTION_TRIGGER_RATIO=0.85
CONTEXT_RECENT_WINDOW_RATIO=0.55
CONTEXT_MIN_RECENT_MESSAGES=6

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

当前推荐配置是全能力开启：数据库 Agent、LLM Text2SQL/Text2Cypher、视觉 LLM、FAISS HNSW、BGE embedding、LLM HyDE、BM25、Cross-Encoder rerank、128K 上下文预算、Redis Stack Checkpoint、Redis 记忆和缓存一起使用。`.env` 里需要补充真实文本模型和视觉模型 API Key；如果 LLM 不可用，动态摘要会使用确定性压缩兜底，其他 LLM 能力也会自然降级。
