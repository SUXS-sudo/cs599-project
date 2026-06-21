# SmartRecipe Multi-Agent System

## 项目简介

SmartRecipe 是一个基于 LangGraph 多 Agent 架构的智能菜谱问答系统，解决传统菜谱检索只能做关键词匹配、无法理解用户复杂意图和饮食偏好记忆的问题。系统支持文本问答、图片菜品识别、结构化数据库查询、知识图谱查询和多轮偏好记忆，能够根据用户的口味、忌口、过敏和健康目标提供个性化菜谱推荐。

## 方向

方向一：Agentic AI 原生开发

## 技术栈

- AI IDE: Trae CN
- LLM: DeepSeek API (deepseek-chat) + Anthropic API (视觉模型)
- 框架: LangGraph + FastAPI
- 容器: Docker (MySQL、Redis、Neo4j)
- 向量检索: FAISS HNSW + BGE Embedding
- 关键词检索: BM25
- 精排: Cross-Encoder (bge-reranker-base)
- 知识图谱: Neo4j (Text2Cypher)
- 结构化查询: MySQL (Text2SQL)
- 会话记忆: Redis Stack + LangGraph Checkpoint
- 上下文工程: 128K Token 预算、滑动窗口、动态摘要
- 测试: pytest

## 目录结构

```
Smart Recipe/
├─ app/
│  ├─ main.py              # FastAPI 入口，注册 /chat、/chat/image、/health、/debug、/ui
│  ├─ graph.py             # LangGraph 多 Agent 工作流编排
│  ├─ state.py             # Agent 之间传递的状态结构
│  ├─ models.py            # API 请求和响应模型
│  ├─ retriever.py         # 菜谱 RAG 检索器
│  ├─ agents/              # 各 Agent 实现
│  │  ├─ router_agent.py   # 意图路由，将问题分流到对应 Agent
│  │  ├─ recipe_agent.py   # 菜谱推荐、做法查询、食材替换
│  │  ├─ nutrition_agent.py# 热量、蛋白质等营养分析
│  │  ├─ sql_agent.py      # MySQL Text2SQL 只读查询
│  │  ├─ cypher_agent.py   # Neo4j Text2Cypher 只读查询
│  │  ├─ fusion_agent.py   # 多源候选融合 (RAG + SQL + Cypher + GraphRAG)
│  │  ├─ answer_agent.py   # 最终自然语言回答生成
│  │  └─ ...
│  ├─ services/            # 基础设施服务
│  │  ├─ mysql_store.py    # MySQL 连接与查询
│  │  ├─ neo4j_store.py    # Neo4j 图谱连接与查询
│  │  ├─ redis_memory.py   # Redis 会话记忆
│  │  ├─ context_budget.py # 128K Token 预算与动态摘要
│  │  ├─ checkpoint_store.py # Redis Stack Checkpoint
│  │  ├─ llm_client.py     # LLM 封装 (DeepSeek / Anthropic)
│  │  ├─ embeddings.py     # Embedding provider (BGE 本地模型)
│  │  └─ ...
│  └─ static/              # 前端页面
│     ├─ index.html        # 对话调试页面
│     └─ database.html     # 数据库只读浏览页面
├─ data/                   # 菜谱数据、评估集、测试图片
├─ scripts/                # 数据导入、连通性检查、评估、PDF 构建等脚本
├─ tests/                  # pytest 自动化测试
├─ docker/                 # Docker MySQL 初始化 SQL
├─ docker-compose.yml      # MySQL、Redis、Neo4j 容器编排
├─ requirements.txt        # Python 依赖
├─ .env_example            # 环境变量配置模板（不提交真实 API Key）
└─ CS599_大作业报告.pdf      # 课程大作业报告
```

## 环境搭建

### 1. 依赖安装

```bash
# 创建 conda 环境
conda create -n smart_recipe python=3.11
conda activate smart_recipe

# 安装 Python 依赖
pip install -r requirements.txt
```

### 2. 环境变量配置

复制 `.env_example` 为 `.env`，填入真实 API Key：

```bash
cp .env_example .env
```

需要配置以下关键项（不要硬编码 API Key 到代码中）：

```env
# DeepSeek 文本模型
SMART_RECIPE_PROVIDER=openai
BASE_URL=https://api.deepseek.com
API_KEY=<你的 DeepSeek API Key>
MODEL=deepseek-chat

# Anthropic 视觉模型
VISION_PROVIDER=anthropic
VISION_BASE_URL=<视觉模型地址>
VISION_API_KEY=<你的视觉 API Key>
VISION_MODEL=mimo-v2.5

# 数据库连接
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3307
NEO4J_URI=bolt://127.0.0.1:7687
REDIS_URL=redis://127.0.0.1:6379/0
```

### 3. 启动步骤

```bash
# 启动数据库容器
docker compose up -d mysql redis-stack neo4j

# 导入菜谱数据到 MySQL 和 Neo4j
python scripts/import_data.py mysql
python scripts/import_data.py neo4j

# 启动 FastAPI 服务
python scripts/start_server.py --host 127.0.0.1 --port 8010
```

启动后访问：
- 对话页面：`http://127.0.0.1:8010/`
- API 文档：`http://127.0.0.1:8010/docs`
- 数据库浏览：`http://127.0.0.1:8010/ui/database.html`

## 项目状态

- [x] Proposal
- [x] MVP
- [√] Final

## 大作业报告与演示视频

| 资源 | 文件 |
|---|---|
| 大作业报告 | [docs/CS599_大作业报告.pdf](docs/CS599_大作业报告.pdf) |
| 演示视频 | [smart_recipe.mp4](smart_recipe.mp4) |
| 详细技术文档 | [readme.md](readme.md) |
