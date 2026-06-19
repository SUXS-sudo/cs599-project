# Services 目录说明

`services/` 放 Agent 共用的基础能力，不直接决定用户意图。

- `llm_client.py`：文本和图文模型统一调用，小米接口配置也在这里读取。
- `embeddings.py`：RAG 向量生成和本地 fallback。
- `memory.py` / `redis_memory.py`：短期会话、长期摘要、偏好存储。
- `mysql_store.py`：MySQL schema、导入、只读查询和偏好落库。
- `neo4j_store.py`：Neo4j 图谱导入和只读查询。
- `query_guard.py`：SQL/Cypher 只读安全校验。
- `graph_rag.py`：RAG 命中后的图谱上下文增强。
- `image_analyzer.py`：菜品图片识别，优先视觉模型，失败走规则 fallback。
- `data_pipeline.py`：菜谱数据清洗和评估种子生成。
- `answer_guard.py`：最终回答证据检查和幻觉防护。
- `logger.py`：统一日志配置。

如果某个能力会被多个 Agent 复用，应放在这里；如果只属于一个 Agent 的路由/决策逻辑，应留在 `agents/`。
