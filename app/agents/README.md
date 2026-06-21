# Agents 目录说明

`agents/` 只放可被 LangGraph 或离线流程调用的 Agent。

- `router_agent.py`：意图识别和目标 Agent 选择。
- `recipe_agent.py`：RAG 菜谱推荐、做法、替换。
- `nutrition_agent.py`：营养和饮食目标解释。
- `sql_agent.py`：Text2SQL 结构化查询。
- `cypher_agent.py`：Text2Cypher 图谱关系查询。
- `fusion_agent.py`：RAG、SQL、Cypher 多源融合。
- `vision_agent.py`：图片识别和相似菜谱检索。
- `rerank_agent.py`：候选菜谱重排。
- `answer_agent.py`：最终回答生成和 Answer Guard。
- `preference_agent.py`：偏好、过敏、忌口抽取和记忆。
- `data_engineering_agent.py`：离线 RecipeParsingAgent，编排异构菜谱解析、清洗、去重和图谱构建。
- `support_agents.py`：轻量支撑 Agent，包含 `SafetyAgent`、`GeneralAgent`、`DataAgent`。

没有独立复杂逻辑的小 Agent 会合并到 `support_agents.py`，避免文件过多。
