# Scripts 目录说明

## 数据构建

- `build_data_pipeline.py`：清洗 `data/recipes.json`，输出 `data/processed/recipes_clean.json` 和评估种子。
- `build_document_index.py`：读取 PDF、DOCX、HTML、TXT、MD 或网页 URL，完成 OCR、文档清洗、菜谱版面解析、结构化整理、智能切块、metadata 保存、向量化和 FAISS 索引保存。

常用文档索引命令：

```powershell
python scripts\build_document_index.py "pdf\家常菜精选265.pdf" --ocr --ocr-force --recipe-refine --ocr-engine rapidocr --ocr-dpi 160 --chunk-size 800 --chunk-overlap 120
```

已有 OCR chunk 需要重新结构化和建索引时：

```powershell
python scripts\build_document_index.py --from-chunks data\processed\家常菜精选265_chunks.jsonl --recipe-refine
```

检索测试。默认使用混合检索：FAISS 向量分数 + 关键词命中 + 菜名/原料 metadata 加权 rerank。

```powershell
python scripts\search_document_faiss.py --index data\processed\家常菜精选265_recipe.index --metadata data\processed\家常菜精选265_recipe_metadata.json --query "老醋花生米怎么做" --top-k 5 --preview-chars 700
```

只看纯向量召回时：

```powershell
python scripts\search_document_faiss.py --index data\processed\家常菜精选265_recipe.index --metadata data\processed\家常菜精选265_recipe_metadata.json --query "老醋花生米怎么做" --mode vector --top-k 5
```

## 数据库导入

- `init_mysql_schema.py`：初始化 MySQL schema。
- `import_recipes_to_mysql.py`：导入菜谱到 MySQL。
- `import_document_chunks_to_mysql.py`：导入 PDF RAG chunk 正文和 metadata 到 MySQL，FAISS 索引仍保存在本地。
- `import_recipes_to_neo4j.py`：导入菜谱图谱到 Neo4j。

```powershell
python scripts\import_document_chunks_to_mysql.py --metadata data\processed\家常菜精选265_recipe_metadata.json --index data\processed\家常菜精选265_recipe.index --reset
```

## 环境检查

- `check_llm.py`：检查 LLM 接口。
- `check_mysql_store.py`：检查 Docker MySQL 连接和统计。
- `check_neo4j_graph.py`：检查 Docker Neo4j 图谱。
- `check_redis_memory.py`：检查 Docker Redis memory。

## 评估

- `evaluate_router.py`：评估 Router 意图识别。
- `evaluate_retrieval.py`：评估 RAG 召回。
- `evaluate_preferences.py`：评估偏好记忆和过滤。
- `evaluate_text2sql.py`：评估 Text2SQL。
- `evaluate_text2cypher.py`：评估 Text2Cypher。
- `evaluate_chat_v2.py`：评估 V2 混合聊天链路。
- `run_ablation.py`：运行方案消融实验，输出 JSON 和 Markdown 对比结果。

## 手动调试

- `test_chat.py`：手动请求 `/chat`。
- `test_retriever.py`：手动查看菜谱检索结果。
- `search_document_faiss.py`：手动查看文档 FAISS 索引的 Top-K chunk。
