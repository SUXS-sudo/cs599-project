# Scripts 目录说明

## 数据构建

- `build_data_pipeline.py`：清洗 `data/recipes.json`，输出 `data/processed/recipes_clean.json` 和评估种子。
- `build_document_index.py`：读取 PDF、DOCX、HTML、TXT、MD 或网页 URL，完成 OCR、文档清洗、菜谱版面解析、结构化整理、智能切块、metadata 保存、向量化和 FAISS 索引保存。
- `build_heterogeneous_recipe_pipeline.py`：一键数据工程 Pipeline，统一解析异构文件、LLM 优化、清洗、去重、FAISS 构建并写入 MySQL 和 Neo4j。

常用文档索引命令：

```powershell
python scripts\build_document_index.py "pdf\家常菜精选265.pdf" --ocr --ocr-force --recipe-refine --ocr-engine rapidocr --ocr-dpi 160 --chunk-size 800 --chunk-overlap 120
```

已有 OCR chunk 需要重新结构化和建索引时：

```powershell
python scripts\build_document_index.py --from-chunks data\processed\家常菜精选265_chunks.jsonl --recipe-refine
```

## 检索测试

- `search_document_faiss.py`：手动查看文档 FAISS 索引的 Top-K chunk。

```powershell
python scripts\search_document_faiss.py --index data\processed\家常菜精选265_recipe.index --metadata data\processed\家常菜精选265_recipe_metadata.json --query "老醋花生米怎么做" --top-k 5 --preview-chars 700
```

只看纯向量召回时：

```powershell
python scripts\search_document_faiss.py --index data\processed\家常菜精选265_recipe.index --metadata data\processed\家常菜精选265_recipe_metadata.json --query "老醋花生米怎么做" --mode vector --top-k 5
```

## 数据导入（import_data.py）

通过子命令选择导入目标：

```powershell
python scripts\import_data.py mysql --data-path data\recipes.json
python scripts\import_data.py neo4j --data-path data\recipes.json --reset
python scripts\import_data.py chunks --metadata data\processed\家常菜精选265_recipe_metadata.json --index data\processed\家常菜精选265_recipe.index --reset
```

| 子命令 | 说明 |
|---|---|
| `mysql` | 导入菜谱到 MySQL。支持 `--reset`、`--dry-run`。 |
| `neo4j` | 导入菜谱图谱到 Neo4j。支持 `--reset`、`--dry-run`。 |
| `chunks` | 导入文档 RAG 分块到 MySQL。支持 `--metadata-dir` 批量模式、`--reset`、`--dry-run`。 |

## 环境检查（check.py）

通过子命令选择检查目标：

```powershell
python scripts\check.py mysql
python scripts\check.py redis
python scripts\check.py neo4j --ingredient 鸡胸肉
python scripts\check.py llm --text-only
python scripts\check.py schema --print-sql
```

| 子命令 | 说明 |
|---|---|
| `mysql` | 检查 MySQL 连接和表行数统计。 |
| `redis` | 检查 Redis 记忆读写。 |
| `neo4j` | 检查 Neo4j 图谱统计和样例查询。 |
| `llm` | 检查文本模型和视觉模型连通性。支持 `--text-only`、`--vision-only`、`--hyde-only`。 |
| `schema` | 初始化或打印 MySQL schema。 |

## 评估（evaluate.py）

通过子命令选择评估目标：

```powershell
python scripts\evaluate.py router --show-errors
python scripts\evaluate.py retrieval --backend faiss --show-errors
python scripts\evaluate.py chat --show-errors
python scripts\evaluate.py text2sql
python scripts\evaluate.py text2cypher
python scripts\evaluate.py document-rag --index ... --metadata ...
python scripts\evaluate.py preferences --backend redis --show-errors
python scripts\evaluate.py safety --show-errors
python scripts\evaluate.py query-understanding --show-errors
```

| 子命令 | 说明 |
|---|---|
| `chat` | 端到端 /chat 评估（HTTP）。 |
| `retrieval` | RAG 检索命中率（Hit@K、MRR）。 |
| `router` | Router 意图分类准确率。 |
| `text2sql` | Text2SQL 端到端评估。 |
| `text2cypher` | Text2Cypher 端到端评估。 |
| `document-rag` | PDF 文档 RAG 评估（FAISS + HyDE + Cross-Encoder）。 |
| `preferences` | 偏好记忆和过敏/忌口过滤评估。 |
| `safety` | 安全边界和回答防护评估（HTTP）。 |
| `query-understanding` | 错别字纠正和意图恢复评估（HTTP）。 |

## 消融实验

- `run_ablation.py`：运行方案消融实验，输出 JSON 和 Markdown 对比结果。
