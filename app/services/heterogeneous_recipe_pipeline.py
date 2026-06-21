from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree

from app.services.data_pipeline import split_steps
from app.services.document_chunking import ChunkingConfig, DocumentChunk, OcrConfig, chunk_documents, chunk_text
from app.services.recipe_chunk_refiner import reconstruct_ocr_text, refine_recipe_chunks
from app.services.recipe_enrichment import validate_and_normalize_recipe


FIELD_ALIASES = {
    "name": ("name", "title", "dish_name", "菜名", "名称"),
    "ingredients": ("ingredients", "ingredient", "原料", "食材", "主料"),
    "seasonings": ("seasonings", "seasoning", "调料", "调味料", "辅料"),
    "steps": ("steps", "method", "description", "制作方法", "做法", "步骤"),
    "category": ("category", "分类", "类别"),
    "cooking_time": ("cooking_time", "time", "烹饪时间", "用时"),
    "difficulty": ("difficulty", "难度"),
    "tags": ("tags", "标签"),
    "calories": ("calories", "calories_per_100g", "热量", "卡路里"),
    "suitable_for": ("suitable_for", "适合人群", "适用场景", "餐次"),
    "tips": ("tips", "小提示", "提示"),
}
COMPONENT_SPLIT_RE = re.compile(r"[，,、；;。\n]+")
QUANTITY_RE = re.compile(
    r"(?:约|各|适量|少许|若干|\d+(?:\.\d+)?(?:[-~～至]\d+(?:\.\d+)?)?\s*(?:克|千克|公斤|斤|两|毫升|升|个|只|片|块|根|颗|勺|匙|杯|把|朵|条|枚|瓣|张|份)?)"
)
TEXT_FIELD_RE = {
    "name": re.compile(r"(?:菜名|名称)[:：]\s*([^\n]+)"),
    "ingredients": re.compile(r"(?:原料|食材|主料)[:：]\s*(.+?)(?=\n(?:调料|调味料|制作方法|做法|步骤|小提示|来源页码)[:：]|$)", re.S),
    "seasonings": re.compile(r"(?:调料|调味料|辅料)[:：]\s*(.+?)(?=\n(?:制作方法|做法|步骤|小提示|来源页码)[:：]|$)", re.S),
    "steps": re.compile(r"(?:制作方法|做法|步骤)[:：]\s*(.+?)(?=\n(?:小提示|来源页码)[:：]|$)", re.S),
    "tips": re.compile(r"(?:小提示|提示)[:：]\s*(.+?)(?=\n来源页码[:：]|$)", re.S),
}
GRAPH_LABELS = {
    "Recipe", "Category", "SourceFile", "SourceRecord", "Ingredient", "IngredientUse",
    "Seasoning", "SeasoningUse", "RecipeStep", "Tag", "Audience", "NutritionFact", "CookingProfile",
}
GRAPH_RELATIONSHIP_TYPES = {
    "IN_CATEGORY", "CONTAINS", "DESCRIBES", "HAS_INGREDIENT_USE", "USES_INGREDIENT",
    "HAS_SEASONING_USE", "USES_SEASONING", "HAS_STEP", "NEXT_STEP", "HAS_TAG",
    "SUITABLE_FOR", "HAS_NUTRITION", "HAS_COOKING_PROFILE",
}
NAME_MERGE_LABELS = {"Recipe", "Category", "Ingredient", "Seasoning", "Tag", "Audience", "SourceFile"}


@dataclass
class HeterogeneousPipelineReport:
    sources: list[dict[str, Any]] = field(default_factory=list)
    raw_record_count: int = 0
    parsed_record_count: int = 0
    cleaned_recipe_count: int = 0
    rejected_count: int = 0
    duplicate_count: int = 0
    llm_enriched_count: int = 0
    automation_rate: float = 0.0
    estimated_manual_minutes_saved: float = 0.0
    manual_minutes_per_record_assumption: float = 2.0
    graph_node_count: int = 0
    graph_relationship_count: int = 0
    graph_node_labels: dict[str, int] = field(default_factory=dict)
    graph_relationship_types: dict[str, int] = field(default_factory=dict)
    rejected_records: list[dict[str, Any]] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def run_heterogeneous_recipe_pipeline(
    sources: list[Path],
    output_dir: Path,
    llm_client=None,
    enable_llm: bool = True,
    manual_minutes_per_record: float = 2.0,
    build_faiss: bool = True,
    import_mysql: bool = True,
    reset_mysql: bool = False,
    import_neo4j: bool = True,
    reset_neo4j: bool = False,
) -> HeterogeneousPipelineReport:
    report = HeterogeneousPipelineReport()
    parsed_rows: list[dict[str, Any]] = []
    document_sources: list[Path] = []
    document_chunks: dict[Path, list[DocumentChunk]] = {}
    _DOCUMENT_SUFFIXES = {".pdf", ".docx", ".html", ".htm", ".txt", ".md"}
    for source in sources:
        records = load_source_records(source, document_chunks_cache=document_chunks)
        source_summary = {"path": str(source), "type": source.suffix.lower().lstrip("."), "records": len(records)}
        report.sources.append(source_summary)
        report.raw_record_count += len(records)
        if source.suffix.lower() in _DOCUMENT_SUFFIXES:
            document_sources.append(source)
        for index, record in enumerate(records):
            parsed, errors = parse_recipe_record(record, source, index, llm_client=llm_client, enable_llm=enable_llm)
            if parsed is None:
                report.rejected_count += 1
                if errors:
                    report.rejected_records.append({
                        "source": str(source),
                        "index": index,
                        "name": str(record.get("name") or record.get("title") or record.get("菜名") or ""),
                        "errors": errors,
                    })
                continue
            if parsed.pop("_llm_enriched", False):
                report.llm_enriched_count += 1
            parsed_rows.append(parsed)
            report.parsed_record_count += 1

    recipes, duplicate_count = merge_recipes(parsed_rows)
    report.duplicate_count = duplicate_count
    report.cleaned_recipe_count = len(recipes)
    report.manual_minutes_per_record_assumption = manual_minutes_per_record
    report.automation_rate = round(report.parsed_record_count / report.raw_record_count, 4) if report.raw_record_count else 0.0
    report.estimated_manual_minutes_saved = round(report.parsed_record_count * manual_minutes_per_record, 1)
    nodes, relationships = build_graph_manifest(recipes)
    report.graph_node_count = len(nodes)
    report.graph_relationship_count = len(relationships)
    report.graph_node_labels = count_by(nodes, "label")
    report.graph_relationship_types = count_by(relationships, "type")

    output_dir.mkdir(parents=True, exist_ok=True)
    recipes_path = output_dir / "recipes_clean.json"
    nodes_path = output_dir / "graph_nodes.jsonl"
    relationships_path = output_dir / "graph_relationships.jsonl"
    report_path = output_dir / "pipeline_report.json"
    recipes_path.write_text(json.dumps(recipes, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(nodes_path, nodes)
    write_jsonl(relationships_path, relationships)
    report.outputs = {
        "recipes": str(recipes_path),
        "graph_nodes": str(nodes_path),
        "graph_relationships": str(relationships_path),
        "report": str(report_path),
    }

    # FAISS index building for document sources
    if build_faiss and document_sources:
        faiss_result = _build_faiss_for_documents(
            [document_chunks[source] for source in document_sources],
            output_dir,
        )
        if faiss_result:
            report.outputs["faiss_index"] = faiss_result["index"]
            report.outputs["faiss_metadata"] = faiss_result["metadata"]

    # MySQL import
    if import_mysql:
        mysql_stats = _import_to_mysql(recipes, output_dir, reset=reset_mysql)
        report.outputs["mysql_import_stats"] = json.dumps(mysql_stats)

    # Neo4j import
    if import_neo4j:
        neo4j_stats = import_graph_manifest_to_neo4j(nodes_path, relationships_path, reset=reset_neo4j)
        report.outputs["neo4j_import_stats"] = json.dumps(neo4j_stats)

    report_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def load_source_records(
    path: Path,
    document_chunks_cache: dict[Path, list[DocumentChunk]] | None = None,
) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
        if isinstance(value, dict) and isinstance(value.get("chunks"), list):
            return [row for row in value["chunks"] if isinstance(row, dict)]
        return [value] if isinstance(value, dict) else []
    if suffix in {".jsonl", ".ndjson"}:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix in {".csv", ".tsv"}:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t" if suffix == ".tsv" else ","))
    if suffix in {".xlsx", ".xlsm"}:
        return load_xlsx_records(path)
    if suffix in {".pdf", ".docx", ".html", ".htm", ".txt", ".md"}:
        chunks = chunk_documents(
            [path],
            config=ChunkingConfig(chunk_size=1_200, chunk_overlap=100),
            ocr_config=OcrConfig(enabled=suffix == ".pdf", show_progress=suffix == ".pdf"),
        )
        if document_chunks_cache is not None:
            document_chunks_cache[path] = chunks
        refined = refine_recipe_chunks([chunk.to_dict() for chunk in chunks])
        if refined:
            return [chunk.to_dict() for chunk in refined]
        return [chunk.to_dict() for chunk in chunks]
    raise ValueError(f"Unsupported source type: {path.suffix}")


def parse_recipe_record(
    record: dict[str, Any],
    source: Path,
    index: int,
    llm_client=None,
    enable_llm: bool = True,
) -> tuple[dict[str, Any] | None, list[str]]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    merged = {**record, **metadata}
    text = str(record.get("text") or "")
    item = {field: first_value(merged, aliases) for field, aliases in FIELD_ALIASES.items()}
    for field, pattern in TEXT_FIELD_RE.items():
        if not item.get(field) and text:
            match = pattern.search(text)
            if match:
                item[field] = match.group(1).strip()
    if not item.get("name") and record.get("title"):
        item["name"] = record["title"]

    def prepare(candidate: dict[str, Any]):
        prepared = candidate.copy()
        prepared_ingredient_details = parse_components(prepared.get("ingredients"))
        prepared_seasoning_details = parse_components(prepared.get("seasonings"))
        prepared["ingredients"] = [detail["name"] for detail in prepared_ingredient_details]
        return prepared, prepared_ingredient_details, prepared_seasoning_details

    deterministic_item, ingredient_details, seasoning_details = prepare(item)
    normalized, errors = validate_and_normalize_recipe(deterministic_item)
    selected_item = deterministic_item
    llm_enriched = False
    if enable_llm and llm_client and llm_client.available:
        enriched = enrich_record_with_llm(record, llm_client, draft=item)
        if enriched:
            candidate = item.copy()
            candidate.update(
                {
                    field: value
                    for field, value in enriched.items()
                    if field in FIELD_ALIASES and value not in (None, "", [])
                }
            )
            llm_item, llm_ingredient_details, llm_seasoning_details = prepare(candidate)
            llm_normalized, llm_errors = validate_and_normalize_recipe(llm_item)
            if llm_normalized is not None:
                selected_item = llm_item
                ingredient_details = llm_ingredient_details
                seasoning_details = llm_seasoning_details
                normalized = llm_normalized
                errors = llm_errors
                llm_enriched = True
    if normalized is None:
        return None, errors

    normalized["seasonings"] = [detail["name"] for detail in seasoning_details]
    normalized["ingredient_details"] = ingredient_details or [
        {"name": name, "raw": name, "quantity": ""} for name in normalized["ingredients"]
    ]
    normalized["seasoning_details"] = seasoning_details
    normalized["tips"] = str(selected_item.get("tips") or "").strip()
    normalized["provenance"] = [
        {
            "source": str(record.get("source") or source),
            "source_type": str(record.get("source_type") or source.suffix.lower().lstrip(".")),
            "record_id": str(record.get("chunk_id") or record.get("id") or f"{source.stem}-{index}"),
            "pages": metadata.get("pages", []),
        }
    ]
    normalized["_llm_enriched"] = llm_enriched
    return normalized, []


def load_xlsx_records(path: Path) -> list[dict[str, Any]]:
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{namespace}si"):
                shared_strings.append("".join(node.text or "" for node in item.iter(f"{namespace}t")))
        rows: list[dict[str, Any]] = []
        sheet_names = sorted(
            name for name in archive.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name)
        )
        for sheet_name in sheet_names:
            root = ElementTree.fromstring(archive.read(sheet_name))
            table: list[dict[int, Any]] = []
            for row in root.iter(f"{namespace}row"):
                values: dict[int, Any] = {}
                for cell in row.findall(f"{namespace}c"):
                    reference = str(cell.attrib.get("r") or "A1")
                    index = excel_column_index(reference)
                    cell_type = cell.attrib.get("t", "")
                    value_node = cell.find(f"{namespace}v")
                    if cell_type == "inlineStr":
                        value = "".join(node.text or "" for node in cell.iter(f"{namespace}t"))
                    else:
                        raw = value_node.text if value_node is not None else ""
                        if cell_type == "s" and raw:
                            value = shared_strings[int(raw)]
                        else:
                            value = raw
                    values[index] = value
                if values:
                    table.append(values)
            if not table:
                continue
            headers = {index: str(value or "").strip() for index, value in table[0].items()}
            for value_row in table[1:]:
                record = {headers[index]: value for index, value in value_row.items() if headers.get(index)}
                if record:
                    rows.append(record)
        return rows


def excel_column_index(reference: str) -> int:
    letters = re.match(r"[A-Za-z]+", reference)
    value = 0
    for letter in (letters.group(0).upper() if letters else "A"):
        value = value * 26 + ord(letter) - ord("A") + 1
    return value - 1


def enrich_record_with_llm(
    record: dict[str, Any],
    llm_client,
    draft: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {"deterministic_draft": draft or {}, "source_record": record}
    prompt = (
        "Optimize the already chunked recipe, correct OCR noise, and preserve only facts present in the input. "
        "你是智能菜谱解析 Agent。把输入整理成结构化菜谱，只返回 JSON，不要解释。"
        "字段必须是 name, ingredients, seasonings, steps, category, cooking_time, difficulty, tags, calories, suitable_for。"
        "不得编造输入中不存在的菜名和关键食材；无法确定的字段使用空字符串、空数组或0。\n"
        f"输入：{json.dumps(record, ensure_ascii=False, default=str)[:8000]}"
    )
    raw = llm_client.generate(prompt, max_tokens=1_000, timeout=30)
    if not raw:
        return {}
    match = re.search(r"\{.*\}", raw, flags=re.S)
    try:
        value = json.loads(match.group(0) if match else raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def merge_recipes(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    merged: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    for row in rows:
        key = normalize_name(row["name"])
        if key not in merged:
            merged[key] = row
            continue
        duplicate_count += 1
        current = merged[key]
        for field in ("ingredients", "seasonings", "tags", "suitable_for"):
            current[field] = unique([*current.get(field, []), *row.get(field, [])])
        for field in ("ingredient_details", "seasoning_details", "provenance"):
            current[field] = dedupe_dicts([*current.get(field, []), *row.get(field, [])])
        if len(str(row.get("steps", ""))) > len(str(current.get("steps", ""))):
            current["steps"] = row["steps"]
        if len(str(row.get("tips", ""))) > len(str(current.get("tips", ""))):
            current["tips"] = row["tips"]
    return sorted(merged.values(), key=lambda item: item["name"]), duplicate_count


def build_graph_manifest(recipes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes: dict[str, dict[str, Any]] = {}
    relationships: dict[str, dict[str, Any]] = {}

    def node(label: str, key: str, properties: dict[str, Any]) -> str:
        node_id = stable_id(label, key)
        nodes.setdefault(node_id, {"id": node_id, "label": label, "properties": properties})
        return node_id

    def relate(source_id: str, rel_type: str, target_id: str, properties: dict[str, Any] | None = None) -> None:
        rel_id = stable_id("REL", f"{source_id}|{rel_type}|{target_id}|{json.dumps(properties or {}, sort_keys=True, ensure_ascii=False)}")
        relationships.setdefault(
            rel_id,
            {"id": rel_id, "source": source_id, "type": rel_type, "target": target_id, "properties": properties or {}},
        )

    for recipe in recipes:
        recipe_id = node("Recipe", recipe["name"], {"name": recipe["name"], "steps": recipe["steps"]})
        category_id = node("Category", recipe["category"], {"name": recipe["category"]})
        relate(recipe_id, "IN_CATEGORY", category_id)
        for provenance in recipe.get("provenance", []):
            source_id = node("SourceFile", provenance["source"], {"path": provenance["source"], "type": provenance["source_type"]})
            record_key = f"{provenance['source']}|{provenance['record_id']}"
            record_id = node("SourceRecord", record_key, provenance)
            relate(source_id, "CONTAINS", record_id)
            relate(record_id, "DESCRIBES", recipe_id)
        for index, detail in enumerate(recipe.get("ingredient_details", []), start=1):
            ingredient_id = node("Ingredient", detail["name"], {"name": detail["name"]})
            use_id = node("IngredientUse", f"{recipe['name']}|{index}|{detail['raw']}", {**detail, "position": index})
            relate(recipe_id, "HAS_INGREDIENT_USE", use_id)
            relate(use_id, "USES_INGREDIENT", ingredient_id)
        for index, detail in enumerate(recipe.get("seasoning_details", []), start=1):
            seasoning_id = node("Seasoning", detail["name"], {"name": detail["name"]})
            use_id = node("SeasoningUse", f"{recipe['name']}|{index}|{detail['raw']}", {**detail, "position": index})
            relate(recipe_id, "HAS_SEASONING_USE", use_id)
            relate(use_id, "USES_SEASONING", seasoning_id)
        previous_step_id = ""
        for index, step_text in enumerate(split_steps(recipe["steps"]), start=1):
            step_id = node("RecipeStep", f"{recipe['name']}|{index}", {"position": index, "text": step_text})
            relate(recipe_id, "HAS_STEP", step_id)
            if previous_step_id:
                relate(previous_step_id, "NEXT_STEP", step_id)
            previous_step_id = step_id
        for tag in recipe.get("tags", []):
            tag_id = node("Tag", tag, {"name": tag})
            relate(recipe_id, "HAS_TAG", tag_id)
        for audience in recipe.get("suitable_for", []):
            audience_id = node("Audience", audience, {"name": audience})
            relate(recipe_id, "SUITABLE_FOR", audience_id)
        nutrition_id = node(
            "NutritionFact",
            recipe["name"],
            {
                "calories_per_100g": recipe.get("calories_per_100g", recipe.get("calories", 0)),
                "protein_g_per_100g": recipe.get("protein_g_per_100g", 0),
                "fat_g_per_100g": recipe.get("fat_g_per_100g", 0),
                "estimated": bool(recipe.get("nutrition_estimated", True)),
            },
        )
        relate(recipe_id, "HAS_NUTRITION", nutrition_id)
        profile_id = node(
            "CookingProfile",
            recipe["name"],
            {
                "cooking_time": recipe.get("cooking_time", ""),
                "cooking_time_minutes": recipe.get("cooking_time_minutes", 0),
                "difficulty": recipe.get("difficulty", ""),
            },
        )
        relate(recipe_id, "HAS_COOKING_PROFILE", profile_id)
    return list(nodes.values()), list(relationships.values())


def parse_components(value: Any) -> list[dict[str, str]]:
    if value is None:
        return []
    raw_parts = value if isinstance(value, list) else COMPONENT_SPLIT_RE.split(str(value))
    details: list[dict[str, str]] = []
    for raw_value in raw_parts:
        raw = str(raw_value).strip(" ：:，,。；;、")
        if not raw:
            continue
        quantity_parts = QUANTITY_RE.findall(raw)
        name = QUANTITY_RE.sub("", raw).strip(" ：:，,。；;、")
        name = re.sub(r"^(主料|辅料|调料|调味料)[:：]?", "", name).strip()
        if not name:
            name = raw
        detail = {"name": name, "raw": raw, "quantity": "、".join(part for part in quantity_parts if part)}
        if detail not in details:
            details.append(detail)
    return details


def first_value(record: dict[str, Any], aliases: Iterable[str]) -> Any:
    for alias in aliases:
        value = record.get(alias)
        if value not in (None, "", []):
            return value
    return None


def normalize_name(value: str) -> str:
    return re.sub(r"[\s：:，,。！？!?（）()《》「」『』]", "", str(value)).lower()


def stable_id(kind: str, value: str) -> str:
    digest = hashlib.sha1(f"{kind}|{value}".encode("utf-8")).hexdigest()[:20]
    return f"{kind.lower()}:{digest}"


def unique(values: list[Any]) -> list[Any]:
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def dedupe_dicts(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen = set()
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row[field])
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def import_graph_manifest_to_neo4j(
    nodes_path: Path,
    relationships_path: Path,
    reset: bool = False,
) -> dict[str, int]:
    from app.services.neo4j_store import Neo4jStore

    nodes = read_jsonl(nodes_path)
    relationships = read_jsonl(relationships_path)
    store = Neo4jStore()
    if reset:
        store.clear_graph()
    store.execute_write(
        "CREATE CONSTRAINT manifest_node_id IF NOT EXISTS "
        "FOR (n:ManifestNode) REQUIRE n.manifest_id IS UNIQUE"
    )

    for label in sorted(GRAPH_LABELS):
        rows = [row for row in nodes if row.get("label") == label]
        if not rows:
            continue
        if label in NAME_MERGE_LABELS:
            query = f"""
            UNWIND $rows AS row
            MERGE (n:{label} {{name: coalesce(row.properties.name, row.properties.path)}})
            SET n:ManifestNode, n += row.properties, n.manifest_id = row.id
            """
        else:
            query = f"""
            UNWIND $rows AS row
            MERGE (n:{label}:ManifestNode {{manifest_id: row.id}})
            SET n += row.properties
            """
        store.execute_write(query, {"rows": rows})

    for rel_type in sorted(GRAPH_RELATIONSHIP_TYPES):
        rows = [row for row in relationships if row.get("type") == rel_type]
        if not rows:
            continue
        query = f"""
        UNWIND $rows AS row
        MATCH (source:ManifestNode {{manifest_id: row.source}})
        MATCH (target:ManifestNode {{manifest_id: row.target}})
        MERGE (source)-[rel:{rel_type} {{manifest_id: row.id}}]->(target)
        SET rel += row.properties
        """
        store.execute_write(query, {"rows": rows})
    return {
        "nodes": len(nodes),
        "relationships": len(relationships),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _build_faiss_for_documents(
    cached_document_chunks: list[list[DocumentChunk]],
    output_dir: Path,
) -> dict[str, str] | None:
    from app.services.document_faiss import build_document_faiss_index

    final_chunks: list[DocumentChunk] = []
    for raw_chunks in cached_document_chunks:
        if not raw_chunks:
            continue
        refined = refine_recipe_chunks([chunk.to_dict() for chunk in raw_chunks])
        if refined:
            final_chunks.extend(
                DocumentChunk(
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    source_type=chunk.source_type,
                    text=chunk.text,
                    start_char=0,
                    end_char=len(chunk.text),
                    metadata=chunk.metadata,
                )
                for chunk in refined
            )
            continue

        first = raw_chunks[0]
        final_chunks.extend(
            chunk_text(
                reconstruct_ocr_text([chunk.to_dict() for chunk in raw_chunks]),
                source=first.source,
                source_type=first.source_type,
                config=ChunkingConfig(chunk_size=800, chunk_overlap=120),
            )
        )
    if not final_chunks:
        return None
    index_obj = build_document_faiss_index(final_chunks)
    index_path = output_dir / "document.index"
    metadata_path = output_dir / "document_recipe_metadata.json"
    index_obj.save(index_path, metadata_path)
    return {"index": str(index_path), "metadata": str(metadata_path)}


def _import_to_mysql(
    recipes: list[dict[str, Any]],
    output_dir: Path,
    reset: bool = False,
) -> dict[str, Any]:
    from app.services.mysql_store import MySQLStore

    store = MySQLStore()
    store.ensure_schema()
    if reset:
        store.reset_recipe_tables()
    stats = store.import_recipes(recipes)
    faiss_metadata_path = output_dir / "document_recipe_metadata.json"
    if faiss_metadata_path.exists():
        meta = json.loads(faiss_metadata_path.read_text(encoding="utf-8"))
        if reset:
            store.reset_document_tables()
        doc_stats = store.import_document_index(
            index_name="heterogeneous_pipeline",
            index_path=str(output_dir / "document.index"),
            metadata_path=str(faiss_metadata_path),
            metadata=meta,
        )
        stats["document_import"] = doc_stats
    return stats


def import_recipes_to_mysql(
    recipes_path: Path,
    reset: bool = False,
) -> dict[str, Any]:
    from app.services.mysql_store import MySQLStore

    recipes = json.loads(recipes_path.read_text(encoding="utf-8"))
    store = MySQLStore()
    store.ensure_schema()
    if reset:
        store.reset_recipe_tables()
    return store.import_recipes(recipes)
