"""SmartRecipe 评估工具。

子命令:
  chat                  端到端 /chat 评估（HTTP）
  retrieval             RAG 检索命中率评估
  router                Router 意图分类评估
  text2sql              Text2SQL 评估
  text2cypher           Text2Cypher 评估
  document-rag          PDF 文档 RAG 评估
  preferences           偏好记忆评估
  safety                安全防御评估（HTTP）
  query-understanding   错别字纠正评估（HTTP）
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _call_chat(base_url: str, message: str, session_id: str, timeout: int) -> dict:
    payload = json.dumps({"message": message, "session_id": session_id, "top_k": 3}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat",
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


# ── chat ───────────────────────────────────────────────────────────────────

def cmd_chat(args: argparse.Namespace) -> int:
    cases = _load_cases(Path(args.eval_file))
    ok = 0
    errors = []
    for index, case in enumerate(cases):
        session_id = f"chat-v2-{case['name']}-{int(time.time())}-{index}"
        responses = []
        failed = False
        for turn, message in enumerate(case["messages"]):
            try:
                response = _call_chat(args.base_url, message, session_id, args.timeout)
            except Exception as exc:
                errors.append({"case": case["name"], "error": f"{type(exc).__name__}: {exc}"})
                failed = True
                break
            responses.append(response)

            expected_intent = case.get("expect_intents", [None] * len(case["messages"]))[turn]
            expected_agent = case.get("expect_agents", [None] * len(case["messages"]))[turn]
            if expected_intent and response.get("intent") != expected_intent:
                errors.append({"case": case["name"], "turn": turn, "expected_intent": expected_intent, "actual": response})
                failed = True
            if expected_agent and response.get("agent") != expected_agent:
                errors.append({"case": case["name"], "turn": turn, "expected_agent": expected_agent, "actual": response})
                failed = True

        if failed:
            continue

        final_response = responses[-1]
        answer = final_response.get("answer", "")
        contains_ok = all(text in answer for text in case.get("contains", []))
        blocked_ok = not _contains_blocked_recipes(final_response.get("recipes", []), case.get("blocked_recipe_ingredients", []))
        if contains_ok and blocked_ok:
            ok += 1
        else:
            errors.append({"case": case["name"], "contains_ok": contains_ok, "blocked_ok": blocked_ok, "final_response": final_response})

    total = len(cases)
    print(f"cases={total}")
    print(f"chat_v2_success={ok / total:.2%} ({ok}/{total})")
    if args.show_errors and errors:
        print("errors:")
        for error in errors:
            print(json.dumps(error, ensure_ascii=False))
    return 0 if ok == total else 1


def _contains_blocked_recipes(recipes: list[dict], blocked: list[str]) -> bool:
    for recipe in recipes:
        for ingredient in recipe.get("ingredients", []):
            if any(block in ingredient or ingredient in block for block in blocked if block):
                return True
    return False


# ── retrieval ──────────────────────────────────────────────────────────────

def _hit_at(ranked_names: list[str], expected_names: set[str], k: int) -> bool:
    return any(name in expected_names for name in ranked_names[:k])


def _first_hit_rank(ranked_names: list[str], expected_names: set[str]) -> int | None:
    for index, name in enumerate(ranked_names, start=1):
        if name in expected_names:
            return index
    return None


def cmd_retrieval(args: argparse.Namespace) -> int:
    os.environ["RAG_BACKEND"] = args.backend

    from app.retriever import RecipeRetriever

    cases = _load_cases(Path(args.eval_file))
    retriever = RecipeRetriever(ROOT_DIR / "data" / "recipes.json")
    max_k = max(1, args.max_k)
    cutoffs = [k for k in (1, 3, 5) if k <= max_k]
    if max_k not in cutoffs:
        cutoffs.append(max_k)

    totals = {k: 0 for k in cutoffs}
    category_stats: dict[str, dict[int, int]] = {}
    category_totals: dict[str, int] = {}
    reciprocal_rank_sum = 0.0
    errors = []

    for case in cases:
        hits = retriever.search(case["query"], top_k=max_k)
        ranked_names = [recipe.name for recipe, _ in hits]
        expected_names = set(case["expected"])
        category = case.get("category", "uncategorized")
        category_totals[category] = category_totals.get(category, 0) + 1
        category_stats.setdefault(category, {k: 0 for k in cutoffs})

        rank = _first_hit_rank(ranked_names, expected_names)
        if rank is not None:
            reciprocal_rank_sum += 1.0 / rank

        for k in cutoffs:
            is_hit = _hit_at(ranked_names, expected_names, k)
            totals[k] += int(is_hit)
            category_stats[category][k] += int(is_hit)

        if not _hit_at(ranked_names, expected_names, max_k):
            errors.append({"query": case["query"], "expected": case["expected"], "actual": ranked_names, "category": category})

    total_cases = len(cases)
    print(f"cases={total_cases}")
    print(f"requested_backend={args.backend}")
    print(f"active_backend={retriever.backend}")
    print(f"embedding_backend={retriever.embedding_backend}")
    if retriever.backend_errors:
        print("backend_errors:")
        for error in retriever.backend_errors:
            print(f"- {error}")

    for k in cutoffs:
        rate = totals[k] / total_cases if total_cases else 0.0
        print(f"hit@{k}={rate:.2%} ({totals[k]}/{total_cases})")
    mrr = reciprocal_rank_sum / total_cases if total_cases else 0.0
    print(f"mrr@{max_k}={mrr:.4f}")

    print("by_category:")
    for category in sorted(category_totals):
        pieces = []
        for k in cutoffs:
            count = category_stats[category][k]
            total = category_totals[category]
            rate = count / total if total else 0.0
            pieces.append(f"hit@{k}={rate:.2%} ({count}/{total})")
        print(f"- {category}: " + ", ".join(pieces))

    if args.show_errors and errors:
        print("errors:")
        for error in errors:
            print(json.dumps(error, ensure_ascii=False))

    return 0


# ── router ─────────────────────────────────────────────────────────────────

def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_bool(primary: str, fallback: str | None = None, default: bool = False) -> bool:
    value = os.getenv(primary)
    if value is None and fallback:
        value = os.getenv(fallback)
    if value is None:
        return default
    return _parse_bool(value)


def cmd_router(args: argparse.Namespace) -> int:
    from app.agents.router_agent import RouterAgent
    from app.services.llm_client import LLMClient
    from app.state import AgentState

    cases = _load_cases(Path(args.eval_file))
    force_enable = args.enable_database_agents or args.enable_v2
    force_disable = args.disable_database_agents or args.disable_v2
    if force_enable and force_disable:
        raise ValueError("Use only one enable/disable database-agent option.")
    if force_enable:
        enable_database_agents = True
    elif force_disable:
        enable_database_agents = False
    else:
        enable_database_agents = _env_bool("ENABLE_DATABASE_AGENTS", "ENABLE_V2", default=False)
    router = RouterAgent(LLMClient() if args.use_llm else None, enable_database_agents=enable_database_agents)

    total = len(cases)
    correct_intent = 0
    correct_agent = 0
    by_intent: dict[str, dict[str, int]] = {}
    errors = []

    for case in cases:
        state = AgentState(user_input=case["message"], session_id="router-eval", top_k=3)
        result = router.run(state)
        expected_intent = case["intent"]
        expected_agent = case.get("target_agent")
        intent_ok = result.intent == expected_intent
        agent_ok = expected_agent is None or result.target_agent == expected_agent
        correct_intent += int(intent_ok)
        correct_agent += int(agent_ok)

        bucket = by_intent.setdefault(expected_intent, {"total": 0, "correct": 0})
        bucket["total"] += 1
        bucket["correct"] += int(intent_ok)

        if not intent_ok or not agent_ok:
            errors.append({
                "message": case["message"],
                "expected_intent": expected_intent,
                "predicted_intent": result.intent,
                "expected_agent": expected_agent,
                "predicted_agent": result.target_agent,
                "router_mode": result.meta.get("router_mode"),
            })

    intent_accuracy = correct_intent / total if total else 0.0
    agent_accuracy = correct_agent / total if total else 0.0
    print(f"cases={total}")
    print(f"enable_database_agents={str(enable_database_agents).lower()}")
    print(f"intent_accuracy={intent_accuracy:.2%} ({correct_intent}/{total})")
    print(f"agent_accuracy={agent_accuracy:.2%} ({correct_agent}/{total})")
    print("by_intent:")
    for intent, stats in sorted(by_intent.items()):
        accuracy = stats["correct"] / stats["total"] if stats["total"] else 0.0
        print(f"- {intent}: {accuracy:.2%} ({stats['correct']}/{stats['total']})")

    if args.show_errors and errors:
        print("errors:")
        for error in errors:
            print(json.dumps(error, ensure_ascii=False))

    return 0 if correct_intent == total and correct_agent == total else 1


# ── text2sql ───────────────────────────────────────────────────────────────

def cmd_text2sql(_args: argparse.Namespace) -> int:
    from app.agents.router_agent import RouterAgent
    from app.agents.sql_agent import SQLAgent
    from app.state import AgentState

    cases = _load_cases(ROOT_DIR / "data" / "evals" / "text2sql_cases.jsonl")
    router = RouterAgent()
    agent = SQLAgent()
    print(f"mysql_target={agent.store.config.user}@{agent.store.config.host}:{agent.store.config.port}/{agent.store.config.database}")
    ok = 0
    for case in cases:
        state = AgentState(user_input=case["message"], session_id="text2sql-eval", top_k=5)
        state = router.run(state)
        if state.intent != case["expected_intent"]:
            print(f"route_miss={case['message']} expected={case['expected_intent']} actual={state.intent}")
            continue
        state = agent.run(state)
        if state.meta.get("sql_status") != "ok":
            print(f"sql_miss={case['message']} status={state.meta.get('sql_status')} output={state.agent_output}")
            continue
        if all(item in state.agent_output for item in case.get("contains", [])):
            ok += 1
        else:
            print(f"answer_miss={case['message']} output={state.agent_output}")
    total = len(cases)
    print(f"cases={total}")
    print(f"text2sql_success={ok / total:.2%} ({ok}/{total})")
    return 0 if ok == total else 1


# ── text2cypher ────────────────────────────────────────────────────────────

def cmd_text2cypher(_args: argparse.Namespace) -> int:
    from app.agents.cypher_agent import CypherAgent
    from app.agents.router_agent import RouterAgent
    from app.state import AgentState

    cases = _load_cases(ROOT_DIR / "data" / "evals" / "text2cypher_cases.jsonl")
    router = RouterAgent()
    agent = CypherAgent()
    ok = 0
    for case in cases:
        state = AgentState(user_input=case["message"], session_id="text2cypher-eval", top_k=5)
        state = router.run(state)
        if state.intent != case["expected_intent"]:
            print(f"route_miss={case['message']} expected={case['expected_intent']} actual={state.intent}")
            continue
        state = agent.run(state)
        if state.meta.get("cypher_status") != "ok":
            print(f"cypher_miss={case['message']} status={state.meta.get('cypher_status')} output={state.agent_output}")
            continue
        if all(item in state.agent_output for item in case.get("contains", [])):
            ok += 1
        else:
            print(f"answer_miss={case['message']} output={state.agent_output}")
    total = len(cases)
    print(f"cases={total}")
    print(f"text2cypher_success={ok / total:.2%} ({ok}/{total})")
    return 0 if ok == total else 1


# ── document-rag ───────────────────────────────────────────────────────────

def cmd_document_rag(args: argparse.Namespace) -> int:
    import importlib

    import numpy as np

    from app.services.document_chunking import DocumentChunk
    from app.services.embeddings import EmbeddingProvider
    from app.services.hyde import HyDEGenerator
    from app.services.llm_client import load_dotenv
    from app.services.query_rewrite import rewrite_recipe_query

    _search_mod = importlib.import_module("scripts.search_document_faiss")
    collect_candidates = _search_mod.collect_candidates
    rank_results = _search_mod.rank_results

    started = time.perf_counter()
    load_dotenv()
    os.environ["HYDE_ENABLED"] = "true" if args.hyde else "false"

    index_path = Path(args.index)
    metadata_path = Path(args.metadata)
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    log("configuration:")
    log(f"  index={index_path}")
    log(f"  metadata={metadata_path}")
    log(f"  top_k={args.top_k} candidate_k={args.candidate_k} limit={args.limit or 'all'}")
    log(f"  query_rewrite={not args.no_query_rewrite} hyde={args.hyde} cross_encoder_rerank={not args.no_cross_encoder_rerank}")

    log("loading FAISS index")
    import faiss

    index = faiss.read_index(str(index_path))
    log(f"loaded FAISS index type={type(index).__name__} dim={index.d} ntotal={index.ntotal}")
    log("loading metadata and chunks")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _set_embedding_model_from_metadata(metadata)
    chunks = [DocumentChunk(**item) for item in metadata.get("chunks", [])]
    if not chunks:
        raise ValueError(f"No chunks found in metadata: {metadata_path}")
    log(f"loaded chunks={len(chunks)} index_type={metadata.get('index_type')} embedding_backend={metadata.get('embedding_backend')}")

    log("initializing query embedding provider")
    provider = EmbeddingProvider([chunk.text for chunk in chunks])
    log(f"query_embedding_backend={provider.backend}")
    cases = _load_document_rag_cases(Path(args.eval_file), chunks) if args.eval_file else _build_document_rag_cases(chunks)
    if args.shuffle:
        random.Random(args.seed).shuffle(cases)
    if args.limit > 0:
        cases = cases[: args.limit]
    log(f"evaluation_cases={len(cases)} source={'eval_file' if args.eval_file else 'metadata_dish_names'}")
    if not args.no_cross_encoder_rerank:
        log("Cross-Encoder rerank is enabled; first evaluated case may show 'Loading weights' while the reranker model loads")
    if args.hyde:
        log("HyDE is enabled; this will call the LLM once per case and may be slow")

    hyde_generator = HyDEGenerator()
    details = []
    totals = {"hit": 0, "recall": 0.0, "precision": 0.0, "average_precision": 0.0, "reciprocal_rank": 0.0}

    for case_index, case in enumerate(cases, start=1):
        ranked = _search_document_case(
            case, index=index, provider=provider, chunks=chunks,
            hyde_generator=hyde_generator, top_k=args.top_k, candidate_k=args.candidate_k,
            expected_dim=index.d, use_query_rewrite=not args.no_query_rewrite,
            use_hyde=args.hyde, use_cross_encoder=not args.no_cross_encoder_rerank,
            cross_encoder_model=args.cross_encoder_model,
            collect_candidates=collect_candidates, rank_results=rank_results,
        )
        metrics = _score_document_case(ranked, set(case["expected_chunk_ids"]), args.top_k)
        for key in totals:
            totals[key] += metrics[key]
        detail = {
            "query": case["query"],
            "expected_chunk_ids": case["expected_chunk_ids"],
            "expected_dish_names": case.get("expected_dish_names", []),
            "ranked": [
                {
                    "rank": idx + 1,
                    "chunk_id": item["chunk"].chunk_id,
                    "dish_name": _dish_name(item["chunk"]),
                    "score": round(float(item["score"]), 6),
                    "sources": item["sources"],
                }
                for idx, item in enumerate(ranked[: args.top_k])
            ],
            "metrics": metrics,
        }
        details.append(detail)
        if args.progress_every > 0 and (case_index == 1 or case_index % args.progress_every == 0 or case_index == len(cases)):
            elapsed = time.perf_counter() - started
            _print_doc_progress(case_index, len(cases), totals, elapsed)

    count = len(cases)
    summary = {
        "cases": count, "top_k": args.top_k, "candidate_k": args.candidate_k,
        "index": str(index_path), "metadata": str(metadata_path),
        "index_type": metadata.get("index_type"), "embedding_backend": metadata.get("embedding_backend"),
        "query_embedding_backend": provider.backend,
        "query_rewrite": not args.no_query_rewrite, "hyde": args.hyde,
        "cross_encoder_rerank": not args.no_cross_encoder_rerank,
        "hit_at_k": _safe_avg(totals["hit"], count),
        "recall_at_k": _safe_avg(totals["recall"], count),
        "precision_at_k": _safe_avg(totals["precision"], count),
        "map_at_k": _safe_avg(totals["average_precision"], count),
        "mrr_at_k": _safe_avg(totals["reciprocal_rank"], count),
    }
    log("final summary")
    _print_doc_summary(summary)
    if args.show_errors:
        for detail in details:
            if detail["metrics"]["hit"] == 0:
                print(json.dumps(detail, ensure_ascii=False))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        log(f"writing detailed output to {output_path}")
        output_path.write_text(json.dumps({"summary": summary, "details": details}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"output={output_path}")
    return 0


def _set_embedding_model_from_metadata(metadata: dict[str, Any]) -> None:
    backend = str(metadata.get("embedding_backend") or "")
    prefix = "local_sentence_transformers:"
    if backend.startswith(prefix):
        os.environ["EMBEDDING_PROVIDER"] = "local"
        os.environ["EMBEDDING_MODEL"] = backend[len(prefix):]


def _build_document_rag_cases(chunks) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = {}
    for chunk in chunks:
        name = _dish_name(chunk)
        if not name:
            continue
        grouped.setdefault(name, []).append(chunk.chunk_id)
    return [{"query": f"{name}怎么做", "expected_chunk_ids": cids, "expected_dish_names": [name]} for name, cids in grouped.items()]


def _load_document_rag_cases(path: Path, chunks) -> list[dict[str, Any]]:
    from app.services.document_chunking import DocumentChunk

    dish_to_chunks: dict[str, list[str]] = {}
    for chunk in chunks:
        name = _dish_name(chunk)
        if name:
            dish_to_chunks.setdefault(name, []).append(chunk.chunk_id)
    cases = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        query = str(row.get("query") or "").strip()
        if not query:
            raise ValueError(f"{path}:{line_number} requires query")
        expected_chunk_ids = [str(item) for item in row.get("expected_chunk_ids", [])]
        expected_dish_names = [str(item) for item in row.get("expected_dish_names", [])]
        for name in expected_dish_names:
            expected_chunk_ids.extend(dish_to_chunks.get(name, []))
        expected_chunk_ids = list(dict.fromkeys(expected_chunk_ids))
        if not expected_chunk_ids:
            raise ValueError(f"{path}:{line_number} requires expected_chunk_ids or resolvable expected_dish_names")
        cases.append({"query": query, "expected_chunk_ids": expected_chunk_ids, "expected_dish_names": expected_dish_names})
    return cases


def _search_document_case(case, index, provider, chunks, hyde_generator, top_k, candidate_k, expected_dim, use_query_rewrite, use_hyde, use_cross_encoder, cross_encoder_model, collect_candidates, rank_results):
    from app.services.query_rewrite import rewrite_recipe_query

    query = case["query"]
    rewrite = rewrite_recipe_query(query)
    expanded_query = rewrite.expanded_query if use_query_rewrite else query
    hyde_query = hyde_generator.generate(query).hypothetical_document if use_hyde else ""
    candidates = collect_candidates(
        index=index, provider=provider, chunks=chunks,
        original_query=query, expanded_query=expanded_query, hyde_query=hyde_query,
        rewrite_core_terms=rewrite.core_terms,
        vector_k=min(candidate_k, len(chunks)), keyword_k=min(candidate_k, len(chunks)),
        expected_dim=expected_dim, include_keyword=True,
    )
    return rank_results(
        query, expanded_query, chunks, candidates,
        mode="hybrid", use_cross_encoder=use_cross_encoder, cross_encoder_model=cross_encoder_model,
    )[:top_k]


def _score_document_case(ranked, expected_ids: set[str], top_k: int) -> dict[str, float]:
    retrieved_ids = [item["chunk"].chunk_id for item in ranked[:top_k]]
    hits = [1 if cid in expected_ids else 0 for cid in retrieved_ids]
    relevant_found = sum(hits)
    precision = relevant_found / top_k if top_k else 0.0
    recall = relevant_found / len(expected_ids) if expected_ids else 0.0
    reciprocal_rank = 0.0
    precision_sum = 0.0
    hit_count = 0
    for rank, hit in enumerate(hits, start=1):
        if not hit:
            continue
        if reciprocal_rank == 0.0:
            reciprocal_rank = 1.0 / rank
        hit_count += 1
        precision_sum += hit_count / rank
    average_precision = precision_sum / len(expected_ids) if expected_ids else 0.0
    return {
        "hit": 1.0 if relevant_found else 0.0,
        "precision": precision,
        "recall": min(recall, 1.0),
        "average_precision": min(average_precision, 1.0),
        "reciprocal_rank": reciprocal_rank,
    }


def _dish_name(chunk) -> str:
    metadata = chunk.metadata or {}
    return str(metadata.get("dish_name") or metadata.get("title") or "").strip()


def _safe_avg(total: float, count: int) -> float:
    return total / count if count else 0.0


def _log(message: str) -> None:
    print(f"[eval] {message}", flush=True)


log = _log


def _print_doc_progress(done: int, total: int, totals: dict[str, float], elapsed: float) -> None:
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = (total - done) / rate if rate > 0 else 0.0
    hit = _safe_avg(totals["hit"], done)
    recall = _safe_avg(totals["recall"], done)
    precision = _safe_avg(totals["precision"], done)
    mrr = _safe_avg(totals["reciprocal_rank"], done)
    _log(f"progress {done}/{total} elapsed={elapsed:.1f}s eta={remaining:.1f}s hit@k={hit:.4f} recall@k={recall:.4f} precision@k={precision:.4f} mrr@k={mrr:.4f}")


def _print_doc_summary(summary: dict[str, Any]) -> None:
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}={value:.4f}")
        else:
            print(f"{key}={value}")


# ── preferences ────────────────────────────────────────────────────────────

def cmd_preferences(args: argparse.Namespace) -> int:
    os.environ["MEMORY_BACKEND"] = args.backend
    os.environ.setdefault("RAG_BACKEND", "bm25")
    os.environ.setdefault("RERANK_ENABLED", "false")

    from app.agents.preference_agent import PreferenceAgent
    from app.agents.recipe_agent import RecipeAgent
    from app.retriever import RecipeRetriever
    from app.services.memory import MemoryStore
    from app.services.redis_memory import RedisMemoryStore
    from app.state import AgentState

    cases = _load_cases(Path(args.eval_file))
    memory_store = RedisMemoryStore(max_messages=10) if args.backend == "redis" else MemoryStore(max_messages=10)
    preference_agent = PreferenceAgent(memory_store)
    recipe_agent = RecipeAgent(RecipeRetriever(ROOT_DIR / "data" / "recipes.json"))

    ok = 0
    errors = []
    for index, case in enumerate(cases):
        session_id = f"{case['session_id']}-{index}"
        for message in case.get("setup_messages", []):
            state = AgentState(user_input=message, session_id=session_id, top_k=3)
            preference_agent.run(state)

        prefs = memory_store.get_preferences(session_id)
        query_state = AgentState(
            user_input=case["query"], session_id=session_id, top_k=3,
            meta={"user_preferences": prefs.to_dict()},
        )
        preference_agent.run(query_state)
        recipe_agent.run(query_state)

        expected_preferences = set(case.get("expected_preferences", []))
        expected_allergies = set(case.get("expected_allergies", []))
        expected_dislikes = set(case.get("expected_dislikes", []))
        blocked = case.get("blocked", [])
        returned = [recipe.name for recipe, _ in query_state.retrieved_docs]
        returned_ingredients = [
            ingredient
            for recipe, _ in query_state.retrieved_docs
            for ingredient in recipe.ingredients
        ]

        preferences_ok = expected_preferences.issubset(set(prefs.preferences))
        allergies_ok = expected_allergies.issubset(set(prefs.allergies))
        dislikes_ok = expected_dislikes.issubset(set(prefs.dislikes))
        filter_ok = not _contains_blocked(returned_ingredients, blocked)
        case_ok = preferences_ok and allergies_ok and dislikes_ok and filter_ok and bool(returned)
        ok += int(case_ok)
        if not case_ok:
            errors.append({
                "session_id": session_id, "query": case["query"],
                "preferences": prefs.to_dict(), "returned": returned,
                "returned_ingredients": returned_ingredients, "blocked": blocked,
                "checks": {
                    "preferences_ok": preferences_ok, "allergies_ok": allergies_ok,
                    "dislikes_ok": dislikes_ok, "filter_ok": filter_ok, "has_results": bool(returned),
                },
            })

    total = len(cases)
    print(f"cases={total}")
    print(f"memory_backend={getattr(memory_store, 'backend', 'memory')}")
    print(f"preference_success={ok / total:.2%} ({ok}/{total})")
    if args.show_errors and errors:
        print("errors:")
        for error in errors:
            print(json.dumps(error, ensure_ascii=False))
    return 0 if ok == total else 1


def _contains_blocked(ingredients: list[str], blocked: list[str]) -> bool:
    return any(block in ingredient or ingredient in block for ingredient in ingredients for block in blocked if block)


# ── safety ─────────────────────────────────────────────────────────────────

def _wilson_upper(failures: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0:
        return 1.0
    p = failures / total
    denominator = 1 + z * z / total
    center = p + z * z / (2 * total)
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return (center + radius) / denominator


def cmd_safety(args: argparse.Namespace) -> int:
    cases = _load_cases(Path(args.eval_file))
    boundary_failures = 0
    answer_guard_failures = 0
    errors: list[dict] = []
    for index, case in enumerate(cases):
        response = _call_chat(args.base_url, case["message"], f"safety-eval-{index}", args.timeout)
        meta = response.get("meta", {})
        actual = meta.get("query_boundary", {}).get("decision")
        if actual != case["expect_decision"]:
            boundary_failures += 1
            errors.append({"name": case["name"], "expected": case["expect_decision"], "actual": actual})
        if meta.get("answer_guard") in {"retryable_unsupported_claims"}:
            answer_guard_failures += 1
            errors.append({"name": case["name"], "answer_guard": meta.get("answer_guard")})

    total = len(cases)
    boundary_rate = boundary_failures / total if total else 1.0
    hallucination_rate = answer_guard_failures / total if total else 1.0
    boundary_upper = _wilson_upper(boundary_failures, total)
    hallucination_upper = _wilson_upper(answer_guard_failures, total)
    print(f"cases={total}")
    print(f"boundary_failure_rate={boundary_rate:.2%} ({boundary_failures}/{total})")
    print(f"answer_guard_failure_rate={hallucination_rate:.2%} ({answer_guard_failures}/{total})")
    print(f"boundary_wilson_95_upper={boundary_upper:.2%}")
    print(f"answer_guard_wilson_95_upper={hallucination_upper:.2%}")
    print(f"target={args.target:.2%}")
    if args.show_errors:
        for error in errors:
            print(json.dumps(error, ensure_ascii=False))
    return 0 if max(boundary_rate, hallucination_rate, boundary_upper, hallucination_upper) < args.target else 1


# ── query-understanding ────────────────────────────────────────────────────

def cmd_query_understanding(args: argparse.Namespace) -> int:
    cases = _load_cases(Path(args.eval_file))
    correction_ok = 0
    intent_ok = 0
    errors: list[dict] = []
    for index, case in enumerate(cases):
        response = _call_chat(args.base_url, case["query"], f"query-typo-{index}", args.timeout)
        understanding = response.get("meta", {}).get("query_understanding", {})
        resolved = understanding.get("resolved_query")
        if resolved == case["expected"]:
            correction_ok += 1
        else:
            errors.append({"name": case["name"], "kind": "correction", "expected": case["expected"], "actual": resolved})
        if response.get("intent") == case["expected_intent"]:
            intent_ok += 1
        else:
            errors.append({"name": case["name"], "kind": "intent", "expected": case["expected_intent"], "actual": response.get("intent")})

    total = len(cases)
    print(f"cases={total}")
    print(f"correction_accuracy={correction_ok / total:.2%} ({correction_ok}/{total})")
    print(f"intent_recovery_accuracy={intent_ok / total:.2%} ({intent_ok}/{total})")
    if args.show_errors:
        for error in errors:
            print(json.dumps(error, ensure_ascii=False))
    return 0 if correction_ok == total and intent_ok == total else 1


# ── main ───────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="SmartRecipe 评估工具。")
    sub = parser.add_subparsers(dest="command", required=True)

    # chat
    p_chat = sub.add_parser("chat", help="端到端 /chat 评估（HTTP）。")
    p_chat.add_argument("--base-url", default="http://127.0.0.1:8010")
    p_chat.add_argument("--eval-file", default=str(ROOT_DIR / "data" / "evals" / "chat_v2_cases.jsonl"))
    p_chat.add_argument("--timeout", type=int, default=120)
    p_chat.add_argument("--show-errors", action="store_true")
    p_chat.set_defaults(func=cmd_chat)

    # retrieval
    p_retrieval = sub.add_parser("retrieval", help="RAG 检索命中率评估。")
    p_retrieval.add_argument("--eval-file", default=str(ROOT_DIR / "data" / "evals" / "rag_retrieval.jsonl"))
    p_retrieval.add_argument("--backend", default=os.getenv("RAG_BACKEND", "bm25"), choices=["bm25", "chroma", "faiss"])
    p_retrieval.add_argument("--max-k", type=int, default=5)
    p_retrieval.add_argument("--show-errors", action="store_true")
    p_retrieval.set_defaults(func=cmd_retrieval)

    # router
    p_router = sub.add_parser("router", help="Router 意图分类评估。")
    p_router.add_argument("--eval-file", default=str(ROOT_DIR / "data" / "evals" / "router_intents.jsonl"))
    p_router.add_argument("--use-llm", action="store_true")
    p_router.add_argument("--show-errors", action="store_true")
    p_router.add_argument("--enable-database-agents", action="store_true")
    p_router.add_argument("--disable-database-agents", action="store_true")
    p_router.add_argument("--enable-v2", action="store_true", help=argparse.SUPPRESS)
    p_router.add_argument("--disable-v2", action="store_true", help=argparse.SUPPRESS)
    p_router.set_defaults(func=cmd_router)

    # text2sql
    p_text2sql = sub.add_parser("text2sql", help="Text2SQL 评估。")
    p_text2sql.set_defaults(func=cmd_text2sql)

    # text2cypher
    p_text2cypher = sub.add_parser("text2cypher", help="Text2Cypher 评估。")
    p_text2cypher.set_defaults(func=cmd_text2cypher)

    # document-rag
    p_doc = sub.add_parser("document-rag", help="PDF 文档 RAG 评估。")
    p_doc.add_argument("--index", default=str(ROOT_DIR / "data" / "processed" / "new_pdf_recipe.index"))
    p_doc.add_argument("--metadata", default=str(ROOT_DIR / "data" / "processed" / "new_pdf_recipe_metadata.json"))
    p_doc.add_argument("--eval-file", default=None)
    p_doc.add_argument("--top-k", type=int, default=5)
    p_doc.add_argument("--candidate-k", type=int, default=30)
    p_doc.add_argument("--limit", type=int, default=0)
    p_doc.add_argument("--shuffle", action="store_true")
    p_doc.add_argument("--seed", type=int, default=42)
    p_doc.add_argument("--hyde", action="store_true")
    p_doc.add_argument("--no-query-rewrite", action="store_true")
    p_doc.add_argument("--no-cross-encoder-rerank", action="store_true")
    p_doc.add_argument("--cross-encoder-model", default=None)
    p_doc.add_argument("--output", default="")
    p_doc.add_argument("--show-errors", action="store_true")
    p_doc.add_argument("--progress-every", type=int, default=10)
    p_doc.set_defaults(func=cmd_document_rag)

    # preferences
    p_prefs = sub.add_parser("preferences", help="偏好记忆评估。")
    p_prefs.add_argument("--eval-file", default=str(ROOT_DIR / "data" / "evals" / "preference_memory_cases.jsonl"))
    p_prefs.add_argument("--backend", choices=["memory", "redis"], default="memory")
    p_prefs.add_argument("--show-errors", action="store_true")
    p_prefs.set_defaults(func=cmd_preferences)

    # safety
    p_safety = sub.add_parser("safety", help="安全防御评估（HTTP）。")
    p_safety.add_argument("--base-url", default="http://127.0.0.1:8010")
    p_safety.add_argument("--eval-file", default=str(ROOT_DIR / "data" / "evals" / "safety_defense_cases.jsonl"))
    p_safety.add_argument("--target", type=float, default=0.04)
    p_safety.add_argument("--timeout", type=int, default=120)
    p_safety.add_argument("--show-errors", action="store_true")
    p_safety.set_defaults(func=cmd_safety)

    # query-understanding
    p_qu = sub.add_parser("query-understanding", help="错别字纠正评估（HTTP）。")
    p_qu.add_argument("--base-url", default="http://127.0.0.1:8010")
    p_qu.add_argument("--eval-file", default=str(ROOT_DIR / "data" / "evals" / "query_typo_cases.jsonl"))
    p_qu.add_argument("--timeout", type=int, default=120)
    p_qu.add_argument("--show-errors", action="store_true")
    p_qu.set_defaults(func=cmd_query_understanding)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
