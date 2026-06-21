from __future__ import annotations

import re


DANGEROUS_SQL_RE = re.compile(r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke)\b", re.I)
DANGEROUS_CYPHER_RE = re.compile(r"\b(create|merge|set|delete|detach|remove|drop|call\s+dbms)\b", re.I)


def validate_readonly_sql(sql: str) -> None:
    text = sql.strip()
    if not text.lower().startswith("select"):
        raise ValueError("Only SELECT SQL is allowed.")
    if DANGEROUS_SQL_RE.search(text):
        raise ValueError("Dangerous SQL keyword is not allowed.")


def validate_readonly_cypher(cypher: str) -> None:
    text = cypher.strip()
    lowered = text.lower()
    if not (lowered.startswith("match") or lowered.startswith("optional match")):
        raise ValueError("Only read-only MATCH Cypher is allowed.")
    if DANGEROUS_CYPHER_RE.search(text):
        raise ValueError("Dangerous Cypher keyword is not allowed.")


def ensure_limit(query: str, limit: int = 10) -> str:
    if re.search(r"\blimit\s+\d+\b", query, flags=re.I):
        return query
    return query.rstrip().rstrip(";") + f"\nLIMIT {limit}"
