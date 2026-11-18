"""On-call knowledge graph schema shared by catalog import, extraction, and retrieval.

Catalog-owned facts (services, teams, dependencies, alerts) come only from YAML. The LLM
extracts incident facts from postmortems and may only use the types listed below.
"""

import re
from typing import Literal

DEPENDENCY_TYPES = ("rpc", "async", "cache", "db")
CATALOG_SOURCE = "catalog"

# Types the LLM may emit (strict=True drops everything else).
ExtractedEntity = Literal["SERVICE", "DATASTORE", "INCIDENT", "SYMPTOM", "MITIGATION"]
ExtractedRelation = Literal["AFFECTED", "ROOT_CAUSE_IN", "SHOWED", "RESOLVED_BY"]
EXTRACTION_SCHEMA = [
    ("INCIDENT", "AFFECTED", "SERVICE"),
    ("INCIDENT", "AFFECTED", "DATASTORE"),
    ("INCIDENT", "ROOT_CAUSE_IN", "SERVICE"),
    ("INCIDENT", "ROOT_CAUSE_IN", "DATASTORE"),
    ("INCIDENT", "SHOWED", "SYMPTOM"),
    ("INCIDENT", "RESOLVED_BY", "MITIGATION"),
]
RELATION_PROPS = [
    ("service", "For SHOWED: the service or datastore where the symptom was observed."),
    ("detail", "A short phrase from the text, such as the metric or the action taken."),
]

# Controlled symptom vocabulary: canonical name -> Chinese and English keywords.
# Matching "the same symptom" across incidents only works on these canonical names.
SYMPTOMS = {
    "latency_high": (
        "延迟",
        "超时",
        "变慢",
        "latency",
        "slow",
        "timeout",
        "timed out",
        "p99",
        "p95",
    ),
    "error_rate_high": (
        "错误率",
        "报错",
        "失败率",
        "error rate",
        "errors",
        "5xx",
        "failed requests",
    ),
    "unavailable": ("不可用", "挂了", "宕机", "无法访问", "unavailable", "unreachable", "down"),
    "consumer_lag": ("积压", "堆积", "消费延迟", "consumer lag", "backlog", "queue depth"),
    "memory_leak": ("内存泄漏", "内存", "oom", "memory leak", "memory"),
    "cpu_high": ("cpu",),
    "cache_failure": ("缓存", "cache"),
    "connection_exhausted": ("连接打满", "连接池", "connection pool", "too many connections"),
    "readiness_failure": ("就绪探针", "readiness", "probe"),
    "lock_contention": ("锁竞争", "锁等待", "lock contention", "deadlock"),
}


def contains_term(text, term):
    """Whole-word match for Latin terms (so "down" misses "download"); substring for Chinese."""
    text, term = text.lower(), term.lower()
    if term.isascii():
        return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None
    return term in text


def match_symptoms(text):
    """Return canonical symptoms whose keywords appear in the text."""
    return {name for name, words in SYMPTOMS.items() if any(contains_term(text, w) for w in words)}


def canonical_symptom(name):
    """Map an extracted symptom phrase to the vocabulary, or None if it does not fit."""
    lowered = name.strip().lower().replace(" ", "_")
    if lowered in SYMPTOMS:
        return lowered
    matches = match_symptoms(name)
    return next(iter(sorted(matches))) if len(matches) == 1 else None
