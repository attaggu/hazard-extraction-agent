import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from kiwipiepy import Kiwi
from openai import OpenAI
from rank_bm25 import BM25Okapi

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_PATH = BASE_DIR / "data" / "index.npz"
BM25_CORPUS_PATH = BASE_DIR / "data" / "bm25_corpus.json"
SEVERITY_PATH = BASE_DIR / "data" / "severity.json"
EMBEDDING_MODEL = "text-embedding-3-small"
HYBRID_ALPHA = 0.5

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
kiwi = Kiwi()

_cache = {}


def tokenize(text: str) -> list[str]:
    return [t.form for t in kiwi.tokenize(text) if t.tag.startswith(("N", "V", "SL"))]


def load_index():
    if "data" not in _cache:
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"{INDEX_PATH} 가 없습니다. 먼저 `python src/build_index.py` 를 실행하세요."
            )
        data = np.load(INDEX_PATH, allow_pickle=True)
        with open(BM25_CORPUS_PATH, encoding="utf-8") as f:
            corpus_tokens = json.load(f)
        severity = {}
        if SEVERITY_PATH.exists():
            with open(SEVERITY_PATH, encoding="utf-8") as f:
                severity = json.load(f)
        _cache["data"] = (
            data["vectors"],
            [json.loads(d) for d in data["docs"]],
            BM25Okapi(corpus_tokens),
            severity,
        )
    return _cache["data"]


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    query_norm = query_vec / np.linalg.norm(query_vec)
    doc_norms = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    return doc_norms @ query_norm


def normalize(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def search_accident_cases(query: str, top_k: int = 3) -> list[dict]:
    """과거 건설안전 사고사례 DB에서 질의와 관련된 사례를 검색합니다 (벡터+BM25 하이브리드)."""
    vectors, docs, bm25, severity = load_index()

    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    query_vec = np.array(response.data[0].embedding, dtype=np.float32)
    cosine_scores = cosine_similarity(query_vec, vectors)
    bm25_scores = np.array(bm25.get_scores(tokenize(query)))

    combined = HYBRID_ALPHA * normalize(cosine_scores) + (1 - HYBRID_ALPHA) * normalize(
        bm25_scores
    )
    top_indices = np.argsort(combined)[::-1][:top_k]

    results = []
    for i in top_indices:
        doc_id = docs[i]["id"]
        sev = severity.get(str(doc_id), {})
        results.append(
            {
                "id": doc_id,
                "공정": docs[i]["공정"],
                "사고유형": docs[i]["사고유형"],
                "내용": docs[i]["내용"],
                "원인": docs[i]["원인"],
                "재발방지대책": docs[i]["재발방지대책"],
                "점수": round(float(combined[i]), 3),
                "사망자수": sev.get("사망자", "미상"),
                "부상자수": sev.get("부상자", "미상"),
                "통계기반등급": sev.get("통계기반등급", "미상"),
            }
        )
    return results


# OpenAI function-calling용 도구 스펙
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_accident_cases",
        "description": "과거 철도/교량/터널 건설현장 사고사례 DB에서 특정 위험요소나 작업 상황과 "
        "관련된 실제 사고사례를 검색합니다. 위험요소를 주장하기 전에 반드시 이 도구로 근거를 확인하세요.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "검색할 작업/위험 상황 (예: '터널 굴착 중 낙석', '전차선 인접 작업')",
                }
            },
            "required": ["query"],
        },
    },
}
