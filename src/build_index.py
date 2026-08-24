import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from kiwipiepy import Kiwi
from openai import OpenAI

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "accident_cases.json"
INDEX_PATH = BASE_DIR / "data" / "index.npz"
BM25_CORPUS_PATH = BASE_DIR / "data" / "bm25_corpus.json"

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 200

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
kiwi = Kiwi()


def tokenize(text: str) -> list[str]:
    return [t.form for t in kiwi.tokenize(text) if t.tag.startswith(("N", "V", "SL"))]


def doc_to_text(doc: dict) -> str:
    return (
        f"[공정: {doc['공정']}] [사고유형: {doc['사고유형']}]\n"
        f"내용: {doc['내용']}\n"
        f"원인: {doc['원인']}\n"
        f"재발방지대책: {doc['재발방지대책']}"
    )


def embed_texts(texts: list[str]) -> np.ndarray:
    all_vectors = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_vectors.extend(item.embedding for item in response.data)
        print(f"  {min(start + BATCH_SIZE, len(texts))}/{len(texts)} 완료")
    return np.array(all_vectors, dtype=np.float32)


def main():
    with open(DATA_PATH, encoding="utf-8") as f:
        docs = json.load(f)

    texts = [doc_to_text(doc) for doc in docs]
    print(f"{len(texts)}건 문서 임베딩 생성 중...")
    vectors = embed_texts(texts)

    np.savez(
        INDEX_PATH,
        vectors=vectors,
        docs=np.array([json.dumps(d, ensure_ascii=False) for d in docs]),
        texts=np.array(texts),
    )
    print(f"인덱스 저장 완료: {INDEX_PATH} (shape={vectors.shape})")

    print("BM25용 토큰화 진행 중...")
    corpus_tokens = [tokenize(t) for t in texts]
    with open(BM25_CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus_tokens, f, ensure_ascii=False)
    print(f"BM25 코퍼스 저장 완료: {BM25_CORPUS_PATH}")


if __name__ == "__main__":
    main()
