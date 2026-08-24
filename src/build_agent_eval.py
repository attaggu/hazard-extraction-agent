import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "accident_cases.json"
EVAL_PATH = BASE_DIR / "data" / "agent_eval_set.json"
DOC_OUT_PATH = BASE_DIR / "data" / "sample_docs" / "eval_composite_spec.txt"

CHAT_MODEL = "gpt-4o-mini"
SAMPLE_SIZE = 15
SEED = 7

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SPEC_SENTENCE_PROMPT = """다음은 실제 건설현장 사고사례입니다.

공정: {공정}
내용: {내용}

이 사고가 나기 "전" 설계문서(시방서)에 있었을 법한, 그 작업 자체를 담담하게 설명하는
문장을 1~2문장으로 작성하세요.
조건:
- "위험", "사고", "주의" 같은 단어는 쓰지 마세요. 그냥 작업 절차/방법을 서술하는 시방서 문체로.
- 문장만 출력하고 다른 설명은 붙이지 마세요."""


def generate_spec_sentence(doc: dict) -> str:
    prompt = SPEC_SENTENCE_PROMPT.format(공정=doc["공정"], 내용=doc["내용"])
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def main():
    with open(DATA_PATH, encoding="utf-8") as f:
        docs = json.load(f)

    random.seed(SEED)
    sample = random.sample(docs, SAMPLE_SIZE)

    eval_set = []
    sentences = []
    for i, doc in enumerate(sample, start=1):
        sentence = generate_spec_sentence(doc)
        eval_set.append({"expected_id": doc["id"], "sentence": sentence, "공정": doc["공정"]})
        sentences.append(f"{i}. {sentence}")
        print(f"[{i}/{SAMPLE_SIZE}] (id={doc['id']}) {sentence}")

    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, ensure_ascii=False, indent=2)
    print(f"\n평가셋 저장 완료: {EVAL_PATH}")

    composite_doc = (
        "[평가용 합성 시방서]\n\n제1장 각종 작업 절차\n\n" + "\n\n".join(sentences)
    )
    DOC_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DOC_OUT_PATH, "w", encoding="utf-8") as f:
        f.write(composite_doc)
    print(f"합성 시방서 저장 완료: {DOC_OUT_PATH}")


if __name__ == "__main__":
    main()
