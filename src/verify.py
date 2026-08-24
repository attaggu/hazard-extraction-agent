import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from report_export import parse_report_to_rows

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "accident_cases.json"
CHAT_MODEL = "gpt-4o-mini"
CITATION_ID_PATTERN = re.compile(r"\[(\d+)\]")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

VERIFY_PROMPT = """아래 [사례 내용]이 [주장된 위험요소]를 실제로 뒷받침하는 근거인지 판단하세요.
같은 작업/같은 사고 메커니즘을 다루면 뒷받침하는 것이고, 공정만 비슷하고 실제 사고 원인이나
상황이 다르면 뒷받침하지 않는 것입니다.

[주장된 위험요소]
{claim}

[사례 내용]
{case_content}

"예" 또는 "아니오"로만 먼저 답하고, 그 뒤에 한 줄로 이유를 쓰세요.
형식: 예/아니오 - 이유"""


def _load_docs_by_id() -> dict:
    with open(DATA_PATH, encoding="utf-8") as f:
        docs = json.load(f)
    return {d["id"]: d for d in docs}


def verify_citation(claim: str, case_content: str) -> dict:
    prompt = VERIFY_PROMPT.format(claim=claim, case_content=case_content)
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = response.choices[0].message.content.strip()
    passed = text.startswith("예")
    return {"passed": passed, "reason": text}


def verify_report(report: str) -> list[dict]:
    """보고서의 각 위험요소-근거사례 쌍이 실제로 정합적인지 별도 LLM 호출로 재검증한다."""
    docs_by_id = _load_docs_by_id()
    rows = parse_report_to_rows(report)

    results = []
    for row in rows:
        ids = [int(m) for m in CITATION_ID_PATTERN.findall(row["근거 사고사례"])]
        if not ids:
            continue
        case_id = ids[0]
        doc = docs_by_id.get(case_id)
        if doc is None:
            continue

        claim = row["위험요소"]
        case_content = f"내용: {doc['내용']}\n원인: {doc['원인']}"
        verdict = verify_citation(claim, case_content)

        results.append(
            {
                "공정": row["공정"],
                "위험요소": claim,
                "근거_id": case_id,
                "통과": verdict["passed"],
                "검증_사유": verdict["reason"],
            }
        )
    return results


if __name__ == "__main__":
    import sys

    print("검증할 보고서를 입력하세요 (Ctrl+D로 종료):")
    report_text = sys.stdin.read()

    results = verify_report(report_text)
    passed = sum(1 for r in results if r["통과"])
    print(f"\n검증 결과: {passed}/{len(results)} 통과\n")
    for r in results:
        mark = "✅" if r["통과"] else "⚠️"
        print(f"{mark} [{r['공정']}] {r['위험요소'][:40]}")
        print(f"   → {r['검증_사유']}")
