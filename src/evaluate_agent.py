import json
import re
from pathlib import Path

from agent import run_agent

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_PATH = BASE_DIR / "data" / "agent_eval_set.json"
DOC_PATH = BASE_DIR / "data" / "sample_docs" / "eval_composite_spec.txt"

CITATION_PATTERN = re.compile(r"근거 사고사례\*\*:\s*\[(\d+)\]")


def main():
    with open(EVAL_PATH, encoding="utf-8") as f:
        eval_set = json.load(f)
    with open(DOC_PATH, encoding="utf-8") as f:
        doc_text = f.read()

    print(f"평가 문항 수: {len(eval_set)}")
    print("에이전트 실행 중 (합성 시방서 전체를 한 번에 분석)...\n")

    def log_call(query, results):
        print(f"  🔍 검색: '{query}'")

    report, tool_calls = run_agent(doc_text, on_tool_call=log_call)
    cited_ids = {int(m) for m in CITATION_PATTERN.findall(report)}

    print(f"\n총 도구 호출 {len(tool_calls)}회, 최종 인용 사례 {len(cited_ids)}건")
    print("=" * 50)

    hits, misses = [], []
    for item in eval_set:
        if item["expected_id"] in cited_ids:
            hits.append(item)
        else:
            misses.append(item)

    recall = len(hits) / len(eval_set)
    print(f"\n인용 재현율 (Citation Recall): {len(hits)}/{len(eval_set)} = {recall:.1%}")
    print("(= 원본이 되는 실제 사고사례가 최종 보고서의 근거로 정확히 인용된 비율)")

    if misses:
        print(f"\n인용되지 못한 원본 사례 ({len(misses)}건):")
        for m in misses:
            print(f"  - (id={m['expected_id']}, {m['공정']}) {m['sentence'][:60]}...")

    print("\n" + "=" * 50)
    print("최종 보고서:\n")
    print(report)


if __name__ == "__main__":
    main()
