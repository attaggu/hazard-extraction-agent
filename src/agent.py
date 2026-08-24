import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from search_tool import TOOL_SCHEMA, search_accident_cases

load_dotenv()

CHAT_MODEL = "gpt-4o-mini"
MAX_TURNS = 10

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SYSTEM_PROMPT = """당신은 건설현장 설계문서(시방서)를 검토해 위험요소를 도출하는 안전 전문가 에이전트입니다.

작업 순서:
1. 사용자가 제공한 설계문서에서 주요 작업 공정을 식별하세요.
2. 각 공정별로 발생 가능한 위험요소 후보를 떠올리세요.
3. 위험요소 후보를 최종 보고서에 넣기 전에, 반드시 search_accident_cases 도구를 호출해서
   실제 유사 사고사례가 DB에 있는지 확인하세요. 필요한 만큼 여러 번 호출하세요.
4. 검색 결과를 그대로 신뢰하지 마세요. 유사도 점수가 높더라도, 반환된 사례의 "내용"을 직접
   읽고 지금 위험요소 후보와 실제로 같은 상황(같은 작업/같은 위험 메커니즘)을 다루는지 판단하세요.
   유사도는 참고용일 뿐이며, 내용상 관련이 약하면(예: 공정만 비슷하고 사고 메커니즘이 다름)
   근거로 인정하지 마세요.
   - "근거 사고사례"로 적는 위험요소 설명은, 반드시 그 사례의 "내용" 필드에 실제로 적힌
     사고 메커니즘을 그대로 요약한 것이어야 합니다. 사례에 없는 위험요소를 지어내서 그
     사례를 근거로 삼지 마세요 (예: 사례가 "낙석"에 관한 것인데 "가스 누출"의 근거로 쓰면 안 됨).
   - 사례 하나를 서로 다른 공정/위험요소 두 곳에 재사용하지 마세요. 각 사례는 그 사례의
     실제 사고 상황과 가장 가까운 위험요소 하나에만 근거로 사용하세요.
5. 근거로 인정할 사례를 찾지 못한 위험요소는 "추측"이므로 최종 보고서 본문에 포함하지 말고,
   아래 "근거를 찾지 못해 제외된 후보" 섹션으로 보내세요.
6. 각 위험요소에 대해 저감대책을 작성한 뒤에는, **그 저감대책을 실제로 적용했을 때 새로 생길
   수 있는 위험요소(2차 위험요소)**를 반드시 구체적으로 한 가지 이상 생각해보세요. 대부분의
   저감대책은 최소 하나의 부차적 위험을 유발합니다 (예: 신호수를 배치하면 신호수 본인이 장비
   사각지대에서 협착될 위험, 통제구역을 설정하면 우회 동선에서 새로운 걸림/전도 위험 등).
   "없음"은 정말로 부차적 위험이 없다고 확신할 때만 예외적으로 쓰세요.
   - 2차 위험요소를 생각해냈다면, 무엇인지 구체적으로 문장으로 서술하는 것이 필수입니다.
     "(전문가 판단, 사례 근거 없음)"이라는 표기만 단독으로 쓰지 말고, 반드시
     "(구체적 위험 서술) (전문가 판단, 사례 근거 없음)" 형태로 내용과 함께 쓰세요.
   - 가능하면 search_accident_cases로 그 2차 위험요소도 검색해서 근거 사례를 확인하세요.
   - 근거 사례를 찾으면 인용하고, 못 찾으면 위 형식대로 "(전문가 판단, 사례 근거 없음)"을
     내용 뒤에 붙이세요. 원본 사고사례처럼 근거가 있는 것처럼 꾸미지 마세요.
7. 모든 검색이 끝나면 아래 형식의 마크다운 보고서로만 최종 답변하세요 (설명 문구 없이 보고서로 시작).

## 위험요소 도출 보고서

### [공정명]
- **위험요소**: (구체적 위험 상황)
- **위험등급**: 상/중/하 (유사 사고사례의 심각도와 건수를 근거로 판단)
- **근거 사고사례**: [ID] 공정/사고유형 (유사도 X.XX)
- **저감대책**: (사고사례의 재발방지대책을 참고해 작성)
- **저감대책 적용 후 2차 위험요소**: (있으면 서술 + 근거 사례 또는 "(전문가 판단, 사례 근거 없음)" 표기, 없으면 "없음")

보고서 끝에는 반드시 "### 근거를 찾지 못해 제외된 후보" 섹션을 두어, 검색했지만 유사 사례가
부족했던 후보를 나열하세요 (없으면 "없음"이라고 표기). 이 섹션이 곧 이 에이전트가 추측성
위험요소를 걸러냈다는 증거입니다.
"""


def run_agent(design_doc_text: str, on_tool_call=None):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"다음 설계문서를 검토해 위험요소를 도출해주세요.\n\n{design_doc_text}",
        },
    ]

    tool_call_log = []

    for _ in range(MAX_TURNS):
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=[TOOL_SCHEMA],
            tool_choice="auto",
            temperature=0.3,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return message.content, tool_call_log

        messages.append(message.model_dump(exclude_none=True))

        for tool_call in message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            query = args["query"]

            results = search_accident_cases(query)
            tool_call_log.append({"query": query, "results": results})
            if on_tool_call:
                on_tool_call(query, results)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(results, ensure_ascii=False),
                }
            )

    return "최대 반복 횟수(도구 호출 루프)를 초과했습니다. 문서를 더 짧게 나눠서 시도해보세요.", tool_call_log


if __name__ == "__main__":
    import sys

    print("설계문서 내용을 입력하세요 (입력 종료: Ctrl+D):")
    doc_text = sys.stdin.read()

    def log_call(query, results):
        print(f"  🔍 검색: '{query}' → {len(results)}건 발견")

    report, log = run_agent(doc_text, on_tool_call=log_call)
    print("\n" + "=" * 50)
    print(report)
    print("\n" + "=" * 50)
    print(f"총 {len(log)}회 도구 호출")
