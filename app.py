import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from agent import run_agent  # noqa: E402

st.set_page_config(page_title="설계문서 위험요소 자동 도출 에이전트", page_icon="🤖")
st.title("🤖 설계문서 위험요소 자동 도출 에이전트")
st.caption("설계문서(시방서)를 입력하면, 에이전트가 위험요소를 도출하고 과거 사고사례를 스스로 검색해 근거를 붙입니다.")

SAMPLE_PATH = Path(__file__).resolve().parent / "data" / "sample_docs" / "tunnel_spec.txt"

with st.expander("샘플 시방서 불러오기"):
    if st.button("터널 공사 시방서 샘플 채우기"):
        st.session_state["doc_text"] = SAMPLE_PATH.read_text(encoding="utf-8")

doc_text = st.text_area(
    "설계문서(시방서) 내용을 붙여넣으세요",
    key="doc_text",
    height=250,
    placeholder="예: 제3장 터널 굴착 및 지보공...",
)

if st.button("위험요소 분석 시작", type="primary", disabled=not doc_text.strip()):
    log_container = st.container()
    log_placeholder = log_container.empty()
    call_log = []

    def on_tool_call(query, results):
        call_log.append(f"🔍 검색: `{query}` → {len(results)}건")
        log_placeholder.markdown("\n\n".join(call_log))

    with st.spinner("에이전트가 문서를 분석하고 사고사례를 검색하는 중..."):
        report, tool_calls = run_agent(doc_text, on_tool_call=on_tool_call)

    st.success(f"분석 완료 (도구 호출 {len(tool_calls)}회)")
    st.markdown(report)

    with st.expander(f"에이전트가 실제로 검색한 내용 보기 (총 {len(tool_calls)}회)"):
        for call in tool_calls:
            st.markdown(f"**검색어**: `{call['query']}`")
            for r in call["results"]:
                st.markdown(
                    f"- (점수 {r['점수']}) [{r['id']}] {r['공정']} / {r['사고유형']} — {r['내용'][:80]}..."
                )
