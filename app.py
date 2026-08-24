import sys
from pathlib import Path

import streamlit as st
from pypdf import PdfReader

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from agent import run_agent  # noqa: E402
from report_export import export_to_excel, parse_report_to_rows  # noqa: E402
from verify import verify_report  # noqa: E402

st.set_page_config(page_title="설계문서 위험요소 자동 도출 에이전트", page_icon="🤖")
st.title("🤖 설계문서 위험요소 자동 도출 에이전트")
st.caption("설계문서(시방서)를 입력하면, 에이전트가 위험요소를 도출하고 과거 사고사례를 스스로 검색해 근거를 붙입니다.")

SAMPLE_PATH = Path(__file__).resolve().parent / "data" / "sample_docs" / "tunnel_spec.txt"


def extract_pdf_text(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


with st.expander("샘플 시방서 불러오기"):
    if st.button("터널 공사 시방서 샘플 채우기"):
        st.session_state["doc_text"] = SAMPLE_PATH.read_text(encoding="utf-8")

uploaded_pdf = st.file_uploader("설계문서(시방서) PDF 업로드", type=["pdf"])
if uploaded_pdf is not None and st.session_state.get("_last_pdf_name") != uploaded_pdf.name:
    extracted = extract_pdf_text(uploaded_pdf)
    if extracted:
        st.session_state["doc_text"] = extracted
        st.session_state["_last_pdf_name"] = uploaded_pdf.name
    else:
        st.warning("PDF에서 텍스트를 추출하지 못했습니다 (스캔 이미지 PDF일 수 있습니다). 텍스트를 직접 붙여넣어 주세요.")

doc_text = st.text_area(
    "설계문서(시방서) 내용 (PDF 업로드 시 자동으로 채워지며, 직접 붙여넣거나 수정할 수도 있습니다)",
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

    rows = parse_report_to_rows(report)
    if rows:
        excel_bytes = export_to_excel(rows)
        st.download_button(
            "📥 위험성평가표 다운로드 (Excel)",
            data=excel_bytes,
            file_name="위험성평가표.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.markdown(report)

    with st.spinner("근거-주장 정합성 자동 재검증 중..."):
        verification = verify_report(report)

    if verification:
        passed = sum(1 for v in verification if v["통과"])
        st.info(
            f"🔍 자동 정합성 검증: {passed}/{len(verification)}건 통과 — "
            "별도 LLM으로 '인용된 사례가 실제로 그 위험요소를 뒷받침하는지'만 다시 확인한 결과입니다. "
            "**최종 판정이 아니라 재확인이 필요한 항목을 먼저 걸러내는 스크리닝용**이니, "
            "⚠️ 표시된 항목은 DFS 보고서에 옮기기 전에 원문을 직접 확인하세요."
        )
        with st.expander("자동 검증 상세 보기"):
            for v in verification:
                mark = "✅" if v["통과"] else "⚠️"
                st.markdown(f"{mark} **[{v['공정']}]** {v['위험요소']}")
                st.caption(v["검증_사유"])

    with st.expander(f"에이전트가 실제로 검색한 내용 보기 (총 {len(tool_calls)}회)"):
        for call in tool_calls:
            st.markdown(f"**검색어**: `{call['query']}`")
            for r in call["results"]:
                st.markdown(
                    f"- (점수 {r['점수']}) [{r['id']}] {r['공정']} / {r['사고유형']} — {r['내용'][:80]}..."
                )
