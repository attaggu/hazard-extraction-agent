import json
from pathlib import Path

import pandas as pd

"""
rag-safety-chatbot/src/prepare_data.py 가 원본 CSV를 필터링해 accident_cases.json(id 1..1594)을
만들 때와 "동일한 필터 조건 + 동일한 순서"로 원본 CSV를 다시 읽어, 위치(순서) 기준으로
사망자/부상자 통계를 매칭해서 심각도 데이터를 만든다.

주의: prepare_data.py의 필터 로직이 바뀌면 이 매칭도 깨지므로, 두 스크립트는 함께 유지되어야 함.
"""

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = (
    BASE_DIR.parent
    / "rag-safety-chatbot"
    / "data"
    / "raw"
    / "국토안전관리원_건설안전사고사례_20250630.csv"
)
ACCIDENT_CASES_PATH = BASE_DIR / "data" / "accident_cases.json"
OUT_PATH = BASE_DIR / "data" / "severity.json"

TARGET_CATEGORIES = ["철도", "교량", "터널"]


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_PATH} 를 찾을 수 없습니다. rag-safety-chatbot 프로젝트의 원본 CSV가 필요합니다."
        )

    df = pd.read_csv(RAW_PATH, encoding="cp949", low_memory=False)
    sub = df[df["시설물 중분류"].isin(TARGET_CATEGORIES)].copy()
    sub = sub[sub["사고경위"].notna()]

    with open(ACCIDENT_CASES_PATH, encoding="utf-8") as f:
        docs = json.load(f)

    if len(sub) != len(docs):
        raise ValueError(
            f"필터링된 원본 행 수({len(sub)})와 accident_cases.json 건수({len(docs)})가 다릅니다. "
            "prepare_data.py의 필터 조건이 바뀌지 않았는지 확인하세요."
        )

    severity = {}
    for doc, (_, row) in zip(docs, sub.iterrows()):
        사망자 = int(row["사망자"]) if pd.notna(row["사망자"]) else 0
        부상자 = int(row["부상자"]) if pd.notna(row["부상자"]) else 0
        if 사망자 >= 1:
            grade = "상"
        elif 부상자 >= 1:
            grade = "중"
        else:
            grade = "하"
        severity[str(doc["id"])] = {"사망자": 사망자, "부상자": 부상자, "통계기반등급": grade}

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(severity, f, ensure_ascii=False, indent=2)

    grade_counts = pd.Series([v["통계기반등급"] for v in severity.values()]).value_counts()
    print(f"{len(severity)}건 심각도 데이터 저장 완료: {OUT_PATH}")
    print(grade_counts)


if __name__ == "__main__":
    main()
