"""
raw_dataset -> sras/data/raw/*.txt + sras/data/qa_pairs.jsonl

각 카테고리(법령·판결문·결정례·해석례)에서 QA 라벨링 JSON을 읽고,
대응하는 원천 CSV를 텍스트로 변환해 raw/ 에 저장한 뒤
qa_pairs.jsonl 을 생성한다.

목표: 800개 QA 쌍 + ~200개 문서 pool (문서당 최대 4개 QA)
"""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DATASET = ROOT / "raw_dataset" / "Training"
LABEL_BASE  = RAW_DATASET / "02.라벨링데이터"
SOURCE_BASE = RAW_DATASET / "01.원천데이터"

OUT_RAW = ROOT / "sras" / "data" / "raw"
OUT_QA  = ROOT / "sras" / "data" / "qa_pairs.jsonl"

OUT_RAW.mkdir(parents=True, exist_ok=True)

CATEGORIES = [
    ("TL_법령_QA",   "TS_법령",   "lawId",       "HJ_B_"),
    ("TL_판결문_QA", "TS_판결문", "precedId",    "HJ_P_"),
    ("TL_결정례_QA", "TS_결정례", "determintId", "HJ_K_"),
    ("TL_해석례_QA", "TS_해석례", "interpreId",  "HJ_H_"),
]

# 카테고리별 목표: (max_docs, max_qa)
# 법령: 문서당 ~3 QA → 80 docs
# 판결문/결정례/해석례: 문서당 ~1 QA → 문서 수 = 목표 QA 수
TARGET = {
    "HJ_B_": (130, 260),   # 법령: ~1.9 QA/doc 실측 → 130 docs
    "HJ_P_": (330, 330),   # 판결문: ~1 QA/doc
    "HJ_K_": (160, 160),   # 결정례
    "HJ_H_": (160, 160),   # 해석례
}
TOTAL_CAP = 800            # 최종 정확히 800개로 자르기
MAX_QA_PER_DOC = 4   # 문서당 QA 상한 (다양성 유지)

random.seed(42)


def csv_to_text(csv_path: Path) -> str:
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            content = row.get("내용", "").strip()
            if content:
                rows.append(content)
    return "\n".join(rows)


def collect_category(
    label_dir: Path,
    source_dir: Path,
    id_field: str,
    prefix: str,
) -> list[dict]:
    """
    한 카테고리에서 QA 쌍을 수집한다.

    - 먼저 QA 파일을 문서 ID 기준으로 그룹화
    - 소스 CSV가 존재하는 문서만 대상으로 max_docs개 랜덤 샘플
    - 문서당 최대 MAX_QA_PER_DOC개 QA 사용
    - 총 max_qa개를 채우면 종료
    """
    max_docs, max_qa = TARGET[prefix]

    # 문서 ID → QA 파일 목록 그룹화
    doc_to_qa_files: dict[str, list[Path]] = defaultdict(list)
    for qa_file in label_dir.glob("*.json"):
        try:
            data = json.load(open(qa_file, encoding="utf-8"))
            doc_id_raw = data["info"].get(id_field, "").strip()
            if doc_id_raw:
                doc_to_qa_files[doc_id_raw].append(qa_file)
        except Exception:
            continue

    # 소스 CSV 존재 여부 필터링 후 문서 샘플링
    valid_doc_ids = [
        doc_id for doc_id in doc_to_qa_files
        if (source_dir / f"{prefix}{doc_id}.csv").exists()
    ]
    random.shuffle(valid_doc_ids)
    sampled_docs = valid_doc_ids[:max_docs]

    results: list[dict] = []

    for doc_id_raw in sampled_docs:
        if len(results) >= max_qa:
            break

        doc_id   = f"{prefix}{doc_id_raw}"
        csv_path = source_dir / f"{prefix}{doc_id_raw}.csv"
        txt_path = OUT_RAW / f"{doc_id}.txt"

        # CSV → txt (아직 없을 때만)
        if not txt_path.exists():
            text = csv_to_text(csv_path)
            if not text:
                continue
            txt_path.write_text(text, encoding="utf-8")

        # 이 문서의 QA 파일들을 랜덤으로 섞어서 최대 MAX_QA_PER_DOC개 사용
        qa_files = doc_to_qa_files[doc_id_raw][:]
        random.shuffle(qa_files)
        added = 0

        for qa_file in qa_files:
            if added >= MAX_QA_PER_DOC or len(results) >= max_qa:
                break
            try:
                data  = json.load(open(qa_file, encoding="utf-8"))
                query  = data["label"]["input"].strip()
                answer = data["label"]["output"].strip()
                if query and answer:
                    results.append({
                        "query":       query,
                        "gold_answer": answer,
                        "gold_doc_id": doc_id,
                    })
                    added += 1
            except Exception:
                continue

    return results


def main():
    # 기존 파일 초기화
    for f in OUT_RAW.glob("*.txt"):
        f.unlink()
    if OUT_QA.exists():
        OUT_QA.unlink()

    all_pairs: list[dict] = []

    for label_name, source_name, id_field, prefix in CATEGORIES:
        label_dir  = LABEL_BASE / label_name
        source_dir = SOURCE_BASE / source_name

        pairs = collect_category(label_dir, source_dir, id_field, prefix)
        doc_count = len({p["gold_doc_id"] for p in pairs})
        print(f"{label_name}: {len(pairs)} QA, {doc_count} docs")
        all_pairs.extend(pairs)

    random.shuffle(all_pairs)
    all_pairs = all_pairs[:TOTAL_CAP]

    with open(OUT_QA, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    doc_count = len(list(OUT_RAW.glob("*.txt")))
    print(f"\n완료: {len(all_pairs)} QA pairs, {doc_count} documents in raw/")
    print(f"qa_pairs.jsonl → {OUT_QA}")


if __name__ == "__main__":
    main()
