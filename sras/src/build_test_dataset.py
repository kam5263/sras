"""
테스트 데이터셋 생성: build_dataset.py와 동일한 방식,
학습셋(qa_pairs.jsonl)에 등장한 doc_id는 완전히 제외.

출력:
  sras/data/test/raw/*.txt        – 테스트 전용 문서
  sras/data/test/qa_pairs_test.jsonl
"""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT        = Path(__file__).resolve().parents[2]
RAW_DATASET = ROOT / "raw_dataset" / "Training"
LABEL_BASE  = RAW_DATASET / "02.라벨링데이터"
SOURCE_BASE = RAW_DATASET / "01.원천데이터"

TRAIN_QA    = ROOT / "sras" / "data" / "qa_pairs.jsonl"
OUT_RAW     = ROOT / "sras" / "data" / "test" / "raw"
OUT_QA      = ROOT / "sras" / "data" / "test" / "qa_pairs_test.jsonl"

OUT_RAW.mkdir(parents=True, exist_ok=True)

CATEGORIES = [
    ("TL_법령_QA",   "TS_법령",   "lawId",       "HJ_B_"),
    ("TL_판결문_QA", "TS_판결문", "precedId",    "HJ_P_"),
    ("TL_결정례_QA", "TS_결정례", "determintId", "HJ_K_"),
    ("TL_해석례_QA", "TS_해석례", "interpreId",  "HJ_H_"),
]

# 테스트셋 목표: (max_docs, max_qa)
# 학습셋 800개의 약 25% = 200개
# 법령 ~2 QA/doc, 나머지 ~1 QA/doc
TARGET = {
    "HJ_B_": (65, 120),   # 실측 ~1.5 QA/doc → 65 docs
    "HJ_P_": (95,  95),   # 실측 ~1.3 QA/doc → 95 docs
    "HJ_K_": (25,  25),
    "HJ_H_": (25,  25),
}
TOTAL_CAP      = 200
MAX_QA_PER_DOC = 4

random.seed(99)   # 학습셋(seed=42)과 다른 seed


def load_train_doc_ids() -> set[str]:
    ids: set[str] = set()
    if TRAIN_QA.exists():
        for line in TRAIN_QA.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ids.add(json.loads(line)["gold_doc_id"])
    return ids


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
    train_doc_ids: set[str],
) -> list[dict]:
    max_docs, max_qa = TARGET[prefix]

    # QA 파일을 문서 ID 기준으로 그룹화
    doc_to_qa_files: dict[str, list[Path]] = defaultdict(list)
    for qa_file in label_dir.glob("*.json"):
        try:
            data = json.load(open(qa_file, encoding="utf-8"))
            doc_id_raw = data["info"].get(id_field, "").strip()
            if doc_id_raw:
                doc_to_qa_files[doc_id_raw].append(qa_file)
        except Exception:
            continue

    # 소스 CSV 존재 + 학습셋 미사용 문서만 필터링
    valid_doc_ids = [
        doc_id for doc_id in doc_to_qa_files
        if (source_dir / f"{prefix}{doc_id}.csv").exists()
        and f"{prefix}{doc_id}" not in train_doc_ids
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

        if not txt_path.exists():
            text = csv_to_text(csv_path)
            if not text:
                continue
            txt_path.write_text(text, encoding="utf-8")

        qa_files = doc_to_qa_files[doc_id_raw][:]
        random.shuffle(qa_files)
        added = 0

        for qa_file in qa_files:
            if added >= MAX_QA_PER_DOC or len(results) >= max_qa:
                break
            try:
                data   = json.load(open(qa_file, encoding="utf-8"))
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
    # 기존 테스트 파일 초기화
    for f in OUT_RAW.glob("*.txt"):
        f.unlink()
    if OUT_QA.exists():
        OUT_QA.unlink()

    train_doc_ids = load_train_doc_ids()
    print(f"학습셋 제외 doc_id: {len(train_doc_ids)}개")

    all_pairs: list[dict] = []

    for label_name, source_name, id_field, prefix in CATEGORIES:
        label_dir  = LABEL_BASE / label_name
        source_dir = SOURCE_BASE / source_name

        pairs = collect_category(label_dir, source_dir, id_field, prefix, train_doc_ids)
        doc_count = len({p["gold_doc_id"] for p in pairs})
        print(f"{label_name}: {len(pairs)} QA, {doc_count} docs")
        all_pairs.extend(pairs)

    random.shuffle(all_pairs)
    all_pairs = all_pairs[:TOTAL_CAP]

    with open(OUT_QA, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    doc_count = len(list(OUT_RAW.glob("*.txt")))
    print(f"\n완료: {len(all_pairs)} QA pairs, {doc_count} documents in test/raw/")

    # 학습셋 doc_id 겹침 검증
    test_ids = {json.loads(l)["gold_doc_id"] for l in OUT_QA.read_text().splitlines() if l}
    overlap  = test_ids & train_doc_ids
    print(f"학습셋 doc_id 겹침: {len(overlap)}개 {'✓' if not overlap else '← 오류'}")


if __name__ == "__main__":
    main()
