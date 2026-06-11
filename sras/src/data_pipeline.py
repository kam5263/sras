"""
Data pipeline: raw .txt docs → chunks → synthetic QA pairs → embedding cache.

Stages
------
1. load_docs()            – read data/raw/*.txt
2. chunk_docs()           – semantic chunking (500–800자, 단락/문장 경계 기준)
3. generate_qa_pairs()    – Ollama (qwen2.5:14b) 청크 단위 Korean QA 생성,
                            heuristic fallback when Ollama is unavailable
4. embed_all()            – jhgan/ko-sroberta-multitask으로 청크 + 쿼리 인코딩; 캐시 저장
5. build_candidate_pool() – 각 QA 쌍에 대해 1 gold 청크 + (n-1) 무작위 distractor 청크 반환

Cache format:
  {
    "chunk_ids":    List[str]           – 청크 ID 목록 (e.g. "HJ_K_482_0")
    "chunk_embs":   Tensor[C, 768]      – 청크 임베딩
    "chunk_to_doc": Dict[str, str]      – chunk_id → doc_id
    "chunk_texts":  Dict[str, str]      – chunk_id → 청크 텍스트 (학습 시 generator 입력용)
    "query_embs":   Tensor[Q, 768]      – QA 쌍별 쿼리 임베딩
    "qa_pairs":     List[Dict]          – {"query", "gold_answer", "gold_chunk_id", "gold_doc_id"}
  }

Run as a script to rebuild the full cache from scratch:
    python src/data_pipeline.py
"""

import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

_DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
_ENCODER_NAME  = "jhgan/ko-sroberta-multitask"
_OLLAMA_MODEL  = "qwen2.5:14b"
_OLLAMA_URL    = "http://localhost:11434/api/generate"


# ─── Document loading ────────────────────────────────────────────────────────

def load_docs(raw_dir: str = "data/raw") -> Dict[str, str]:
    """Returns {doc_id: text} for every .txt file in raw_dir."""
    raw_path = Path(raw_dir)
    docs: Dict[str, str] = {}
    for txt_file in sorted(raw_path.glob("*.txt")):
        doc_id = txt_file.stem
        docs[doc_id] = txt_file.read_text(encoding="utf-8").strip()
    return docs


# ─── Semantic chunking ────────────────────────────────────────────────────────

def chunk_docs(
    docs: Dict[str, str],
    min_chars: int = 300,
    max_chars: int = 800,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    문서를 의미 단위 청크로 분할한다.

    전략:
    - 빈 줄(\\n\\n)로 단락을 분리한 뒤, max_chars 내에서 인접 단락을 병합
    - 단락이 max_chars를 초과하면 문장 경계(마침표/줄바꿈)로 재분할
    - min_chars 미만의 잔여 텍스트는 마지막 청크에 병합

    Returns:
        chunks:       {chunk_id: chunk_text}      e.g. {"HJ_K_482_0": "..."}
        chunk_to_doc: {chunk_id: doc_id}
    """
    chunks: Dict[str, str] = {}
    chunk_to_doc: Dict[str, str] = {}

    for doc_id, text in docs.items():
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text.strip()) if p.strip()]
        chunk_idx = 0

        def _save(buf: str) -> None:
            nonlocal chunk_idx
            buf = buf.strip()
            if not buf:
                return
            cid = f"{doc_id}_{chunk_idx}"
            chunks[cid] = buf
            chunk_to_doc[cid] = doc_id
            chunk_idx += 1

        def _split_by_sentence(long_text: str) -> None:
            """max_chars를 초과하는 텍스트를 문장 단위로 추가 분할."""
            sents = re.split(r"(?<=[.!?。])\s+|\n", long_text)
            buf = ""
            for sent in sents:
                sent = sent.strip()
                if not sent:
                    continue
                if buf and len(buf) + len(sent) + 1 > max_chars:
                    _save(buf)
                    buf = sent
                else:
                    buf = (buf + " " + sent).strip() if buf else sent
            if buf:
                _save(buf)

        current = ""
        for para in paragraphs:
            if not current:
                current = para
            elif len(current) + len(para) + 2 <= max_chars:
                current = current + "\n\n" + para
            else:
                # current가 충분히 크면 저장; 아니면 계속 병합
                if len(current) >= min_chars:
                    if len(current) > max_chars:
                        _split_by_sentence(current)
                    else:
                        _save(current)
                    current = para
                else:
                    current = current + "\n\n" + para

        # 남은 텍스트 처리
        if current:
            if len(current) > max_chars:
                _split_by_sentence(current)
            else:
                _save(current)

        # 빈 문서 보호: 청크가 하나도 없으면 전체 텍스트를 하나의 청크로
        if chunk_idx == 0 and text.strip():
            _save(text.strip())

    return chunks, chunk_to_doc


# ─── Question generation ──────────────────────────────────────────────────────

def _answer_spans(text: str) -> List[str]:
    """heuristic fallback: 한국어 텍스트에서 답변 후보 구절 추출."""
    sentences = re.split(r"[.!?。\n]", text)
    spans: List[str] = []
    for sent in sentences:
        words = sent.strip().split()
        if len(words) >= 2:
            chunk = " ".join(words[:min(5, len(words))])
            if chunk:
                spans.append(chunk)
            if len(spans) >= 3:
                break
    return spans or [text[:60]]


def _generate_qa_ollama(
    text: str,
    model: str = _OLLAMA_MODEL,
    ollama_url: str = _OLLAMA_URL,
) -> List[Dict[str, str]]:
    """
    Ollama (qwen2.5:14b)로 청크 단위 한국어 QA 쌍 생성.
    청크가 이미 500~800자이므로 전체 텍스트를 프롬프트에 사용한다.
    """
    prompt = (
        "다음 한국어 문서 청크에서 질문-답변 쌍을 최대 3개 생성하세요.\n"
        "답변은 반드시 아래 청크 내용에서만 찾을 수 있어야 합니다.\n\n"
        f"[청크]\n{text}\n\n"
        "출력 형식 (JSON 배열만, 다른 텍스트 없이):\n"
        '[{"question": "질문", "answer": "답변"}, ...]'
    )
    try:
        resp = requests.post(
            ollama_url,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        pairs = json.loads(raw[start:end])
        return [
            {"question": p.get("question", ""), "answer": p.get("answer", "")}
            for p in pairs
            if p.get("question") and p.get("answer")
        ]
    except Exception as exc:
        print(f"  Ollama QA 생성 실패: {exc}")
        return []


def generate_qa_pairs(
    chunks: Dict[str, str],
    chunk_to_doc: Dict[str, str],
    output_path: str = "data/qa_pairs.jsonl",
    use_model: bool = True,
    seed: int = 42,
) -> List[Dict]:
    """
    청크 단위 합성 QA 쌍 생성.

    Each record: {
        "query":         str,
        "gold_answer":   str,
        "gold_chunk_id": str,   # 정답의 출처 청크
        "gold_doc_id":   str,   # 참조용 원문서 ID
    }
    """
    random.seed(seed)
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    ollama_available = False
    if use_model:
        try:
            ping = requests.get("http://localhost:11434/api/tags", timeout=5)
            ollama_available = ping.status_code == 200
            if ollama_available:
                print(f"Ollama 서버 확인 완료 (모델: {_OLLAMA_MODEL})")
            else:
                print("Ollama 서버 응답 없음 — heuristic fallback 사용")
        except Exception:
            print("Ollama 서버 미실행 — heuristic fallback 사용")

    qa_pairs: List[Dict] = []
    for chunk_id, text in tqdm(chunks.items(), desc="QA 쌍 생성"):
        doc_id = chunk_to_doc[chunk_id]
        if ollama_available:
            pairs = _generate_qa_ollama(text)
            for p in pairs:
                qa_pairs.append({
                    "query":         p["question"],
                    "gold_answer":   p["answer"],
                    "gold_chunk_id": chunk_id,
                    "gold_doc_id":   doc_id,
                })
            if not pairs:
                for span in _answer_spans(text):
                    qa_pairs.append({
                        "query":         f"{span}에 대해 설명하시오.",
                        "gold_answer":   span,
                        "gold_chunk_id": chunk_id,
                        "gold_doc_id":   doc_id,
                    })
        else:
            for span in _answer_spans(text):
                qa_pairs.append({
                    "query":         f"{span}에 대해 설명하시오.",
                    "gold_answer":   span,
                    "gold_chunk_id": chunk_id,
                    "gold_doc_id":   doc_id,
                })

    with open(output_path_obj, "w", encoding="utf-8") as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"저장 완료: {len(qa_pairs)}개 QA 쌍 → {output_path_obj}")
    return qa_pairs


def load_qa_pairs(path: str = "data/qa_pairs.jsonl") -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def resolve_gold_chunks(
    qa_pairs: List[Dict],
    chunks: Dict[str, str],
    chunk_to_doc: Dict[str, str],
) -> List[Dict]:
    """
    gold_chunk_id가 없는 QA 쌍에 대해 gold_doc_id의 청크 중
    gold_answer와 단어 겹침이 가장 많은 청크를 gold_chunk_id로 할당한다.

    build_dataset.py가 만든 구형 qa_pairs.jsonl(gold_doc_id만 존재)을
    청크 기반 파이프라인에서 재사용할 때 사용한다.
    """
    # doc_id → 해당 문서의 chunk_id 목록 (역인덱스)
    doc_to_chunks: Dict[str, List[str]] = {}
    for cid, did in chunk_to_doc.items():
        doc_to_chunks.setdefault(did, []).append(cid)

    resolved: List[Dict] = []
    missing = 0
    for qa in qa_pairs:
        if "gold_chunk_id" in qa:
            resolved.append(qa)
            continue

        doc_id = qa.get("gold_doc_id", "")
        cids = doc_to_chunks.get(doc_id, [])
        if not cids:
            # 해당 doc_id의 청크가 없으면 건너뜀
            missing += 1
            continue

        answer_words = set(re.sub(r"[^\w]", " ", qa.get("gold_answer", "")).split())
        best_cid, best_score = cids[0], -1
        for cid in cids:
            chunk_words = set(re.sub(r"[^\w]", " ", chunks[cid]).split())
            score = len(answer_words & chunk_words)
            if score > best_score:
                best_score, best_cid = score, cid

        resolved.append({**qa, "gold_chunk_id": best_cid})

    if missing:
        print(f"  경고: doc_id 매핑 실패로 {missing}개 QA 쌍 제외됨")
    return resolved


# ─── Embedding cache ──────────────────────────────────────────────────────────

def embed_all(
    chunks: Dict[str, str],
    chunk_to_doc: Dict[str, str],
    qa_pairs: List[Dict],
    cache_path: str = "data/cache/embeddings.pt",
    encoder_name: str = _ENCODER_NAME,
) -> Dict:
    """
    jhgan/ko-sroberta-multitask으로 모든 청크와 쿼리를 인코딩; cache_path에 저장.

    Cache format (dict saved via torch.save):
      {
        "chunk_ids":    List[str]          – 청크 ID 목록
        "chunk_embs":   Tensor[C, 768]     – 청크 임베딩
        "chunk_to_doc": Dict[str, str]     – chunk_id → doc_id
        "chunk_texts":  Dict[str, str]     – chunk_id → 텍스트 (generator 입력용)
        "query_embs":   Tensor[Q, 768]     – QA 쌍별 쿼리 임베딩
        "qa_pairs":     List[Dict]         – gold_chunk_id 포함
      }
    """
    cache_path_obj = Path(cache_path)
    cache_path_obj.parent.mkdir(parents=True, exist_ok=True)

    print(f"인코더 로드: {encoder_name} …")
    encoder = SentenceTransformer(encoder_name, device=str(_DEVICE))

    chunk_ids        = list(chunks.keys())
    chunk_texts_list = [chunks[c] for c in chunk_ids]
    queries          = [qa["query"] for qa in qa_pairs]

    print(f"청크 인코딩 중 … ({len(chunk_ids)}개)")
    chunk_embs = encoder.encode(
        chunk_texts_list,
        convert_to_tensor=True,
        device=str(_DEVICE),
        show_progress_bar=True,
    )  # [C, 768]

    print(f"쿼리 인코딩 중 … ({len(queries)}개)")
    query_embs = encoder.encode(
        queries,
        convert_to_tensor=True,
        device=str(_DEVICE),
        show_progress_bar=True,
    )  # [Q, 768]

    cache = {
        "chunk_ids":    chunk_ids,
        "chunk_embs":   chunk_embs.cpu(),
        "chunk_to_doc": chunk_to_doc,
        "chunk_texts":  chunks,
        "query_embs":   query_embs.cpu(),
        "qa_pairs":     qa_pairs,
    }
    torch.save(cache, cache_path_obj)
    print(f"캐시 저장 완료 → {cache_path_obj}")
    return cache


def load_cache(cache_path: str = "data/cache/embeddings.pt") -> Dict:
    return torch.load(cache_path, map_location="cpu", weights_only=False)


# ─── Candidate pool builder ───────────────────────────────────────────────────

def build_candidate_pool(
    cache: Dict,
    qa_idx: int,
    n_candidates: int = 8,
    seed: Optional[int] = None,
    hard_negatives: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, int, List[str]]:
    """
    QA 쌍 qa_idx에 대해 n_candidates개 청크 풀을 구성한다.

    hard_negatives=False (기본): distractor를 무작위 샘플링.
    hard_negatives=True        : query와 코사인 유사도가 높지만 gold가 아닌 청크를
                                 distractor로 사용 — 모델이 더 세밀한 구별을 학습.

    Pool은 항상 셔플되므로 gold_pos가 0이 아닐 수 있다.

    Returns:
        query_emb:      Tensor[768]         – 쿼리 임베딩
        chunk_embs:     Tensor[n, 768]      – 후보 청크 임베딩 (셔플됨)
        gold_pos:       int                 – 풀 내 gold 청크 인덱스
        pool_chunk_ids: List[str]           – 풀 내 청크 ID 목록 (셔플됨)
    """
    chunk_ids  = cache["chunk_ids"]
    chunk_embs = cache["chunk_embs"]   # [C, 768]
    query_embs = cache["query_embs"]   # [Q, 768]
    qa_pairs   = cache["qa_pairs"]

    gold_chunk_id = qa_pairs[qa_idx]["gold_chunk_id"]
    gold_idx      = chunk_ids.index(gold_chunk_id)
    gold_emb      = chunk_embs[gold_idx]              # [768]
    query_emb     = query_embs[qa_idx]                # [768]

    rng = random.Random(seed)
    n_distractors = min(n_candidates - 1, len(chunk_ids) - 1)

    if hard_negatives:
        # 쿼리와 코사인 유사도 기준 상위 청크 (gold 제외)
        q_norm   = F.normalize(query_emb.unsqueeze(0), dim=-1)   # [1, D]
        c_norm   = F.normalize(chunk_embs, dim=-1)                # [C, D]
        sims     = (c_norm @ q_norm.T).squeeze(-1).clone()        # [C]
        sims[gold_idx] = -2.0                                     # gold 제외
        chosen = torch.topk(sims, n_distractors).indices.tolist()
    else:
        distractor_indices = [i for i in range(len(chunk_ids)) if i != gold_idx]
        chosen = rng.sample(distractor_indices, n_distractors)

    distractor_embs = chunk_embs[chosen]              # [n-1, 768]

    # gold를 position 0에 두고 pool 구성
    pre_shuffle_ids  = [gold_chunk_id] + [chunk_ids[i] for i in chosen]
    pre_shuffle_pool = torch.cat([gold_emb.unsqueeze(0), distractor_embs], dim=0)  # [n, 768]

    # 셔플: gold_pos 고정 편향 방지
    order          = list(range(n_candidates))
    rng.shuffle(order)
    pool           = pre_shuffle_pool[order]
    pool_chunk_ids = [pre_shuffle_ids[i] for i in order]
    gold_pos       = order.index(0)                   # gold가 셔플 후 위치

    return query_emb, pool, gold_pos, pool_chunk_ids


# ─── Script entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir",      default="data/raw")
    parser.add_argument("--qa-out",       default="data/qa_pairs.jsonl")
    parser.add_argument("--cache-out",    default="data/cache/embeddings.pt")
    parser.add_argument("--min-chars",    type=int, default=300,
                        help="청크 최소 글자 수")
    parser.add_argument("--max-chars",    type=int, default=800,
                        help="청크 최대 글자 수")
    parser.add_argument("--no-qg-model",  action="store_true",
                        help="Ollama 없이 heuristic fallback만 사용")
    parser.add_argument("--skip-qg",      action="store_true",
                        help="기존 qa_pairs.jsonl 사용; QA 생성 생략")
    args = parser.parse_args()

    docs = load_docs(args.raw_dir)
    print(f"문서 {len(docs)}개 로드 완료.")

    chunks, chunk_to_doc = chunk_docs(docs, min_chars=args.min_chars, max_chars=args.max_chars)
    print(f"청크 {len(chunks)}개 생성 완료. (평균 {sum(len(t) for t in chunks.values()) // len(chunks):,}자)")

    if args.skip_qg and Path(args.qa_out).exists():
        print(f"기존 QA 쌍 로드: {args.qa_out} …")
        qa_pairs = load_qa_pairs(args.qa_out)
        print(f"{len(qa_pairs)}개 QA 쌍 로드 완료.")
        if qa_pairs and "gold_chunk_id" not in qa_pairs[0]:
            print("gold_chunk_id 없음 → 단어 겹침으로 자동 매핑 중 …")
            qa_pairs = resolve_gold_chunks(qa_pairs, chunks, chunk_to_doc)
            with open(args.qa_out, "w", encoding="utf-8") as f:
                for pair in qa_pairs:
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            print(f"매핑 완료: {len(qa_pairs)}개 QA 쌍 → {args.qa_out} 업데이트")
    else:
        qa_pairs = generate_qa_pairs(
            chunks,
            chunk_to_doc,
            output_path=args.qa_out,
            use_model=not args.no_qg_model,
        )

    cache = embed_all(chunks, chunk_to_doc, qa_pairs, cache_path=args.cache_out)
    print(
        f"임베딩: 청크={cache['chunk_embs'].shape}, "
        f"쿼리={cache['query_embs'].shape}"
    )
