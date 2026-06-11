# SRAS: 경량 강화학습 기반 문서 선택기 (한국어 법률 QA 적용)

> **서강대학교 강화학습 프로젝트** · 2026년 1학기  

## 프로젝트 개요

SRAS(Sparse Reward-Aware Selector)는 RAG(Retrieval-Augmented Generation) 파이프라인에서 문서 선택 단계를 강화학습(PPO)으로 학습하는 초경량 모델입니다.

본 프로젝트는 Muttur(2026)의 논문 "SRAS: A Lightweight RL-based Document Selector for Edge-Native RAG Pipelines"의 결과를 재연하되, 데이터셋을 **한국어 법률 QA**(법령·판결문·결정례·해석례)로 교체하고 임베딩 인코더를 `jhgan/ko-sroberta-multitask`(768차원)으로 대체했습니다.

### 참고 논문

Muttur, R. C. (2026). *SRAS: A Lightweight Reinforcement Learning-based Document Selector for Edge-Native RAG Pipelines.* arXiv:2601.01785.

---

## 핵심 특징

| 항목 | 내용 |
|---|---|
| 모델 파라미터 수 | ~98K (논문 197K 대비: hidden_dim=64 사용) |
| 모델 크기 | ~0.38 MB |
| 추론 지연 (CPU) | < 1s |
| 임베딩 인코더 | `jhgan/ko-sroberta-multitask` (768-dim, 고정) |
| QA 생성 | Ollama `qwen2.5:14b` (한국어, 없을 시 heuristic fallback) |
| RL 알고리즘 | PPO (custom loop, pure PyTorch) |
| 학습 데이터 | 한국어 법률 문서 642개, QA 쌍 800개 |
| 테스트 데이터 | 학습셋과 완전 분리된 문서 193개, QA 쌍 200개 |

---

## 데이터셋

**AI Hub 법률 QA 데이터셋** (법령·판결문·결정례·해석례)을 사용합니다. 학습셋과 테스트셋은 **doc_id 수준에서 완전히 분리**되어 있으며, 동일한 문서가 양쪽에 중복되지 않습니다.

### 원시 데이터 구조

원시 데이터는 `raw_dataset/`에 위치하며, 전처리 스크립트로 `sras/data/` 하위 파일들을 생성합니다.

```
raw_dataset/
├── Training/
│   ├── 01.원천데이터/     # CSV 형식 원본 문서
│   └── 02.라벨링데이터/   # JSON 형식 QA 라벨
└── Validation/
    ├── 01.원천데이터/
    └── 02.라벨링데이터/
```

### 학습/테스트 분할

| 속성 | 학습셋 | 테스트셋 |
|---|---|---|
| QA 쌍 수 | 800 | 200 |
| 문서 수 | 642 | 193 |
| doc_id 겹침 | — | 0 (보장) |
| 랜덤 시드 | 42 | 99 |
| QA 파일 | `data/qa_pairs.jsonl` | `data/test/qa_pairs_test.jsonl` |
| 임베딩 캐시 | `data/cache/embeddings.pt` | `data/test/embeddings_test.pt` |

테스트셋은 학습셋(시드 42)과 다른 시드(99)로 샘플링하며, 학습셋에 등장한 모든 doc_id를 사전에 제외합니다.

### 카테고리별 구성

| 카테고리 | 접두사 | 학습 문서 | 학습 QA | 테스트 문서 | 테스트 QA |
|---|---|---|---|---|---|
| 법령 | `HJ_B_` | ~250 | ~320 | 65 | 120 |
| 판결문 | `HJ_P_` | ~280 | ~280 | 95 | 95 |
| 결정례 | `HJ_K_` | ~60 | ~100 | 25 | 25 |
| 해석례 | `HJ_H_` | ~52 | ~100 | 25 | 25 |
| **합계** | — | **642** | **800** | **193** | **200** |

각 QA 쌍의 포맷은 동일합니다:

```json
{"query": "법무부장관은 중재산업 진흥을 위해 어떤 계획을 수립해야 하나요?",
 "gold_answer": "법무부장관은 5년마다 중재산업 진흥 기본계획을 수립·시행해야 합니다.",
 "gold_doc_id": "HJ_B_012707",
 "gold_chunk_id": "HJ_B_012707_01"}
```

---

## 디렉터리 구조

```
rl/
├── raw_dataset/               # 원시 데이터 (git 제외 — AI Hub 라이선스)
├── sras/
│   ├── data/
│   │   ├── raw/               # 학습용 .txt 문서 (642개, build_dataset.py 생성)
│   │   ├── qa_pairs.jsonl     # 학습 QA 쌍 (800개)
│   │   ├── cache/             # 학습 임베딩 캐시 .pt (git 제외 — 재생성 필요)
│   │   └── test/
│   │       ├── raw/               # 테스트 전용 .txt 문서 (193개, 학습셋 미포함)
│   │       ├── qa_pairs_test.jsonl  # 테스트 QA 쌍 (200개)
│   │       └── embeddings_test.pt   # 테스트 임베딩 캐시
│   ├── src/
│   │   ├── build_dataset.py       # raw_dataset → data/raw + qa_pairs.jsonl (학습)
│   │   ├── build_test_dataset.py  # raw_dataset → data/test/raw + qa_pairs_test.jsonl
│   │   ├── data_pipeline.py       # txt → QA 생성 → 임베딩 캐시 (학습/테스트 모두 사용)
│   │   ├── model.py               # SRASSelector (핵심 모델)
│   │   ├── reward.py              # Relaxed F1 + BERTScore 보상 함수
│   │   ├── warmup.py              # 지도 사전학습 (Cross-Entropy)
│   │   ├── ppo_trainer.py         # 커스텀 PPO 학습 루프
│   │   ├── evaluate.py            # 결정론적 평가 (Hit Rate, MRR)
│   │   └── pipeline.py            # 프로덕션 추론 파이프라인
│   ├── checkpoints/           # 모델 체크포인트 (git 제외 — 아래 링크 참조)
│   ├── configs/
│   │   └── default.yaml       # 하이퍼파라미터 설정
│   └── REPORT.md              # 실험 보고서
├── requirements.txt
└── README.md
```

---

## 설치

Python 3.10 이상, Apple M4 (MPS) 또는 CPU 환경을 지원합니다.

```bash
# 가상환경 생성 및 활성화
python -m venv .rl
source .rl/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

> **Apple MPS 참고:** MPS 커널 미지원 연산이 발생하면 환경 변수 `PYTORCH_ENABLE_MPS_FALLBACK=1`을 설정하세요.

---

## 실행 방법

모든 명령어는 `sras/` 디렉터리 내에서 실행합니다.

```bash
cd sras
```

### 1단계: 학습 데이터셋 전처리

`raw_dataset/`에서 `data/raw/*.txt`와 `data/qa_pairs.jsonl`을 생성합니다.

```bash
python src/build_dataset.py
```

### 2단계: 테스트 데이터셋 생성

학습셋에 등장한 doc_id를 모두 제외한 뒤, 나머지 문서에서 독립적으로 샘플링합니다. 완료 시 학습셋과의 doc_id 겹침을 자동으로 검증합니다.

```bash
python src/build_test_dataset.py
# 학습셋 제외 doc_id: 642개
# TL_법령_QA: 120 QA, 65 docs
# TL_판결문_QA: 95 QA, 95 docs
# TL_결정례_QA: 25 QA, 25 docs
# TL_해석례_QA: 25 QA, 25 docs
# 완료: 200 QA pairs, 193 documents in test/raw/
# 학습셋 doc_id 겹침: 0개 ✓
```

### 3단계: 임베딩 캐시 생성

`jhgan/ko-sroberta-multitask`로 문서·쿼리를 인코딩하고 캐시합니다. 학습셋과 테스트셋 각각 실행합니다.

```bash
# 학습셋 캐시 생성
python src/data_pipeline.py --skip-qg   # 기존 qa_pairs.jsonl 재사용

# 테스트셋 캐시 생성 (경로를 테스트 디렉터리로 지정)
python src/data_pipeline.py \
  --skip-qg \
  --raw-dir   data/test/raw \
  --qa-out    data/test/qa_pairs_test.jsonl \
  --cache-out data/test/embeddings_test.pt
```

> Ollama `qwen2.5:14b`로 QA를 새로 생성하려면 `--skip-qg`를 제거하세요. Ollama 없이 실행하려면 `--no-qg-model`을 추가하세요.

### 4단계: 지도 사전학습 (Warmup)

Gold 문서를 정답 레이블로 Cross-Entropy 사전학습합니다 (PPO 전 안정성 확보).

```bash
python src/warmup.py
```

### 5단계: PPO 강화학습

Relaxed F1 + BERTScore 혼합 보상으로 PPO를 수행합니다.

```bash
python src/ppo_trainer.py
```

### 6단계: 평가

학습된 모델을 SRAS, Top-k Cosine, Random 기준선과 비교합니다. 공정한 평가를 위해 학습 중 보지 못한 **테스트셋 캐시**를 사용합니다.

```bash
# 테스트셋 기준 평가 (권장)
python src/evaluate.py \
  --checkpoint checkpoints/sras_best.pt \
  --cache      data/test/embeddings_test.pt

# Hard negative distractor — 코사인 유사도 상위 청크를 방해 문서로 사용 (사용x)
python src/evaluate.py \
  --checkpoint checkpoints/sras_best.pt \
  --cache      data/test/embeddings_test.pt \
  --hard-negatives

# 학습셋 캐시로 in-sample 성능 확인
python src/evaluate.py --checkpoint checkpoints/sras_best.pt
```

### 7단계: 프로덕션 추론 (선택)

단일 쿼리에 대한 엔드-투-엔드 추론을 실행합니다.

```bash
python src/pipeline.py --query "계약 해지 요건은 무엇인가?"
```

---

## 모델 아키텍처

```
scoring: s_i = wᵀ · tanh(W_q · q + W_d · d_i)
```

| 레이어 | 크기 |
|---|---|
| W_q (쿼리 투영) | 768 → 64 |
| W_d (문서 투영) | 768 → 64 |
| w (스코어링 헤드) | 64 → 1 |

모든 선형 레이어는 `bias=False`.

---

## 하이퍼파라미터

`sras/configs/default.yaml`에서 확인 및 수정 가능합니다.

| 파라미터 | 값 |
|---|---|
| PPO Epochs | 25 |
| Batch Size | 8 |
| 후보 문서 수 (n) | 8 |
| 선택 문서 수 (K) | 3 |
| Learning Rate | 1e-5 |
| Discount γ | 0.99 |
| Clip ε | 0.2 |
| 보상 가중치 α | 0.6 (Relaxed F1) |
| Optimizer | AdamW |

---

## 평가 지표

| 지표 | 설명 |
|---|---|
| **Hit Rate @K** | Gold 문서가 상위 K개 선택에 포함될 확률 |
| **MRR** | Mean Reciprocal Rank — Gold 문서 순위의 역수 평균 |
| **Relaxed F1** | 정규화된 토큰 수준 F1 (소문자·구두점·불용어 제거) |
| **BERTScore F1** | `klue/roberta-base` 기반 의미적 유사도 |
| **Latency (ms)** | CPU 단일 쿼리 추론 시간 (batch=1) |
| **Model Size (MB)** | 직렬화된 모델 파일 크기 |

Hit Rate @K와 MRR은 테스트셋(200 QA, 193개 문서) 기준으로 보고합니다.

---

## 학습된 모델 다운로드

[GitHub Releases v1.0](https://github.com/kam5263/sras/releases/tag/v1.0)에서 다운로드하거나 아래 명령어를 사용하세요.

| 파일 | 설명 | 링크 |
|---|---|---|
| `sras_best.pt` | PPO 학습 최종 체크포인트 | [다운로드](https://github.com/kam5263/sras/releases/download/v1.0/sras_best.pt) |
| `sras_warmup.pt` | 지도 사전학습(Warmup) 체크포인트 | [다운로드](https://github.com/kam5263/sras/releases/download/v1.0/sras_warmup.pt) |
| `sras_best_hardneg.pt` | Hard Negative Distractor 실험용 | [다운로드](https://github.com/kam5263/sras/releases/download/v1.0/sras_best_hardneg.pt) |

```bash
# 체크포인트 디렉터리로 다운로드
mkdir -p sras/checkpoints
wget -P sras/checkpoints https://github.com/kam5263/sras/releases/download/v1.0/sras_best.pt
wget -P sras/checkpoints https://github.com/kam5263/sras/releases/download/v1.0/sras_warmup.pt
```

---

## 논문 원본 대비 주요 차이점

| 항목 | 논문 (Muttur 2026) | 본 구현 |
|---|---|---|
| 데이터셋 | 영어 범용 문서 (905개) | 한국어 법률 문서 (642개 학습 + 193개 테스트) |
| 임베딩 인코더 | `all-MiniLM-L6-v2` (384-dim) | `jhgan/ko-sroberta-multitask` (768-dim) |
| QA 생성 | `valhalla/t5-base-qg-hl` | Ollama `qwen2.5:14b` (한국어) |
| 파라미터 수 | ~197K | ~98K (hidden_dim=64) |
| 학습 디바이스 | CPU (Intel i5) | Apple M4 MPS |

---

## 참고문헌

- Muttur, R. C. (2026). SRAS: A Lightweight RL-based Document Selector for Edge-Native RAG Pipelines. *arXiv:2601.01785.*
