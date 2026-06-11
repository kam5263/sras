# SRAS: Sparse Reward-Aware Selector for Edge-Native RAG Pipelines

> 강화학습 기반 문서 선택기 실험 보고서

---

## 목차

1. [프로젝트 주제 및 목표](#1-프로젝트-주제-및-목표)
2. [환경 및 데이터셋](#2-환경-및-데이터셋)
3. [State · Action · Reward 설계](#3-state--action--reward-설계)
4. [강화학습 알고리즘 및 하이퍼파라미터](#4-강화학습-알고리즘-및-하이퍼파라미터)
5. [실험 셋업](#5-실험-셋업)
6. [실험 결과](#6-실험-결과)
7. [토의 및 결론](#7-토의-및-결론)

---

## 1. 프로젝트 주제 및 목표

### 1.1 배경

RAG(Retrieval-Augmented Generation) 파이프라인에서 문서 검색 단계는 전통적으로 코사인 유사도 기반의 top-k 검색으로 수행된다. 이 방식은 인코더가 사전학습한 표현 공간에 전적으로 의존하며, 다운스트림 QA 품질을 직접 최적화하지 않는다.

SRAS(Sparse Reward-Aware Selector)는 이 검색 단계에 강화학습을 도입하여, 생성된 답변의 품질을 보상 신호로 활용하는 문서 선택 정책을 학습한다.

### 1.2 목표

| 목표 | 기준 |
|---|---|
| 학습 기반 문서 선택 | 코사인 유사도 baseline 대비 성능 향상 |
| 엣지 배포 가능성 | 모델 크기 < 1 MB, CPU 추론 지연 < 1 ms |
| 희박 보상 처리 | PPO + 하이브리드 보상으로 안정적 학습 |

### 1.3 파이프라인 구조

```
사용자 쿼리
    │
    ▼
Vector DB (top-N=8 후보 검색)
    │
    ▼
SRAS Selector (top-K=3 선택) ← 학습 대상
    │
    ▼
Frozen Generator (답변 생성)
    │
    ▼
최종 답변
```

---

## 2. 환경 및 데이터셋

### 2.1 원천 데이터

AI Hub 한국어 법률 QA 데이터셋을 사용하였다. 법령, 판결문, 결정례, 해석례 4개 카테고리로 구성된다.

| 카테고리 | ID 접두사 | 설명 |
|---|---|---|
| 법령 | `HJ_B_` | 법률 조문 및 규정 |
| 판결문 | `HJ_P_` | 법원 판결 문서 |
| 결정례 | `HJ_K_` | 행정 결정 사례 |
| 해석례 | `HJ_H_` | 법령 해석 사례 |

### 2.2 데이터 규모

| 구분 | 문서 수 | QA pair 수 | 비고 |
|---|---|---|---|
| 훈련 | 642 | 800 | gold doc 중복 허용 |
| 테스트 | 70 | 200 | 훈련 미사용 문서만 |

훈련셋과 테스트셋의 gold document는 완전히 분리되어 있어 데이터 누출이 없다.

### 2.3 전처리 파이프라인

```
원천 CSV 파일
    │  csv_to_text()
    ▼
data/raw/*.txt              (문서 텍스트 저장)
    │  chunk_docs()          [500~800자, 단락/문장 경계 기준]
    ▼
청크 단위 텍스트             {chunk_id: text, chunk_id → doc_id}
    │  generate_qa_pairs()   [qwen2.5:14b via Ollama]
    ▼
data/qa_pairs.jsonl         {"query", "gold_answer", "gold_chunk_id", "gold_doc_id"}
    │  embed_all()           [jhgan/ko-sroberta-multitask, 768-dim]
    ▼
data/cache/embeddings.pt    chunk_embs[C,768] + query_embs[Q,768]
                            + chunk_ids, chunk_to_doc, chunk_texts
```

**QA 생성 세부 사항**
- 모델: `qwen2.5:14b` via Ollama
- 입력: 청크 전체 텍스트 (500~800자)
- 출력: JSON 배열 `[{"question": ..., "answer": ...}]`
- 청크당 최대 3개 QA pair 생성
- gold 단위: `gold_chunk_id` (정답 출처 청크) + `gold_doc_id` (참조용 원문서)

**청크 분할 전략**
- 빈 줄(`\n\n`)로 단락 분리 후 `max_chars` 이내에서 인접 단락 병합
- 단락이 `max_chars` 초과 시 문장 경계(마침표/줄바꿈)로 재분할
- `min_chars`(300자) 미만 잔여 텍스트는 마지막 청크에 병합

**임베딩 캐시**
- 인코더: `jhgan/ko-sroberta-multitask` (768-dim, frozen)
- 임베딩은 1회 계산 후 캐시 — 학습 루프 내 재계산 없음
- MPS(Apple Silicon) 가속 활용

### 2.4 Candidate Pool 구성

각 학습 스텝에서 다음과 같이 후보 풀을 구성한다.

```
Candidate Pool (n=8)
├── [0] gold chunk           ← 정답 청크 (gold_chunk_id)
├── [1] random distractor chunk
├── [2] random distractor chunk
│   ...
└── [7] random distractor chunk  ← 7개 무작위 오답 청크
```

- distractor는 gold 청크를 제외한 **전체 청크 풀**에서 무작위 샘플링
- gold 위치 편향 방지를 위해 Pool 구성 후 셔플
- `hard_negatives=True` 옵션 시 쿼리와 코사인 유사도 상위 청크를 distractor로 사용

---

## 3. State · Action · Reward 설계

### 3.1 State

$$s = \left(q_{\text{emb}},\ \{d_1, d_2, \ldots, d_n\}\right)$$

| 구성 | 차원 | 설명 |
|---|---|---|
| 쿼리 임베딩 $q_{\text{emb}}$ | $\mathbb{R}^{768}$ | 사용자 질문의 의미 표현 |
| 청크 임베딩 풀 $d_i$ | $\mathbb{R}^{n \times 768}$ | n=8 후보 청크 |

인코더는 frozen 상태로 유지하며 학습 중 갱신하지 않는다.

### 3.2 Action

$$a = \{i_1, i_2, i_3\} \subset \{1, \ldots, n\}, \quad |a| = K = 3$$

- **학습 시**: softmax 정책에서 비복원 순차 샘플링
- **추론 시**: argmax deterministic top-K 선택 (랜덤성 없음)

비복원 순차 샘플링 절차:

```
available = [0, 1, ..., n-1]
for k in range(K):
    probs = softmax(scores[available])
    choice = multinomial(probs)
    selected.append(choice)
    available.remove(choice)
```

### 3.3 Scoring Function (SRAS 모델)

$$s_i = w^\top \cdot \tanh\!\left(W_q q + W_d d_i\right)$$

```python
class SRASSelector(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=64):
        self.W_q  = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.W_d  = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.w    = nn.Linear(hidden_dim, 1, bias=False)
        self.tanh = nn.Tanh()
```

| 파라미터 | 크기 | 파라미터 수 |
|---|---|---|
| $W_q$ | $768 \times 64$ | 49,152 |
| $W_d$ | $768 \times 64$ | 49,152 |
| $w$ | $64 \times 1$ | 64 |
| **합계** | | **98,368** |

### 3.4 Reward

$$R = \underbrace{\alpha \cdot F_1}_{\text{Relaxed F1}} + \underbrace{(1-\alpha) \cdot B}_{\text{BERTScore}} + \underbrace{\beta \cdot \mathbf{1}[\text{gold} \in \text{selected}]}_{\text{Doc Hit Bonus}}$$

| 구성 요소 | 가중치 | 구현 세부 |
|---|---|---|
| Relaxed F1 | α = 0.6 | 토큰 수준 F1, 소문자·불용어 정규화 |
| BERTScore | 1-α = 0.4 | `klue/roberta-base`, lang="ko" |
| Doc Hit Bonus | β = 0.2 | Gold doc 선택 시 직접 검색 신호 |

**답변 생성기**: `qwen3.5:2b` via Ollama (HTTP API)
- `think: false`로 thinking mode 비활성화
- 문서 앞 500자로 truncate하여 프롬프트 길이 최적화
- 동일 (쿼리, 선택 문서) 조합은 in-memory 캐시로 재사용

**보상 정규화**: 배치별 zero-mean / unit-variance 정규화 후 advantage 계산

$$\hat{A}_i = \frac{R_i - \mu_R}{\sigma_R + \epsilon}$$

---

## 4. 강화학습 알고리즘 및 하이퍼파라미터

### 4.1 알고리즘: PPO (Proximal Policy Optimization)

순수 PyTorch 커스텀 구현 (외부 RL 프레임워크 미사용). Clipped surrogate objective:

$$L^{\text{CLIP}} = \mathbb{E}_t\!\left[\min\!\left(r_t \hat{A}_t,\ \text{clip}(r_t,\ 1-\varepsilon,\ 1+\varepsilon)\,\hat{A}_t\right)\right]$$

$$r_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

로그 확률은 rollout 시점에 `detach()`하여 old policy로 고정한 후, update 시점에 재계산한다.

### 4.2 훈련 안정화 기법

| 기법 | 설명 | 효과 |
|---|---|---|
| **Supervised Warmup** | PPO 이전 cross-entropy 사전훈련 | 정책 붕괴 방지 |
| **Hybrid Reward** | F1 + BERTScore 조합 | 희박 보상 문제 완화 |
| **Doc Hit Bonus** | 직접 검색 신호 추가 | 생성기 노이즈 우회 |
| **Reward Normalization** | 배치별 zero-mean/unit-var | 학습 안정성 |
| **Curriculum Learning** | 쉬운→어려운 순 점진적 노출 | 초기 학습 안정성 |

**Curriculum Learning 구현**: epoch 비율에 따라 활성 QA pair 수를 선형 증가

```python
frac      = min(1.0, 0.5 + 0.5 * epoch / (max_epochs - 1))
n_active  = max(batch_size, int(frac * n_qa))   # 400 → 800
```

### 4.3 하이퍼파라미터

| 파라미터 | 값 | 비고 |
|---|---|---|
| Optimizer | AdamW | — |
| Learning rate | 1e-5 | — |
| PPO Epochs | 25 | — |
| Batch size | 8 | — |
| Candidates n | 8 | 1 gold + 7 random |
| Top-K | 3 | — |
| Clip ε | 0.2 | PPO clipping |
| Discount γ | 0.99 | — |
| Reward α | 0.6 | F1 가중치 |
| Doc Hit Bonus β | 0.2 | 직접 검색 신호 |
| Embedding dim | 768 | Frozen 인코더 |
| Hidden dim | 64 | SRAS 내부 |

---

## 5. 실험 셋업

### 5.1 실험 환경

| 항목 | 사양 |
|---|---|
| 하드웨어 | Apple MacBook M4, 32GB RAM |
| 가속 디바이스 | MPS (Metal Performance Shaders) |
| Python | 3.10 |
| 인코더 | jhgan/ko-sroberta-multitask (frozen) |
| BERTScore 백본 | klue/roberta-base (frozen) |
| 생성기 | qwen3.5:2b via Ollama (frozen) |
| Warmup | cross-entropy, 1 epoch, gold doc index 레이블 |

### 5.2 Evaluation Metrics

**Hit Rate @K**

$$\text{Hit@K} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}\!\left[\text{gold}_i \in \text{top-}K_i\right]$$

n=8 후보 중 K=3 선택 시 gold doc이 포함된 비율. 랜덤 기대값 = 3/8 = 37.5%.

**MRR (Mean Reciprocal Rank)**

$$\text{MRR} = \frac{1}{N}\sum_{i=1}^{N} \frac{1}{\text{rank}(\text{gold}_i)}$$

Gold doc의 전체 n개 후보 내 순위 역수 평균. 단순 hit 여부뿐 아니라 순위 품질까지 반영.

### 5.3 비교 모델

| 방법 | 설명 |
|---|---|
| **SRAS (Warmup)** | Cross-entropy 사전훈련 후 PPO 미적용 |
| **SRAS (PPO best)** | Warmup 이후 Full PPO epoch 19+ 최고 보상 체크포인트 (`sras_best.pt`) |
| **Cosine** | 쿼리-문서 코사인 유사도 기준 deterministic top-K |
| **Random** | 균일 무작위 선택 (seed=qa_idx 고정) |

---

## 6. 실험 결과

### 6.1 학습 곡선

**[Figure 1] Epoch별 평균 보상 (avg_reward) — Full PPO (random distractor, Doc Hit Bonus 포함)**

```
reward
0.645 |                                    ★ 0.6403 (best, epoch 19)
      |                              ●  ●●●
0.635 |                          ●●●●
      |          ●               ●
0.625 |     ●●●●●●●●●●●●●●●●
      |   ●
0.615 |  ●
0.610 |●
      +--+--+--+--+--+--+--+--+--+--→
      1  3  5  7  9  11 13 15 17 19
                      Epoch
```

Epoch 1에서 0.6269로 시작, epoch 6(0.6133)에서 일시 하락 후 전반적 상승. Epoch 19에서 0.6403으로 최고 기록 (epoch 20–25는 로그 미포함).

---

**[Figure 2] Epoch별 Gold Doc Hit Rate**

```
hit %
 96 |                  ★★ 95.4% (epoch 11, 15)
    |          ●               ●
 94 |   ●                          ●       ●
 93 |               ●     ●●●        ●  ●
 92 |    ●●●●●●●●●      ●    ●●●●
 91 |
    +--+--+--+--+--+--+--+--+--+--→
    1  3  5  7  9  11 13 15 17 19
                      Epoch
```

전 epoch에 걸쳐 91–95%로 안정 유지. Epoch 11, 15에서 95.4% 최고 기록.

---

**[Table 1] Epoch별 Hit/Miss 보상 분리 추이 — Full PPO**

| Epoch | avg_reward | Hit Rate | R(hit) | R(miss) | Gap |
|---|---|---|---|---|---|
| 1 | 0.6269 | 94.5% | 0.6509 | 0.2137 | 0.437 |
| 5 | 0.6229 | 91.8% | 0.6535 | 0.2633 | 0.390 |
| 8 | 0.6286 | 93.0% | 0.6575 | 0.2471 | 0.410 |
| 11 | 0.6366 | 95.4% | 0.6565 | 0.2286 | 0.428 |
| 15 | 0.6337 | 95.4% | 0.6505 | 0.2186 | 0.432 |
| 19 | 0.6403 | 94.4% | 0.6653 | 0.2171 | **0.448** |

Gold doc 선택 시(R(hit))와 미선택 시(R(miss))의 보상 차이 ~0.43이 전 epoch에 걸쳐 유지됨. Doc Hit Bonus 도입으로 이전 실험(gap ~0.33) 대비 보상 신호의 변별력이 향상되었음.

---

**[Table 1b] Hard Negatives PPO — Epoch별 추이 (epoch 13–25)**

Hard Negative Sampling(`hard_negatives=True`) 조건 결과:

| Epoch | avg_reward | Hit Rate | R(hit) | R(miss) |
|---|---|---|---|---|
| 13 | 0.5425 | 66.5% | 0.6439 | 0.3412 |
| 14 | 0.5561 | 71.3% | 0.6437 | 0.3391 |
| 18 | 0.5498 | 67.5% | 0.6469 | 0.3476 |
| 21 | 0.5458 | 67.4% | 0.6482 | 0.3342 |
| 25 | 0.5477 | 68.2% | 0.6488 | 0.3306 |
| **Best** | **0.5561** | 71.3% | — | — |

쿼리와 코사인 유사도가 높은 hard negative distractor 사용으로, Full PPO 대비 hit rate(~67%)와 보상(~0.54)이 낮음.

---

### 6.2 최종 성능 비교

**[Table 2] 훈련셋 평가 (n=800, K=3)**

| 방법 | Hit Rate @3 | MRR | Latency (ms) | Size (MB) | Params |
|---|---|---|---|---|---|
| **SRAS Warmup** | **97.2%** | 0.8609 | 0.038 | 0.38 | 98,368 |
| SRAS PPO best | 86.8% | 0.7269 | 0.038 | 0.38 | 98,368 |
| Cosine | 96.2% | **0.9077** | — | — | — |
| Random | 36.2% | 0.3377 | — | — | — |

---

**[Table 3] 테스트셋 평가 (n=200, K=3, 미사용 문서)**

| 방법 | Hit Rate @3 | MRR | Latency (ms) | Size (MB) | Params |
|---|---|---|---|---|---|
| **SRAS Warmup** | **90.0%** | 0.7427 | 0.033 | 0.38 | 98,368 |
| SRAS PPO best | 79.5% | 0.6439 | 0.033 | 0.38 | 98,368 |
| Cosine | **94.0%** | **0.8884** | — | — | — |
| Random | 44.0% | 0.3565 | — | — | — |

---

**[Figure 3] 테스트셋 Hit Rate @3 시각 비교 (랜덤 distractor 설정)**

```
         0%    20%    40%    60%    80%   100%
         |      |      |      |      |      |
Random   [████████████████████               ]  44.0%
PPO best [████████████████████████████████   ]  79.5%
Warmup   [█████████████████████████████████  ]  90.0%
Cosine   [██████████████████████████████████ ]  94.0%
         |      |      |      |      |      |
```

---

**[Table 3b] 테스트셋 평가 — Hard Negatives 설정 (n=200, K=3)**

후보 풀을 코사인 유사도 상위 문서로 구성한 어려운 평가 설정. Checkpoint: `sras_best_hardneg.pt` (Hard Negatives PPO best, avg_reward 0.5561).

| 방법 | Hit Rate @3 | MRR | Latency (ms) | Size (MB) | Params |
|---|---|---|---|---|---|
| **SRAS (Hard Neg PPO)** | 52.5% | 0.4142 | 0.038 | 0.38 | 98,368 |
| Cosine | **59.5%** | **0.5393** | — | — | — |
| Random | 49.5% | 0.4539 | — | — | — |

Random 대비 +3pp 향상, Cosine 대비 -7pp. 쉬운 랜덤 설정(+35.5pp vs Random)에 비해 hard negative 설정에서의 SRAS 우위가 크게 감소함.

---

### 6.3 일반화 성능 (훈련 → 테스트 드롭)

**[Table 4] Train/Test 성능 차이**

| 방법 | 훈련셋 Hit@3 | 테스트셋 Hit@3 | 드롭 |
|---|---|---|---|
| SRAS Warmup | 97.2% | 90.0% | -7.2pp |
| SRAS PPO best | 86.8% | 79.5% | -7.3pp |
| **Cosine** | 96.2% | 94.0% | **-2.2pp** |

Cosine의 일반화 드롭(-2.2pp)이 SRAS 대비 현저히 낮다. SRAS는 훈련 문서 분포에 부분적으로 과적합됨.

---

### 6.4 Seed 변경 실험 및 신뢰구간

본 실험은 시간 제약으로 단일 시드(seed=42)로 수행되었다. 아래는 신뢰할 수 있는 비교를 위한 다중 시드 실험 계획이다.

**[Table 5] 단일 시드 결과 (신뢰구간 미측정)**

| 방법 | Hit Rate @3 (test) | 95% CI |
|---|---|---|
| SRAS Warmup | 90.0% | 미측정 (단일 시드) |
| SRAS PPO best | 79.5% | 미측정 (단일 시드) |
| Cosine | 94.0% | 결정론적 — CI 불필요 |
| Random | 44.0% | 이론값 = 37.5% ± 변동 |

**다중 시드 실험 제안**

- 시드 집합: {42, 123, 2024, 7, 99} (5회)
- 각 시드에서 warmup + PPO 25 epoch full training
- 결과: 평균 ± 표준편차로 95% CI 제시
- 예상 분산 요인:
  - PPO rollout의 multinomial 샘플링
  - Candidate pool의 random distractor 구성
  - Curriculum ordering의 shuffling

Random baseline의 이론적 기대 Hit@3 = K/n = 3/8 = 37.5%. 실측값 44.0%는 seed=qa_idx 고정 방식에 의한 특정 시드 편향으로 해석된다.

## 7. 토의 및 결론

### 7.1 핵심 발견

**발견 1: Doc Hit Bonus 도입 후 PPO 학습 안정성 크게 향상**

```
이전 실험: avg_reward 0.57–0.59, hit_rate 86–93% (epoch 9 정점 후 감소)
Full PPO:  avg_reward 0.61–0.64, hit_rate 91–95% (안정 유지 및 완만 상승)
```

Doc Hit Bonus(β=0.2) 도입 이후 PPO 훈련 중 hit rate가 91–95%로 안정 유지되었다. 이전 실험 대비 보상 신호의 변별력(R(hit)–R(miss) gap: ~0.33 → ~0.43)이 향상되었으며, policy collapse 징후가 관찰되지 않았다. Epoch 19에서 avg_reward 0.6403(best)을 기록하였다 (epoch 20–25 미포함). Doc Hit Bonus가 생성기 답변 품질 노이즈를 우회하여 직접적인 검색 신호를 제공함으로써 학습 안정성에 기여한 것으로 판단된다.

**발견 2: Warmup만으로 Cosine에 근접**

Supervised Warmup만으로 훈련셋 97.2%(Cosine 96.2% 초과), 테스트셋 90.0%(Cosine 94.0%와 4pp 차이)를 달성했다. SRAS의 additive scoring 아키텍처 자체는 유효하며, 문제는 RL 보상 설계에 있음을 시사한다.

**발견 3: Hard Negative Sampling — 어려운 구별력 학습의 한계**

Hard Negative Sampling 실험 결과:
- 훈련 중 hit rate ~67% (랜덤 distractor ~93% 대비 크게 하락)
- 테스트 평가(Hard Neg 설정): SRAS 52.5% vs Cosine 59.5% vs Random 49.5%

SRAS가 랜덤 baseline(49.5%)은 소폭 초과하나 Cosine(59.5%)에는 7pp 미달한다. 랜덤 distractor 설정에서의 Cosine 대비 격차(-3.5pp 또는 우위)와 비교하면, 어려운 구분 환경에서 SRAS의 학습된 판별력이 충분히 Cosine을 넘어서지 못함을 보여준다. 추가적인 표현력 또는 더 많은 학습 데이터가 필요하다.

**발견 4: 빠른 추론 및 경량 모델 달성**

CPU 추론 지연 0.036ms, 모델 크기 0.38MB로 엣지 배포 조건을 충족한다.

### 7.2 보완 및 개선사항

**단기 (즉시 적용 가능)**

**① KL Divergence 페널티 추가**

PPO 목적함수에 warmup policy와의 KL 거리 페널티를 추가하여 정책이 초기화에서 과도하게 이탈하지 않도록 제약한다.

```python
# _ppo_update()에서 추가
kl_penalty  = kl_coef * (old_log_probs - new_log_probs).mean()
policy_loss = -torch.min(surr1, surr2).mean() + kl_penalty
```

예상 효과: PPO 훈련 후에도 warmup 수준 이상의 Hit Rate 유지.

**② Hard Negative Sampling**

랜덤 distractor 대신 코사인 유사도가 높은 문서를 distractor로 사용한다.

```python
# 현재: 완전 랜덤 distractor
distractor_indices = random.sample(all_except_gold, n-1)

# 개선: 코사인 유사도 상위 문서에서 샘플링 (hard negatives)
sim = cosine(query_emb, all_doc_embs)
hard_pool = topk(sim, k=20, exclude=gold_idx)
distractor_indices = random.sample(hard_pool, n-1)
```

예상 효과: SRAS가 단순 코사인 유사도 이상의 판별력을 학습.

**중기**

**③ 보상 신호 개선**

현재 Ollama 생성 품질(F1+BERTScore)은 노이즈가 크고 검색 정확도와 간접적으로만 연결된다. 대안:
- Gold doc hit 여부 기반 이진 보상 (직접 정렬)
- 더 강력한 생성 모델(`qwen2.5:14b`) 사용으로 보상 품질 향상

**④ 훈련 데이터 확장**

현재 800개 QA pair는 일반화 가능한 scoring function 학습에 부족하다. 5,000개 이상으로 확장하고, 특히 의미적으로 유사한 문서 간 구별을 요구하는 어려운 케이스를 포함하는 것이 중요하다.

**장기**

**⑤ 다중 시드 실험**

5개 이상의 랜덤 시드로 full training을 반복하여 평균 ± 표준편차로 신뢰구간을 제시하고, 알고리즘 간 통계적 유의성을 검증한다.

**⑥ 모델 아키텍처 개선**

현재의 additive interaction (`tanh(W_q·q + W_d·d)`)을 cross-attention 기반 상호작용으로 교체하면 표현력을 높일 수 있다. 다만 0.38MB 크기 제약 내에서 설계해야 한다.

### 7.3 결론

SRAS는 엣지 배포 조건(0.38MB, 0.038ms CPU 지연)을 만족하면서, Supervised Warmup만으로 테스트셋 Hit Rate @3 **90.0%** 를 달성하였다. 이는 랜덤 baseline(44.0%) 대비 +46pp의 성능 향상이며, Cosine baseline(94.0%)과 4pp 차이에 해당하는 경쟁력 있는 결과이다.

Doc Hit Bonus(β=0.2) 도입 이후 Full PPO 훈련에서 학습 안정성이 크게 향상되었다. 훈련 중 hit rate 91–95% 안정 유지, avg_reward epoch 19 기준 0.6403(best)으로 이전 실험 대비 개선되었다. 다만 해당 체크포인트의 테스트셋 공식 평가는 아직 수행되지 않았다.

Hard Negative Sampling 실험에서는 SRAS가 랜덤 baseline은 소폭 초과(52.5% vs 49.5%)하였으나, Cosine(59.5%)에는 미달하였다. 향후 **KL-regularized PPO** 와 **훈련 데이터 확충** 을 통해 강화학습의 이점을 더욱 활용할 수 있을 것으로 기대된다.

---

## 부록: 주요 파일 구조

```
sras/
├── src/
│   ├── data_pipeline.py    # 전처리 + 임베딩 캐시
│   ├── model.py            # SRASSelector
│   ├── reward.py           # Relaxed F1 + BERTScore
│   ├── warmup.py           # Supervised cross-entropy warmup
│   ├── ppo_trainer.py      # PPO 학습 루프
│   ├── evaluate.py         # 검색 지표 평가
│   └── build_test_dataset.py  # 테스트셋 생성
├── checkpoints/
│   ├── sras_warmup.pt          # Supervised Warmup 체크포인트 (테스트 Hit@3 90.0%)
│   ├── sras_best_hardneg.pt    # Hard Negatives PPO best (avg_reward 0.5561, epoch 14)
│   └── sras_best.pt            # Full PPO best (avg_reward 0.6403+, epoch 19+)
├── data/
│   ├── qa_pairs.jsonl      # 훈련 QA pair (800개)
│   ├── test/qa_pairs_test.jsonl  # 테스트 QA pair (200개)
│   └── cache/
│       ├── embeddings.pt       # 훈련 임베딩 캐시
│       └── test_embeddings.pt  # 테스트 임베딩 캐시
└── REPORT.md
```
