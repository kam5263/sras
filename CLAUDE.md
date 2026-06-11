# CLAUDE.md — SRAS (Sparse Reward-Aware Selector)

Guidance for Claude Code when working in this repository.

## Project Overview

SRAS is an ultra-lightweight (~98K params, ~0.38MB) RL-based **document selector** for
edge-native RAG pipelines. Instead of a fixed top-k retriever, SRAS learns a nonlinear
query-document scoring policy via **PPO**, trained on a hybrid QA-derived reward
(Relaxed F1 + BERTScore). It replaces the "select k of n candidates" step in a RAG pipeline.

This implementation uses **pure PyTorch with a custom PPO loop** (no RL frameworks like
Stable-Baselines3 or TRL). This is deliberate: it gives full control over the
forward/backward pass and lets us fully exploit Apple M4 acceleration via the `mps` device.

Reference paper: *"SRAS: A Lightweight RL-based Document Selector for Edge-Native RAG Pipelines"* (Muttur).

## Hardware & Device Policy

- Target machine: **MacBook M4, 32GB RAM**.
- ALWAYS resolve the device once and reuse it:
  ```python
  device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
  ```
- Move the model and all tensors to `device`. Do NOT mix `mps` and `cpu` tensors in an op.
- `sentence-transformers` and HuggingFace models also honor `mps` — pass the device through.
- Some ops lack `mps` kernels; if you hit a `NotImplementedError`, the fallback is to set
  `PYTORCH_ENABLE_MPS_FALLBACK=1` for that process — do not silently move the whole model to CPU.

## Tech Stack

- **PyTorch** — model, custom PPO loop, training.
- **sentence-transformers** (`jhgan/ko-sroberta-multitask`, 768-dim) — frozen embedding encoder.
- **Ollama** (`qwen2.5:14b`) — synthetic QA generation; (`qwen3.5:2b`) — frozen answer generator during PPO.
- **bert-score** — semantic reward (`klue/roberta-base` backbone, Korean).
- **ragas** — final-stage evaluation (Context Precision, Faithfulness).
- Python 3.10+. Manage deps in `requirements.txt` / venv.

## Suggested Directory Structure

```
sras/
├── data/
│   ├── raw/                 # original .txt documents (one doc per file)
│   ├── qa_pairs.jsonl       # {"query","gold_answer","gold_doc_id"} per line
│   └── cache/               # embeddings.pt / .npy caches
├── src/
│   ├── data_pipeline.py     # txt -> QA pairs -> embedding cache
│   ├── model.py             # SRASSelector (see spec below)
│   ├── reward.py            # Relaxed F1 + BERTScore reward engine
│   ├── ppo_trainer.py       # rollout buffer + custom PPO update loop
│   ├── warmup.py            # supervised cross-entropy pretraining
│   ├── evaluate.py          # deterministic top-k inference + RAGAS metrics
│   └── pipeline.py          # production: VectorDB -> SRAS -> generator
├── checkpoints/             # *.pt model weights
├── configs/                 # hyperparameter configs
└── CLAUDE.md
```

## Core Model Spec (do not change without reason)

Scoring: `s_i = wᵀ · tanh(W_q·q + W_d·d_i)`. Shared hidden space, additive interaction,
linear scoring head. Keep it ~98K params / ~0.38MB. `embedding_dim=768`, `hidden_dim=64`,
all projections `bias=False`.

```python
class SRASSelector(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=64):
        super().__init__()
        self.W_q = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.W_d = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.w   = nn.Linear(hidden_dim, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, query_emb, doc_embs):
        # query_emb: [B, D]   doc_embs: [B, N, D]   ->  scores: [B, N]
        B, N, Dm = doc_embs.shape
        h_q = self.W_q(query_emb).unsqueeze(1).expand(-1, N, -1)
        h_d = self.W_d(doc_embs.view(-1, Dm)).view(B, N, -1)
        return self.w(self.tanh(h_q + h_d)).squeeze(-1)
```

## Pipeline (4 stages)

1. **Data pipeline** (`data_pipeline.py`): read `.txt` docs → generate synthetic QA pairs
   (Ollama `qwen2.5:14b`, heuristic fallback) → encode all docs + queries with
   `jhgan/ko-sroberta-multitask` → cache 768-dim vectors to `.pt`. Embeddings are computed ONCE and reused.
2. **Selection**: per QA pair build a candidate pool of **n=8** (1 gold + 7 random distractors).
   SRAS scores them; policy π = softmax(scores); sample **K=3** without replacement.
3. **Reward**: feed the 3 selected docs to frozen generator → answer → reward.
4. **PPO update**: clipped surrogate objective, backprop, AdamW step.

## Reward Engine

`R = 0.6 · Relaxed_F1 + 0.4 · BERTScore` (α=0.6 per paper grid search).
- **Relaxed F1**: token-level F1 on normalized text (lowercase, strip punctuation, remove stopwords).
- **BERTScore**: semantic F1 (`klue/roberta-base`). Cache/warm the BERTScore model once — it is the
  heaviest reward component.
- Per batch, normalize rewards to **zero mean / unit variance** before computing advantages.

## Custom PPO Loop

Hyperparameters (paper defaults — keep unless experimenting):

| Param | Value |
|---|---|
| Epochs | 25 |
| Batch size | 8 |
| Candidates n | 8 |
| Top-k K | 3 |
| Learning rate | 1e-5 |
| Discount γ | 0.99 |
| Clip ε | 0.2 |
| Optimizer | AdamW |
| GAE-like advantage | yes |

Clipped surrogate objective:
```python
ratios = torch.exp(new_log_probs - old_log_probs)
surr1 = ratios * advantages
surr2 = torch.clamp(ratios, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * advantages
policy_loss = -torch.min(surr1, surr2).mean()
optimizer.zero_grad(); policy_loss.backward(); optimizer.step()
```

Record `old_log_probs` at rollout time (detached); recompute `new_log_probs` during the update.

### Stabilization (all three matter — ablations confirm)

- **Supervised warmup**: 1–2 epochs of cross-entropy with the gold doc index as the label,
  BEFORE PPO. Prevents the policy from collapsing to garbage selections. Removing it hurts most early.
- **Reward shaping** (the hybrid reward itself): the single most important factor — removing it
  stalls learning (~0.02 reward). Never train on a bare sparse reward.
- **Reward normalization**: per-batch zero-mean/unit-variance (above).
- **Curriculum learning**: start with easy QA pairs (high top-1 overlap), increase difficulty.
  Improves training smoothness/stability.

## Evaluation

- **Inference is deterministic**: pick the top-K by score (argmax/topk), NOT sampling.
- Report **Relaxed F1**, **BERTScore F1**, **latency** (CPU, batch size 1), **model size (MB)**.
- RAGAS: **Context Precision** (are useful docs ranked high?) and **Faithfulness**
  (does the generation stay grounded in selected context?).
- Baselines to beat/compare: Top-k Cosine, Random, Supervised (FF).
- Deployment target: CPU-only, <1s latency, sub-1MB model.

## Conventions for Claude

- Pure PyTorch only — do NOT introduce RL frameworks (SB3, TRL, RLlib, etc.).
- Freeze the embedding encoder and the QA generator — only SRAS weights train.
- Embeddings are precomputed and cached; never re-embed inside the training loop.
- Keep the model tiny (~98K params). Flag any change that materially grows model size.
- Use `device` consistently; never hardcode `"cpu"` or `"cuda"`.
- Keep selection/reward/PPO modular and independently testable.
- Prefer small, runnable prototypes (≤100 docs) before scaling, per the paper's prototyping setup.

## Common Commands

```bash
cd sras
python src/build_dataset.py        # raw_dataset → data/raw + qa_pairs.jsonl
python src/data_pipeline.py --skip-qg  # build embedding cache (reuse existing QA pairs)
python src/warmup.py               # supervised warmup (cross-entropy)
python src/ppo_trainer.py          # PPO training
python src/evaluate.py --checkpoint checkpoints/sras_best.pt  # deterministic eval
python src/pipeline.py --query "계약 해지 요건은 무엇인가?"   # production inference
```

## Production Flow

```
User query → Vector DB (top-N=8) → SRAS selector (top-K=3) → frozen generator → answer
```