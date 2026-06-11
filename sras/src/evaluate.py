"""
Deterministic evaluation of SRAS.

Metrics (retrieval, no generator needed)
-----------------------------------------
- Gold chunk Hit Rate @K : gold 청크가 선택된 K개에 포함될 확률
- MRR @K                 : Mean Reciprocal Rank (gold 청크의 순위 역수 평균)
- Latency (CPU, batch=1) : 배포 가능성
- Model size (MB)

Baselines
---------
- Top-k Cosine : query-chunk 코사인 유사도 기준 상위 K
- Random       : 무작위 선택

Run:
    python src/evaluate.py --checkpoint checkpoints/sras_best.pt
"""

import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from data_pipeline import build_candidate_pool, load_cache
from model import SRASSelector, build_model

_DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
_CPU    = torch.device("cpu")


# ─── Deterministic top-K selection ───────────────────────────────────────────

@torch.no_grad()
def sras_select(
    model: SRASSelector,
    query_emb: torch.Tensor,
    doc_embs: torch.Tensor,
    top_k: int = 3,
) -> List[int]:
    """Deterministic top-K by score (argmax, no sampling)."""
    model.eval()
    Q = query_emb.unsqueeze(0).to(_DEVICE)
    D = doc_embs.unsqueeze(0).to(_DEVICE)
    scores = model(Q, D).squeeze(0)
    _, indices = torch.topk(scores, k=min(top_k, scores.shape[0]))
    return indices.tolist()


@torch.no_grad()
def sras_rank_all(
    model: SRASSelector,
    query_emb: torch.Tensor,
    doc_embs: torch.Tensor,
) -> List[int]:
    """Full ranking of all docs by score (highest first). Used for MRR."""
    model.eval()
    Q = query_emb.unsqueeze(0).to(_DEVICE)
    D = doc_embs.unsqueeze(0).to(_DEVICE)
    scores = model(Q, D).squeeze(0)
    return torch.argsort(scores, descending=True).tolist()


# ─── Baseline selectors ───────────────────────────────────────────────────────

def cosine_rank_all(query_emb: torch.Tensor, doc_embs: torch.Tensor) -> List[int]:
    q = F.normalize(query_emb.unsqueeze(0), dim=-1)
    d = F.normalize(doc_embs, dim=-1)
    sims = (d @ q.T).squeeze(-1)
    return torch.argsort(sims, descending=True).tolist()


def random_rank_all(n: int, seed: Optional[int] = None) -> List[int]:
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    return order


# ─── Latency benchmark ────────────────────────────────────────────────────────

def benchmark_latency(model: SRASSelector, n: int = 8, n_runs: int = 200) -> float:
    """Mean latency (ms) on CPU, batch-size 1."""
    cpu_model = model.to(_CPU).eval()
    emb_dim = model.W_q.weight.shape[1]
    q = torch.randn(1, emb_dim)
    d = torch.randn(1, n, emb_dim)
    for _ in range(20):          # warmup
        cpu_model(q, d)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        cpu_model(q, d)
        times.append((time.perf_counter() - t0) * 1000)
    model.to(_DEVICE)
    return sum(times) / len(times)


# ─── Evaluation loop ─────────────────────────────────────────────────────────

def evaluate(
    model: SRASSelector,
    cache: Dict,
    n_candidates: int = 8,
    top_k: int = 3,
    max_samples: Optional[int] = None,
    hard_negatives: bool = False,
) -> Dict:
    """
    Evaluate SRAS vs baselines.

    Returns nested dict:
        {
          "sras":   {"hit_rate": ..., "mrr": ...},
          "cosine": {...},
          "random": {...},
          "latency_ms": ..., "size_mb": ..., "params": ...,
        }
    """
    qa_pairs = cache["qa_pairs"]
    if max_samples:
        qa_pairs = qa_pairs[:max_samples]
    n_total = len(qa_pairs)

    results = {
        "sras":   {"hits": 0, "rr_sum": 0.0},
        "cosine": {"hits": 0, "rr_sum": 0.0},
        "random": {"hits": 0, "rr_sum": 0.0},
    }

    for qa_idx, qa in enumerate(tqdm(qa_pairs, desc="Evaluating")):
        q_emb, pool, gold_pos, pool_doc_ids = build_candidate_pool(
            cache, qa_idx, n_candidates=n_candidates,
            seed=qa_idx, hard_negatives=hard_negatives,
        )

        rankings = {
            "sras":   sras_rank_all(model, q_emb, pool),
            "cosine": cosine_rank_all(q_emb, pool),
            "random": random_rank_all(n_candidates, seed=qa_idx),
        }

        for name, ranking in rankings.items():
            # Hit @K: gold doc가 top-K 안에 있는가
            top_k_set = set(ranking[:top_k])
            if gold_pos in top_k_set:
                results[name]["hits"] += 1

            # MRR: gold doc의 전체 랭킹에서의 순위 (1-indexed)
            rank = ranking.index(gold_pos) + 1
            results[name]["rr_sum"] += 1.0 / rank

    summary = {}
    for name, vals in results.items():
        summary[name] = {
            "hit_rate": vals["hits"] / n_total,
            "mrr":      vals["rr_sum"] / n_total,
        }

    summary["latency_ms"] = benchmark_latency(model, n=n_candidates)
    summary["size_mb"]    = model.size_mb()
    summary["params"]     = model.count_params()
    summary["n_samples"]  = n_total

    return summary


def print_summary(summary: Dict):
    n = summary.get("n_samples", "?")
    k = 3
    print("\n" + "=" * 60)
    print(f"SRAS Evaluation  (n={n}, K={k})")
    print("=" * 60)
    print(f"  {'Method':<10} | Hit Rate @{k}  | MRR")
    print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*8}")
    for name in ["sras", "cosine", "random"]:
        if name in summary:
            m = summary[name]
            print(f"  {name.upper():<10} | {m['hit_rate']:>10.1%}  | {m['mrr']:.4f}")
    print()
    print(f"  Latency (CPU, batch=1) : {summary.get('latency_ms', 0):.3f} ms")
    print(f"  Model size             : {summary.get('size_mb', 0):.2f} MB")
    print(f"  Parameters             : {summary.get('params', 0):,}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   default="checkpoints/sras_best.pt")
    parser.add_argument("--cache",        default="data/cache/embeddings.pt")
    parser.add_argument("--n-candidates", type=int, default=8)
    parser.add_argument("--top-k",        type=int, default=3)
    parser.add_argument("--max-samples",  type=int, default=None)
    parser.add_argument("--hard-negatives", action="store_true",
                        help="random distractor 대신 코사인 유사도 상위 청크를 distractor로 사용.")
    args = parser.parse_args()

    cache = load_cache(args.cache)
    model = build_model()

    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=_DEVICE))
        print(f"Loaded checkpoint: {ckpt}")
    else:
        print(f"Warning: checkpoint not found at {ckpt}. Evaluating with random weights.")

    summary = evaluate(
        model,
        cache,
        n_candidates=args.n_candidates,
        top_k=args.top_k,
        max_samples=args.max_samples,
        hard_negatives=args.hard_negatives,
    )
    print_summary(summary)
