"""
Supervised warmup: 1–2 epochs of cross-entropy on the gold chunk index.

This is run BEFORE PPO to prevent the policy from collapsing to random
selections in early training. The gold chunk is always at position 0 in each
candidate pool (see data_pipeline.build_candidate_pool).

Run:
    python src/warmup.py
"""

import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data_pipeline import build_candidate_pool, load_cache
from model import SRASSelector, build_model

_DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def warmup_train(
    model: SRASSelector,
    cache: dict,
    n_candidates: int = 8,
    epochs: int = 2,
    batch_size: int = 8,
    lr: float = 1e-3,
    checkpoint_path: str = "checkpoints/sras_warmup.pt",
    seed: int = 42,
) -> SRASSelector:
    """Cross-entropy warmup; gold doc is at pool index 0 for every sample."""
    random.seed(seed)
    torch.manual_seed(seed)

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n_qa = len(cache["qa_pairs"])
    indices = list(range(n_qa))

    for epoch in range(epochs):
        random.shuffle(indices)
        total_loss = 0.0
        n_batches  = 0

        for batch_start in tqdm(
            range(0, n_qa, batch_size),
            desc=f"Warmup epoch {epoch + 1}/{epochs}",
        ):
            batch_idx = indices[batch_start: batch_start + batch_size]
            if not batch_idx:
                continue

            query_embs = []
            doc_embs   = []
            gold_labels = []

            for qa_idx in batch_idx:
                q_emb, pool, gold_pos, _ = build_candidate_pool(
                    cache, qa_idx, n_candidates=n_candidates, seed=qa_idx
                )
                query_embs.append(q_emb)
                doc_embs.append(pool)
                gold_labels.append(gold_pos)  # always 0

            # Stack into tensors and move to device
            Q = torch.stack(query_embs).to(_DEVICE)          # [B, 384]
            D = torch.stack(doc_embs).to(_DEVICE)            # [B, n, 384]
            labels = torch.tensor(gold_labels, device=_DEVICE)  # [B]

            scores = model(Q, D)                              # [B, n]
            loss   = criterion(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch + 1} — avg cross-entropy loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), checkpoint_path)
    print(f"Warmup checkpoint saved → {checkpoint_path}")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache",       default="data/cache/embeddings.pt")
    parser.add_argument("--checkpoint",  default="checkpoints/sras_warmup.pt")
    parser.add_argument("--epochs",      type=int,   default=2)
    parser.add_argument("--batch-size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--n-candidates",type=int,   default=8)
    args = parser.parse_args()

    cache = load_cache(args.cache)
    model = build_model()
    print(f"Model params: {model.count_params():,}  size: {model.size_mb():.2f} MB")

    warmup_train(
        model,
        cache,
        n_candidates=args.n_candidates,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_path=args.checkpoint,
    )
