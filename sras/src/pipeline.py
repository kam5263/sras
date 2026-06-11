"""
Production inference pipeline.

Flow:
    User query → Vector DB (top-N=8) → SRAS selector (top-K=3) → frozen generator → answer

The VectorDB is simulated with an in-memory cosine index over the cached embeddings.
In production, replace with an actual vector store (FAISS, Chroma, Qdrant, etc.).

Usage:
    python src/pipeline.py --query "What is photosynthesis?"
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from data_pipeline import load_cache
from evaluate import sras_select
from model import SRASSelector, build_model
from ppo_trainer import FrozenGenerator

_DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ─── In-memory Vector DB ──────────────────────────────────────────────────────

class VectorDB:
    """
    Lightweight in-memory cosine-similarity vector store.
    For production, swap with FAISS / Chroma / Qdrant.
    """

    def __init__(self, doc_ids: List[str], doc_embs: torch.Tensor):
        self.doc_ids  = doc_ids
        self.doc_embs = F.normalize(doc_embs.float(), dim=-1)  # [M, 384] normalized

    @classmethod
    def from_cache(cls, cache: Dict) -> "VectorDB":
        return cls(cache["chunk_ids"], cache["chunk_embs"])

    def retrieve(self, query_emb: torch.Tensor, n: int = 8) -> Tuple[List[str], torch.Tensor]:
        """
        Retrieve top-n documents by cosine similarity.

        Returns:
            doc_ids:  list of n retrieved doc IDs
            doc_embs: Tensor [n, 384]
        """
        q = F.normalize(query_emb.float().unsqueeze(0), dim=-1)   # [1, 384]
        sims = (self.doc_embs @ q.T).squeeze(-1)                   # [M]
        _, indices = torch.topk(sims, k=min(n, sims.shape[0]))
        top_ids  = [self.doc_ids[i] for i in indices.tolist()]
        top_embs = self.doc_embs[indices]                          # [n, 384]
        return top_ids, top_embs


# ─── Query encoder (ko-sroberta-multitask, frozen) ───────────────────────────

class QueryEncoder:
    def __init__(self, encoder_name: str = "jhgan/ko-sroberta-multitask"):
        from sentence_transformers import SentenceTransformer
        self._encoder = SentenceTransformer(encoder_name, device=str(_DEVICE))

    @torch.no_grad()
    def encode(self, query: str) -> torch.Tensor:
        emb = self._encoder.encode(
            query, convert_to_tensor=True, device=str(_DEVICE)
        )
        return emb.cpu()   # return on CPU; move to device inside model call


# ─── End-to-end RAG pipeline ─────────────────────────────────────────────────

class SRASPipeline:
    def __init__(
        self,
        model: SRASSelector,
        vector_db: VectorDB,
        generator: FrozenGenerator,
        query_encoder: QueryEncoder,
        doc_texts: Optional[Dict[str, str]] = None,
        n_retrieve: int = 8,
        top_k: int = 3,
    ):
        self.model         = model
        self.vector_db     = vector_db
        self.generator     = generator
        self.query_encoder = query_encoder
        self.doc_texts     = doc_texts or {}   # doc_id -> raw text
        self.n_retrieve    = n_retrieve
        self.top_k         = top_k

    def run(self, query: str, verbose: bool = False) -> Dict:
        t_start = time.perf_counter()

        # 1. Encode query
        q_emb = self.query_encoder.encode(query)  # [384]

        # 2. Vector DB retrieval (top-N)
        retrieved_ids, retrieved_embs = self.vector_db.retrieve(q_emb, n=self.n_retrieve)

        # 3. SRAS selection (top-K, deterministic)
        sel_indices = sras_select(self.model, q_emb, retrieved_embs, self.top_k)
        selected_ids   = [retrieved_ids[i] for i in sel_indices]
        selected_texts = [self.doc_texts.get(did, did) for did in selected_ids]

        # 4. Frozen generator
        answer = self.generator.generate(query, selected_texts)

        latency_ms = (time.perf_counter() - t_start) * 1000

        if verbose:
            print(f"\nQuery     : {query}")
            print(f"Retrieved : {retrieved_ids}")
            print(f"Selected  : {selected_ids}")
            print(f"Answer    : {answer}")
            print(f"Latency   : {latency_ms:.1f} ms")

        return {
            "query":         query,
            "retrieved_ids": retrieved_ids,
            "selected_ids":  selected_ids,
            "selected_texts": selected_texts,
            "answer":        answer,
            "latency_ms":    latency_ms,
        }


def build_pipeline(
    checkpoint_path: str = "checkpoints/sras_best.pt",
    cache_path: str       = "data/cache/embeddings.pt",
    n_retrieve: int       = 8,
    top_k: int            = 3,
) -> SRASPipeline:
    """Convenience factory that wires all components together."""
    cache   = load_cache(cache_path)

    model = build_model()
    ckpt  = Path(checkpoint_path)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=_DEVICE))
        print(f"Loaded SRAS checkpoint: {ckpt}")
    else:
        print(f"Warning: no checkpoint at {ckpt} — using random weights.")

    model.eval()

    vector_db     = VectorDB.from_cache(cache)
    query_encoder = QueryEncoder()
    generator     = FrozenGenerator()

    return SRASPipeline(
        model         = model,
        vector_db     = vector_db,
        generator     = generator,
        query_encoder = query_encoder,
        doc_texts     = cache.get("chunk_texts", {}),
        n_retrieve    = n_retrieve,
        top_k         = top_k,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query",       required=True, help="Natural language query")
    parser.add_argument("--checkpoint",  default="checkpoints/sras_best.pt")
    parser.add_argument("--cache",       default="data/cache/embeddings.pt")
    parser.add_argument("--n-retrieve",  type=int, default=8)
    parser.add_argument("--top-k",       type=int, default=3)
    args = parser.parse_args()

    pipeline = build_pipeline(
        checkpoint_path = args.checkpoint,
        cache_path      = args.cache,
        n_retrieve      = args.n_retrieve,
        top_k           = args.top_k,
    )

    result = pipeline.run(args.query, verbose=True)
