"""SRASSelector — core scoring model (~197K params, ~0.76 MB)."""

import torch
import torch.nn as nn


class SRASSelector(nn.Module):
    """
    Scores each candidate document against a query.

    Scoring: s_i = wᵀ · tanh(W_q·q + W_d·d_i)
    Shared hidden space, additive interaction, linear scoring head.
    """

    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 64):
        super().__init__()
        self.W_q = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.W_d = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.w   = nn.Linear(hidden_dim, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, query_emb: torch.Tensor, doc_embs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_emb: [B, D]
            doc_embs:  [B, N, D]
        Returns:
            scores: [B, N]  (raw logits, not softmaxed)
        """
        B, N, Dm = doc_embs.shape
        h_q = self.W_q(query_emb).unsqueeze(1).expand(-1, N, -1)        # [B, N, H]
        h_d = self.W_d(doc_embs.view(-1, Dm)).view(B, N, -1)             # [B, N, H]
        return self.w(self.tanh(h_q + h_d)).squeeze(-1)                  # [B, N]

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def size_mb(self) -> float:
        return sum(p.numel() * p.element_size() for p in self.parameters()) / 1024 ** 2


def build_model(embedding_dim: int = 768, hidden_dim: int = 64) -> SRASSelector:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SRASSelector(embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(device)
    return model


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model()
    print(f"Parameters : {model.count_params():,}")
    print(f"Size       : {model.size_mb():.2f} MB")
    print(f"Device     : {device}")

    B, N, D = 4, 8, 768
    q = torch.randn(B, D, device=device)
    d = torch.randn(B, N, D, device=device)
    scores = model(q, d)
    print(f"Score shape: {scores.shape}")   # [4, 8]
