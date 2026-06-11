"""
Custom PPO training loop for SRAS.

Algorithm
---------
For each epoch:
  1. Rollout: sample K docs per query using π = softmax(scores),
     record (query_emb, doc_embs, selected_indices, old_log_probs) – detached.
  2. Generate answers with the frozen T5 generator.
  3. Compute hybrid rewards (Relaxed F1 + BERTScore); normalize per batch.
  4. PPO update: recompute log probs, compute clipped surrogate loss, AdamW step.

Curriculum support: sort QA pairs by "easiness" (cosine sim of gold doc to query)
and anneal over epochs so early epochs see easier examples.

Run:
    python src/ppo_trainer.py
"""

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import re
import requests

from data_pipeline import build_candidate_pool, load_cache
from model import SRASSelector, build_model
from reward import RewardEngine, relaxed_f1

_DEVICE        = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
_GENERATOR_NAME  = "qwen3.5:2b"
_OLLAMA_BASE_URL = "http://localhost:11434"


# ─── Rollout buffer ───────────────────────────────────────────────────────────

@dataclass
class RolloutStep:
    query_emb:        torch.Tensor   # [384]
    doc_embs:         torch.Tensor   # [n, 384]
    selected_indices: torch.Tensor   # [K] – sampled doc positions (pool-local)
    old_log_probs:    torch.Tensor   # [K] – log π(a) at rollout time (detached)
    pool_chunk_ids:   list = field(default_factory=list)  # pool-local chunk_id mapping
    reward:           float = 0.0


# ─── Log prob of sampling K items without replacement from softmax ────────────

def _log_prob_selection(
    scores: torch.Tensor,
    selected: torch.Tensor,
    K: int,
) -> torch.Tensor:
    """
    Sequential without-replacement log prob using the chain rule:
      log P = Σ_k log(exp(s_{i_k}) / Σ_{j not yet selected} exp(s_j))

    Args:
        scores:   [n] raw logits (unnormalized)
        selected: [K] chosen indices (LongTensor)
        K:        number selected
    Returns:
        log_prob: scalar tensor
    """
    log_prob = torch.zeros(1, device=scores.device)
    available = torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)
    for k in range(K):
        avail_scores = scores.masked_fill(~available, float("-inf"))
        log_prob += F.log_softmax(avail_scores, dim=-1)[selected[k]]
        available[selected[k]] = False
    return log_prob.squeeze()


# ─── Text generation ──────────────────────────────────────────────────────────

_DOC_TRUNCATE_CHARS = None  # 문서 전체 사용 (None = 제한 없음)

class FrozenGenerator:
    """
    Ollama 기반 generative QA generator.
    로컬 Ollama 서버(localhost:11434)에 HTTP 요청을 보내 답변을 생성한다.
    모델 로딩/디바이스 관리는 Ollama가 담당하므로 PyTorch 메모리를 사용하지 않는다.

    최적화:
    - 문서를 _DOC_TRUNCATE_CHARS 자로 잘라 프롬프트 길이를 줄임
    - (query, sel_doc_ids) → answer 인메모리 캐시로 동일 선택 재생성 방지
    """

    _SYSTEM_PROMPT = (
        "당신은 주어진 문서만을 근거로 질문에 답하는 법률 QA 도우미입니다. "
        "반드시 문서 내용에 근거하여 한 문장~세 문장으로 간결하게 답하세요. "
        "문서에 근거가 없으면 '알 수 없습니다'라고 답하세요."
    )

    def __init__(self, model_name: str = _GENERATOR_NAME, base_url: str = _OLLAMA_BASE_URL):
        self._model  = model_name
        self._url    = f"{base_url}/api/chat"
        self._cache: Dict[tuple, str] = {}
        self._cache_hits = 0
        print(f"Generator 설정: Ollama {model_name} @ {base_url}")
        try:
            resp   = requests.get(f"{base_url}/api/tags", timeout=5)
            models = [m["name"] for m in resp.json().get("models", [])]
            if not any(model_name in m for m in models):
                print(f"  경고: '{model_name}' 모델이 Ollama에 없음. 사용 가능: {models}")
        except Exception as e:
            print(f"  경고: Ollama 서버 연결 실패 ({e}). `ollama serve` 확인 필요.")

    def generate(self, query: str, selected_docs: List[str]) -> str:
        # 캐시 키: (query, 선택된 doc 텍스트 앞 100자 튜플) — temperature=0이므로 결정론적
        doc_snippets = tuple(d.strip()[:100] for d in selected_docs if d.strip())
        cache_key    = (query, doc_snippets)
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]

        truncated = [
            d.strip()[:_DOC_TRUNCATE_CHARS] if _DOC_TRUNCATE_CHARS else d.strip()
            for d in selected_docs if d.strip()
        ]
        context   = "\n\n".join(truncated)
        if not context:
            return ""

        payload = {
            "model":  self._model,
            "think":  False,   # Ollama native: Qwen3 thinking 완전 비활성화
            "messages": [
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user",   "content": f"[문서]\n{context}\n\n[질문]\n{query}"},
            ],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 256},
        }
        resp   = requests.post(self._url, json=payload, timeout=120)
        resp.raise_for_status()
        raw    = resp.json()["message"]["content"]
        # 처음 3번만 raw 응답 출력 (디버그)
        if len(self._cache) < 3:
            print(f"  [DBG raw] {repr(raw[:300])}")
        answer = raw.strip()
        # Qwen3 thinking 태그 제거
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
        # 첫 번째 non-empty 줄 반환
        for line in answer.split("\n"):
            line = line.strip()
            if line:
                answer = line
                break

        self._cache[cache_key] = answer
        return answer

    def cache_stats(self) -> str:
        total = len(self._cache) + self._cache_hits
        return f"cache={len(self._cache)} unique, {self._cache_hits} hits / {total} total"


# ─── Curriculum ordering ──────────────────────────────────────────────────────

def _curriculum_order(cache: Dict, n_candidates: int) -> List[int]:
    """
    Sort QA pair indices by cosine similarity of query embedding to its gold chunk
    (descending = easy first). Easy = gold chunk is already very similar to the query.
    """
    chunk_ids  = cache["chunk_ids"]
    chunk_embs = cache["chunk_embs"]   # [C, D]
    q_embs     = cache["query_embs"]   # [Q, D]
    qa_pairs   = cache["qa_pairs"]

    scores = []
    for i, qa in enumerate(qa_pairs):
        gold_idx = chunk_ids.index(qa["gold_chunk_id"])
        q  = F.normalize(q_embs[i], dim=-1)
        d  = F.normalize(chunk_embs[gold_idx], dim=-1)
        scores.append((i, (q * d).sum().item()))

    scores.sort(key=lambda x: -x[1])   # descending cosine → easy first
    return [idx for idx, _ in scores]


# ─── PPO trainer ─────────────────────────────────────────────────────────────

class PPOTrainer:
    def __init__(
        self,
        model: SRASSelector,
        cache: Dict,
        reward_engine: RewardEngine,
        generator: FrozenGenerator,
        chunk_texts: Optional[Dict[str, str]] = None,
        n_candidates: int = 8,
        top_k: int = 3,
        epochs: int = 25,
        batch_size: int = 8,
        lr: float = 1e-5,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        ppo_epochs: int = 4,
        checkpoint_path: str = "checkpoints/sras_best.pt",
        curriculum: bool = True,
        log_samples: int = 2,
        doc_hit_bonus: float = 0.2,
        hard_negatives: bool = False,
    ):
        self.model          = model
        self.cache          = cache
        self.reward_engine  = reward_engine
        self.generator      = generator
        self.chunk_texts    = chunk_texts or {}
        self.n_candidates   = n_candidates
        self.top_k          = top_k
        self.epochs         = epochs
        self.batch_size     = batch_size
        self.gamma          = gamma
        self.clip_epsilon   = clip_epsilon
        self.ppo_epochs     = ppo_epochs
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.doc_hit_bonus  = doc_hit_bonus
        self.hard_negatives = hard_negatives

        self.log_samples    = log_samples
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)

        self.qa_order: List[int]
        if curriculum:
            self.qa_order = _curriculum_order(cache, n_candidates)
        else:
            self.qa_order = list(range(len(cache["qa_pairs"])))

    # ── Rollout ────────────────────────────────────────────────────────────────

    def _rollout_batch(self, batch_qa_idx: List[int]) -> List[RolloutStep]:
        """Sample actions (document selections) for a batch of QA pairs."""
        self.model.eval()
        steps = []
        with torch.no_grad():
            for qa_idx in batch_qa_idx:
                q_emb, pool, _, pool_chunk_ids = build_candidate_pool(
                    self.cache, qa_idx, self.n_candidates,
                    seed=qa_idx, hard_negatives=self.hard_negatives,
                )
                Q = q_emb.unsqueeze(0).to(_DEVICE)   # [1, 384]
                D = pool.unsqueeze(0).to(_DEVICE)     # [1, n, 384]

                scores  = self.model(Q, D).squeeze(0)  # [n]

                # Guard: if weights became NaN (e.g. after a bad update), reset scores
                if torch.isnan(scores).any():
                    scores = torch.zeros_like(scores)

                probs   = F.softmax(scores, dim=-1)     # [n]

                # Sample K docs without replacement using sequential sampling
                selected = []
                available = list(range(self.n_candidates))
                for _ in range(min(self.top_k, len(available))):
                    avail_probs = probs[available]
                    avail_probs = avail_probs.clamp(min=0.0)   # no negatives after float errors
                    s = avail_probs.sum()
                    avail_probs = avail_probs / s if s > 0 else torch.ones_like(avail_probs) / len(available)
                    choice = available[torch.multinomial(avail_probs, 1).item()]
                    selected.append(choice)
                    available.remove(choice)

                sel_tensor = torch.tensor(selected, dtype=torch.long, device=_DEVICE)
                old_lp = _log_prob_selection(scores.detach(), sel_tensor, self.top_k)

                steps.append(RolloutStep(
                    query_emb        = q_emb.cpu(),
                    doc_embs         = pool.cpu(),
                    selected_indices = sel_tensor.cpu(),
                    old_log_probs    = old_lp.detach().cpu(),
                    pool_chunk_ids   = pool_chunk_ids,
                ))
        return steps

    # ── Reward computation ─────────────────────────────────────────────────────

    def _compute_rewards(
        self,
        steps: List[RolloutStep],
        qa_indices: List[int],
        log_samples: int = 0,
    ) -> Tuple[torch.Tensor, List[bool]]:
        """Generate answers and compute hybrid rewards for the batch.

        Returns:
            rewards: float tensor [B] — hybrid reward (+ optional doc hit bonus)
            hits:    list[bool] [B]  — whether gold doc was in the selected set
        """
        qa_pairs = self.cache["qa_pairs"]

        predictions  = []
        gold_answers = []
        queries      = []
        sel_ids_list = []
        gold_hits: List[bool] = []

        for step, qa_idx in zip(steps, qa_indices):
            qa       = qa_pairs[qa_idx]
            selected = step.selected_indices.tolist()

            sel_ids   = [step.pool_chunk_ids[i] for i in selected if i < len(step.pool_chunk_ids)]
            sel_texts = [self.chunk_texts.get(cid, cid) for cid in sel_ids]
            gold_id   = qa.get("gold_chunk_id", "")

            answer = self.generator.generate(qa["query"], sel_texts)
            predictions.append(answer)
            gold_answers.append(qa["gold_answer"])
            queries.append(qa["query"])
            sel_ids_list.append(sel_ids)
            gold_hits.append(gold_id in sel_ids)

        raw_rewards = self.reward_engine.compute_batch(predictions, gold_answers)

        # Direct retrieval signal: bypass generator noise when gold doc is selected
        if self.doc_hit_bonus > 0.0:
            hit_tensor  = torch.tensor(gold_hits, dtype=torch.float32)
            raw_rewards = raw_rewards + self.doc_hit_bonus * hit_tensor

        if log_samples > 0:
            f1_scores = torch.tensor(
                [relaxed_f1(p, g) for p, g in zip(predictions, gold_answers)],
                dtype=torch.float32,
            )
            print("\n" + "─" * 70)
            for i in range(min(log_samples, len(predictions))):
                gold_id = qa_pairs[qa_indices[i]].get("gold_chunk_id", "?")
                hit = gold_hits[i]
                r_base = raw_rewards[i] - self.doc_hit_bonus * float(hit)
                bert_approx = (r_base - 0.6 * f1_scores[i]) / 0.4
                print(f"[샘플 {i+1}]")
                print(f"  질문    : {queries[i][:80]}")
                print(f"  정답    : {gold_answers[i][:80]}")
                print(f"  생성답  : {predictions[i][:80]}")
                print(f"  선택doc : {sel_ids_list[i]}  (gold={gold_id}, hit={'✓' if hit else '✗'})")
                print(f"  F1={f1_scores[i]:.4f}  BERTScore≈{bert_approx:.4f}  R={raw_rewards[i]:.4f}")
            print("─" * 70 + "\n")

        return raw_rewards, gold_hits

    # ── PPO update ────────────────────────────────────────────────────────────

    def _ppo_update(self, steps: List[RolloutStep], rewards: torch.Tensor):
        """PPO update with multiple inner epochs over the same rollout batch."""
        advantages    = RewardEngine.normalize(rewards).to(_DEVICE)
        old_log_probs = torch.stack([s.old_log_probs for s in steps]).to(_DEVICE)

        losses = []
        for _ in range(self.ppo_epochs):
            self.model.train()
            new_log_probs_list = []
            for step in steps:
                Q      = step.query_emb.unsqueeze(0).to(_DEVICE)
                D      = step.doc_embs.unsqueeze(0).to(_DEVICE)
                scores = self.model(Q, D).squeeze(0)
                sel    = step.selected_indices.to(_DEVICE)
                new_lp = _log_prob_selection(scores, sel, self.top_k)
                new_log_probs_list.append(new_lp)

            new_log_probs = torch.stack(new_log_probs_list)

            log_ratio   = torch.clamp(new_log_probs - old_log_probs, -10.0, 10.0)
            ratios      = torch.exp(log_ratio)
            surr1       = ratios * advantages
            surr2       = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            if torch.isnan(policy_loss):
                break

            self.optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            losses.append(policy_loss.item())

        return sum(losses) / max(len(losses), 1) if losses else float("nan")

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self, warmup_ckpt: Optional[str] = None):
        if warmup_ckpt and Path(warmup_ckpt).exists():
            self.model.load_state_dict(torch.load(warmup_ckpt, map_location=_DEVICE))
            print(f"Loaded warmup weights from {warmup_ckpt}")

        best_reward = float("-inf")
        n_qa = len(self.qa_order)

        for epoch in range(self.epochs):
            # Curriculum: linearly anneal difficulty over epochs
            # Use the first portion of (easy) examples early, full set later
            frac = min(1.0, 0.5 + 0.5 * epoch / max(self.epochs - 1, 1))
            n_active = max(self.batch_size, int(frac * n_qa))
            active_indices = self.qa_order[:n_active]

            epoch_rewards  = []
            epoch_losses   = []
            hit_rewards:  List[float] = []
            miss_rewards: List[float] = []

            for batch_start in tqdm(
                range(0, n_active, self.batch_size),
                desc=f"PPO epoch {epoch + 1}/{self.epochs}",
            ):
                batch_qa_idx = active_indices[batch_start: batch_start + self.batch_size]
                if not batch_qa_idx:
                    continue

                # 매 epoch 첫 번째 배치에서만 샘플 2개 출력
                log_n = self.log_samples if batch_start == 0 else 0

                steps             = self._rollout_batch(batch_qa_idx)
                rewards, hits     = self._compute_rewards(steps, batch_qa_idx, log_samples=log_n)
                loss              = self._ppo_update(steps, rewards)

                epoch_rewards.append(rewards.mean().item())
                epoch_losses.append(loss)

                for r, h in zip(rewards.tolist(), hits):
                    (hit_rewards if h else miss_rewards).append(r)

            avg_r    = sum(epoch_rewards) / max(len(epoch_rewards), 1)
            avg_l    = sum(epoch_losses)  / max(len(epoch_losses),  1)
            n_hits   = len(hit_rewards)
            n_miss   = len(miss_rewards)
            hit_rate = n_hits / max(n_hits + n_miss, 1)
            avg_r_hit  = sum(hit_rewards)  / max(n_hits, 1)
            avg_r_miss = sum(miss_rewards) / max(n_miss, 1)
            print(
                f"  Epoch {epoch + 1:3d} | "
                f"avg_reward={avg_r:.4f}  policy_loss={avg_l:.4f}  "
                f"active_pairs={n_active}\n"
                f"           | hit_rate={hit_rate:.1%}  "
                f"R(hit)={avg_r_hit:.4f}  R(miss)={avg_r_miss:.4f}  "
                f"{self.generator.cache_stats()}"
            )

            if avg_r > best_reward:
                best_reward = avg_r
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"  ✓ New best ({best_reward:.4f}) → {self.checkpoint_path}")

        print(f"\nTraining complete. Best reward: {best_reward:.4f}")


# ─── Script entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache",         default="data/cache/embeddings.pt")
    parser.add_argument("--warmup-ckpt",   default="checkpoints/sras_warmup.pt")
    parser.add_argument("--checkpoint",    default="checkpoints/sras_best.pt")
    parser.add_argument("--epochs",        type=int,   default=25)
    parser.add_argument("--batch-size",    type=int,   default=8)
    parser.add_argument("--lr",            type=float, default=1e-5)
    parser.add_argument("--ppo-epochs",    type=int,   default=4)
    parser.add_argument("--n-candidates",  type=int,   default=8)
    parser.add_argument("--top-k",         type=int,   default=3)
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--log-samples",    type=int,   default=2,
                        help="매 epoch 첫 배치에서 출력할 샘플 수 (0=비활성화)")
    parser.add_argument("--doc-hit-bonus", type=float, default=0.2,
                        help="gold doc 선택 시 추가 보상 (0=비활성화). "
                             "generator 노이즈를 보완하는 직접 retrieval 신호.")
    parser.add_argument("--hard-negatives", action="store_true",
                        help="random distractor 대신 query와 코사인 유사도 상위 청크를 "
                             "distractor로 사용 (harder evaluation pool).")
    args = parser.parse_args()

    cache       = load_cache(args.cache)
    chunk_texts = cache.get("chunk_texts", {})
    model       = build_model()
    reward_eng  = RewardEngine(alpha=0.6, device=str(_DEVICE))
    generator   = FrozenGenerator()

    trainer = PPOTrainer(
        model           = model,
        cache           = cache,
        reward_engine   = reward_eng,
        generator       = generator,
        chunk_texts     = chunk_texts,
        n_candidates    = args.n_candidates,
        top_k           = args.top_k,
        epochs          = args.epochs,
        batch_size      = args.batch_size,
        lr              = args.lr,
        ppo_epochs      = args.ppo_epochs,
        checkpoint_path = args.checkpoint,
        curriculum      = not args.no_curriculum,
        log_samples     = args.log_samples,
        doc_hit_bonus   = args.doc_hit_bonus,
        hard_negatives  = args.hard_negatives,
    )

    trainer.train(warmup_ckpt=args.warmup_ckpt)
