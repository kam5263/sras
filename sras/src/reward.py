"""
Reward engine: R = 0.6 · Relaxed_F1 + 0.4 · BERTScore
한국어 텍스트 기준으로 동작한다 (klue/roberta-base, lang="ko").
"""

import re
import string
from typing import List

import torch
from bert_score import BERTScorer
import bert_score.utils as _bsu


def _safe_sent_encode(tokenizer, sent):
    """
    bert_score.utils.sent_encode 패치: transformers ≥5.x 호환 + 빈 문자열 처리.
    """
    sent = sent.strip() if sent else ""
    return tokenizer.encode(
        sent if sent else "",
        add_special_tokens=True,
        max_length=getattr(tokenizer, "model_max_length", 512),
        truncation=True,
    )


_bsu.sent_encode = _safe_sent_encode

# 한국어 불용어: 문장 부호나 단독으로 나타나는 기능어
_STOPWORDS = frozenset([
    "이", "가", "은", "는", "을", "를", "의", "에", "와", "과",
    "도", "만", "에서", "으로", "로", "까지", "부터", "이나", "나",
    "이며", "며", "이고", "고", "이다", "다", "하는", "한", "있는",
    "없는", "하고", "이라", "라", "그", "이", "저", "것", "수",
    "있", "없", "되", "하", "않", "못", "안",
])


def _normalize(text: str) -> List[str]:
    """소문자화, 구두점 제거, 불용어 제거 후 토큰 리스트 반환."""
    text = text.lower()
    # ASCII 구두점 + 한국어 특수문자 제거
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[。，、·…「」『』【】〔〕《》〈〉]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in _STOPWORDS]


def relaxed_f1(prediction: str, gold: str) -> float:
    """정규화된 예측/정답 간 토큰 수준 F1."""
    pred_tokens = _normalize(prediction)
    gold_tokens = _normalize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_set = set(pred_tokens)
    gold_set = set(gold_tokens)
    common = pred_set & gold_set
    if not common:
        return 0.0
    precision = len(common) / len(pred_set)
    recall    = len(common) / len(gold_set)
    return 2 * precision * recall / (precision + recall)


class RewardEngine:
    """
    하이브리드 보상 R = alpha·F1 + (1-alpha)·BERTScore.

    BERTScorer는 생성 시 한 번 워밍업하고 학습 전체에서 재사용된다.
    배치 처리를 위해 compute_batch()를 사용한다.
    """

    def __init__(self, alpha: float = 0.6, device: str = "cpu"):
        self.alpha = alpha
        self.device = device
        # klue/roberta-base: 한국어 특화 RoBERTa 모델
        # rescale_with_baseline=False: 한국어 사전 계산 baseline 없음
        # num_layers=12: bert_score의 model2layers 딕셔너리에 없는 모델은
        # 레이어 수를 직접 지정해야 KeyError를 피할 수 있다.
        self._scorer = BERTScorer(
            model_type="klue/roberta-base",
            num_layers=12,
            lang="ko",
            rescale_with_baseline=False,
            device=device,
        )

    def compute_batch(
        self,
        predictions: List[str],
        gold_answers: List[str],
    ) -> torch.Tensor:
        """
        Args:
            predictions:  list of B generated answers
            gold_answers: list of B reference answers
        Returns:
            rewards: float tensor of shape [B]
        """
        safe_preds  = [p if p.strip() else "." for p in predictions]
        safe_golds  = [g if g.strip() else "." for g in gold_answers]

        _, _, bert_f1 = self._scorer.score(safe_preds, safe_golds)
        bert_f1 = bert_f1.float()   # [B]

        f1_scores = torch.tensor(
            [relaxed_f1(p, g) for p, g in zip(predictions, gold_answers)],
            dtype=torch.float32,
        )

        rewards = self.alpha * f1_scores + (1.0 - self.alpha) * bert_f1
        return rewards

    @staticmethod
    def normalize(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """배치별 zero-mean / unit-variance 정규화."""
        if rewards.numel() <= 1:
            return torch.zeros_like(rewards)
        std = rewards.std()
        if torch.isnan(std) or std < eps:
            return rewards - rewards.mean()
        return (rewards - rewards.mean()) / (std + eps)


if __name__ == "__main__":
    engine = RewardEngine(alpha=0.6, device="cpu")
    preds  = ["광합성은 산소를 생성한다", "하늘은 초록색이다"]
    golds  = ["광합성은 부산물로 산소를 발생시킨다", "하늘은 파란색이다"]
    r = engine.compute_batch(preds, golds)
    print(f"Rewards: {r}")
    print(f"Normalized: {RewardEngine.normalize(r)}")
