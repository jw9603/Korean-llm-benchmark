"""
LLM Evaluation 공통 모듈

EvaluationResult, BaseEvaluator 등 공통 클래스를 정의합니다.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta


@dataclass
class EvaluationResult:
    """
    단일 데이터셋 평가 결과를 저장하는 데이터 클래스.

    Attributes:
        dataset_name: 데이터셋 이름 (예: "kmmlu", "csatqa")
        model_name: 평가한 모델의 전체 이름
        num_samples: 평가한 샘플 수
        score: 정확도 (0.0 ~ 1.0)
        score_norm: 길이 정규화된 정확도 (MCQA만 해당, Generation은 None)
        metric: 사용된 메트릭 이름 ("accuracy" 또는 "exact_match")
        timestamp: 평가 시간 (ISO 형식)
        details: 각 샘플별 상세 결과 리스트
    """
    dataset_name: str
    model_name: str
    num_samples: int
    score: float
    score_norm: float | None
    metric: str
    timestamp: str
    details: list[dict] = field(default_factory=list)


def get_kst_timestamp() -> str:
    """현재 시간을 KST(한국 표준시) ISO 형식으로 반환."""
    kst = timezone(timedelta(hours=9))
    return datetime.now(kst).isoformat()


class BaseEvaluator(ABC):
    """
    평가기 추상 베이스 클래스.

    모든 evaluator(vLLM, Chat API, API)가 구현해야 하는 공통 인터페이스를 정의합니다.
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """결과 저장 시 사용할 모델 ID."""
        pass

    @abstractmethod
    def evaluate(
        self,
        dataset_name: str,
        limit: int | None = None,
        save_details: bool = True,
    ) -> EvaluationResult:
        """
        데이터셋 평가 실행.

        Args:
            dataset_name: 평가할 데이터셋 이름
            limit: 평가할 샘플 수 제한 (None이면 전체)
            save_details: 상세 결과 저장 여부

        Returns:
            EvaluationResult: 평가 결과
        """
        pass
