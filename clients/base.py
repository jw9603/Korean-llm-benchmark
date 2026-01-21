"""
LLM 클라이언트 기본 인터페이스

모든 LLM 클라이언트가 구현해야 하는 추상 인터페이스를 정의합니다.
lm-evaluation-harness의 LM 클래스와 유사한 구조입니다.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LoglikelihoodResult:
    """
    Loglikelihood 계산 결과.

    Attributes:
        logprob: continuation의 log probability 합계
        num_tokens: continuation의 토큰 수 (길이 정규화용)
        is_greedy: greedy decoding 시 동일한 결과가 나오는지 여부
    """
    logprob: float
    num_tokens: int
    is_greedy: bool = True


class BaseClient(ABC):
    """
    LLM API 클라이언트 기본 클래스.

    lm-evaluation-harness의 LM 클래스와 유사한 인터페이스를 제공합니다.
    """

    @abstractmethod
    def loglikelihood(
        self,
        context: str,
        continuation: str,
    ) -> LoglikelihoodResult:
        """
        조건부 확률 log P(continuation|context) 계산.

        Args:
            context: 조건이 되는 텍스트 (질문 + 프롬프트)
            continuation: 확률을 계산할 텍스트 (선택지)

        Returns:
            LoglikelihoodResult: logprob, 토큰 수, is_greedy 포함
        """
        pass

    @abstractmethod
    def loglikelihood_batch(
        self,
        requests: list[tuple[str, str]],
    ) -> list[LoglikelihoodResult]:
        """
        여러 (context, continuation) 쌍의 loglikelihood 배치 계산.

        Args:
            requests: (context, continuation) 튜플 리스트

        Returns:
            LoglikelihoodResult 리스트
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0,
        stop: list[str] | None = None,
    ) -> str:
        """
        텍스트 생성.

        Args:
            prompt: 입력 프롬프트
            max_tokens: 최대 생성 토큰 수
            temperature: 샘플링 온도 (0 = greedy)
            stop: 생성 중단 문자열 리스트

        Returns:
            생성된 텍스트
        """
        pass

    @abstractmethod
    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 2048,
        temperature: float = 0,
        stop: list[str] | None = None,
    ) -> list[str]:
        """
        여러 프롬프트에 대해 텍스트 배치 생성.

        Args:
            prompts: 입력 프롬프트 리스트
            max_tokens: 최대 생성 토큰 수
            temperature: 샘플링 온도 (0 = greedy)
            stop: 생성 중단 문자열 리스트

        Returns:
            생성된 텍스트 리스트
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """모델 이름 반환"""
        pass
