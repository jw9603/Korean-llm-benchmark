"""
OpenAI API 클라이언트

lm-evaluation-harness 표준 방식(loglikelihood)으로 MCQA 평가를 지원합니다.

지원 API:
1. Completions API (babbage-002, davinci-002)
   - echo=True, logprobs로 prompt logprobs 계산 가능
   - lm-evaluation-harness 표준 방식과 동일

2. Chat Completions API (gpt-4, gpt-4o, etc.)
   - prompt logprobs 미지원
   - generate_until 방식만 사용 가능

3. OpenAI-compatible API (vLLM, SGLang 등)
   - base_url 지정하여 로컬 서버 사용
   - Completions API 엔드포인트 필요
"""
import asyncio
import os
import time
from typing import Any

from openai import AsyncOpenAI, OpenAI

from .base import BaseClient, LoglikelihoodResult


class OpenAIClient(BaseClient):
    """
    OpenAI Completions API 기반 클라이언트.

    lm-evaluation-harness의 loglikelihood 방식을 구현합니다.
    - echo=True: 입력 프롬프트도 응답에 포함
    - logprobs: 각 토큰의 log probability 반환

    주의: OpenAI 공식 API는 babbage-002, davinci-002만 Completions API 지원
    gpt-4 등 Chat 모델은 Chat Completions API만 지원 (loglikelihood 불가)

    vLLM, SGLang 등 OpenAI-compatible 서버는 Completions API 지원
    """

    # 지원 모델 목록 (Completions API)
    SUPPORTED_COMPLETIONS_MODELS = ["babbage-002", "davinci-002"]

    def __init__(
        self,
        model: str = "davinci-002",
        base_url: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        OpenAI 클라이언트 초기화.

        Args:
            model: 모델 이름 (예: "davinci-002", "meta-llama/Llama-3.1-8B")
            base_url: API 베이스 URL (None이면 OpenAI 공식 API)
                예: "http://localhost:8000/v1" (vLLM 서버)
            api_key: API 키 (None이면 환경변수 OPENAI_API_KEY 사용)
            max_retries: 실패 시 재시도 횟수
            retry_delay: 재시도 간 대기 시간 (초)
        """
        self._model = model
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # API 키 설정
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "API 키가 필요합니다. OPENAI_API_KEY 환경변수를 설정하거나 "
                    "api_key 인자를 전달하세요."
                )

        # OpenAI 클라이언트 생성
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        # 공식 OpenAI API 사용 시 모델 확인
        if base_url is None and model not in self.SUPPORTED_COMPLETIONS_MODELS:
            print(f"경고: {model}은 OpenAI Completions API를 지원하지 않습니다.")
            print(f"지원 모델: {self.SUPPORTED_COMPLETIONS_MODELS}")
            print("Chat 모델(gpt-4 등)은 loglikelihood를 지원하지 않습니다.")

        print(f"OpenAI 클라이언트 초기화 완료: {model}")
        if base_url:
            print(f"  Base URL: {base_url}")

    @property
    def model_name(self) -> str:
        return self._model

    def _call_completions_api(
        self,
        prompt: str,
        max_tokens: int = 1,
        temperature: float = 0,
        logprobs: int | None = 1,
        echo: bool = True,
        stop: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Completions API 호출 (재시도 로직 포함).

        Args:
            prompt: 입력 프롬프트
            max_tokens: 최대 생성 토큰 수
            temperature: 샘플링 온도
            logprobs: 반환할 상위 logprobs 수 (None이면 비활성화)
            echo: True면 프롬프트도 응답에 포함
            stop: 생성 중단 문자열

        Returns:
            API 응답 딕셔너리
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.completions.create(
                    model=self._model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    logprobs=logprobs,
                    echo=echo,
                    stop=stop,
                )
                return response.model_dump()

            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"API 호출 실패 (시도 {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise

    def loglikelihood(
        self,
        context: str,
        continuation: str,
    ) -> LoglikelihoodResult:
        """
        조건부 확률 log P(continuation|context) 계산.

        OpenAI Completions API의 echo=True와 logprobs를 사용하여
        lm-evaluation-harness와 동일한 방식으로 계산합니다.

        Args:
            context: 조건이 되는 텍스트
            continuation: 확률을 계산할 텍스트

        Returns:
            LoglikelihoodResult: logprob 합계와 토큰 수
        """
        # 전체 텍스트 = context + continuation
        full_text = context + continuation

        # API 호출
        response = self._call_completions_api(
            prompt=full_text,
            max_tokens=0,  # 생성 없이 logprobs만 필요
            temperature=0,
            logprobs=1,
            echo=True,  # 프롬프트 토큰들의 logprobs 반환
        )

        # 응답 파싱
        choice = response["choices"][0]
        token_logprobs = choice["logprobs"]["token_logprobs"]
        tokens = choice["logprobs"]["tokens"]
        top_logprobs = choice["logprobs"]["top_logprobs"]

        # context의 토큰 수 계산
        # context만 따로 tokenize하여 길이 확인
        context_response = self._call_completions_api(
            prompt=context,
            max_tokens=0,
            logprobs=1,
            echo=True,
        )
        context_tokens = context_response["choices"][0]["logprobs"]["tokens"]
        context_len = len(context_tokens)

        # continuation 부분의 logprob 합산
        logprob_sum = 0.0
        is_greedy = True
        num_tokens = 0

        for i in range(context_len, len(tokens)):
            lp = token_logprobs[i]
            if lp is not None:
                logprob_sum += lp
                num_tokens += 1

                # greedy 여부 확인
                if top_logprobs and i < len(top_logprobs):
                    top = top_logprobs[i]
                    if top:
                        max_logprob = max(top.values())
                        if lp < max_logprob - 1e-6:
                            is_greedy = False

        return LoglikelihoodResult(
            logprob=logprob_sum,
            num_tokens=num_tokens,
            is_greedy=is_greedy,
        )

    def loglikelihood_batch(
        self,
        requests: list[tuple[str, str]],
    ) -> list[LoglikelihoodResult]:
        """
        여러 (context, continuation) 쌍의 loglikelihood 배치 계산.

        OpenAI API는 진정한 배치를 지원하지 않으므로
        순차적으로 처리합니다.

        Args:
            requests: (context, continuation) 튜플 리스트

        Returns:
            LoglikelihoodResult 리스트
        """
        results = []
        for context, continuation in requests:
            result = self.loglikelihood(context, continuation)
            results.append(result)
        return results

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
            temperature: 샘플링 온도
            stop: 생성 중단 문자열

        Returns:
            생성된 텍스트
        """
        # OpenAI API는 stop 시퀀스를 최대 4개까지만 허용
        if stop and len(stop) > 4:
            stop = stop[:4]

        response = self._call_completions_api(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=None,
            echo=False,
            stop=stop,
        )
        return response["choices"][0]["text"]

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
            temperature: 샘플링 온도
            stop: 생성 중단 문자열

        Returns:
            생성된 텍스트 리스트
        """
        results = []
        for prompt in prompts:
            result = self.generate(prompt, max_tokens, temperature, stop)
            results.append(result)
        return results


class OpenAIChatClient(BaseClient):
    """
    OpenAI Chat Completions API 기반 클라이언트.

    주의: Chat API는 prompt logprobs를 지원하지 않습니다.
    따라서 loglikelihood 메서드는 NotImplementedError를 발생시킵니다.
    generate_until 방식의 평가만 가능합니다.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        reasoning_effort: str | None = None,
    ):
        """
        OpenAI Chat 클라이언트 초기화.

        Args:
            model: 모델 이름 (예: "gpt-4o", "gpt-4-turbo")
            base_url: API 베이스 URL
            api_key: API 키
            max_retries: 실패 시 재시도 횟수
            retry_delay: 재시도 간 대기 시간
            reasoning_effort: GPT-5/o1/o3 reasoning effort (low, medium, high)
        """
        self._model = model
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.reasoning_effort = reasoning_effort

        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("API 키가 필요합니다.")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        effort_str = f" (reasoning: {reasoning_effort})" if reasoning_effort else ""
        print(f"OpenAI Chat 클라이언트 초기화 완료: {model}{effort_str}")

    @property
    def model_name(self) -> str:
        return self._model

    def loglikelihood(
        self,
        context: str,
        continuation: str,
    ) -> LoglikelihoodResult:
        """
        Chat API는 loglikelihood를 지원하지 않습니다.
        """
        raise NotImplementedError(
            "OpenAI Chat Completions API는 prompt logprobs를 지원하지 않습니다. "
            "MCQA 평가를 위해서는 Completions API(babbage-002, davinci-002) 또는 "
            "OpenAI-compatible 서버(vLLM 등)를 사용하세요."
        )

    def loglikelihood_batch(
        self,
        requests: list[tuple[str, str]],
    ) -> list[LoglikelihoodResult]:
        raise NotImplementedError(
            "OpenAI Chat Completions API는 prompt logprobs를 지원하지 않습니다."
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0,
        stop: list[str] | None = None,
    ) -> str:
        """
        Chat API로 텍스트 생성.
        """
        messages = [{"role": "user", "content": prompt}]

        # OpenAI API는 stop 시퀀스를 최대 4개까지만 허용
        if stop and len(stop) > 4:
            stop = stop[:4]

        # K-EXAONE 모델은 temperature 파라미터를 서버에서 고정하므로 제외
        is_k_exaone = "k-exaone" in self._model.lower()
        # 새 OpenAI 모델(gpt-5, o1, o3 등)은 max_completion_tokens 사용
        use_max_completion_tokens = any(x in self._model.lower() for x in ["gpt-5", "o1", "o3"])

        for attempt in range(self.max_retries):
            try:
                if is_k_exaone:
                    response = self.client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        max_tokens=max_tokens,
                        stop=stop,
                    )
                elif use_max_completion_tokens:
                    # GPT-5, o1, o3: reasoning_effort 지원, temperature 미지원 (기본값 1만 허용)
                    kwargs = {
                        "model": self._model,
                        "messages": messages,
                        "max_completion_tokens": max_tokens,
                    }
                    if self.reasoning_effort:
                        kwargs["reasoning_effort"] = self.reasoning_effort
                    # GPT-5/o1/o3는 stop, temperature 파라미터 미지원
                    response = self.client.chat.completions.create(**kwargs)
                else:
                    response = self.client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop=stop,
                    )
                message = response.choices[0].message
                # reasoning 모델 (gpt-oss 등)은 content가 비어있고 reasoning 필드에 응답
                content = message.content or ""
                if not content and hasattr(message, "reasoning") and message.reasoning:
                    content = message.reasoning
                # dict 형태인 경우도 처리 (Ollama 등)
                if not content and isinstance(message, dict):
                    content = message.get("content") or message.get("reasoning") or ""
                return content

            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"API 호출 실패 (시도 {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 2048,
        temperature: float = 0,
        stop: list[str] | None = None,
    ) -> list[str]:
        results = []
        for prompt in prompts:
            result = self.generate(prompt, max_tokens, temperature, stop)
            results.append(result)
        return results

    async def generate_async(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0,
        stop: list[str] | None = None,
    ) -> str:
        """
        비동기 Chat API로 텍스트 생성.
        """
        messages = [{"role": "user", "content": prompt}]

        if stop and len(stop) > 4:
            stop = stop[:4]

        is_k_exaone = "k-exaone" in self._model.lower()
        use_max_completion_tokens = any(x in self._model.lower() for x in ["gpt-5", "o1", "o3"])

        for attempt in range(self.max_retries):
            try:
                if is_k_exaone:
                    response = await self.async_client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        max_tokens=max_tokens,
                        stop=stop,
                    )
                elif use_max_completion_tokens:
                    # GPT-5, o1, o3: temperature 미지원 (기본값 1만 허용)
                    kwargs = {
                        "model": self._model,
                        "messages": messages,
                        "max_completion_tokens": max_tokens,
                    }
                    if self.reasoning_effort:
                        kwargs["reasoning_effort"] = self.reasoning_effort
                    response = await self.async_client.chat.completions.create(**kwargs)
                else:
                    response = await self.async_client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop=stop,
                    )
                message = response.choices[0].message
                # reasoning 모델 (gpt-oss 등)은 content가 비어있고 reasoning 필드에 응답
                content = message.content or ""
                if not content and hasattr(message, "reasoning") and message.reasoning:
                    content = message.reasoning
                # dict 형태인 경우도 처리 (Ollama 등)
                if not content and isinstance(message, dict):
                    content = message.get("content") or message.get("reasoning") or ""
                return content

            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"API 호출 실패 (시도 {attempt + 1}/{self.max_retries}): {e}")
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise

    async def generate_batch_async(
        self,
        prompts: list[str],
        max_tokens: int = 2048,
        temperature: float = 0,
        stop: list[str] | None = None,
        max_concurrency: int = 10,
    ) -> list[str]:
        """
        비동기 병렬 배치 생성.

        Args:
            prompts: 프롬프트 리스트
            max_tokens: 최대 토큰 수
            temperature: 샘플링 온도
            stop: 중단 문자열
            max_concurrency: 최대 동시 요청 수

        Returns:
            생성된 텍스트 리스트 (순서 보장)
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def limited_generate(idx: int, prompt: str) -> tuple[int, str]:
            async with semaphore:
                result = await self.generate_async(prompt, max_tokens, temperature, stop)
                return idx, result

        tasks = [limited_generate(i, p) for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks)

        # 순서대로 정렬
        sorted_results = sorted(results, key=lambda x: x[0])
        return [r[1] for r in sorted_results]
