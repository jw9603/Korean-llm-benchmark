"""
OpenAI API 기반 평가 엔진

OpenAI Completions API 또는 OpenAI-compatible API를 사용하여
lm-evaluation-harness 표준 방식(loglikelihood)으로 모델을 평가합니다.

사용 예시:
    # OpenAI 공식 API (davinci-002)
    evaluator = APIEvaluator(
        model="davinci-002",
    )

    # vLLM 서버 (OpenAI-compatible)
    evaluator = APIEvaluator(
        model="meta-llama/Llama-3.1-8B-Instruct",
        base_url="http://localhost:8000/v1",
    )

    result = evaluator.evaluate_dataset("kmmlu")
"""
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .clients import OpenAIClient
from .clients.base import BaseClient
from .core import EvaluationResult, get_kst_timestamp
from .dataset_configs import DatasetLoader, EvalSample, get_available_datasets


class APIEvaluator:
    """
    OpenAI API 기반 모델 평가기.

    OpenAI Completions API를 사용하여 lm-evaluation-harness와
    동일한 loglikelihood 방식으로 MCQA를 평가합니다.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        client: BaseClient | None = None,
    ):
        """
        APIEvaluator 초기화.

        Args:
            model: 모델 이름
            base_url: API 베이스 URL (None이면 OpenAI 공식 API)
            api_key: API 키
            client: 커스텀 클라이언트 (None이면 OpenAIClient 생성)
        """
        self.model_name = model
        self.model_id = model.split("/")[-1]

        # 클라이언트 초기화
        if client is not None:
            self.client = client
        else:
            self.client = OpenAIClient(
                model=model,
                base_url=base_url,
                api_key=api_key,
            )

    def _evaluate_mcqa(
        self,
        samples: list[EvalSample],
        save_details: bool = True,
    ) -> tuple[list[float], list[float], list[dict]]:
        """
        MCQA를 loglikelihood 방식으로 평가.

        각 샘플에 대해:
        1. 모든 선택지의 log P(choice|context) 계산
        2. 가장 높은 확률의 선택지를 예측으로 선택
        3. 정답과 비교하여 점수 계산

        Args:
            samples: 평가할 EvalSample 리스트
            save_details: 상세 결과 저장 여부

        Returns:
            (scores, scores_norm, details)
        """
        scores = []
        scores_norm = []
        details = []

        for sample in tqdm(samples, desc="MCQA 평가 중"):
            context = sample.prompt

            # 각 선택지의 loglikelihood 계산
            choice_logprobs = []
            choice_lengths = []

            for choice in sample.choices:
                result = self.client.loglikelihood(context, choice)
                choice_logprobs.append(result.logprob)
                choice_lengths.append(result.num_tokens)

            # 예측 결정
            logprobs_arr = np.array(choice_logprobs)
            lengths_arr = np.array(choice_lengths)

            # 일반 예측: 가장 높은 logprob
            pred = int(np.argmax(logprobs_arr))
            # 정규화 예측: logprob / 토큰 수
            pred_norm = int(np.argmax(logprobs_arr / lengths_arr))

            # 점수 계산
            score = 1.0 if pred == sample.gold else 0.0
            score_norm = 1.0 if pred_norm == sample.gold else 0.0

            scores.append(score)
            scores_norm.append(score_norm)

            if save_details:
                details.append({
                    "prompt": sample.prompt,
                    "choices": sample.choices,
                    "gold": sample.gold,
                    "pred": pred,
                    "pred_norm": pred_norm,
                    "logprobs": choice_logprobs,
                    "score": score,
                    "score_norm": score_norm,
                })

        return scores, scores_norm, details

    def _evaluate_generation(
        self,
        samples: list[EvalSample],
        loader: DatasetLoader,
        save_details: bool = True,
    ) -> tuple[list[float], list[dict]]:
        """
        생성형 문제 평가.

        Args:
            samples: 평가할 EvalSample 리스트
            loader: 데이터셋 로더
            save_details: 상세 결과 저장 여부

        Returns:
            (scores, details)
        """
        gen_kwargs = loader.generation_kwargs
        max_tokens = gen_kwargs.get("max_tokens", 2048)
        temperature = gen_kwargs.get("temperature", 0)
        stop = gen_kwargs.get("stop", [])

        scores = []
        details = []

        for sample in tqdm(samples, desc="생성 평가 중"):
            # 텍스트 생성
            prediction = self.client.generate(
                prompt=sample.prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )
            prediction = prediction.strip()

            # 점수 계산
            score = loader.compute_score(prediction, sample)
            scores.append(score)

            if save_details:
                details.append({
                    "prompt": sample.prompt,
                    "prediction": prediction,
                    "gold": sample.gold,
                    "score": score,
                })

        return scores, details

    def evaluate_dataset(
        self,
        dataset_name: str,
        save_details: bool = True,
    ) -> EvaluationResult:
        """
        특정 데이터셋에서 모델 평가.

        Args:
            dataset_name: 평가할 데이터셋 이름
            save_details: 상세 결과 저장 여부

        Returns:
            EvaluationResult
        """
        # 데이터셋 로더 초기화
        loader = DatasetLoader(dataset_name)

        print(f"\n{'='*50}")
        print(f"평가 중: {dataset_name}")
        print(f"평가 유형: {loader.output_type}")
        print(f"모델: {self.model_name}")
        print(f"{'='*50}")

        # 데이터셋 로드
        dataset = loader.load()
        samples = loader.format_all(dataset)
        print(f"{len(samples)}개 샘플 로드됨")

        # output_type에 따라 평가
        if loader.output_type == "multiple_choice":
            scores, scores_norm, details = self._evaluate_mcqa(
                samples, save_details
            )
            final_score = sum(scores) / len(scores)
            final_score_norm = sum(scores_norm) / len(scores_norm)
        else:  # generate_until
            scores, details = self._evaluate_generation(
                samples, loader, save_details
            )
            final_score = sum(scores) / len(scores)
            final_score_norm = None

        # 결과 출력
        print(f"\n{dataset_name} 결과:")
        print(f"  정확도: {final_score:.4f}")
        if final_score_norm is not None:
            print(f"  정규화 정확도: {final_score_norm:.4f}")

        return EvaluationResult(
            dataset_name=dataset_name,
            model_name=self.model_name,
            num_samples=len(samples),
            score=final_score,
            score_norm=final_score_norm,
            metric=loader.config.metric,
            timestamp=get_kst_timestamp(),
            details=details if save_details else [],
        )

    def evaluate_all(
        self,
        datasets: list[str] | None = None,
        save_details: bool = True,
    ) -> dict[str, EvaluationResult]:
        """
        여러 데이터셋에서 모델 평가.

        Args:
            datasets: 평가할 데이터셋 이름 리스트 (None이면 전체)
            save_details: 상세 결과 저장 여부

        Returns:
            dict[str, EvaluationResult]
        """
        if datasets is None:
            datasets = get_available_datasets()

        results = {}
        for dataset_name in datasets:
            try:
                result = self.evaluate_dataset(dataset_name, save_details)
                results[dataset_name] = result
            except Exception as e:
                print(f"{dataset_name} 평가 중 에러: {e}")
                import traceback
                traceback.print_exc()
                continue

        return results
