"""
vLLM 기반 평가 엔진

모델 로딩, 추론, 점수 계산을 담당합니다.

평가 모드:
1. Loglikelihood (MCQA - 객관식 문제):
   - 각 선택지에 대해 log P(choice|context) 계산
   - lm-evaluation-harness와 동일한 방식
   - 예: context="질문: 한국의 수도는?\n정답:", choice="서울"
   - log P("서울"|context) 계산

2. Generation (생성형 문제):
   - 모델이 답변 텍스트를 생성
   - 생성된 답변과 정답을 비교 (exact match)
   - 주로 수학 문제에 사용

핵심 개념:
- prompt_logprobs: vLLM에서 입력 토큰들의 log probability를 반환
- TokensPrompt: 토큰 ID를 직접 전달하여 토큰화 불일치 방지
- Length normalization: 긴 선택지에 불리한 bias를 보정
"""
import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams, TokensPrompt

from .core import EvaluationResult, get_kst_timestamp
from .dataset_configs import DatasetLoader, EvalSample, get_available_datasets


class Evaluator:
    """
    vLLM 기반 모델 평가기.

    vLLM을 사용하여 대규모 언어 모델을 효율적으로 평가합니다.
    - Continuous batching으로 높은 처리량
    - PagedAttention으로 메모리 효율적인 KV cache 관리
    - Tensor Parallel로 여러 GPU에 모델 분산 가능

    사용 예시:
        evaluator = Evaluator(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            tensor_parallel_size=1,
        )
        result = evaluator.evaluate_dataset("kmmlu")
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
    ):
        """
        Evaluator 초기화 및 모델 로드.

        Args:
            model_name: HuggingFace 모델 이름 또는 로컬 경로
                예: "meta-llama/Llama-3.1-8B-Instruct"
            tensor_parallel_size: GPU 분산 수
                1이면 단일 GPU, 2 이상이면 여러 GPU에 모델 분산
                32B 모델을 80GB GPU 1장에서 돌리기 어려울 때 2장 이상 사용
            gpu_memory_utilization: GPU 메모리 사용률 (0.0 ~ 1.0)
                0.9면 GPU 메모리의 90%까지 사용
                KV cache 메모리 할당에 영향
            max_model_len: 최대 시퀀스 길이 제한
                None이면 모델 설정값 사용 (예: Qwen3-32B는 40960)
                큰 모델에서 KV cache 메모리 부족 시 줄여야 함
                예: 32B 모델에서 메모리 부족 시 16384로 제한
        """
        self.model_name = model_name
        self.model_id = model_name.split("/")[-1]  # "org/model" -> "model"

        print(f"모델 로딩 중: {model_name}")

        # vLLM LLM 엔진 생성
        # - 모델 가중치 다운로드 (첫 실행 시)
        # - GPU에 모델 로드
        # - KV cache 메모리 할당
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,  # 커스텀 모델 코드 허용
        )

        # Tokenizer 가져오기
        # chat_template 적용에 사용
        self.tokenizer = self.llm.get_tokenizer()
        print(f"모델 로드 완료")

    def _apply_chat_template(
        self,
        prompt: str,
        add_generation_prompt: bool = True,
    ) -> str:
        """
        모델의 chat template을 프롬프트에 적용.

        대부분의 Instruct 모델은 특정 형식의 입력을 기대합니다.
        예를 들어 Llama-3의 경우:
            <|begin_of_text|><|start_header_id|>user<|end_header_id|>
            {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        Args:
            prompt: 원본 프롬프트 텍스트
            add_generation_prompt: True면 assistant 응답 시작 부분 추가
                MCQA에서는 True로 설정하여 모델이 답변을 시작하도록 유도

        Returns:
            chat template이 적용된 프롬프트 문자열
        """
        # OpenAI 형식의 메시지 리스트로 변환
        messages = [{"role": "user", "content": prompt}]

        try:
            # tokenizer의 chat_template 사용
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,  # 문자열로 반환
                add_generation_prompt=add_generation_prompt,
            )
            return formatted
        except Exception:
            # chat_template이 없는 모델은 원본 그대로 반환
            return prompt

    def _compute_loglikelihood(
        self,
        context: str,
        continuation: str,
    ) -> tuple[float, int]:
        """
        조건부 확률 log P(continuation|context) 계산.

        lm-evaluation-harness의 loglikelihood 방식을 구현합니다.
        context 뒤에 continuation이 올 확률의 로그값을 계산합니다.

        예시:
            context = "질문: 한국의 수도는?\n정답:"
            continuation = "서울"
            -> log P("서울"|context) 계산

        Args:
            context: 조건이 되는 텍스트 (질문 + 프롬프트)
            continuation: 확률을 계산할 텍스트 (선택지)

        Returns:
            (logprob_sum, num_tokens): 로그 확률 합계와 토큰 수
            - logprob_sum: continuation의 각 토큰 log P의 합
            - num_tokens: continuation의 토큰 수 (길이 정규화에 사용)
        """
        # context와 continuation을 각각 토큰화
        # add_special_tokens=False: BOS/EOS 토큰 추가 안 함
        context_tokens = self.tokenizer.encode(context, add_special_tokens=False)
        continuation_tokens = self.tokenizer.encode(continuation, add_special_tokens=False)

        # 전체 입력 = context + continuation
        full_tokens = context_tokens + continuation_tokens
        full_text = self.tokenizer.decode(full_tokens)

        # vLLM 샘플링 파라미터 설정
        sampling_params = SamplingParams(
            temperature=0,       # greedy decoding
            max_tokens=1,        # 생성은 필요 없음, logprob만 필요
            prompt_logprobs=1,   # 입력 토큰들의 logprob 반환
        )

        # 추론 실행
        outputs = self.llm.generate([full_text], sampling_params)

        # prompt_logprobs에서 continuation 부분의 logprob 추출
        prompt_logprobs = outputs[0].prompt_logprobs

        if prompt_logprobs is None:
            return float("-inf"), len(continuation_tokens)

        # continuation 토큰들의 logprob 합산
        # context 부분은 건너뜀 (조건이므로)
        context_len = len(context_tokens)
        logprob_sum = 0.0

        for i, logprob_dict in enumerate(prompt_logprobs):
            if i < context_len:
                continue  # context 토큰은 건너뜀
            if logprob_dict is None:
                continue

            # 현재 위치의 토큰 ID
            token_id = full_tokens[i]

            # 해당 토큰의 logprob 가져오기
            if token_id in logprob_dict:
                logprob = logprob_dict[token_id]
                # vLLM은 Logprob 객체를 반환할 수 있음
                if hasattr(logprob, 'logprob'):
                    logprob = logprob.logprob
                logprob_sum += logprob

        return logprob_sum, len(continuation_tokens)

    def _evaluate_mcqa_loglikelihood(
        self,
        samples: list[EvalSample],
        save_details: bool = True,
    ) -> tuple[list[float], list[float], list[dict]]:
        """
        MCQA를 loglikelihood 방식으로 평가 (단일 샘플씩).

        각 샘플에 대해:
        1. 모든 선택지의 log P(choice|context) 계산
        2. 가장 높은 확률의 선택지를 예측으로 선택
        3. 정답과 비교하여 점수 계산

        Note: _evaluate_mcqa_batch가 더 효율적이므로 일반적으로 사용되지 않음

        Args:
            samples: 평가할 EvalSample 리스트
            save_details: True면 각 샘플별 상세 결과 저장

        Returns:
            (scores, scores_norm, details):
            - scores: 각 샘플의 정답 여부 (0.0 또는 1.0)
            - scores_norm: 길이 정규화된 예측 기준 정답 여부
            - details: 상세 결과 딕셔너리 리스트
        """
        scores = []
        scores_norm = []
        details = []

        for sample in tqdm(samples, desc="MCQA 평가 중"):
            # chat template 적용
            context = self._apply_chat_template(sample.prompt)

            # 각 선택지의 loglikelihood 계산
            choice_logprobs = []
            choice_lengths = []

            for choice in sample.choices:
                logprob, length = self._compute_loglikelihood(context, choice)
                choice_logprobs.append(logprob)
                choice_lengths.append(length)

            # 예측 결정
            logprobs_arr = np.array(choice_logprobs)
            lengths_arr = np.array(choice_lengths)

            # 일반 예측: 가장 높은 logprob
            pred = int(np.argmax(logprobs_arr))
            # 정규화 예측: logprob를 토큰 수로 나눈 값 (긴 선택지 bias 보정)
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

    def _evaluate_mcqa_batch(
        self,
        samples: list[EvalSample],
        save_details: bool = True,
    ) -> tuple[list[float], list[float], list[dict]]:
        """
        MCQA 배치 평가 (효율적인 방식).

        lm-evaluation-harness와 동일하게 TokensPrompt를 사용하여
        토큰 ID를 직접 vLLM에 전달합니다.

        배치 처리 흐름:
        1. 모든 (context, choice) 쌍을 토큰화
        2. vLLM에 배치로 전달하여 logprob 계산
        3. 결과를 샘플별로 그룹화하여 예측 결정

        Args:
            samples: 평가할 EvalSample 리스트
            save_details: True면 각 샘플별 상세 결과 저장

        Returns:
            (scores, scores_norm, details):
            - scores: 각 샘플의 정답 여부 (0.0 또는 1.0)
            - scores_norm: 길이 정규화된 예측 기준 정답 여부
            - details: 상세 결과 딕셔너리 리스트
        """
        # 1단계: 모든 요청을 토큰화
        # 각 샘플의 각 선택지에 대해 (context + choice) 토큰 준비
        tokenized_requests = []

        print("토큰화 중...")
        for sample_idx, sample in enumerate(tqdm(samples, desc="샘플")):
            # chat template 적용
            context = self._apply_chat_template(sample.prompt)
            context_tokens = self.tokenizer.encode(context, add_special_tokens=False)

            # 각 선택지에 대해 토큰화
            for choice_idx, choice in enumerate(sample.choices):
                continuation_tokens = self.tokenizer.encode(choice, add_special_tokens=False)
                full_tokens = context_tokens + continuation_tokens

                tokenized_requests.append({
                    "sample_idx": sample_idx,      # 원본 샘플 인덱스
                    "choice_idx": choice_idx,      # 선택지 인덱스
                    "context_len": len(context_tokens),      # context 토큰 수
                    "continuation_len": len(continuation_tokens),  # choice 토큰 수
                    "full_tokens": full_tokens,    # 전체 토큰 리스트
                })

        # 2단계: 배치 추론
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            prompt_logprobs=1,  # 입력 토큰의 logprob 반환
        )

        print("배치 추론 중...")

        # TokensPrompt 사용: 토큰 ID를 직접 전달
        # 이렇게 하면 vLLM이 다시 토큰화하지 않아 불일치 방지
        prompts = [
            TokensPrompt(prompt_token_ids=req["full_tokens"])
            for req in tokenized_requests
        ]
        outputs = self.llm.generate(prompts, sampling_params)

        # 3단계: 결과 정리
        # 샘플별로 logprob 그룹화
        sample_results = {i: {"logprobs": [], "lengths": []} for i in range(len(samples))}

        for req, output in zip(tokenized_requests, outputs):
            sample_idx = req["sample_idx"]
            context_len = req["context_len"]
            full_tokens = req["full_tokens"]

            prompt_logprobs = output.prompt_logprobs

            # continuation 부분의 logprob 합산
            logprob_sum = 0.0
            if prompt_logprobs:
                for i in range(context_len, len(full_tokens)):
                    if i >= len(prompt_logprobs) or prompt_logprobs[i] is None:
                        continue
                    token_id = full_tokens[i]
                    logprob_dict = prompt_logprobs[i]
                    if token_id in logprob_dict:
                        lp = logprob_dict[token_id]
                        if hasattr(lp, 'logprob'):
                            lp = lp.logprob
                        logprob_sum += lp

            sample_results[sample_idx]["logprobs"].append(logprob_sum)
            sample_results[sample_idx]["lengths"].append(req["continuation_len"])

        # 4단계: 점수 계산
        scores = []
        scores_norm = []
        details = []

        for sample_idx, sample in enumerate(samples):
            logprobs = np.array(sample_results[sample_idx]["logprobs"])
            lengths = np.array(sample_results[sample_idx]["lengths"])

            # 예측: argmax(logprob)
            pred = int(np.argmax(logprobs))
            # 정규화 예측: argmax(logprob / length)
            pred_norm = int(np.argmax(logprobs / lengths))

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
                    "logprobs": logprobs.tolist(),
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
        생성형 문제 평가 (수학 문제 등).

        모델이 답변을 생성하고, 생성된 답변과 정답을 비교합니다.
        주로 수학 문제에서 사용되며, $\boxed{답}$ 형식의 답변을 추출합니다.

        Args:
            samples: 평가할 EvalSample 리스트
            loader: 데이터셋 로더 (generation_kwargs, compute_score 메서드 사용)
            save_details: True면 각 샘플별 상세 결과 저장

        Returns:
            (scores, details):
            - scores: 각 샘플의 정답 여부 (0.0 또는 1.0)
            - details: 상세 결과 딕셔너리 리스트
        """
        # chat template 적용
        formatted_prompts = [
            self._apply_chat_template(sample.prompt)
            for sample in samples
        ]

        # 생성 파라미터 설정 (YAML에서 로드)
        gen_kwargs = loader.generation_kwargs
        sampling_params = SamplingParams(
            max_tokens=gen_kwargs.get("max_tokens", 2048),   # 최대 생성 토큰 수
            temperature=gen_kwargs.get("temperature", 0),    # 0 = greedy
            stop=gen_kwargs.get("stop", []),                 # 생성 중단 토큰
        )

        # 배치 생성
        print("생성 추론 중...")
        outputs = self.llm.generate(formatted_prompts, sampling_params)

        # 점수 계산
        scores = []
        details = []

        for sample, output in zip(samples, outputs):
            # 생성된 텍스트 추출
            prediction = output.outputs[0].text.strip()

            # 점수 계산 (exact match 또는 math equivalence)
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
        use_batch: bool = True,
    ) -> EvaluationResult:
        """
        특정 데이터셋에서 모델 평가.

        데이터셋의 output_type에 따라 적절한 평가 방식 선택:
        - multiple_choice: loglikelihood 기반 MCQA 평가
        - generate_until: 텍스트 생성 후 exact match 평가

        Args:
            dataset_name: 평가할 데이터셋 이름 (예: "kmmlu", "csatqa")
            save_details: True면 각 샘플별 상세 결과 저장
            use_batch: True면 배치 평가 사용 (더 효율적)

        Returns:
            EvaluationResult: 평가 결과 객체
        """
        # 데이터셋 로더 초기화
        loader = DatasetLoader(dataset_name)

        print(f"\n{'='*50}")
        print(f"평가 중: {dataset_name}")
        print(f"평가 유형: {loader.output_type}")
        print(f"{'='*50}")

        # 데이터셋 로드 및 포맷팅
        dataset = loader.load()
        samples = loader.format_all(dataset)
        print(f"{len(samples)}개 샘플 로드됨")

        # output_type에 따라 평가 방식 선택
        if loader.output_type == "multiple_choice":
            # MCQA: loglikelihood 기반 평가
            if use_batch:
                # 배치 평가 (효율적)
                scores, scores_norm, details = self._evaluate_mcqa_batch(
                    samples, save_details
                )
            else:
                # 단일 샘플 평가 (디버깅용)
                scores, scores_norm, details = self._evaluate_mcqa_loglikelihood(
                    samples, save_details
                )
            final_score = sum(scores) / len(scores)
            final_score_norm = sum(scores_norm) / len(scores_norm)
        else:  # generate_until
            # Generation: 텍스트 생성 평가
            scores, details = self._evaluate_generation(
                samples, loader, save_details
            )
            final_score = sum(scores) / len(scores)
            final_score_norm = None  # Generation에는 정규화 점수 없음

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
            datasets: 평가할 데이터셋 이름 리스트
                None이면 모든 사용 가능한 데이터셋 평가
            save_details: True면 각 샘플별 상세 결과 저장

        Returns:
            dict[str, EvaluationResult]: 데이터셋 이름 -> 평가 결과 매핑
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


def save_results(
    results: dict[str, EvaluationResult],
    output_dir: str | Path,
    model_id: str,
) -> Path:
    """
    평가 결과를 JSON 파일로 저장.
    기존 결과가 있으면 병합합니다.

    저장 구조:
    - {output_dir}/{model_id}/results.json: 전체 요약 결과
    - {output_dir}/{model_id}/{dataset}_details.json: 각 데이터셋별 상세 결과

    Args:
        results: 데이터셋 이름 -> 평가 결과 매핑
        output_dir: 저장할 디렉토리 경로
        model_id: 모델 ID (디렉토리명에 사용)

    Returns:
        Path: 저장된 결과 파일 경로
    """
    output_dir = Path(output_dir)
    model_dir = output_dir / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    # 요약 결과 저장
    results_file = model_dir / "results.json"

    # 기존 결과 로드 (있으면)
    existing_data = {}
    if results_file.exists():
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing_data = {}

    # 새 결과 생성
    new_results_data = {
        name: {
            "dataset_name": r.dataset_name,
            "model_name": r.model_name,
            "num_samples": r.num_samples,
            "score": r.score,
            "score_norm": r.score_norm,
            "metric": r.metric,
            "timestamp": r.timestamp,
        }
        for name, r in results.items()
    }

    # 기존 결과와 병합 (새 결과가 우선)
    merged_data = {**existing_data, **new_results_data}

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    # 상세 결과 저장 (데이터셋별)
    for name, result in results.items():
        if result.details:
            detail_file = model_dir / f"{name}_details.json"
            with open(detail_file, "w", encoding="utf-8") as f:
                json.dump(result.details, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장됨: {results_file}")
    return results_file


def parse_leaderboard(leaderboard_file: Path) -> dict[str, dict]:
    """
    기존 결과를 JSON 파일들에서 파싱합니다.
    각 모델 디렉토리의 results.json 또는 *_details.json 파일을 읽어서 데이터를 가져옵니다.

    디렉토리 구조:
    - results/{model_id}/results.json (우선)
    - results/{model_id}/*_details.json (fallback)

    Args:
        leaderboard_file: leaderboard.md 파일 경로 (디렉토리 확인용)

    Returns:
        dict: 모델별 데이터셋 점수 딕셔너리
              {model_id: {dataset_name: {"score": float}, ...}, ...}
    """
    results_dir = leaderboard_file.parent
    existing_results = {}

    # 각 모델 디렉토리에서 결과 로드
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_id = model_dir.name
        existing_results[model_id] = {}

        # 1. results.json에서 로드 시도
        results_file = model_dir / "results.json"
        if results_file.exists():
            try:
                with open(results_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for ds_name, ds_data in data.items():
                    if isinstance(ds_data, dict) and "score" in ds_data:
                        existing_results[model_id][ds_name] = {"score": ds_data["score"]}
            except (json.JSONDecodeError, IOError):
                pass

        # 2. *_details.json에서 추가 로드 (results.json에 없는 데이터셋)
        for details_file in model_dir.glob("*_details.json"):
            ds_name = details_file.stem.replace("_details", "")

            # 이미 results.json에서 로드한 경우 스킵
            if ds_name in existing_results[model_id]:
                continue

            try:
                with open(details_file, "r", encoding="utf-8") as f:
                    details = json.load(f)

                if isinstance(details, list) and len(details) > 0:
                    # details는 각 샘플의 결과 리스트, score 평균 계산
                    scores = [item.get("score", 0) for item in details if "score" in item]
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        existing_results[model_id][ds_name] = {"score": avg_score}
            except (json.JSONDecodeError, IOError):
                continue

        # 빈 결과 제거
        if not existing_results[model_id]:
            del existing_results[model_id]

    return existing_results


def update_leaderboard(
    results: dict[str, EvaluationResult],
    output_dir: str | Path,
    existing_results: dict[str, dict] | None = None,
    model_id: str | None = None,
) -> Path:
    """
    리더보드 마크다운 파일 업데이트.
    데이터셋별로 표를 분리하고, 각 표 내에서 점수 내림차순으로 정렬합니다.

    Args:
        results: 새로운 평가 결과
        output_dir: 저장할 디렉토리 경로
        existing_results: 기존 결과 (누적 저장 시 사용)
        model_id: 모델 ID (None이면 model_name에서 추출)

    Returns:
        Path: 저장된 리더보드 파일 경로
    """
    output_dir = Path(output_dir)
    leaderboard_file = output_dir / "leaderboard.md"

    # 기존 결과가 있으면 병합
    all_results = existing_results.copy() if existing_results else {}

    # 새 결과 추가
    if results:
        if model_id is None:
            model_name = list(results.values())[0].model_name
            model_id = model_name.split("/")[-1]

        if model_id not in all_results:
            all_results[model_id] = {}

        for name, r in results.items():
            all_results[model_id][name] = {
                "score": r.score,
                "score_norm": r.score_norm,
            }

    # 마크다운 생성
    all_datasets = get_available_datasets()

    # 통합 데이터셋 그룹 정의
    dataset_groups = {
        "HRM8K": {
            "members": ["hrm8k", "hrm8k_mmmlu"],
            "weight_by_size": False,
        },
        "Ko-MuSR": {
            "members": ["ko_musr_mm", "ko_musr_op", "ko_musr_ta"],
            "weight_by_size": True,
        },
    }

    # 각 데이터셋의 샘플 수 (가중 평균 계산용)
    dataset_sample_counts = {
        "hrm8k": 7541,
        "hrm8k_mmmlu": 470,
        "ko_musr_mm": 250,
        "ko_musr_op": 256,
        "ko_musr_ta": 250,
    }

    # 그룹에 속한 세부 데이터셋들
    grouped_datasets = set()
    for group_info in dataset_groups.values():
        grouped_datasets.update(group_info["members"])

    # 리더보드에 표시할 데이터셋: 개별 + 통합 그룹
    display_datasets = []
    for ds in all_datasets:
        if ds not in grouped_datasets:
            display_datasets.append(ds)
    display_datasets.extend(dataset_groups.keys())

    # 데이터셋 설명 매핑
    dataset_descriptions = {
        "click": "한국 문화 지식",
        "csatqa": "수능 문제",
        "haerae": "한국 문화/사회 상식",
        "kmmlu": "한국어 MMLU",
        "kmmlu_pro": "KMMLU 고급 버전",
        "kobalt": "한국어 이해력 테스트",
        "kormedmcqa": "한국 의료 면허시험",
        "HRM8K": "한국어 수학 (hrm8k + hrm8k_mmmlu 평균)",
        "Ko-MuSR": "한국어 다단계 추론 (MM + OP + TA 가중 평균)",
    }

    # 마크다운 시작
    md_lines = [
        "# LLM 평가 리더보드",
        "",
        f"최종 업데이트: {get_kst_timestamp()[:19].replace('T', ' ')} KST",
        "",
        "## 데이터셋 설명",
        "",
    ]

    for ds in display_datasets:
        desc = dataset_descriptions.get(ds, ds)
        md_lines.append(f"- **{ds}**: {desc}")

    md_lines.append("")

    # 각 데이터셋별 표 생성
    for dataset in display_datasets:
        md_lines.append(f"### {dataset}")
        md_lines.append("")
        md_lines.append("| 모델 | 점수 |")
        md_lines.append("| :--- | ---: |")

        # 해당 데이터셋에 대한 모델별 점수 수집
        model_scores = []

        for model_id, model_results in all_results.items():
            if dataset in dataset_groups:
                # 통합 그룹: 멤버들의 평균 계산
                group_info = dataset_groups[dataset]
                group_members = group_info["members"]
                weight_by_size = group_info["weight_by_size"]

                group_scores = []
                group_sizes = []
                for member in group_members:
                    if member in model_results:
                        score = model_results[member]
                        if isinstance(score, dict):
                            score = score.get("score", 0)
                        group_scores.append(score)
                        group_sizes.append(dataset_sample_counts.get(member, 1))

                if group_scores:
                    if weight_by_size:
                        weighted_sum = sum(s * n for s, n in zip(group_scores, group_sizes))
                        avg_score = weighted_sum / sum(group_sizes)
                    else:
                        avg_score = sum(group_scores) / len(group_scores)
                    model_scores.append((model_id, avg_score))
            else:
                # 개별 데이터셋
                if dataset in model_results:
                    score = model_results[dataset]
                    if isinstance(score, dict):
                        score = score.get("score", 0)
                    model_scores.append((model_id, score))

        # 점수 내림차순 정렬
        model_scores.sort(key=lambda x: x[1], reverse=True)

        # 표에 행 추가
        for model_id, score in model_scores:
            md_lines.append(f"| {model_id} | {score:.4f} |")

        if not model_scores:
            md_lines.append("| - | - |")

        md_lines.append("")

    # 전체 평균 표
    md_lines.append("### 전체 평균")
    md_lines.append("")
    md_lines.append("| 모델 | 평균 |")
    md_lines.append("| :--- | ---: |")

    model_averages = []
    for model_id, model_results in all_results.items():
        scores = []

        for dataset in display_datasets:
            if dataset in dataset_groups:
                group_info = dataset_groups[dataset]
                group_members = group_info["members"]
                weight_by_size = group_info["weight_by_size"]

                group_scores = []
                group_sizes = []
                for member in group_members:
                    if member in model_results:
                        score = model_results[member]
                        if isinstance(score, dict):
                            score = score.get("score", 0)
                        group_scores.append(score)
                        group_sizes.append(dataset_sample_counts.get(member, 1))

                if group_scores:
                    if weight_by_size:
                        weighted_sum = sum(s * n for s, n in zip(group_scores, group_sizes))
                        avg_score = weighted_sum / sum(group_sizes)
                    else:
                        avg_score = sum(group_scores) / len(group_scores)
                    scores.append(avg_score)
            else:
                if dataset in model_results:
                    score = model_results[dataset]
                    if isinstance(score, dict):
                        score = score.get("score", 0)
                    scores.append(score)

        if scores:
            avg = sum(scores) / len(scores)
            model_averages.append((model_id, avg))

    # 평균 내림차순 정렬
    model_averages.sort(key=lambda x: x[1], reverse=True)

    for model_id, avg in model_averages:
        md_lines.append(f"| {model_id} | {avg:.4f} |")

    if not model_averages:
        md_lines.append("| - | - |")

    md_lines.append("")

    with open(leaderboard_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"리더보드 업데이트됨: {leaderboard_file}")
    return leaderboard_file