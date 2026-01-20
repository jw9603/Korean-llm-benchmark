"""
OpenAI Chat API 기반 평가 엔진

GPT-4o 등 Chat Completions API만 지원하는 모델용입니다.
prompt logprobs를 지원하지 않으므로, MCQA를 생성 방식으로 평가합니다.

MCQA 평가 방식:
- 모델에게 "정답만 출력하세요: A, B, C, D 중 하나" 형태로 요청
- 생성된 답변에서 A/B/C/D를 추출하여 정답과 비교

사용 예시:
    evaluator = ChatEvaluator(model="gpt-4o")
    result = evaluator.evaluate_dataset("kmmlu")

    # 병렬 처리 (asyncio)
    result = evaluator.evaluate_dataset("kmmlu", max_concurrency=10)
"""
import asyncio
import re

from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from .clients.openai_client import OpenAIChatClient
from .core import EvaluationResult, get_kst_timestamp
from .dataset_configs import DatasetLoader, EvalSample, get_available_datasets


class ChatEvaluator:
    """
    OpenAI Chat API 기반 모델 평가기.

    GPT-4, GPT-4o 등 Chat 모델용입니다.
    MCQA를 생성 방식으로 평가합니다 (loglikelihood 불가).
    """

    # 정답 추출용 정규식
    ANSWER_PATTERN = re.compile(r"^\s*([A-D])\s*[.:]?\s*", re.IGNORECASE)
    ANSWER_PATTERN_ANYWHERE = re.compile(r"\b([A-D])\b", re.IGNORECASE)

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
        default_max_tokens: int | None = None,
    ):
        """
        ChatEvaluator 초기화.

        Args:
            model: Chat 모델 이름 (예: "gpt-4o", "gpt-4-turbo")
            base_url: API 베이스 URL
            api_key: API 키
            reasoning_effort: GPT-5/o1/o3 reasoning effort (low, medium, high)
            default_max_tokens: 기본 최대 토큰 수 (None이면 데이터셋 설정 사용)
        """
        self.model_name = model
        self.reasoning_effort = reasoning_effort
        self.default_max_tokens = default_max_tokens

        # 모델 ID에 reasoning effort 표기
        base_model_id = model.split("/")[-1]
        if reasoning_effort:
            self.model_id = f"{base_model_id}({reasoning_effort})"
        else:
            self.model_id = base_model_id

        self.client = OpenAIChatClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
        )

    def _extract_answer(
        self, response: str, num_choices: int = 4, use_number: bool = False
    ) -> int | None:
        """
        모델 응답에서 정답 추출.
        문자형(A/B/C/D) 또는 숫자형(1/2/3/4) 선택지 모두 지원.

        Args:
            response: 모델 응답 텍스트
            num_choices: 선택지 개수
            use_number: True면 숫자형(1,2,3,4), False면 문자형(A,B,C,D)

        Returns:
            선택지 인덱스 (0-indexed) 또는 None (추출 실패)
        """
        if not response or not response.strip():
            return None

        response = response.strip()

        if use_number:
            # 숫자형 선택지: 1, 2, 3, 4, ...
            choices = [str(i + 1) for i in range(num_choices)]  # ['1', '2', '3', '4', ...]

            def to_idx(val: str) -> int | None:
                val = val.strip()
                if val in choices:
                    return int(val) - 1
                return None

            # 1. 첫 줄이 단일 숫자인 경우
            first_line = response.split('\n')[0].strip()
            if first_line in choices:
                return to_idx(first_line)

            # 2. ### ANSWER 섹션
            if "### ANSWER" in response:
                answer_section = response.split("### ANSWER", 1)[1].strip()
                first_word = answer_section.split()[0] if answer_section.split() else ""
                if first_word in choices:
                    return to_idx(first_word)

            # 3. "Answer: 1" 또는 "정답: 1" 형식
            answer_match = re.search(r'(?:answer|답변|정답)[\s:：]*(\d+)', response, re.IGNORECASE)
            if answer_match and answer_match.group(1) in choices:
                return to_idx(answer_match.group(1))

            # 4. "(1)" 또는 "[1]" 형식
            bracket_match = re.search(r'[\(\[](\d+)[\)\]]', response)
            if bracket_match and bracket_match.group(1) in choices:
                return to_idx(bracket_match.group(1))

            # 5. "1)" 또는 "1." 형식 (줄 시작)
            option_match = re.search(r'^(\d+)[\.\)]', response, re.MULTILINE)
            if option_match and option_match.group(1) in choices:
                return to_idx(option_match.group(1))

            # 6. XML 태그 <answer>1</answer>
            xml_match = re.search(r'<answer>(\d+)</answer>', response, re.IGNORECASE)
            if xml_match and xml_match.group(1) in choices:
                return to_idx(xml_match.group(1))

            # 7. 마지막 줄이 단일 숫자인 경우
            last_line = response.split('\n')[-1].strip()
            if last_line in choices:
                return to_idx(last_line)

            # 8. 응답 시작 부분에서 숫자 찾기
            start_match = re.match(r'^\s*(\d+)\s*[.:]?\s*', response)
            if start_match and start_match.group(1) in choices:
                return to_idx(start_match.group(1))

            # 9. 응답 어디서든 숫자 찾기 (첫 번째 매치) - fallback
            anywhere_match = re.search(r'\b(\d+)\b', response)
            if anywhere_match and anywhere_match.group(1) in choices:
                return to_idx(anywhere_match.group(1))

        else:
            # 문자형 선택지: A, B, C, D, ...
            choices = [chr(65 + i) for i in range(num_choices)]  # ['A', 'B', 'C', 'D', ...]

            def to_idx(letter: str) -> int | None:
                letter = letter.upper()
                if letter in choices:
                    return ord(letter) - ord('A')
                return None

            # 1. 첫 줄이 단일 문자인 경우
            first_line = response.split('\n')[0].strip()
            if len(first_line) == 1 and first_line.upper() in choices:
                return to_idx(first_line)

            # 2. ### ANSWER 섹션
            if "### ANSWER" in response:
                answer_section = response.split("### ANSWER", 1)[1].strip()
                first_char = answer_section.split()[0] if answer_section.split() else answer_section[:1]
                if first_char.upper() in choices:
                    return to_idx(first_char)

            # 3. "Answer: A" 또는 "정답: A" 형식
            answer_match = re.search(r'(?:answer|답변|정답)[\s:：]*([A-J])', response, re.IGNORECASE)
            if answer_match and answer_match.group(1).upper() in choices:
                return to_idx(answer_match.group(1))

            # 4. "(A)" 또는 "[A]" 형식
            bracket_match = re.search(r'[\(\[]([A-J])[\)\]]', response, re.IGNORECASE)
            if bracket_match and bracket_match.group(1).upper() in choices:
                return to_idx(bracket_match.group(1))

            # 5. "A)" 또는 "A." 형식 (줄 시작)
            option_match = re.search(r'^([A-J])[\.\)]', response, re.IGNORECASE | re.MULTILINE)
            if option_match and option_match.group(1).upper() in choices:
                return to_idx(option_match.group(1))

            # 6. XML 태그 <answer>A</answer>
            xml_match = re.search(r'<answer>([A-J])</answer>', response, re.IGNORECASE)
            if xml_match and xml_match.group(1).upper() in choices:
                return to_idx(xml_match.group(1))

            # 7. 마지막 줄이 단일 문자인 경우
            last_line = response.split('\n')[-1].strip()
            if len(last_line) == 1 and last_line.upper() in choices:
                return to_idx(last_line)

            # 8. 응답 시작 부분에서 A/B/C/D 찾기
            match = self.ANSWER_PATTERN.match(response)
            if match:
                letter = match.group(1).upper()
                if letter in choices:
                    return to_idx(letter)

            # 9. 응답 어디서든 A/B/C/D 찾기 (첫 번째 매치) - fallback
            match = self.ANSWER_PATTERN_ANYWHERE.search(response)
            if match:
                letter = match.group(1).upper()
                if letter in choices:
                    return to_idx(letter)

        return None

    def _format_mcqa_prompt(self, sample: EvalSample, use_number: bool = False) -> str:
        """
        MCQA 샘플을 Chat API용 프롬프트로 포맷.

        Args:
            sample: EvalSample
            use_number: True면 숫자형(1,2,3,4), False면 문자형(A,B,C,D)

        Returns:
            포맷된 프롬프트
        """
        num_choices = len(sample.choices)

        if use_number:
            # 숫자형 선택지
            labels = [str(i + 1) for i in range(num_choices)]  # ['1', '2', '3', '4', ...]
        else:
            # 문자형 선택지
            labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"][:num_choices]

        # 프롬프트 구성
        prompt = sample.prompt

        # 선택지가 프롬프트에 이미 포함되어 있는지 확인
        if not any(f"{l}." in prompt or f"{l})" in prompt or f"({l})" in prompt for l in labels):
            # 선택지 추가
            choices_text = "\n".join(
                f"{l}. {choice}" for l, choice in zip(labels, sample.choices)
            )
            prompt = f"{prompt}\n\n{choices_text}"

        # 답변 형식 지시 (강화된 프롬프트)
        choice_str = ", ".join(labels[:-1]) + f", or {labels[-1]}"

        if use_number:
            prompt += f"""

CRITICAL: You MUST respond with EXACTLY ONE NUMBER ONLY ({choice_str}).

ABSOLUTELY NO explanations, reasoning, or additional text.

Just the number. Period.

Examples of CORRECT responses:
1
2
3

Examples of WRONG responses (NEVER do this):
- "1. The answer is..."
- "I think the answer is 3"
- "Answer: 2"

RESPOND WITH ONLY ONE NUMBER:"""
        else:
            prompt += f"""

CRITICAL: You MUST respond with EXACTLY ONE LETTER ONLY ({choice_str}).

ABSOLUTELY NO explanations, reasoning, or additional text.

Just the letter. Period.

Examples of CORRECT responses:
A
B
C

Examples of WRONG responses (NEVER do this):
- "A. The answer is..."
- "I think the answer is C"
- "Answer: B"

RESPOND WITH ONLY ONE LETTER:"""

        return prompt

    def _evaluate_mcqa_generation(
        self,
        samples: list[EvalSample],
        save_details: bool = True,
        use_number: bool = False,
        max_tokens: int = 4000,
        max_concurrency: int = 1,
    ) -> tuple[list[float], list[dict]]:
        """
        MCQA를 생성 방식으로 평가.

        Args:
            samples: 평가할 EvalSample 리스트
            save_details: 상세 결과 저장 여부
            use_number: True면 숫자형(1,2,3,4), False면 문자형(A,B,C,D)
            max_tokens: 생성 최대 토큰 수
            max_concurrency: 최대 동시 요청 수 (1이면 순차 처리)

        Returns:
            (scores, details)
        """
        label_type = "숫자" if use_number else "문자"

        # 병렬 처리
        if max_concurrency > 1:
            return asyncio.run(
                self._evaluate_mcqa_generation_async(
                    samples, save_details, use_number, max_tokens, max_concurrency
                )
            )

        # 순차 처리
        scores = []
        details = []

        for sample in tqdm(samples, desc=f"MCQA 평가 중 ({label_type}형)"):
            prompt = self._format_mcqa_prompt(sample, use_number=use_number)

            # 생성
            response = self.client.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0,
            )

            # 정답 추출
            pred = self._extract_answer(response, len(sample.choices), use_number=use_number)

            # 점수 계산
            if pred is not None:
                score = 1.0 if pred == sample.gold else 0.0
            else:
                score = 0.0  # 추출 실패 시 오답 처리
                pred = -1

            scores.append(score)

            if save_details:
                details.append({
                    "prompt": sample.prompt,
                    "choices": sample.choices,
                    "gold": sample.gold,
                    "pred": pred,
                    "response": response,
                    "score": score,
                })

        return scores, details

    async def _evaluate_mcqa_generation_async(
        self,
        samples: list[EvalSample],
        save_details: bool = True,
        use_number: bool = False,
        max_tokens: int = 4000,
        max_concurrency: int = 10,
    ) -> tuple[list[float], list[dict]]:
        """
        MCQA를 비동기 병렬로 평가.
        """
        label_type = "숫자" if use_number else "문자"
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_sample(idx: int, sample: EvalSample) -> tuple[int, float, dict | None]:
            async with semaphore:
                prompt = self._format_mcqa_prompt(sample, use_number=use_number)

                response = await self.client.generate_async(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0,
                )

                pred = self._extract_answer(response, len(sample.choices), use_number=use_number)

                if pred is not None:
                    score = 1.0 if pred == sample.gold else 0.0
                else:
                    score = 0.0
                    pred = -1

                detail = None
                if save_details:
                    detail = {
                        "prompt": sample.prompt,
                        "choices": sample.choices,
                        "gold": sample.gold,
                        "pred": pred,
                        "response": response,
                        "score": score,
                    }

                return idx, score, detail

        tasks = [process_sample(i, s) for i, s in enumerate(samples)]

        scores = [0.0] * len(samples)
        details = [None] * len(samples) if save_details else []

        for coro in atqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"MCQA 평가 중 ({label_type}형, 병렬={max_concurrency})",
        ):
            idx, score, detail = await coro
            scores[idx] = score
            if save_details and detail:
                details[idx] = detail

        return scores, details if save_details else []

    def _evaluate_generation(
        self,
        samples: list[EvalSample],
        loader: DatasetLoader,
        save_details: bool = True,
        max_concurrency: int = 1,
    ) -> tuple[list[float], list[dict]]:
        """
        생성형 문제 평가 (수학 등).

        Args:
            samples: 평가할 EvalSample 리스트
            loader: 데이터셋 로더
            save_details: 상세 결과 저장 여부
            max_concurrency: 최대 동시 요청 수 (1이면 순차 처리)

        Returns:
            (scores, details)
        """
        gen_kwargs = loader.generation_kwargs
        max_tokens = gen_kwargs.get("max_tokens", 2048)
        temperature = gen_kwargs.get("temperature", 0)
        stop = gen_kwargs.get("stop", [])

        # 병렬 처리
        if max_concurrency > 1:
            return asyncio.run(
                self._evaluate_generation_async(
                    samples, loader, save_details, max_concurrency
                )
            )

        # 순차 처리
        scores = []
        details = []

        for sample in tqdm(samples, desc="생성 평가 중"):
            prediction = self.client.generate(
                prompt=sample.prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop if stop else None,
            )
            prediction = prediction.strip()

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

    async def _evaluate_generation_async(
        self,
        samples: list[EvalSample],
        loader: DatasetLoader,
        save_details: bool = True,
        max_concurrency: int = 10,
    ) -> tuple[list[float], list[dict]]:
        """
        생성형 문제를 비동기 병렬로 평가.
        """
        gen_kwargs = loader.generation_kwargs
        max_tokens = gen_kwargs.get("max_tokens", 2048)
        temperature = gen_kwargs.get("temperature", 0)
        stop = gen_kwargs.get("stop", [])

        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_sample(idx: int, sample: EvalSample) -> tuple[int, float, dict | None]:
            async with semaphore:
                prediction = await self.client.generate_async(
                    prompt=sample.prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop if stop else None,
                )
                prediction = prediction.strip()

                score = loader.compute_score(prediction, sample)

                detail = None
                if save_details:
                    detail = {
                        "prompt": sample.prompt,
                        "prediction": prediction,
                        "gold": sample.gold,
                        "score": score,
                    }

                return idx, score, detail

        tasks = [process_sample(i, s) for i, s in enumerate(samples)]

        scores = [0.0] * len(samples)
        details = [None] * len(samples) if save_details else []

        for coro in atqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"생성 평가 중 (병렬={max_concurrency})",
        ):
            idx, score, detail = await coro
            scores[idx] = score
            if save_details and detail:
                details[idx] = detail

        return scores, details if save_details else []

    def _detect_use_number(self, loader: DatasetLoader) -> bool:
        """
        데이터셋 설정에서 숫자형/문자형 선택지 여부 감지.

        숫자형 데이터셋: csatqa, kmmlu_pro
        문자형 데이터셋: kmmlu, haerae, kobalt, click

        Args:
            loader: DatasetLoader

        Returns:
            True면 숫자형(1,2,3,4), False면 문자형(A,B,C,D)
        """
        # answer_type이 number_string이면 숫자형
        answer_type = getattr(loader.config, 'answer_type', None)
        if answer_type == 'number_string':
            return True

        # choices 필드 확인
        choices = loader.config.choices
        if choices:
            # "(1)", "(2)" 또는 "1", "2" 형식이면 숫자형
            first_choice = str(choices[0])
            if first_choice.isdigit() or first_choice.strip('()').isdigit():
                return True

        # 데이터셋 이름으로 판단 (fallback)
        dataset_name = loader.config.name
        number_datasets = {'csatqa', 'kmmlu_pro'}
        if dataset_name in number_datasets:
            return True

        return False

    def evaluate_dataset(
        self,
        dataset_name: str,
        save_details: bool = True,
        max_concurrency: int = 1,
        limit: int | None = None,
    ) -> EvaluationResult:
        """
        특정 데이터셋에서 모델 평가.

        Args:
            dataset_name: 평가할 데이터셋 이름
            save_details: 상세 결과 저장 여부
            max_concurrency: 최대 동시 요청 수 (1이면 순차 처리)
            limit: 평가할 샘플 수 제한 (None이면 전체)

        Returns:
            EvaluationResult
        """
        loader = DatasetLoader(dataset_name)

        # 숫자형/문자형 감지
        use_number = self._detect_use_number(loader)
        label_type = "숫자형(1,2,3,...)" if use_number else "문자형(A,B,C,...)"

        print(f"\n{'='*50}")
        print(f"평가 중: {dataset_name}")
        print(f"평가 유형: {loader.output_type} (Chat API - 생성 방식)")
        print(f"선택지 유형: {label_type}")
        print(f"모델: {self.model_name}")
        if max_concurrency > 1:
            print(f"병렬 처리: {max_concurrency} 동시 요청")
        if limit:
            print(f"샘플 제한: {limit}개")
        print(f"{'='*50}")

        dataset = loader.load()
        samples = loader.format_all(dataset)

        # 샘플 수 제한
        if limit and limit < len(samples):
            samples = samples[:limit]
            print(f"{limit}개 샘플로 제한됨 (전체: {len(loader.format_all(dataset))}개)")
        else:
            print(f"{len(samples)}개 샘플 로드됨")

        # generation_kwargs에서 max_tokens 가져오기
        # CLI에서 지정한 default_max_tokens가 있으면 우선 사용
        gen_kwargs = loader.generation_kwargs
        if self.default_max_tokens is not None:
            max_tokens = self.default_max_tokens
        else:
            max_tokens = gen_kwargs.get("max_tokens", 4000)

        # 평가 실행
        if loader.output_type == "multiple_choice":
            # MCQA를 생성 방식으로 평가
            scores, details = self._evaluate_mcqa_generation(
                samples, save_details, use_number=use_number,
                max_tokens=max_tokens, max_concurrency=max_concurrency
            )
        else:
            # generate_until
            scores, details = self._evaluate_generation(
                samples, loader, save_details, max_concurrency=max_concurrency
            )

        final_score = sum(scores) / len(scores)

        print(f"\n{dataset_name} 결과:")
        print(f"  정확도: {final_score:.4f}")

        return EvaluationResult(
            dataset_name=dataset_name,
            model_name=self.model_name,
            num_samples=len(samples),
            score=final_score,
            score_norm=None,  # 생성 방식은 정규화 점수 없음
            metric=loader.config.metric,
            timestamp=get_kst_timestamp(),
            details=details if save_details else [],
        )

    def evaluate_all(
        self,
        datasets: list[str] | None = None,
        save_details: bool = True,
        max_concurrency: int = 1,
    ) -> dict[str, EvaluationResult]:
        """
        여러 데이터셋에서 모델 평가.

        Args:
            datasets: 평가할 데이터셋 이름 리스트 (None이면 전체)
            save_details: 상세 결과 저장 여부
            max_concurrency: 최대 동시 요청 수 (1이면 순차 처리)

        Returns:
            dict[str, EvaluationResult]
        """
        if datasets is None:
            datasets = get_available_datasets()

        results = {}
        for dataset_name in datasets:
            try:
                result = self.evaluate_dataset(dataset_name, save_details, max_concurrency)
                results[dataset_name] = result
            except Exception as e:
                print(f"{dataset_name} 평가 중 에러: {e}")
                import traceback
                traceback.print_exc()
                continue

        return results
