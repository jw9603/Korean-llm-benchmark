"""
YAML-based dataset loader.
Loads dataset configuration from YAML files and provides unified interface.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset, Dataset, concatenate_datasets, Features, Value
from jinja2 import Template

from ..utils.math_utils import is_equiv, postprocess


@dataclass
class EvalSample:
    """A single evaluation sample."""
    prompt: str
    choices: list[str] | None
    gold: Any
    doc: dict


@dataclass
class DatasetConfig:
    """Dataset configuration loaded from YAML."""
    name: str
    dataset_path: str
    output_type: str  # "multiple_choice" or "generate_until"
    split: str
    prompt_template: str
    field_mapping: dict[str, str]
    metric: str
    choices: list[str] | None = None
    subsets: list[str] | None = None
    gold_offset: int = 0
    generation_kwargs: dict = field(default_factory=dict)
    answer_type: str = "index"  # "index", "string_choice", "text_match", "number_string"
    dataset_config: str | None = None  # HuggingFace dataset config name
    answer_extraction: str | None = None  # "cot" for Chain-of-Thought extraction
    answer_pattern: str | None = None  # regex pattern for answer extraction
    # Few-shot settings
    num_fewshot: int = 0  # Number of few-shot examples
    fewshot_split: str | None = None  # Split to use for few-shot examples (e.g., "validation")
    fewshot_template: str | None = None  # Template for few-shot examples (with answer)
    # Filter settings
    filter_column: str | None = None  # Column to filter by (e.g., "category")
    filter_value: str | None = None  # Value to filter for (e.g., "biology")

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "DatasetConfig":
        """Load config from YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        gold_offset = data.get("field_mapping", {}).pop("gold_offset", 0)

        return cls(
            name=data["name"],
            dataset_path=data["dataset_path"],
            output_type=data["output_type"],
            split=data["split"],
            prompt_template=data["prompt_template"],
            field_mapping=data.get("field_mapping", {}),
            metric=data["metric"],
            choices=data.get("choices"),
            subsets=data.get("subsets"),
            gold_offset=gold_offset,
            generation_kwargs=data.get("generation_kwargs", {}),
            answer_type=data.get("answer_type", "index"),
            dataset_config=data.get("dataset_config"),
            answer_extraction=data.get("answer_extraction"),
            answer_pattern=data.get("answer_pattern"),
            num_fewshot=data.get("num_fewshot", 0),
            fewshot_split=data.get("fewshot_split"),
            fewshot_template=data.get("fewshot_template"),
            filter_column=data.get("filter_column"),
            filter_value=data.get("filter_value"),
        )


class DatasetLoader:
    """Unified dataset loader using YAML configuration."""

    CONFIGS_DIR = Path(__file__).parent / "configs"

    def __init__(self, dataset_name: str):
        yaml_path = self.CONFIGS_DIR / f"{dataset_name}.yaml"
        if not yaml_path.exists():
            raise ValueError(f"Dataset config not found: {yaml_path}")

        self.config = DatasetConfig.from_yaml(yaml_path)
        self._template = Template(self.config.prompt_template)
        self._fewshot_template = Template(self.config.fewshot_template) if self.config.fewshot_template else None
        self._fewshot_examples: list[dict] | None = None

    # @property: 속성 메서드로 사용
    @property
    def name(self) -> str:
        return self.config.name

    @property
    def output_type(self) -> str:
        return self.config.output_type

    @property
    def generation_kwargs(self) -> dict:
        return self.config.generation_kwargs

    def _load_fewshot_examples(self) -> list[dict]:
        """Load few-shot examples from fewshot_split."""
        if self._fewshot_examples is not None:
            return self._fewshot_examples

        if not self.config.fewshot_split or self.config.num_fewshot == 0:
            self._fewshot_examples = []
            return self._fewshot_examples

        kwargs = {
            "path": self.config.dataset_path,
            "split": self.config.fewshot_split,
            "token": True,
            "trust_remote_code": True,
        }
        if self.config.dataset_config:
            kwargs["name"] = self.config.dataset_config

        fewshot_ds = load_dataset(**kwargs)
        # Take first N examples
        self._fewshot_examples = [fewshot_ds[i] for i in range(min(self.config.num_fewshot, len(fewshot_ds)))]
        return self._fewshot_examples

    def _format_fewshot_example(self, doc: dict) -> str:
        """Format a single few-shot example (with answer)."""
        if self._fewshot_template:
            # Map fields for template
            template_vars = {}
            for template_key, doc_key in self.config.field_mapping.items():
                if doc_key in doc:
                    template_vars[template_key] = doc[doc_key]

            # Add choices to template vars if needed
            choices = self.config.choices
            choices_key = self.config.field_mapping.get("choices")
            if choices_key and choices_key in doc:
                choices = doc[choices_key]
            if choices:
                template_vars["choices"] = choices

            return self._fewshot_template.render(**template_vars).strip()
        return ""

    def load(self) -> Dataset:
        """Load the dataset from HuggingFace. Handles subsets if specified."""
        if self.config.subsets:
            datasets = []
            for subset in self.config.subsets:
                try:
                    ds = load_dataset(
                        self.config.dataset_path,
                        name=subset,
                        split=self.config.split,
                        token=True,  # Use HF token for gated datasets
                        trust_remote_code=True,
                    )

                    # 모든 컬럼을 string으로 변환하여 스키마 통일
                    # 1. 값을 string으로 변환
                    def convert_to_string(example, subset_name=subset):
                        result = {"_subset": subset_name}
                        for key, value in example.items():
                            if value is None:
                                result[key] = ""
                            elif isinstance(value, str):
                                result[key] = value
                            else:
                                result[key] = str(value)
                        return result

                    ds = ds.map(convert_to_string, remove_columns=ds.column_names)

                    # 2. Features 스키마도 string으로 캐스팅
                    new_features = Features({
                        col: Value("string") for col in ds.column_names
                    })
                    ds = ds.cast(new_features)

                    datasets.append(ds)
                except Exception as e:
                    print(f"Warning: Failed to load subset {subset}: {e}")
                    continue

            if not datasets:
                raise ValueError(f"No subsets could be loaded for {self.config.name}")
            return concatenate_datasets(datasets)
        else:
            # Use dataset_config if specified (e.g., for KoBALT-700)
            kwargs = {
                "path": self.config.dataset_path,
                "split": self.config.split,
                "token": True,  # Use HF token for gated datasets
                "trust_remote_code": True,
            }
            if self.config.dataset_config:
                kwargs["name"] = self.config.dataset_config
            dataset = load_dataset(**kwargs)

            # Apply filter if specified (e.g., filter by category)
            if self.config.filter_column and self.config.filter_value:
                dataset = dataset.filter(
                    lambda x: x[self.config.filter_column] == self.config.filter_value
                )
            return dataset

    def format_sample(self, doc: dict) -> EvalSample:
        """Format a single document into an evaluation sample."""
        # Get choices - either from config or from document
        choices = self.config.choices
        choices_key = self.config.field_mapping.get("choices")
        if choices_key and choices_key in doc:
            choices = doc[choices_key]
            if isinstance(choices, str):
                # Parse string representation of list
                import ast
                try:
                    choices = ast.literal_eval(choices)
                except:
                    pass

        # Map fields for template
        template_vars = {}
        for template_key, doc_key in self.config.field_mapping.items():
            if doc_key in doc:
                template_vars[template_key] = doc[doc_key]

        # Add choices to template vars if needed
        if choices:
            template_vars["choices"] = choices

        # Add few-shot examples if configured
        fewshot_prefix = ""
        if self.config.num_fewshot > 0 and self._fewshot_template:
            fewshot_examples = self._load_fewshot_examples()
            fewshot_texts = [self._format_fewshot_example(ex) for ex in fewshot_examples]
            if fewshot_texts:
                fewshot_prefix = "\n\n".join(fewshot_texts) + "\n\n"

        # Render prompt
        prompt = fewshot_prefix + self._template.render(**template_vars).strip()

        # Get gold answer
        gold_key = self.config.field_mapping.get("gold", "gold")
        gold = doc.get(gold_key)

        # Handle different answer types
        if self.config.answer_type == "string_choice":
            # Convert string choice ("a", "b", etc.) to index
            choice_map = {c.lower(): i for i, c in enumerate(choices)}
            gold = choice_map.get(str(gold).lower(), 0)
        elif self.config.answer_type == "text_match":
            # Find the index of gold answer in choices list
            gold_str = str(gold).strip()
            gold = 0  # default
            for i, choice in enumerate(choices):
                if str(choice).strip() == gold_str:
                    gold = i
                    break
        elif self.config.answer_type == "number_string":
            # Convert "1", "2", etc. to 0-indexed
            try:
                gold = int(str(gold)) - 1
            except (ValueError, TypeError):
                gold = 0
        elif self.config.gold_offset:
            # gold_offset 적용 (문자열이든 정수든 처리)
            try:
                gold = int(str(gold)) + self.config.gold_offset
            except (ValueError, TypeError):
                gold = 0

        # For generation tasks, postprocess the gold answer
        if self.config.output_type == "generate_until":
            gold = postprocess(str(gold))

        return EvalSample(
            prompt=prompt,
            choices=choices,
            gold=gold,
            doc=doc,
        )

    def format_all(self, dataset: Dataset) -> list[EvalSample]:
        """Format all documents in the dataset."""
        return [self.format_sample(doc) for doc in dataset]

    def compute_score(self, prediction: str, sample: EvalSample) -> float:
        """Compute score for a single prediction."""
        # Ko-MuSR: 특별한 채점 방식 (정답: X 형식)
        if self.config.answer_type == "ko_musr":
            return self._compute_ko_musr(prediction, sample)
        # CoT extraction (e.g., MMLU-Pro "the answer is (X)" pattern)
        elif self.config.answer_extraction == "cot":
            return self._compute_cot_extraction(prediction, sample)
        elif self.config.metric == "exact_match":
            return self._compute_exact_match(prediction, sample)
        else:  # accuracy
            return self._compute_accuracy(prediction, sample)

    def _compute_accuracy(self, prediction: str, sample: EvalSample) -> float:
        """Compute accuracy for multiple choice."""
        pred = prediction.strip()
        correct_choice = sample.choices[sample.gold]

        # Normalize prediction
        pred_upper = pred.upper()
        correct_upper = correct_choice.upper()

        # Check various formats
        if pred_upper == correct_upper:
            return 1.0

        # Strip parentheses: "(A)" -> "A"
        pred_stripped = re.sub(r"[()]", "", pred_upper)
        correct_stripped = re.sub(r"[()]", "", correct_upper)

        if pred_stripped == correct_stripped:
            return 1.0

        # Check if starts with correct choice
        if pred_upper.startswith(correct_upper):
            return 1.0
        if pred_stripped.startswith(correct_stripped):
            return 1.0

        return 0.0

    def _compute_exact_match(self, prediction: str, sample: EvalSample) -> float:
        """Compute exact match for generation tasks (math)."""
        if is_equiv(prediction, sample.gold):
            return 1.0
        return 0.0

    def _compute_cot_extraction(self, prediction: str, sample: EvalSample) -> float:
        """
        Chain-of-Thought answer extraction.

        Extracts answer from model response using regex pattern (e.g., "the answer is (X)").
        Used for MMLU-Pro style evaluation.

        Args:
            prediction: Model's generated response
            sample: Evaluation sample (gold is the correct answer letter, e.g., "A")

        Returns:
            1.0 if correct, 0.0 otherwise
        """
        gold = str(sample.gold).upper()
        extracted = None

        # 1. Primary: Extract answer using config's regex pattern
        if self.config.answer_pattern:
            pattern = re.compile(self.config.answer_pattern, re.IGNORECASE)
            matches = pattern.findall(prediction)
            if matches:
                extracted = matches[-1].upper()

        # 2. Fallback: Try common answer formats if primary pattern fails
        if not extracted:
            # Try: "Final Answer: X", "Answer: X", "정답: X"
            fallback_pattern = re.compile(
                r'(?:Final Answer|Answer|정답|The answer)[:\s]*\*?\*?\(?([A-J])\)?\*?\*?',
                re.IGNORECASE
            )
            matches = fallback_pattern.findall(prediction)
            if matches:
                extracted = matches[-1].upper()

        # 3. Fallback: Look for standalone letter in last few lines (e.g., "**I**" or just "I")
        if not extracted:
            last_lines = prediction.strip().split('\n')[-5:]
            for line in reversed(last_lines):
                line = line.strip()
                # Match: **A**, *A*, (A), or standalone A-J at end of line
                m = re.search(r'\*\*([A-J])\*\*|\*([A-J])\*|\(([A-J])\)$|^([A-J])$', line, re.IGNORECASE)
                if m:
                    extracted = (m.group(1) or m.group(2) or m.group(3) or m.group(4)).upper()
                    break

        # Compare with gold answer
        if extracted and extracted == gold:
            return 1.0

        return 0.0

    def _compute_ko_musr(self, prediction: str, sample: EvalSample) -> float:
        """
        Ko-MuSR 채점 로직.

        모델 응답에서 "정답: X" 또는 "ANSWER: X" 형식을 찾아
        마지막 정답 라인의 숫자를 추출하여 비교합니다.

        Args:
            prediction: 모델이 생성한 응답
            sample: 평가 샘플 (gold는 0-indexed)

        Returns:
            1.0 if correct, 0.0 otherwise
        """
        lines = prediction.split("\n")

        # "정답" 또는 "answer"가 포함된 라인 찾기
        answer_lines = []
        for line in lines:
            lower = line.lower()
            if "정답" in lower or "answer" in lower:
                answer_lines.append(line)

        if not answer_lines:
            return 0.0

        # 마지막 정답 라인에서 숫자 추출
        final_line = answer_lines[-1].strip().replace(" ", "")

        # gold는 0-indexed, 정답은 1-indexed
        gold_answer = int(sample.gold) + 1

        if str(gold_answer) in final_line:
            return 1.0

        return 0.0


def get_available_datasets() -> list[str]:
    """Get list of available dataset names."""
    configs_dir = Path(__file__).parent / "configs"
    # f.stem: 확장자를 제외한 파일명
    return [f.stem for f in configs_dir.glob("*.yaml")]


def load_dataset_by_name(name: str) -> DatasetLoader:
    """Load a dataset by name."""
    return DatasetLoader(name)
