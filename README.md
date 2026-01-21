# LLM Evaluation - Korean Benchmarks

한국어 LLM 벤치마크 평가 도구. vLLM 또는 OpenAI-compatible API를 사용하여 모델을 평가합니다.

## Updates

- **2026-01-21**: Open Korean Benchmark evaluation

</details>

## 지원 데이터셋

| 데이터셋 | 설명 | 유형 | 선택지 | 메트릭 |
|----------|------|------|--------|--------|
| **kmmlu** | 한국어 MMLU | MCQA (4지선다) | A,B,C,D | Accuracy |
| **kmmlu_pro** | 한국어 MMLU Pro | MCQA (5지선다) | 1,2,3,4,5 | Accuracy |
| **mmlu_pro(zero)** | MMLU-Pro (영어, 0-shot) | Generation (CoT) | A~J | Exact Match |
| **mmlu_pro(five)** | MMLU-Pro (영어, 5-shot) | Generation (CoT) | A~J | Exact Match |
| **csatqa** | 한국 수능 국어 | MCQA (5지선다) | 1,2,3,4,5 | Accuracy |
| **haerae** | 한국 문화/사회 | MCQA (5지선다) | A,B,C,D,E | Accuracy |
| **hrm8k** | 한국어 수학 문제 | Generation | - | Exact Match |
| **hrm8k_mmmlu** | 한국어 수학 (MMMLU) | Generation | 1,2,3,4 | Exact Match |
| **kobalt** | 한국어 BALT | MCQA (10지선다) | A~J | Accuracy |
| **click** | CLIcK 벤치마크 | MCQA | A,B,C,D,E | Accuracy |
| **ko_musr_mm** | 한국어 MuSR (Murder Mystery) | Generation | 1,2,3 | Accuracy |
| **ko_musr_op** | 한국어 MuSR (Object Placement) | Generation | 1,2,3 | Accuracy |
| **ko_musr_ta** | 한국어 MuSR (Team Allocation) | Generation | 1,2,3 | Accuracy |
| **kormedmcqa** | 한국 의료 면허시험 | MCQA (5지선다) | A,B,C,D,E | Accuracy |

### 데이터셋 라이센스

| 데이터셋 | 라이센스 | 상업적 사용 |
|----------|----------|-------------|
| [kmmlu](https://huggingface.co/datasets/HAERAE-HUB/KMMLU) | CC-BY-SA-4.0 | ✅ 가능 |
| [kmmlu_pro](https://huggingface.co/datasets/LGAI-EXAONE/KMMLU-Pro) | CC-BY-NC-ND-4.0 | ❌ 불가 |
| [csatqa](https://huggingface.co/datasets/HAERAE-HUB/csatqa) | 연구 목적만 허용 | ❌ 불가 |
| [haerae](https://huggingface.co/datasets/HAERAE-HUB/HAE_RAE_BENCH_1.0) | CC-BY-NC-ND | ❌ 불가 |
| [hrm8k, hrm8k_mmmlu](https://huggingface.co/datasets/HAERAE-HUB/HRM8K) | MIT | ✅ 가능 |
| [kobalt](https://huggingface.co/datasets/snunlp/KoBALT-700) | CC-BY-NC-4.0 | ❌ 불가 |
| [click](https://huggingface.co/datasets/EunsuKim/CLIcK) | 명시 안 됨 | ⚠️ 확인 필요 |
| [ko_musr_*](https://huggingface.co/datasets/thunder-research-group/SNU_Ko-MuSR) | MIT | ✅ 가능 |
| [kormedmcqa](https://huggingface.co/datasets/sean0042/KorMedMCQA) | CC-BY-NC-2.0 | ❌ 불가 |
| [mmlu_pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | MIT | ✅ 가능 |

## 지원 백엔드

| 백엔드 | 설명 | 평가 방식 |
|--------|------|-----------|
| **vllm** | 로컬 GPU (vLLM) | Loglikelihood |
| **openai** | OpenAI Completions API | Loglikelihood |
| **chat** | OpenAI Chat API | Generation |
| **api** | 자동 감지 | 자동 선택 |

## 설치

### 기본 환경 (vLLM 0.13+)

대부분의 모델 평가에 사용하는 기본 환경입니다.

```bash
# 가상환경 생성
conda create -n llm python=3.11
conda activate llm

# 의존성 설치
pip install -r requirements.txt
```

### VAETKI 환경 (vLLM 0.11.2)

NC AI의 VAETKI MoE 모델 평가를 위한 별도 환경입니다.

```bash
# 가상환경 생성
conda create -n vaetki python=3.11
conda activate vaetki

# 의존성 설치
pip install -r requirements-vaetki.txt

# VAETKI 플러그인 설치
git clone --depth 1 --branch dev https://github.com/cyr0930/Megatron-LM-wbl.git
PYTHONPATH="$PWD/Megatron-LM-wbl/vllm_plugin" pip install Megatron-LM-wbl/vllm_plugin
rm -rf Megatron-LM-wbl
```

### 환경 변수 설정

API 백엔드 사용 시 `.env` 파일을 생성하세요:

```bash
# OpenAI API
OPENAI_API_KEY=sk-xxx

# Friendly AI (K-EXAONE)
FRIENDLI_API_KEY=xxx
```

## 사용법

> **Note:** `scripts/run_eval.sh` 스크립트로 모든 평가를 수행할 수 있습니다.

### 통합 스크립트 (run_eval.sh)

```bash
# 도움말
./llm_evaluation/scripts/run_eval.sh --help

# 전체 데이터셋 평가
./llm_evaluation/scripts/run_eval.sh --model gpt-4o

# 단일 데이터셋 평가
./llm_evaluation/scripts/run_eval.sh --dataset kmmlu --model gpt-4o

# 샘플 수 제한 (테스트용)
./llm_evaluation/scripts/run_eval.sh --dataset hrm8k --model gpt-4o --limit 100
```

### 스크립트 옵션

| 옵션 | 단축 | 설명 |
|------|------|------|
| `--model` | `-m` | 모델 이름 또는 프리셋 (필수) |
| `--dataset` | `-d` | 단일 데이터셋 (없으면 전체 평가) |
| `--limit` | `-l` | 샘플 수 제한 |
| `--concurrency` | `-c` | 동시 요청 수 (API) |
| `--tensor-parallel-size` | | GPU 병렬 수 (vLLM) |
| `--max-model-len` | | 최대 컨텍스트 길이 (vLLM) |
| `--base-url` | | 커스텀 API URL |
| `--api-key` | | API 키 |
| `--output-dir` | `-o` | 결과 저장 디렉토리 |

### 모델 프리셋

| 프리셋 | 모델 | 백엔드 |
|--------|------|--------|
| `gpt-5.2` | gpt-5.2-2025-12-11 | OpenAI API |
| `gpt-4o` | gpt-4o | OpenAI API |
| `gpt-4o-mini` | gpt-4o-mini | OpenAI API |
| `k-exaone` | K-EXAONE-236B | Friendly AI |

### 예시

```bash
# API 모델 (프리셋)
./llm_evaluation/scripts/run_eval.sh --model gpt-4o --dataset kmmlu

# 커스텀 API 엔드포인트
./llm_evaluation/scripts/run_eval.sh \
    --model krevas/gpt-oss-120b \
    --base-url https://api.example.com/v1 \
    --api-key your-key \
    --dataset kobalt

# vLLM 로컬 (HuggingFace 모델)
./llm_evaluation/scripts/run_eval.sh \
    --model Qwen/Qwen3-32B \
    --tensor-parallel-size 2

# CLI 직접 사용
python -m llm_evaluation --backend vllm --model Qwen/Qwen3-32B --datasets kmmlu csatqa
python -m llm_evaluation --backend api --model gpt-4o --datasets kmmlu

# GPT-5/o1/o3 reasoning effort 옵션
python -m llm_evaluation --backend api --model gpt-5.2-2025-12-11 --reasoning-effort medium --datasets kmmlu
```


## 평가 방식

### MCQA (Multiple Choice QA)

**Loglikelihood 방식** (vLLM, OpenAI Completions API)
- 각 선택지에 대해 `log P(choice|context)` 계산
- 가장 높은 확률의 선택지를 정답으로 선택
- lm-evaluation-harness 표준 방식
- `acc`: Raw accuracy (log P(choice|context) 기준)
- `acc_norm`: 토큰 길이로 정규화된 accuracy
  - 긴 선택지는 토큰이 많아 logprob 합이 낮아지는 bias 존재
  - `log P / num_tokens`로 정규화하여 길이 bias 보정
  - 예: "서울특별시" (3토큰) vs "서울" (1토큰) → 정규화 없이는 "서울"이 유리

**Generation 방식** (Chat API)
- Chat API는 prompt logprobs를 지원하지 않아 loglikelihood 계산 불가
- 대신 모델에게 직접 정답을 생성하도록 요청
- 강화된 영어 프롬프트로 단일 문자 응답 유도:
  ```
  질문: 다음 중 대한민국의 수도는?
  A. 부산
  B. 서울
  C. 대전
  D. 인천

  정답만 출력하세요 (A, B, C, D 중 하나):

  CRITICAL: You MUST respond with EXACTLY ONE LETTER ONLY (A, B, C, D).
  ABSOLUTELY NO explanations, reasoning, or additional text.
  Just the letter. Period.
  ```
- 9가지 패턴으로 응답 파싱:
  1. 첫 줄 단일 문자 (예: `A`)
  2. `### ANSWER` 섹션
  3. `Answer: A` / `정답: A` 형식
  4. `(A)` 또는 `[A]` 형식
  5. `A)` 또는 `A.` 형식 (줄 시작)
  6. XML 태그 `<answer>A</answer>`
  7. 마지막 줄 단일 문자
  8. 응답 내 단일 문자 A/B/C/D
  9. 정규식 fallback
- GPT-4o, GPT-5, K-EXAONE 등 Chat 전용 모델에서 사용

### Generation (수학/추론 문제)

일부 데이터셋은 선택지가 있지만 `generate_until` 방식으로 평가합니다:

| 데이터셋 | 설명 | 이유 |
|----------|------|------|
| **hrm8k_mmmlu** | 수학 4지선다 | `\boxed{N}` 형식으로 생성 |
| **ko_musr_*** | 추론 문제 | CoT 추론 후 "정답: X" 생성 |
| **mmlu_pro_*** | MMLU-Pro (영어, 0/5-shot) | CoT 추론 후 "the answer is (X)" 추출 |

**왜 MCQA가 아닌 Generation인가?**
- 단순 확률 비교가 아닌 **추론 과정 생성**을 유도
- 프롬프트에서 단계별 추론 후 정답 출력 요청
- lm-evaluation-harness 원본 벤치마크 표준 방식

**정답 추출 방식:**
- 수학: `\boxed{N}` 형식에서 추출, 수학적 동치성 비교 (예: `0.5` = `1/2`)
- 추론: "정답: X" 형식에서 추출
- MMLU-Pro: `the answer is (X)` 패턴에서 마지막 매치 추출

## CLI 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--backend` | 평가 백엔드 (vllm/api/openai/chat) | vllm |
| `--model` | 모델 이름 | (필수) |
| `--datasets` | 평가할 데이터셋 | 전체 |
| `--output-dir` | 결과 저장 디렉토리 | ./llm_evaluation/results |
| `--tensor-parallel-size` | GPU 분산 수 (vLLM) | 1 |
| `--gpu-memory-utilization` | GPU 메모리 사용률 (vLLM) | 0.9 |
| `--max-model-len` | 최대 시퀀스 길이 (vLLM) | auto |
| `--base-url` | API 베이스 URL | - |
| `--api-key` | API 키 | OPENAI_API_KEY |
| `--reasoning-effort` | GPT-5/o1/o3 reasoning 수준 (low/medium/high) | None |

## 결과 출력

```
llm_evaluation/results/
├── {model_id}_results.json           # 요약 결과
├── {model_id}_{dataset}_details.json # 상세 결과
└── leaderboard.md                    # 리더보드

# model_id 예시:
# - vLLM: "Qwen3-32B"
# - Chat API: "gpt-4o"
# - reasoning_effort 사용 시: "gpt-5.2-2025-12-11(medium)"
```

## 개발 상태

| 기능 | 상태 |
|------|------|
| vLLM 백엔드 | ✅ 완료 |
| OpenAI Completions API | ✅ 완료 |
| OpenAI Chat API | ✅ 완료 |
| K-EXAONE 지원 | ⏸️ 잠정 중단 |

### 참고: skt/A.X-3.1 테스트 환경

`skt/A.X-3.1` 모델은 H100 80GB 1장에 기본 설정으로 로드되지 않아 `max_model_len=8192`로 제한하여 테스트했습니다.

```bash
./llm_evaluation/scripts/run_eval.sh skt/A.X-3.1 1 8192
```

### 참고: K-EXAONE (Friendly AI)

K-EXAONE 모델은 Friendly AI serverless API를 통해 지원되나, 너무 느려서.. 현재 **잠정 중단** 상태입니다.

**문제점:**
- 응답 속도가 매우 느림 (~17초/요청, GPT-4o 대비 약 35배)
- Rate limiting으로 대규모 평가 불가 (429 에러 빈발)
- Serverless 특성상 cold start 지연 발생

## TODO

- [ ] [Ko-IFEval](https://huggingface.co/datasets/allganize/IFEval-Ko) 데이터셋 추가하기
- [ ] [KBL](https://huggingface.co/datasets/lbox/kbl) 데이터셋 추가하기


## References

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) - EleutherAI LLM 평가 프레임워크
- [evaluate-llm-on-korean-dataset](https://github.com/daekeun-ml/evaluate-llm-on-korean-dataset) - 한국어 데이터셋 LLM 평가

<details>
<summary>벤치마크</summary>

```bibtex
@misc{son2024kmmlumeasuringmassivemultitask,
      title={KMMLU: Measuring Massive Multitask Language Understanding in Korean},
      author={Guijin Son and Hanwool Lee and Sungdong Kim and Seungone Kim and Niklas Muennighoff and Taekyoon Choi and Cheonbok Park and Kang Min Yoo and Stella Biderman},
      year={2024},
      eprint={2402.11548},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2402.11548},
}

@misc{son2024haeraebenchevaluationkorean,
      title={HAE-RAE Bench: Evaluation of Korean Knowledge in Language Models},
      author={Guijin Son and Hanwool Lee and Suwan Kim and Huiseo Kim and Jaecheol Lee and Je Won Yeom and Jihyu Jung and Jung Woo Kim and Songseong Kim},
      year={2024},
      eprint={2309.02706},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2309.02706},
}

@misc{kim2024click,
      title={CLIcK: A Benchmark Dataset of Cultural and Linguistic Intelligence in Korean},
      author={Eunsu Kim and Juyoung Suk and Philhoon Oh and Haneul Yoo and James Thorne and Alice Oh},
      year={2024},
      eprint={2403.06412},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
}

@misc{ko2025hrm8k,
      title={HRM8K: A Bilingual Math Reasoning Benchmark for Korean and English},
      author={Hyunwoo Ko and Guijin Son and Dasol Choi},
      year={2025},
      eprint={2501.02448},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.02448},
}

@misc{lee2025kobalt,
      title={KoBALT: A Benchmark for Evaluating Korean Linguistic Phenomena in Large Language Models},
      author={Dohyun Lee and Seunghyun Hwang and Seungtaek Choi and Hwisang Jeon and Sohyun Park and Sungjoon Park and Yungi Kim},
      year={2025},
      eprint={2505.16125},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.16125},
}

@misc{park2025komusrmultistepsoftreasoning,
      title={Ko-MuSR: A Multistep Soft Reasoning Benchmark for LLMs Capable of Understanding Korean},
      author={Chanwoo Park and Suyoung Park and JiA Kang and Jongyeon Park and Sangho Kim and Hyunji M. Park and Sumin Bae and Mingyu Kang and Jaejin Lee},
      year={2025},
      eprint={2510.24150},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.24150},
}

@misc{hong2025kmmlureduxkmmluproprofessionalkorean,
      title={From KMMLU-Redux to KMMLU-Pro: A Professional Korean Benchmark Suite for LLM Evaluation},
      author={Seokhee Hong and Sunkyoung Kim and Guijin Son and Soyeon Kim and Yeonjung Hong and Jinsik Lee},
      year={2025},
      eprint={2507.08924},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.08924},
}

@misc{kweon2024kormedmcqamultichoicequestionanswering,
      title={KorMedMCQA: Multi-Choice Question Answering Benchmark for Korean Healthcare Professional Licensing Examinations},
      author={Sunjun Kweon and Byungjin Choi and Gyouk Chu and Junyeong Song and Daeun Hyeon and Sujin Gan and Jueon Kim and Minkyu Kim and Rae Woong Park and Edward Choi},
      year={2024},
      eprint={2403.01469},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2403.01469},
}

@misc{wang2024mmluprorobustchallengingmultitask,
      title={MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark},
      author={Yubo Wang and Xueguang Ma and Ge Zhang and Yuansheng Ni and Abhranil Chandra and Shiguang Guo and Weiming Ren and Aaran Arulraj and Xuan He and Ziyan Jiang and Tianle Li and Max Ku and Kai Wang and Alex Zhuang and Rongqi Fan and Xiang Yue and Wenhu Chen},
      year={2024},
      eprint={2406.01574},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.01574},
}

% csatqa: https://huggingface.co/datasets/HAERAE-HUB/csatqa (논문 없음)
```

</details>
