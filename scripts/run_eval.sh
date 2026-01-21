#!/bin/bash
# LLM Evaluation Script
# Usage: ./scripts/run_eval.sh [options]
#
# 모드:
#   1. 전체 평가: ./scripts/run_eval.sh --model <model>
#   2. 단일 데이터셋: ./scripts/run_eval.sh --dataset <dataset> --model <model>
#
# 데이터셋:
#   kmmlu, kmmlu_pro, haerae, click, csatqa, hrm8k, hrm8k_mmmlu,
#   kobalt, ko_musr_mm, ko_musr_op, ko_musr_ta, kormedmcqa
#
# 모델 프리셋:
#   gpt-5.2, gpt-4o, gpt-4o-mini, k-exaone
#   또는 HuggingFace 모델 경로 (Qwen/Qwen3-32B 등)
#
# 예시:
#   ./scripts/run_eval.sh --model gpt-4o                     # 전체 평가
#   ./scripts/run_eval.sh --dataset kmmlu --model gpt-4o     # 단일 데이터셋
#   ./scripts/run_eval.sh --dataset hrm8k --model gpt-4o --limit 100

set -e

# 기본값
DATASET=""
MODEL=""
BACKEND=""
BASE_URL=""
API_KEY=""
TP_SIZE=1
MAX_MODEL_LEN=32768
LIMIT=""
CONCURRENCY=""
OUTPUT_DIR="./results"
EXTRA_ARGS=""

# 사용 가능한 데이터셋
DATASETS=(
    "kmmlu"
    "kmmlu_pro"
    "haerae"
    "click"
    "csatqa"
    "hrm8k"
    "hrm8k_mmmlu"
    "kobalt"
    "ko_musr_mm"
    "ko_musr_op"
    "ko_musr_ta"
    "kormedmcqa"
    "mmlu_pro(zero)"
    "mmlu_pro(five)"
)

# API 프리셋 목록
API_PRESETS=("gpt-5.2" "gpt-4o" "gpt-4o-mini" "k-exaone")

# 도움말
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "필수 옵션:"
    echo "  --model, -m <model>      모델 이름 또는 프리셋"
    echo ""
    echo "선택 옵션:"
    echo "  --dataset, -d <dataset>  단일 데이터셋 (없으면 전체 평가)"
    echo "  --limit, -l <N>          샘플 수 제한"
    echo "  --concurrency, -c <N>    동시 요청 수 (API, 기본값: 1)"
    echo "  --tensor-parallel-size <N>  GPU 병렬 수 (vLLM, 기본값: 1)"
    echo "  --max-model-len <N>      최대 컨텍스트 길이 (vLLM, 기본값: 16384)"
    echo "  --output-dir, -o <dir>   결과 저장 디렉토리 (기본값: ./results)"
    echo "  --base-url <url>         커스텀 API URL"
    echo "  --api-key <key>          API 키 (환경변수 우선)"
    echo "  --help, -h               도움말"
    echo ""
    echo "데이터셋:"
    printf "  %s\n" "${DATASETS[@]}"
    echo ""
    echo "모델 프리셋:"
    echo "  gpt-5.2      - OpenAI GPT-5.2"
    echo "  gpt-4o       - OpenAI GPT-4o"
    echo "  gpt-4o-mini  - OpenAI GPT-4o Mini"
    echo "  k-exaone     - LG K-EXAONE"
    echo ""
    echo "HuggingFace 모델 (vLLM):"
    echo "  Qwen/Qwen3-32B, meta-llama/Llama-3.1-8B-Instruct, ..."
    echo ""
    echo "예시:"
    echo "  $0 --model gpt-4o                           # 전체 평가"
    echo "  $0 --dataset kmmlu --model gpt-4o           # 단일 데이터셋"
    echo "  $0 --dataset hrm8k --model gpt-4o --limit 100"
    echo "  $0 --model Qwen/Qwen3-32B --tensor-parallel-size 2"
    exit 0
}

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --model|-m)
            MODEL="$2"
            shift 2
            ;;
        --dataset|-d)
            DATASET="$2"
            shift 2
            ;;
        --limit|-l)
            LIMIT="$2"
            shift 2
            ;;
        --concurrency|-c)
            CONCURRENCY="$2"
            shift 2
            ;;
        --tensor-parallel-size)
            TP_SIZE="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --output-dir|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --base-url)
            BASE_URL="$2"
            shift 2
            ;;
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# 모델 필수 확인
if [ -z "$MODEL" ]; then
    echo "ERROR: 모델을 지정해주세요. (--model)"
    echo ""
    show_help
fi

# API 프리셋인지 확인
is_api_preset() {
    local model="$1"
    for preset in "${API_PRESETS[@]}"; do
        if [ "$preset" == "$model" ]; then
            return 0
        fi
    done
    return 1
}

# 데이터셋 유효성 검사 (지정된 경우)
if [ -n "$DATASET" ]; then
    valid_dataset=false
    for ds in "${DATASETS[@]}"; do
        if [ "$ds" == "$DATASET" ]; then
            valid_dataset=true
            break
        fi
    done

    if [ "$valid_dataset" = false ]; then
        echo "ERROR: 알 수 없는 데이터셋: $DATASET"
        echo "사용 가능한 데이터셋:"
        printf "  %s\n" "${DATASETS[@]}"
        exit 1
    fi
fi

# .env 파일 로드
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"

if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

# 프로젝트 루트 설정
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 백엔드 및 모델 설정
MODEL_INPUT="$MODEL"
if is_api_preset "$MODEL_INPUT"; then
    BACKEND="api"
    case $MODEL_INPUT in
        gpt-5.2)
            MODEL="gpt-5.2-2025-12-11"
            [ -z "$API_KEY" ] && API_KEY="$OPENAI_API_KEY"
            ;;
        gpt-4o)
            MODEL="gpt-4o"
            [ -z "$API_KEY" ] && API_KEY="$OPENAI_API_KEY"
            ;;
        gpt-4o-mini)
            MODEL="gpt-4o-mini"
            [ -z "$API_KEY" ] && API_KEY="$OPENAI_API_KEY"
            ;;
        k-exaone)
            MODEL="LGAI-EXAONE/K-EXAONE-236B-A23B"
            BASE_URL="https://api.friendli.ai/serverless/v1"
            [ -z "$API_KEY" ] && API_KEY="$FRIENDLI_API_KEY"
            ;;
    esac
elif [ -n "$BASE_URL" ]; then
    # 커스텀 API URL 지정된 경우
    BACKEND="chat"
else
    # HuggingFace 모델 → vLLM 백엔드
    BACKEND="vllm"
fi

# 헤더 출력
echo "=============================================="
echo "LLM Evaluation - Korean Benchmarks"
echo "=============================================="
if [ -n "$DATASET" ]; then
    echo "Dataset: $DATASET"
else
    echo "Dataset: 전체 (${#DATASETS[@]}개)"
fi
echo "Model: $MODEL"
echo "Backend: $BACKEND"
if [ -n "$BASE_URL" ]; then
    echo "Base URL: $BASE_URL"
fi
if [ "$BACKEND" == "vllm" ]; then
    echo "Tensor Parallel Size: $TP_SIZE"
    echo "Max Model Len: $MAX_MODEL_LEN"
fi
if [ -n "$LIMIT" ]; then
    echo "Limit: $LIMIT"
fi
if [ -n "$CONCURRENCY" ]; then
    echo "Concurrency: $CONCURRENCY"
fi
echo "Output: $OUTPUT_DIR"
echo "=============================================="

# 공통 인자 빌드
COMMON_ARGS="--output-dir $OUTPUT_DIR"

if [ -n "$DATASET" ]; then
    COMMON_ARGS="$COMMON_ARGS --datasets $DATASET"
fi

if [ -n "$LIMIT" ]; then
    COMMON_ARGS="$COMMON_ARGS --limit $LIMIT"
fi

if [ -n "$CONCURRENCY" ]; then
    COMMON_ARGS="$COMMON_ARGS --max-concurrency $CONCURRENCY"
fi

# 백엔드별 실행
if [ "$BACKEND" == "vllm" ]; then
    # vLLM 백엔드 (HuggingFace 모델)
    python -m llm_evaluation \
        --backend vllm \
        --model "$MODEL" \
        --tensor-parallel-size "$TP_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        $COMMON_ARGS \
        $EXTRA_ARGS

elif [ "$BACKEND" == "chat" ]; then
    # Chat API 백엔드 (커스텀 URL)
    if [ -z "$API_KEY" ]; then
        echo "ERROR: API 키가 설정되지 않았습니다. (--api-key 또는 환경변수)"
        exit 1
    fi
    python -m llm_evaluation \
        --backend chat \
        --model "$MODEL" \
        --base-url "$BASE_URL" \
        --api-key "$API_KEY" \
        $COMMON_ARGS \
        $EXTRA_ARGS

elif [ -n "$BASE_URL" ]; then
    # API 백엔드 (커스텀 URL)
    if [ -z "$API_KEY" ]; then
        echo "ERROR: API 키가 설정되지 않았습니다. (--api-key 또는 환경변수)"
        exit 1
    fi
    python -m llm_evaluation \
        --backend api \
        --model "$MODEL" \
        --base-url "$BASE_URL" \
        --api-key "$API_KEY" \
        $COMMON_ARGS \
        $EXTRA_ARGS

else
    # API 백엔드 (OpenAI)
    if [ -z "$API_KEY" ]; then
        echo "ERROR: API 키가 설정되지 않았습니다. (--api-key 또는 환경변수)"
        exit 1
    fi
    python -m llm_evaluation \
        --backend api \
        --model "$MODEL" \
        --api-key "$API_KEY" \
        $COMMON_ARGS \
        $EXTRA_ARGS
fi

echo "=============================================="
echo "Evaluation complete!"
if [ -n "$DATASET" ]; then
    echo "Dataset: $DATASET"
fi
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
