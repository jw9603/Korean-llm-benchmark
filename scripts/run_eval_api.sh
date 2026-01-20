#!/bin/bash
# LLM Evaluation Script - API 모델 평가
# Usage: ./scripts/run_eval_api.sh [model_preset]
# 실행 위치: llm_evaluation 폴더 안에서 실행
#
# 지원 모델:
#   1) gpt-5.2      - OpenAI GPT-5.2 (기본값)
#   2) gpt-4o       - OpenAI GPT-4o
#   3) gpt-4o-mini  - OpenAI GPT-4o Mini
#   4) k-exaone     - LG K-EXAONE (Friendly AI)
#   5) custom       - 직접 입력

set -e

# 스크립트 디렉토리 기준으로 .env 파일 로드
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"

if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

# 프로젝트 루트 설정 (PYTHONPATH)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

OUTPUT_DIR="./results"

# 모델 선택 함수
select_model() {
    echo "=============================================="
    echo "LLM Evaluation - API 모델 선택"
    echo "=============================================="
    echo "1) gpt-5.2      - OpenAI GPT-5.2"
    echo "2) gpt-4o       - OpenAI GPT-4o"
    echo "3) gpt-4o-mini  - OpenAI GPT-4o Mini"
    echo "4) k-exaone     - LG K-EXAONE (Friendly AI)"
    echo "5) custom       - 직접 입력"
    echo "=============================================="
    read -p "선택 (1-5, 기본값 1): " choice
    choice=${choice:-1}

    case $choice in
        1|gpt-5.2)
            MODEL="gpt-5.2-2025-12-11"
            BASE_URL=""
            API_KEY="$OPENAI_API_KEY"
            ;;
        2|gpt-4o)
            MODEL="gpt-4o"
            BASE_URL=""
            API_KEY="$OPENAI_API_KEY"
            ;;
        3|gpt-4o-mini)
            MODEL="gpt-4o-mini"
            BASE_URL=""
            API_KEY="$OPENAI_API_KEY"
            ;;
        4|k-exaone)
            MODEL="LGAI-EXAONE/K-EXAONE-236B-A23B"
            BASE_URL="https://api.friendli.ai/serverless/v1"
            API_KEY="$FRIENDLI_API_KEY"
            ;;
        5|custom)
            read -p "모델명: " MODEL
            read -p "Base URL (없으면 Enter): " BASE_URL
            read -p "API Key 환경변수명 (기본값 OPENAI_API_KEY): " API_KEY_ENV
            API_KEY_ENV=${API_KEY_ENV:-OPENAI_API_KEY}
            API_KEY="${!API_KEY_ENV}"
            ;;
        *)
            echo "잘못된 선택입니다."
            exit 1
            ;;
    esac
}

# --dataset, --limit 인자 파싱
DATASETS=""
LIMIT=""
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset|--datasets)
            DATASETS="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL_ARGS[@]}"

# 인자가 있으면 직접 설정, 없으면 대화형 선택
if [ -n "$1" ]; then
    case $1 in
        gpt-5.2|1)
            MODEL="gpt-5.2-2025-12-11"
            BASE_URL=""
            API_KEY="$OPENAI_API_KEY"
            ;;
        gpt-4o|2)
            MODEL="gpt-4o"
            BASE_URL=""
            API_KEY="$OPENAI_API_KEY"
            ;;
        gpt-4o-mini|3)
            MODEL="gpt-4o-mini"
            BASE_URL=""
            API_KEY="$OPENAI_API_KEY"
            ;;
        k-exaone|4)
            MODEL="LGAI-EXAONE/K-EXAONE-236B-A23B"
            BASE_URL="https://api.friendli.ai/serverless/v1"
            API_KEY="$FRIENDLI_API_KEY"
            ;;
        *)
            # 직접 모델명 지정
            MODEL="$1"
            BASE_URL="${2:-}"
            API_KEY_ENV="${3:-OPENAI_API_KEY}"
            API_KEY="${!API_KEY_ENV}"
            ;;
    esac
else
    select_model
fi

echo "=============================================="
echo "LLM Evaluation - API"
echo "=============================================="
echo "Model: $MODEL"
echo "Backend: api (자동 감지)"
if [ -n "$BASE_URL" ]; then
    echo "Base URL: $BASE_URL"
fi
echo "Output: $OUTPUT_DIR"
echo "=============================================="

# API 키 확인
if [ -z "$API_KEY" ]; then
    echo "ERROR: API 키가 설정되지 않았습니다."
    echo ".env 파일에 해당 API 키를 설정하세요."
    exit 1
fi

# Run evaluation with auto-detection
CMD="python -m llm_evaluation --backend api --model \"$MODEL\" --api-key \"$API_KEY\" --output-dir \"$OUTPUT_DIR\""

if [ -n "$BASE_URL" ]; then
    CMD="$CMD --base-url \"$BASE_URL\""
fi

if [ -n "$DATASETS" ]; then
    CMD="$CMD --datasets \"$DATASETS\""
fi

if [ -n "$LIMIT" ]; then
    CMD="$CMD --limit $LIMIT"
fi

eval $CMD

echo "=============================================="
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
