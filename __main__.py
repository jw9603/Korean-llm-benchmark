"""
LLM Evaluation CLI

사용 예시:
    # vLLM (기본)
    python -m llm_evaluation --model Qwen/Qwen3-32B --datasets kmmlu csatqa

    # OpenAI API (자동 감지)
    python -m llm_evaluation --backend api --model gpt-4o --datasets kmmlu

    # OpenAI-compatible API (vLLM 서버)
    python -m llm_evaluation --backend api --model meta-llama/Llama-3.1-8B-Instruct \
        --base-url http://localhost:8000/v1 --datasets kmmlu
"""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from .dataset_configs import get_available_datasets

# .env 파일에서 환경변수 로드 (OPENAI_API_KEY 등)
load_dotenv()


def detect_api_type(model: str, base_url: str | None, api_key: str | None) -> str:
    """
    API 타입 자동 감지.

    Completions API를 시도하고, 실패하면 Chat API로 fallback.

    Returns:
        "openai" (Completions API) 또는 "chat" (Chat API)
    """
    from openai import OpenAI

    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=10.0)

    print(f"API 타입 감지 중: {model}")

    try:
        # Completions API 시도 (echo=True로 logprobs 테스트)
        client.completions.create(
            model=model,
            prompt="test",
            max_tokens=1,
            echo=True,
            logprobs=1,
        )
        print("  → Completions API 지원 (loglikelihood 방식)")
        return "openai"
    except Exception as e:
        error_msg = str(e).lower()
        print(f"  → Completions API 실패: {type(e).__name__}")
        # Completions API를 지원하지 않는 경우 Chat으로 fallback
        # 404, 모델 없음, invalid, 지원 안함 등 모든 경우
        print("  → Chat API 사용 (생성 방식)")
        return "chat"


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Evaluation - Korean Benchmark Evaluation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # vLLM backend (default)
  python -m llm_evaluation --model Qwen/Qwen3-32B --datasets kmmlu csatqa

  # OpenAI API
  python -m llm_evaluation --backend openai --model davinci-002 --datasets kmmlu

  # OpenAI-compatible API (vLLM server)
  python -m llm_evaluation --backend openai --model meta-llama/Llama-3.1-8B \\
      --base-url http://localhost:8000/v1 --datasets kmmlu
        """,
    )

    # 백엔드 선택
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "api", "openai", "chat"],
        default="vllm",
        help="평가 백엔드: vllm(로컬GPU), api(자동감지), openai(Completions), chat(Chat)",
    )

    # 모델 설정
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace 모델 이름 또는 OpenAI 모델 이름",
    )

    # 데이터셋 선택
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help=f"평가할 데이터셋 (기본: 전체). 사용 가능: {get_available_datasets()}",
    )

    # 출력 설정
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./llm_evaluation/results",
        help="결과 저장 디렉토리 (기본: ./llm_evaluation/results)",
    )

    # vLLM 설정
    vllm_group = parser.add_argument_group("vLLM options")
    vllm_group.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="GPU 분산 수 (기본: 1)",
    )
    vllm_group.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU 메모리 사용률 (기본: 0.9)",
    )
    vllm_group.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="최대 시퀀스 길이 (기본: 모델 설정)",
    )

    # OpenAI API 설정
    openai_group = parser.add_argument_group("OpenAI API options")
    openai_group.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="OpenAI-compatible API 베이스 URL (예: http://localhost:8000/v1)",
    )
    openai_group.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API 키 (기본: OPENAI_API_KEY 환경변수)",
    )
    openai_group.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["low", "medium", "high"],
        default=None,
        help="GPT-5/o1/o3 reasoning effort (기본: None=비활성화)",
    )
    openai_group.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="병렬 요청 수 (기본: 1=순차 처리, Chat API 전용)",
    )
    openai_group.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="최대 생성 토큰 수 (기본: 데이터셋 설정값, Chat API 전용)",
    )

    # 테스트용 샘플 수 제한
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="평가할 샘플 수 제한 (테스트용, 기본: 전체)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # API 백엔드 자동 감지
    if args.backend == "api":
        args.backend = detect_api_type(args.model, args.base_url, args.api_key)

    if args.backend == "vllm":
        # vLLM 백엔드
        from .dataset_configs import get_available_datasets
        from .evaluator import (
            Evaluator,
            parse_leaderboard,
            save_results,
            update_leaderboard,
        )

        evaluator = Evaluator(
            model_name=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )

        # 데이터셋별로 평가 실행 및 즉시 저장
        datasets_to_eval = args.datasets if args.datasets else get_available_datasets()
        results = {}
        model_id = args.model.split("/")[-1]

        for dataset_name in datasets_to_eval:
            try:
                result = evaluator.evaluate_dataset(dataset_name, save_details=True)
                results[dataset_name] = result

                # 즉시 결과 저장
                save_results({dataset_name: result}, output_dir, model_id)

                # 리더보드 업데이트
                leaderboard_file = output_dir / "leaderboard.md"
                existing_results = parse_leaderboard(leaderboard_file) if leaderboard_file.exists() else {}
                update_leaderboard(results, output_dir, existing_results)

                print(f"  → 결과 저장됨: {output_dir}/{model_id}_{dataset_name}_details.json")

            except Exception as e:
                print(f"{dataset_name} 평가 중 에러: {e}")
                import traceback
                traceback.print_exc()
                continue

    elif args.backend == "openai":
        # OpenAI Completions API 백엔드 (loglikelihood 방식)
        from .api_evaluator import APIEvaluator
        from .dataset_configs import get_available_datasets
        from .evaluator import (
            parse_leaderboard,
            save_results,
            update_leaderboard,
        )

        evaluator = APIEvaluator(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
        )

        # 데이터셋별로 평가 실행 및 즉시 저장
        datasets_to_eval = args.datasets if args.datasets else get_available_datasets()
        results = {}
        model_id = args.model.split("/")[-1]

        for dataset_name in datasets_to_eval:
            try:
                result = evaluator.evaluate_dataset(dataset_name, save_details=True)
                results[dataset_name] = result

                # 즉시 결과 저장
                save_results({dataset_name: result}, output_dir, model_id)

                # 리더보드 업데이트
                leaderboard_file = output_dir / "leaderboard.md"
                existing_results = parse_leaderboard(leaderboard_file) if leaderboard_file.exists() else {}
                update_leaderboard(results, output_dir, existing_results)

                print(f"  → 결과 저장됨: {output_dir}/{model_id}_{dataset_name}_details.json")

            except Exception as e:
                print(f"{dataset_name} 평가 중 에러: {e}")
                import traceback
                traceback.print_exc()
                continue

    elif args.backend == "chat":
        # OpenAI Chat API 백엔드 (생성 방식 - GPT-4o 등)
        from .chat_evaluator import ChatEvaluator
        from .dataset_configs import get_available_datasets
        from .evaluator import (
            parse_leaderboard,
            save_results,
            update_leaderboard,
        )

        evaluator = ChatEvaluator(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            reasoning_effort=args.reasoning_effort,
            default_max_tokens=args.max_tokens,
        )

        # 데이터셋별로 평가 실행 및 즉시 저장
        datasets_to_eval = args.datasets if args.datasets else get_available_datasets()
        results = {}

        max_concurrency = args.concurrency
        if max_concurrency > 1:
            print(f"병렬 처리 모드: {max_concurrency} 동시 요청")

        for dataset_name in datasets_to_eval:
            try:
                result = evaluator.evaluate_dataset(
                    dataset_name, save_details=True, max_concurrency=max_concurrency,
                    limit=args.limit
                )
                results[dataset_name] = result

                # 즉시 결과 저장
                model_id = evaluator.model_id
                save_results({dataset_name: result}, output_dir, model_id)

                # 리더보드 업데이트
                leaderboard_file = output_dir / "leaderboard.md"
                existing_results = parse_leaderboard(leaderboard_file) if leaderboard_file.exists() else {}
                update_leaderboard(results, output_dir, existing_results, model_id=model_id)

                print(f"  → 결과 저장됨: {output_dir}/{model_id}_{dataset_name}_details.json")

            except Exception as e:
                print(f"{dataset_name} 평가 중 에러: {e}")
                import traceback
                traceback.print_exc()
                continue

    # 결과 요약 출력
    print("\n" + "=" * 50)
    print("평가 완료!")
    print("=" * 50)
    for name, result in results.items():
        print(f"  {name}: {result.score:.4f}")
    print(f"\n결과 저장됨: {output_dir}")


if __name__ == "__main__":
    main()
