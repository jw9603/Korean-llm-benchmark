#!/usr/bin/env python3
"""
LLM Evaluation - Streamlit Web UI

한국어 벤치마크 평가 웹 인터페이스입니다.
HuggingFace 모델을 입력하고, 벤치마크 데이터셋을 선택하여 평가를 실행합니다.

실행 방법:
    cd llm_evaluation/webapp
    streamlit run app.py
"""
import json
import subprocess
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# 상위 디렉토리를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def get_available_datasets() -> list[str]:
    """사용 가능한 데이터셋 목록을 반환합니다. (vLLM 의존성 없이)"""
    configs_dir = Path(__file__).resolve().parent.parent / "datasets" / "configs"
    return sorted([
        f.stem for f in configs_dir.glob("*.yaml")
        if f.is_file()
    ])

# 페이지 설정
st.set_page_config(
    page_title="LLM Evaluation",
    page_icon="🎯",
    layout="wide",
)

# 상수 정의
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # default-package
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# 데이터셋 그룹 정의
DATASET_GROUPS = {
    "HRM8K": ["hrm8k", "hrm8k_mmmlu"],
    "Ko-MuSR": ["ko_musr_mm", "ko_musr_op", "ko_musr_ta"],
}

# 데이터셋 설명
DATASET_DESCRIPTIONS = {
    "click": "한국 문화 지식",
    "csatqa": "수능 문제 (대학수학능력시험)",
    "haerae": "한국 문화/사회 상식",
    "kmmlu": "한국어 MMLU (45개 분야)",
    "kmmlu_pro": "KMMLU 고급 버전",
    "kobalt": "한국어 이해력 테스트",
    "hrm8k": "한국어 수학 문제",
    "hrm8k_mmmlu": "번역된 MMMLU 수학 문제",
    "ko_musr_mm": "Ko-MuSR Murder Mysteries",
    "ko_musr_op": "Ko-MuSR Object Placements",
    "ko_musr_ta": "Ko-MuSR Team Allocation",
}


def load_existing_results() -> dict:
    """
    기존 평가 결과를 로드합니다.

    Returns:
        dict: {model_id: {dataset: score, ...}, ...}
    """
    results = {}

    if not RESULTS_DIR.exists():
        return results

    for model_dir in RESULTS_DIR.iterdir():
        if not model_dir.is_dir():
            continue

        results_file = model_dir / "results.json"
        if not results_file.exists():
            continue

        try:
            with open(results_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            model_id = model_dir.name
            results[model_id] = {}

            for ds_name, ds_data in data.items():
                if isinstance(ds_data, dict) and "score" in ds_data:
                    results[model_id][ds_name] = ds_data["score"]
        except (json.JSONDecodeError, IOError):
            continue

    return results


def create_leaderboard_df(results: dict, dataset: str = None) -> pd.DataFrame:
    """
    리더보드 데이터프레임 생성

    Args:
        results: 평가 결과
        dataset: 특정 데이터셋만 표시 (None이면 전체 평균)

    Returns:
        pd.DataFrame: 리더보드 데이터프레임
    """
    if not results:
        return pd.DataFrame()

    rows = []

    for model_id, model_results in results.items():
        if dataset:
            # 특정 데이터셋의 점수
            if dataset in DATASET_GROUPS:
                # 그룹 평균
                group_scores = []
                for member in DATASET_GROUPS[dataset]:
                    if member in model_results:
                        group_scores.append(model_results[member])
                if group_scores:
                    score = sum(group_scores) / len(group_scores)
                    rows.append({"모델": model_id, "점수": score})
            elif dataset in model_results:
                rows.append({"모델": model_id, "점수": model_results[dataset]})
        else:
            # 전체 평균
            scores = list(model_results.values())
            if scores:
                avg = sum(scores) / len(scores)
                rows.append({"모델": model_id, "평균 점수": avg})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    score_col = "점수" if dataset else "평균 점수"
    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # 1부터 시작

    return df


def run_evaluation(model_name: str, datasets: list, gpu_settings: dict):
    """
    평가 실행 (subprocess로 실행)

    Args:
        model_name: HuggingFace 모델 이름
        datasets: 평가할 데이터셋 리스트
        gpu_settings: GPU 설정

    Returns:
        subprocess.Popen: 실행 중인 프로세스
    """
    cmd = [
        sys.executable, "-m", "llm_evaluation.main",
        "--model", model_name,
        "--datasets", *datasets,
        "--tensor-parallel-size", str(gpu_settings.get("tp", 1)),
        "--gpu-memory-utilization", str(gpu_settings.get("memory", 0.9)),
    ]

    if gpu_settings.get("max_model_len"):
        cmd.extend(["--max-model-len", str(gpu_settings["max_model_len"])])

    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(BASE_DIR),
    )


def main():
    st.title("🎯 LLM 평가 시스템")
    st.markdown("HuggingFace 모델을 한국어 벤치마크에서 평가합니다.")

    # 사이드바: 설정
    with st.sidebar:
        st.header("⚙️ 설정")

        # GPU 설정
        st.subheader("GPU 설정")
        tensor_parallel = st.number_input(
            "Tensor Parallel Size",
            min_value=1,
            max_value=8,
            value=1,
            help="사용할 GPU 개수"
        )
        gpu_memory = st.slider(
            "GPU 메모리 사용률",
            min_value=0.5,
            max_value=1.0,
            value=0.9,
            step=0.05,
        )
        max_model_len = st.number_input(
            "Max Model Length (0=자동)",
            min_value=0,
            max_value=131072,
            value=0,
            help="KV cache 메모리 부족 시 줄여서 사용 (예: 16384)"
        )

        st.divider()

        # 결과 디렉토리 정보
        st.subheader("결과 저장 위치")
        st.code(str(RESULTS_DIR))

    # 탭 구성
    tab1, tab2 = st.tabs(["🚀 평가 실행", "🏆 리더보드"])

    # 평가 실행 탭
    with tab1:
        st.header("모델 평가")

        # 모델 입력
        model_name = st.text_input(
            "HuggingFace 모델 이름",
            placeholder="예: meta-llama/Llama-3.1-8B-Instruct",
            help="HuggingFace Hub의 모델 이름 또는 로컬 경로"
        )

        # 데이터셋 선택
        st.subheader("벤치마크 데이터셋 선택")

        available_datasets = get_available_datasets()

        # 세션 상태 초기화
        if "selected_datasets" not in st.session_state:
            st.session_state.selected_datasets = set()

        # 전체 선택/해제 버튼
        col1, col2 = st.columns(2)
        with col1:
            if st.button("전체 선택"):
                st.session_state.selected_datasets = set(available_datasets)
                st.rerun()
        with col2:
            if st.button("전체 해제"):
                st.session_state.selected_datasets = set()
                st.rerun()

        # 그룹별로 표시
        grouped_datasets = set()
        for members in DATASET_GROUPS.values():
            grouped_datasets.update(members)

        general_datasets = [ds for ds in available_datasets if ds not in grouped_datasets]

        cols = st.columns(3)

        with cols[0]:
            st.markdown("**일반 벤치마크**")
            for ds in general_datasets:
                desc = DATASET_DESCRIPTIONS.get(ds, ds)
                checked = ds in st.session_state.selected_datasets
                if st.checkbox(f"{ds}", value=checked, key=f"cb_{ds}", help=desc):
                    st.session_state.selected_datasets.add(ds)
                elif ds in st.session_state.selected_datasets:
                    st.session_state.selected_datasets.discard(ds)

        with cols[1]:
            st.markdown("**HRM8K (한국어 수학)**")
            for ds in DATASET_GROUPS.get("HRM8K", []):
                if ds in available_datasets:
                    desc = DATASET_DESCRIPTIONS.get(ds, ds)
                    checked = ds in st.session_state.selected_datasets
                    if st.checkbox(f"{ds}", value=checked, key=f"cb_{ds}", help=desc):
                        st.session_state.selected_datasets.add(ds)
                    elif ds in st.session_state.selected_datasets:
                        st.session_state.selected_datasets.discard(ds)

        with cols[2]:
            st.markdown("**Ko-MuSR (다단계 추론)**")
            for ds in DATASET_GROUPS.get("Ko-MuSR", []):
                if ds in available_datasets:
                    desc = DATASET_DESCRIPTIONS.get(ds, ds)
                    checked = ds in st.session_state.selected_datasets
                    if st.checkbox(f"{ds}", value=checked, key=f"cb_{ds}", help=desc):
                        st.session_state.selected_datasets.add(ds)
                    elif ds in st.session_state.selected_datasets:
                        st.session_state.selected_datasets.discard(ds)

        st.divider()

        # 선택된 데이터셋 표시
        selected_list = list(st.session_state.selected_datasets)
        if selected_list:
            st.info(f"선택된 데이터셋: {', '.join(sorted(selected_list))}")

        # 평가 실행 버튼
        if st.button("🚀 평가 시작", type="primary", use_container_width=True):
            if not model_name:
                st.error("모델 이름을 입력해주세요.")
            elif not selected_list:
                st.error("최소 하나의 데이터셋을 선택해주세요.")
            else:
                gpu_settings = {
                    "tp": tensor_parallel,
                    "memory": gpu_memory,
                    "max_model_len": max_model_len if max_model_len > 0 else None,
                }

                # 실행 정보 표시
                st.info(f"**모델**: {model_name}")

                # subprocess로 실행
                with st.spinner("평가 실행 중... (시간이 오래 걸릴 수 있습니다)"):
                    process = run_evaluation(
                        model_name,
                        selected_list,
                        gpu_settings
                    )

                    # 출력 표시
                    output_container = st.empty()
                    output_text = ""

                    for line in iter(process.stdout.readline, ""):
                        output_text += line
                        output_container.code(output_text[-10000:], language="text")  # 마지막 10000자만 표시

                    process.wait()

                    if process.returncode == 0:
                        st.success("평가 완료!")
                        st.balloons()
                    else:
                        st.error(f"평가 실패 (exit code: {process.returncode})")

    # 리더보드 탭
    with tab2:
        st.header("리더보드")

        # 새로고침 버튼
        if st.button("🔄 결과 새로고침", key="refresh_leaderboard"):
            st.rerun()

        # 결과 로드
        results = load_existing_results()

        if not results:
            st.warning("아직 평가 결과가 없습니다. 먼저 모델을 평가해주세요.")
        else:
            # 전체 결과 테이블
            st.subheader("📊 전체 결과")

            # 표시할 데이터셋 (그룹 포함)
            grouped_datasets = set()
            for members in DATASET_GROUPS.values():
                grouped_datasets.update(members)

            individual_datasets = [
                ds for ds in get_available_datasets()
                if ds not in grouped_datasets
            ]

            # 데이터프레임 생성
            rows = []
            for model_id, model_results in results.items():
                row = {"모델": model_id}

                # 개별 데이터셋
                for ds in individual_datasets:
                    row[ds] = model_results.get(ds)

                # HRM8K 그룹 평균
                hrm_scores = [model_results.get(m) for m in DATASET_GROUPS["HRM8K"] if model_results.get(m)]
                row["HRM8K"] = sum(hrm_scores) / len(hrm_scores) if hrm_scores else None

                # Ko-MuSR 그룹 평균
                musr_scores = [model_results.get(m) for m in DATASET_GROUPS["Ko-MuSR"] if model_results.get(m)]
                row["Ko-MuSR"] = sum(musr_scores) / len(musr_scores) if musr_scores else None

                # 전체 평균
                all_scores = [v for v in row.values() if isinstance(v, (int, float))]
                row["평균"] = sum(all_scores) / len(all_scores) if all_scores else None

                rows.append(row)

            df = pd.DataFrame(rows)

            # 평균 기준 내림차순 정렬
            df = df.sort_values("평균", ascending=False).reset_index(drop=True)
            df.index = df.index + 1  # 순위 1부터

            # 컬럼 순서 정리
            cols = ["모델"] + individual_datasets + ["HRM8K", "Ko-MuSR", "평균"]
            df = df[[c for c in cols if c in df.columns]]

            # 스타일 적용
            def highlight_best(s):
                if s.dtype == 'float64':
                    is_max = s == s.max()
                    return ['background-color: #2e7d32; color: white' if v else '' for v in is_max]
                return ['' for _ in s]

            styled_df = df.style.format(
                {col: "{:.4f}" for col in df.columns if col != "모델"},
                na_rep="-"
            ).apply(highlight_best)

            st.dataframe(styled_df, use_container_width=True, height=400)

            st.divider()

            # 데이터셋별 상세 보기
            st.subheader("📈 데이터셋별 순위")

            all_views = individual_datasets + ["HRM8K", "Ko-MuSR"]
            selected_dataset = st.selectbox("데이터셋 선택", all_views)

            if selected_dataset:
                ds_df = create_leaderboard_df(results, selected_dataset)
                if not ds_df.empty:
                    st.dataframe(ds_df, use_container_width=True)


if __name__ == "__main__":
    main()