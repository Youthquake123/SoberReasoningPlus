import os
import argparse
import logging
import pandas as pd
from pathlib import Path
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# 预期的配置
EXPECTED_TEMPS = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
EXPECTED_TOP_PS = [0.7, 0.8, 0.9, 0.95, 1.0]

BENCHMARK_SOBER_STEPS = {
    "aime24": list(range(16)),
    "aime25": list(range(16)),
    "amc23": list(range(16)),
    "math_500": list(range(3)),
    "minerva": list(range(3)),
    "olympiadbench": list(range(3))
}

EXPECTED_PER_TEMP = sum(len(steps) * len(EXPECTED_TOP_PS) for steps in BENCHMARK_SOBER_STEPS.values())  # 285


def parse_experiment_dir(dir_name):
    """解析实验目录名"""
    parts = dir_name.split("-")
    if len(parts) < 4:
        return None
    try:
        sober_steps = int(parts[0])
        top_p = float(parts[2])
        benchmark = parts[3]
        return {"sober_steps": sober_steps, "top_p": top_p, "benchmark": benchmark}
    except (ValueError, IndexError):
        return None


def check_temp_complete(temp_dir):
    """检查某个 temp 目录下某个模型是否完成"""
    model_status = {}

    for model_dir in temp_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        completed = 0

        for exp_dir in model_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            config = parse_experiment_dir(exp_dir.name)
            if config is None:
                continue

            results_subdir = exp_dir / "results"
            if results_subdir.exists():
                json_files = list(results_subdir.glob("**/*.json"))
                if len(json_files) > 0:
                    completed += 1

        model_status[model_name] = {
            "completed": completed,
            "expected": EXPECTED_PER_TEMP,
            "is_complete": completed == EXPECTED_PER_TEMP,
            "path": model_dir
        }

    return model_status


def convert_parquet_to_csv(parquet_path):
    """转换单个 parquet 文件"""
    try:
        df = pd.read_parquet(parquet_path)
        csv_path = parquet_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        return True
    except Exception as e:
        logging.error(f"Error converting {parquet_path}: {e}")
        return False


def convert_model_dir(model_dir):
    """转换某个模型目录下的所有 parquet 文件"""
    parquet_files = list(model_dir.glob("**/*.parquet"))

    if not parquet_files:
        logging.warning(f"No parquet files in {model_dir.name}")
        return 0, 0

    success = 0
    for pf in parquet_files:
        # 跳过已经转换过的
        csv_path = pf.with_suffix('.csv')
        if csv_path.exists():
            success += 1
            continue
        if convert_parquet_to_csv(pf):
            success += 1

    return success, len(parquet_files)


def main():
    parser = argparse.ArgumentParser(description="转换已完成实验的 parquet 到 csv")
    parser.add_argument(
        "--results_dir", type=str,
        default="/home/zax/SoberReasoningPlus/results",
        help="结果目录路径"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="只检查不转换"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="只转换指定模型"
    )

    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    logging.info(f"扫描目录: {results_dir}")

    # 遍历每个 temp 目录
    for temp_dir in sorted(results_dir.iterdir()):
        if not temp_dir.is_dir() or not temp_dir.name.startswith("temp_"):
            continue

        temp_name = temp_dir.name
        logging.info(f"\n{'='*60}")
        logging.info(f"检查 {temp_name}")
        logging.info(f"{'='*60}")

        model_status = check_temp_complete(temp_dir)

        for model_name, status in sorted(model_status.items()):
            # 筛选模型
            if args.model and args.model not in model_name:
                continue

            completed = status["completed"]
            expected = status["expected"]
            is_complete = status["is_complete"]
            model_path = status["path"]

            status_icon = "✓" if is_complete else "✗"
            logging.info(f"  {model_name}: {completed}/{expected} {status_icon}")

            if is_complete:
                if args.dry_run:
                    logging.info(f"    [DRY RUN] 会转换此目录的 parquet 文件")
                else:
                    logging.info(f"    转换 parquet -> csv ...")
                    success, total = convert_model_dir(model_path)
                    logging.info(f"    完成: {success}/{total} 文件")


if __name__ == "__main__":
    main()
