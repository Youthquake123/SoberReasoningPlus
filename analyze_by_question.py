import os
import argparse
import logging
import pandas as pd
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# 配置
EXPECTED_TEMPS = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
EXPECTED_TOP_PS = [0.7, 0.8, 0.9, 0.95, 1.0]

BENCHMARK_SOBER_STEPS = {
    "aime24": list(range(16)),
    # "aime25": list(range(16)),
    # "amc23": list(range(16)),
    # "math_500": list(range(3)),
    # "minerva": list(range(3)),
    # "olympiadbench": list(range(3))
}

# 模型名到 HuggingFace 模型 ID 的映射
MODEL_TO_HF = {
    "Qwen-Qwen3-1.7B": "Qwen/Qwen3-1.7B",
    # "Qwen-Qwen3-4B": "Qwen/Qwen3-4B",
    # "Qwen-Qwen3-8B": "Qwen/Qwen3-8B",
    # "Qwen-Qwen3-14B": "Qwen/Qwen3-14B",
    # "deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    # "deepseek-ai-DeepSeek-R1-Distill-Qwen-7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    # "deepseek-ai-DeepSeek-R1-Distill-Qwen-14B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    # "deepseek-ai-DeepSeek-R1-0528-Qwen3-8B": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
}


def get_model_short_name(model_dir_name):
    """从目录名提取模型短名"""
    # 去掉后缀如 -bfloat16-34816
    parts = model_dir_name.split("-bfloat16")[0]
    return parts


def parse_experiment_dir(dir_name):
    """解析实验目录名"""
    parts = dir_name.split("-")
    if len(parts) < 4:
        return None
    try:
        sober_steps = int(parts[0])
        top_p = float(parts[2])
        benchmark = parts[3]
        return {
            "sober_steps": sober_steps,
            "top_p": top_p,
            "benchmark": benchmark,
        }
    except (ValueError, IndexError):
        return None


def load_tokenizer(model_name):
    """加载模型的 tokenizer"""
    short_name = get_model_short_name(model_name)
    hf_model = MODEL_TO_HF.get(short_name)

    if hf_model is None:
        logging.warning(f"未找到模型 {short_name} 的 HuggingFace ID，尝试直接使用")
        hf_model = short_name.replace("-", "/", 1)

    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
        logging.info(f"加载 tokenizer: {hf_model}")
        return tokenizer
    except Exception as e:
        logging.error(f"无法加载 tokenizer {hf_model}: {e}")
        return None


def get_token_length(text, tokenizer):
    """计算文本的 token 长度"""
    if tokenizer is None or text is None or pd.isna(text):
        return 0
    try:
        if isinstance(text, list):
            text = text[0] if text else ""
        return len(tokenizer.encode(str(text)))
    except:
        return 0


def extract_metric(metrics_value):
    """从 metrics 列提取 extractive_match 值"""
    if isinstance(metrics_value, dict):
        return metrics_value.get('extractive_match', 0.0)
    elif isinstance(metrics_value, str):
        import ast
        try:
            d = ast.literal_eval(metrics_value)
            return d.get('extractive_match', 0.0)
        except:
            return 0.0
    return 0.0


def collect_data_for_benchmark(results_dir, benchmark, models_filter=None):
    """
    收集指定 benchmark 的所有数据
    返回: {(model, temp, top_p, question_id): [(is_correct, token_length), ...]}
    """
    results_dir = Path(results_dir)
    data = defaultdict(list)

    # 遍历所有 temp 目录
    for temp_dir in results_dir.iterdir():
        if not temp_dir.is_dir() or not temp_dir.name.startswith("temp_"):
            continue

        try:
            temp = float(temp_dir.name.replace("temp_", ""))
        except ValueError:
            continue

        if temp not in EXPECTED_TEMPS:
            continue

        # 遍历模型目录
        for model_dir in temp_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name

            # 过滤模型
            if models_filter and not any(m in model_name for m in models_filter):
                continue

            # 遍历实验目录
            for exp_dir in model_dir.iterdir():
                if not exp_dir.is_dir():
                    continue

                config = parse_experiment_dir(exp_dir.name)
                if config is None or config["benchmark"] != benchmark:
                    continue

                top_p = config["top_p"]
                if top_p not in EXPECTED_TOP_PS:
                    continue

                sober_steps = config["sober_steps"]

                # 查找 parquet 文件
                details_dir = exp_dir / "details"
                if not details_dir.exists():
                    continue

                parquet_files = list(details_dir.glob("**/*.parquet"))
                if not parquet_files:
                    continue

                # 读取 parquet
                for pf in parquet_files:
                    try:
                        df = pd.read_parquet(pf)

                        for idx, row in df.iterrows():
                            question_id = idx  # 行索引作为题目 ID
                            is_correct = extract_metric(row.get('metrics', {}))
                            prediction = row.get('predictions', '')

                            key = (model_name, temp, top_p, question_id)
                            data[key].append({
                                'sober_steps': sober_steps,
                                'is_correct': is_correct,
                                'prediction': prediction
                            })
                    except Exception as e:
                        logging.error(f"读取 {pf} 失败: {e}")

    return data


def analyze_benchmark(results_dir, benchmark, output_dir, models_filter=None, tokenizers=None):
    """分析单个 benchmark 并输出 CSV"""
    logging.info(f"分析 benchmark: {benchmark}")

    data = collect_data_for_benchmark(results_dir, benchmark, models_filter)

    if not data:
        logging.warning(f"未找到 {benchmark} 的数据")
        return

    # 统计结果
    results = []

    # 按 (model, temp, top_p, question_id) 分组
    for (model_name, temp, top_p, question_id), records in data.items():
        total_seeds = len(records)
        correct_count = sum(1 for r in records if r['is_correct'] >= 0.5)
        wrong_count = total_seeds - correct_count
        accuracy = correct_count / total_seeds if total_seeds > 0 else 0

        # 计算 token 长度
        tokenizer = tokenizers.get(model_name) if tokenizers else None

        correct_lengths = []
        wrong_lengths = []
        all_lengths = []

        for r in records:
            length = get_token_length(r['prediction'], tokenizer)
            all_lengths.append(length)
            if r['is_correct'] >= 0.5:
                correct_lengths.append(length)
            else:
                wrong_lengths.append(length)

        avg_len_correct = sum(correct_lengths) / len(correct_lengths) if correct_lengths else 0
        avg_len_wrong = sum(wrong_lengths) / len(wrong_lengths) if wrong_lengths else 0
        avg_len_all = sum(all_lengths) / len(all_lengths) if all_lengths else 0

        results.append({
            'model': get_model_short_name(model_name),
            'temp': temp,
            'top_p': top_p,
            'question_id': question_id,
            'total_seeds': total_seeds,
            'correct_count': correct_count,
            'wrong_count': wrong_count,
            'accuracy': round(accuracy, 4),
            'avg_len_correct': round(avg_len_correct, 2),
            'avg_len_wrong': round(avg_len_wrong, 2),
            'avg_len_all': round(avg_len_all, 2)
        })

    # 排序
    results.sort(key=lambda x: (x['model'], x['temp'], x['top_p'], x['question_id']))

    # 输出 CSV
    df = pd.DataFrame(results)
    output_path = Path(output_dir) / f"{benchmark}_stats.csv"
    df.to_csv(output_path, index=False)
    logging.info(f"输出: {output_path} ({len(results)} 行)")

    return df


def main():
    parser = argparse.ArgumentParser(description="按题目统计实验结果")
    parser.add_argument(
        "--results_dir", type=str,
        default="/home/zax/SoberReasoningPlus/results",
        help="结果目录路径"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/home/zax/SoberReasoningPlus/stats",
        help="输出目录路径"
    )
    parser.add_argument(
        "--benchmarks", type=str, nargs="+",
        default=["aime24", "aime25", "amc23", "math_500", "minerva", "olympiadbench"],
        help="要分析的 benchmarks"
    )
    parser.add_argument(
        "--models", type=str, nargs="+",
        default=None,
        help="要分析的模型（部分匹配），默认全部"
    )
    parser.add_argument(
        "--no_tokenizer", action="store_true",
        help="不使用 tokenizer 计算长度（用字符数代替）"
    )

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载 tokenizers
    tokenizers = {}
    if not args.no_tokenizer:
        logging.info("加载 tokenizers...")
        results_dir = Path(args.results_dir)

        # 找出所有模型
        models_found = set()
        for temp_dir in results_dir.iterdir():
            if temp_dir.is_dir() and temp_dir.name.startswith("temp_"):
                for model_dir in temp_dir.iterdir():
                    if model_dir.is_dir():
                        if args.models is None or any(m in model_dir.name for m in args.models):
                            models_found.add(model_dir.name)

        for model_name in models_found:
            tokenizer = load_tokenizer(model_name)
            if tokenizer:
                tokenizers[model_name] = tokenizer

    # 分析每个 benchmark
    for benchmark in args.benchmarks:
        analyze_benchmark(
            args.results_dir,
            benchmark,
            args.output_dir,
            args.models,
            tokenizers
        )

    logging.info("完成!")


if __name__ == "__main__":
    main()
