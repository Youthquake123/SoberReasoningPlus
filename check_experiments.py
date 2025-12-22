import os
import argparse
from pathlib import Path
from collections import defaultdict

# 预期的配置
EXPECTED_TEMPS = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
EXPECTED_TOP_PS = [0.7, 0.8, 0.9, 0.95, 1.0]

# 不同 benchmark 有不同的 sober_steps 范围
# aime24, aime25, amc23: 16 seeds -> sober_steps 0-15
# math_500, minerva, olympiadbench: 3 seeds -> sober_steps 0-2
BENCHMARK_SOBER_STEPS = {
    "aime24": list(range(16)),      # 0-15
    "aime25": list(range(16)),      # 0-15
    "amc23": list(range(16)),       # 0-15
    "math_500": list(range(3)),     # 0-2
    "minerva": list(range(3)),      # 0-2
    "olympiadbench": list(range(3)) # 0-2
}
EXPECTED_BENCHMARKS = list(BENCHMARK_SOBER_STEPS.keys())


def parse_experiment_dir(dir_name):
    """
    解析实验目录名，提取配置参数
    格式: {sober_steps}-{param}-{top_p}-{benchmark}-{num_samples}-{batch}-{max_model_len}-{max_new_tokens}
    例如: 0-0.0-0.7-aime24-1-128-131072-32768
    """
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


def check_model_experiments(results_dir):
    """检查所有模型的实验完成情况"""
    results_dir = Path(results_dir)

    # 收集所有实验结果
    # 结构: {model: {temp: {(sober_steps, top_p, benchmark): has_result}}}
    model_results = defaultdict(lambda: defaultdict(dict))

    # 遍历所有温度目录
    for temp_dir in results_dir.iterdir():
        if not temp_dir.is_dir() or not temp_dir.name.startswith("temp_"):
            continue

        try:
            temp = float(temp_dir.name.replace("temp_", ""))
        except ValueError:
            continue

        # 遍历模型目录
        for model_dir in temp_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name

            # 遍历实验目录
            for exp_dir in model_dir.iterdir():
                if not exp_dir.is_dir():
                    continue

                config = parse_experiment_dir(exp_dir.name)
                if config is None:
                    continue

                # 检查是否有结果文件 (results目录下有json文件)
                results_subdir = exp_dir / "results"
                has_result = False
                if results_subdir.exists():
                    json_files = list(results_subdir.glob("**/*.json"))
                    has_result = len(json_files) > 0

                key = (config["sober_steps"], config["top_p"], config["benchmark"])
                model_results[model_name][temp][key] = has_result

    return model_results


def generate_report(model_results):
    """生成检查报告"""
    # 计算预期总数
    # aime24, aime25, amc23: 16 * 5 * 3 = 240
    # math_500, minerva, olympiadbench: 3 * 5 * 3 = 45
    # 每个 temp: 240 + 45 = 285
    expected_per_temp = sum(len(steps) * len(EXPECTED_TOP_PS) for steps in BENCHMARK_SOBER_STEPS.values())
    expected_total = expected_per_temp * len(EXPECTED_TEMPS)

    print("=" * 80)
    print("实验完成情况检查报告")
    print("=" * 80)
    print(f"\n预期配置:")
    print(f"  - 温度 (temps): {EXPECTED_TEMPS}")
    print(f"  - Top-p: {EXPECTED_TOP_PS}")
    print(f"  - aime24/aime25/amc23: sober_steps 0-15 (16个)")
    print(f"  - math_500/minerva/olympiadbench: sober_steps 0-2 (3个)")
    print(f"  - 每个温度预期实验数: {expected_per_temp} (16*5*3 + 3*5*3 = 240 + 45)")
    print(f"  - 总预期实验数: {expected_total}")
    print("=" * 80)

    all_models_complete = []
    incomplete_models = []

    for model_name in sorted(model_results.keys()):
        temps_data = model_results[model_name]

        print(f"\n模型: {model_name}")
        print("-" * 60)

        total_completed = 0
        total_missing = 0
        missing_details = []

        for temp in EXPECTED_TEMPS:
            temp_key = temp
            exp_data = temps_data.get(temp_key, {})

            completed = sum(1 for v in exp_data.values() if v)
            missing = expected_per_temp - completed
            total_completed += completed
            total_missing += missing

            status = "✓" if missing == 0 else "✗"
            print(f"  temp_{temp}: {completed}/{expected_per_temp} {status}")

            # 收集缺失的具体配置
            if missing > 0:
                for benchmark, sober_steps_list in BENCHMARK_SOBER_STEPS.items():
                    for sober_steps in sober_steps_list:
                        for top_p in EXPECTED_TOP_PS:
                            key = (sober_steps, top_p, benchmark)
                            if not exp_data.get(key, False):
                                missing_details.append((temp, sober_steps, top_p, benchmark))

        completion_rate = total_completed / expected_total * 100
        print(f"\n  总计: {total_completed}/{expected_total} ({completion_rate:.1f}%)")

        if total_missing == 0:
            all_models_complete.append(model_name)
            print("  状态: ✓ 完成")
        else:
            incomplete_models.append((model_name, total_completed, expected_total, missing_details))
            print(f"  状态: ✗ 缺少 {total_missing} 个实验")

    # 总结
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)

    print(f"\n已完成所有实验的模型 ({len(all_models_complete)}):")
    for m in all_models_complete:
        print(f"  ✓ {m}")

    print(f"\n未完成实验的模型 ({len(incomplete_models)}):")
    for m, completed, total, missing in incomplete_models:
        print(f"  ✗ {m}: {completed}/{total} ({completed/total*100:.1f}%)")

    return all_models_complete, incomplete_models


def show_missing_details(incomplete_models, model_filter=None, limit=20):
    """显示缺失实验的详细信息"""
    for model_name, completed, total, missing_details in incomplete_models:
        if model_filter and model_filter not in model_name:
            continue

        print(f"\n{model_name} 缺失的实验 (前{limit}个):")
        for i, (temp, sober_steps, top_p, benchmark) in enumerate(missing_details[:limit]):
            print(f"  - temp={temp}, sober_steps={sober_steps}, top_p={top_p}, benchmark={benchmark}")

        if len(missing_details) > limit:
            print(f"  ... 还有 {len(missing_details) - limit} 个")


def main():
    parser = argparse.ArgumentParser(description="检查实验完成情况")
    parser.add_argument(
        "--results_dir", type=str,
        default="/home/zax/SoberReasoningPlus/results",
        help="结果目录路径"
    )
    parser.add_argument(
        "--show_missing", action="store_true",
        help="显示缺失实验的详细信息"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="只显示指定模型的缺失详情"
    )
    parser.add_argument(
        "--limit", type=int, default=20,
        help="显示缺失实验的数量限制"
    )

    args = parser.parse_args()

    print(f"正在扫描目录: {args.results_dir}")
    model_results = check_model_experiments(args.results_dir)

    if not model_results:
        print("未找到任何实验结果!")
        return

    complete, incomplete = generate_report(model_results)

    if args.show_missing and incomplete:
        print("\n" + "=" * 80)
        print("缺失实验详情")
        print("=" * 80)
        show_missing_details(incomplete, args.model, args.limit)


if __name__ == "__main__":
    main()
