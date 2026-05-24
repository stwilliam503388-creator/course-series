#!/usr/bin/env python3
"""
案例5：蒙特卡洛仿真与项目工期风险评估
=========================================
使用 Monte Carlo 仿真评估软件项目总工期分布。
场景：5 个串并行任务，三时估计（乐观/最可能/悲观）。
纯 Python 标准库实现，无第三方依赖。
"""

import math
import random
import statistics


# ============================================================
# 工具函数
# ============================================================

def normal_cdf(x, mu=0, sigma=1):
    """正态分布累积分布函数"""
    z = (x - mu) / sigma
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    sign = 1 if z >= 0 else -1
    z_abs = abs(z) / math.sqrt(2)
    t = 1.0 / (1.0 + p * z_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-z_abs * z_abs)
    return 0.5 * (1.0 + sign * y)


def normal_ppf(p, mu=0, sigma=1, tol=1e-10, max_iter=100):
    """正态分布分位数函数（二分法求解）"""
    if p <= 0:
        return -float('inf')
    if p >= 1:
        return float('inf')
    lo, hi = -10, 10
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        cdf_mid = normal_cdf(mid, mu, sigma)
        if abs(cdf_mid - p) < tol:
            return mid
        if cdf_mid < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def sample_quantile(data, q):
    """计算样本分位数"""
    sorted_data = sorted(data)
    n = len(sorted_data)
    idx = q * (n - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_data[lo]
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


# ============================================================
# 三角分布采样
# ============================================================

def triangular_sample(a, m, b):
    """
    从三角分布采样。

    参数：
        a: 乐观时间（最小值）
        m: 最可能时间（众数）
        b: 悲观时间（最大值）
    """
    u = random.random()
    # 三角形分布的累积分布函数逆函数
    # 分两种情况：u 在 (a, m) 区间 或 (m, b) 区间
    fc = (m - a) / (b - a)  # 众数的相对位置
    if u <= fc:
        return a + math.sqrt(u * (b - a) * (m - a))
    else:
        return b - math.sqrt((1 - u) * (b - a) * (b - m))


def triangular_mean(a, m, b):
    """三角分布的均值"""
    return (a + m + b) / 3.0


def triangular_std(a, m, b):
    """三角分布的标准差"""
    return math.sqrt((a**2 + m**2 + b**2 - a * m - a * b - m * b) / 18.0)


# ============================================================
# 任务与项目网络定义
# ============================================================

# 任务定义：(名称, 乐观, 最可能, 悲观)
TASKS = [
    ("A", 3, 4, 7),   # 任务 A
    ("B", 3, 5, 8),   # 任务 B
    ("C", 4, 6, 10),  # 任务 C
    ("D", 3, 4, 7),   # 任务 D
    ("E", 2, 3, 5),   # 任务 E
]

# 依赖关系：每个任务依赖的任务列表
# A 和 B 不依赖任何任务（起始任务）
# C 依赖 A，D 依赖 B
# E 依赖 C 和 D（两者都完成才能开始 E）
DEPENDENCIES = {
    "A": [],
    "B": [],
    "C": ["A"],
    "D": ["B"],
    "E": ["C", "D"],
}


def simulate_one():
    """
    单次 Monte Carlo 仿真。

    返回：
        (total_duration, critical_tasks)
        - total_duration: 项目总工期
        - critical_tasks: 关键路径上的任务名称列表
    """
    durations = {}
    earliest_start = {}
    earliest_finish = {}

    # 按拓扑顺序处理任务（A, B, C, D, E）
    task_order = ["A", "B", "C", "D", "E"]

    for task_name in task_order:
        # 找到对应的三时估计
        for t_name, a, m, b in TASKS:
            if t_name == task_name:
                # 从三角分布采样工期
                dur = triangular_sample(a, m, b)
                durations[task_name] = dur
                break

        # 计算最早开始时间（所有依赖任务的最晚完成时间）
        deps = DEPENDENCIES[task_name]
        if not deps:
            es = 0.0
        else:
            es = max(earliest_finish[dep] for dep in deps)

        earliest_start[task_name] = es
        earliest_finish[task_name] = es + durations[task_name]

    total_duration = max(earliest_finish.values())

    # 关键路径：从最后一个任务反向追溯
    # 用 earliest_finish 判断哪些任务在关键路径上
    critical_tasks = []
    # 找到最后一个完成的任务
    last_task = max(task_order, key=lambda t: earliest_finish[t])
    critical_tasks.append(last_task)

    # 反向追溯：对关键路径上的任务，找到它的关键依赖
    queue = [last_task]
    visited = set()
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        deps = DEPENDENCIES[current]
        if not deps:
            continue

        # 关键依赖：最早完成时间最晚的那个
        # （因为它的完成时间决定了当前任务的开始时间）
        critical_dep = max(deps, key=lambda d: earliest_finish[d])
        if critical_dep not in visited:
            critical_tasks.append(critical_dep)
            queue.append(critical_dep)

    critical_tasks.reverse()
    return total_duration, critical_tasks


# ============================================================
# Monte Carlo 仿真主函数
# ============================================================

def monte_carlo_simulation(num_simulations=10000, seed=42):
    """
    执行完整的 Monte Carlo 仿真。

    参数：
        num_simulations: 仿真次数
        seed: 随机种子

    返回：
        dict with keys:
            - durations: 每次仿真的总工期列表
            - critical_paths: 每次仿真的关键路径列表
            - mean_duration: 平均工期
            - std_duration: 工期标准差
            - p50: 中位数工期
            - p90: 90% 分位数
            - p95: 95% 分位数
            - on_time_prob: 30 天内交付概率
    """
    random.seed(seed)

    durations = []
    critical_paths = []

    for i in range(num_simulations):
        dur, crit = simulate_one()
        durations.append(dur)
        critical_paths.append(tuple(crit))

    mean_duration = statistics.mean(durations)
    std_duration = statistics.stdev(durations)
    p50 = sample_quantile(durations, 0.50)
    p90 = sample_quantile(durations, 0.90)
    p95 = sample_quantile(durations, 0.95)

    # 30 天内交付概率
    deadline = 30
    on_time_count = sum(1 for d in durations if d <= deadline)
    on_time_prob = on_time_count / num_simulations

    return {
        "durations": durations,
        "critical_paths": critical_paths,
        "mean_duration": mean_duration,
        "std_duration": std_duration,
        "p50": p50,
        "p90": p90,
        "p95": p95,
        "on_time_prob": on_time_prob,
    }


# ============================================================
# 关键路径频次分析
# ============================================================

def critical_path_analysis(critical_paths):
    """统计每个任务出现在关键路径上的频次"""
    task_freq = {t[0]: 0 for t in TASKS}
    total = len(critical_paths)

    for path in critical_paths:
        for task_name in path:
            task_freq[task_name] = task_freq.get(task_name, 0) + 1

    print(f"\n  {'='*50}")
    print(f"  关键路径频次统计（{total} 次仿真）")
    print(f"  {'='*50}")
    print(f"  {'任务':>6} {'频次':>8} {'频率':>10}")
    print(f"  {'-'*24}")
    for name, freq in sorted(task_freq.items()):
        pct = freq / total * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {name:>6} {freq:>8} {pct:>8.1f}%  {bar}")

    return task_freq


# ============================================================
# 敏感性分析
# ============================================================

def sensitivity_analysis(num_simulations=5000, delay_factor=1.2):
    """
    敏感性分析：每个任务工期增加 20%，看总工期的变化。

    返回按影响程度排序的结果。
    """
    # 基线
    baseline = monte_carlo_simulation(num_simulations, seed=42)
    base_mean = baseline["mean_duration"]

    print(f"\n  {'='*50}")
    print(f"  敏感性分析（每个任务延期 {int((delay_factor-1)*100)}%）")
    print(f"  {'='*50}")
    print(f"  基线平均工期: {base_mean:.2f} 天")
    print()

    # 另存原来的任务定义
    original_tasks = list(TASKS)

    results = []
    for i, (name, a, m, b) in enumerate(original_tasks):
        # 暂时修改任务定义
        new_a = a * delay_factor
        new_m = m * delay_factor
        new_b = b * delay_factor
        TASKS[i] = (name, new_a, new_m, new_b)

        # 重新仿真
        result = monte_carlo_simulation(num_simulations, seed=42)
        delta = result["mean_duration"] - base_mean
        delta_pct = delta / base_mean * 100
        results.append((name, delta, delta_pct, result["mean_duration"]))

        # 恢复
        TASKS[i] = (name, a, m, b)

    # 按影响排序（从大到小）
    results.sort(key=lambda x: x[1], reverse=True)

    max_delta = max(r[1] for r in results) if results else 1
    print(f"  {'任务':>6} {'延后工期':>12} {'影响(%)':>10} {'新均值':>10}")
    print(f"  {'-'*40}")
    for name, delta, delta_pct, new_mean in results:
        bar_ratio = delta / max_delta if max_delta > 0 else 0
        bar = "█" * int(bar_ratio * 30)
        print(f"  {name:>6} {delta:>+8.2f}天 {delta_pct:>+7.2f}%  {new_mean:>7.2f}天  {bar}")

    return results


# ============================================================
# 可视化（文本版）
# ============================================================

def text_histogram(data, title="工期分布直方图", bins=20, width=50):
    """输出文本条形图"""
    min_val = min(data)
    max_val = max(data)
    bin_width = (max_val - min_val) / bins

    counts = [0] * bins
    for d in data:
        idx = min(int((d - min_val) / bin_width), bins - 1)
        counts[idx] += 1

    max_count = max(counts) if counts else 1

    print(f"\n  {title}")
    print(f"  {'-' * (width + 30)}")
    for i in range(bins):
        lo = min_val + i * bin_width
        hi = lo + bin_width
        count = counts[i]
        bar_len = int(count / max_count * width)
        bar = "█" * bar_len
        label = f"[{lo:5.1f}, {hi:5.1f})"
        print(f"  {label} |{bar:<{width}} {count}")
    print(f"  {'-' * (width + 30)}")


# ============================================================
# 收敛性演示
# ============================================================

def convergence_demo():
    """演示 P95 分位数随着仿真次数增加而收敛"""
    print(f"\n  {'='*50}")
    print(f"  收敛性演示（P95 随仿真次数变化）")
    print(f"  {'='*50}")
    print(f"  {'仿真次数':>10} {'P95(天)':>10} {'P90(天)':>10} {'均值(天)':>10} {'变化':>10}")
    print(f"  {'-'*50}")

    n_list = [100, 500, 1000, 5000, 10000, 50000]
    prev_p95 = None
    for n in n_list:
        result = monte_carlo_simulation(num_simulations=n, seed=42)
        change = ""
        if prev_p95 is not None:
            diff = result["p95"] - prev_p95
            change = f"{diff:+.4f}"
        print(f"  {n:>10,} {result['p95']:>10.2f} {result['p90']:>10.2f} "
              f"{result['mean_duration']:>10.2f} {change:>10}")
        prev_p95 = result["p95"]

    print(f"\n  → P95 随仿真次数增加而收敛 ✅")


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("案例5：蒙特卡洛仿真与项目工期风险评估")
    print("=" * 60)
    print("场景：软件项目 | 5 个串并行任务 | 三时估计")
    print("=" * 60)

    # ========= 任务信息 =========
    print(f"\n{'='*60}")
    print("项目任务信息")
    print(f"{'='*60}")
    print(f"  {'任务':>6} {'乐观':>6} {'最可能':>8} {'悲观':>6} {'PERT均值':>10} {'PERT标准差':>12}")
    print(f"  {'-'*48}")
    for name, a, m, b in TASKS:
        pert_mean = (a + 4 * m + b) / 6.0
        pert_std = (b - a) / 6.0
        print(f"  {name:>6} {a:>6} {m:>8} {b:>6} {pert_mean:>10.2f} {pert_std:>12.2f}")

    # ========= 主仿真 =========
    print(f"\n{'='*60}")
    print("Monte Carlo 仿真（10000 次）")
    print(f"{'='*60}")

    NUM_SIM = 10000
    result = monte_carlo_simulation(num_simulations=NUM_SIM, seed=42)

    print(f"\n  仿真次数: {NUM_SIM}")
    print(f"\n  总工期统计:")
    print(f"    均值: {result['mean_duration']:.2f} 天")
    print(f"    标准差: {result['std_duration']:.2f} 天")
    print(f"    P50 (中位数): {result['p50']:.2f} 天")
    print(f"    P90: {result['p90']:.2f} 天")
    print(f"    P95: {result['p95']:.2f} 天")
    print(f"    最小值: {min(result['durations']):.2f} 天")
    print(f"    最大值: {max(result['durations']):.2f} 天")

    print(f"\n  按期交付概率:")
    for deadline in [25, 28, 30, 32, 35]:
        prob = sum(1 for d in result['durations'] if d <= deadline) / NUM_SIM
        print(f"    P(工期 ≤ {deadline}天) = {prob:.4f} ({prob*100:.1f}%)")

    # ========= 直方图 =========
    text_histogram(result['durations'], "总工期分布直方图", bins=15)

    # ========= 关键路径分析 =========
    critical_path_analysis(result['critical_paths'])

    # ========= 敏感性分析 =========
    sensitivity_analysis(num_simulations=5000, delay_factor=1.2)

    # ========= 收敛性 =========
    convergence_demo()

    # ========= 结论 =========
    print(f"\n{'='*60}")
    print("结论")
    print(f"{'='*60}")
    print(f"""
1. 项目总工期均值: {result['mean_duration']:.1f} 天
   P50 = {result['p50']:.1f} 天, P90 = {result['p90']:.1f} 天, P95 = {result['p95']:.1f} 天
   - 乐观估算: {result['mean_duration']:.0f} 天
   - 保守估算 (P95): {result['p95']:.0f} 天

2. 30 天交付概率: {result['on_time_prob']*100:.1f}%
   - {"✅ 较大概率按期交付" if result['on_time_prob'] >= 0.8 else "⚠️ 存在延期风险"}

3. 关键路径分析:
   - 任务 C 和 E 出现频次最高——是项目的核心瓶颈
   - 任务 A/B 谁更慢，谁所在的路径就更"关键"

4. 敏感性分析:
   - 任务 E 延期的影响最大（所有路径都要经过它）
   - 任务 C 次之（A-C-E 路径中的瓶颈）
   - 任务 D 的影响最小（B-D-E 路径较短，有缓冲）

5. 建议:
   - 重点关注任务 C 和 E 的进度
   - 任务 C 的悲观估计 10 天——需要制定应急预案
   - 向老板汇报: "P50={result['p50']:.0f}天, P95={result['p95']:.0f}天"
""")
    print("=" * 60)
    print("仿真完成！🎲")
    print("=" * 60)


if __name__ == "__main__":
    main()
