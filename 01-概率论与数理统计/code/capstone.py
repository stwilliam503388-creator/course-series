#!/usr/bin/env python3
"""
🏆 毕业项目：质量管控的完整统计分析
========================================
将本课程所有技能（分布拟合、置信区间、假设检验、贝叶斯更新）
串联起来解决一个真实的电子制造质量控制问题。

场景：电阻生产——标称值 100Ω，公差 ±5%（95Ω~105Ω）。
使用纯 Python 标准库实现，无第三方依赖。
"""

import math
import random
import statistics


# ============================================================
# 工具函数
# ============================================================

def normal_pdf(x, mu=0, sigma=1):
    """正态分布概率密度函数"""
    return (1.0 / (sigma * math.sqrt(2 * math.pi))) * \
           math.exp(-0.5 * ((x - mu) / sigma) ** 2)


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


def generate_normal(n, mu=0, sigma=1):
    """Box-Muller 变换生成正态分布样本"""
    samples = []
    for _ in range(n // 2 + 1):
        u1 = random.random()
        u2 = random.random()
        z1 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        z2 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
        samples.append(mu + sigma * z1)
        samples.append(mu + sigma * z2)
    return samples[:n]


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


def chi2_ppf(p, df, tol=1e-10, max_iter=1000):
    """卡方分布分位数（使用 Wilson-Hilferty 近似作为初值 + 二分法）"""
    if p <= 0:
        return 0.0
    if p >= 1:
        return float('inf')

    # Wilson-Hilferty 近似
    z = normal_ppf(p)
    approx = df * (1 - 2.0 / (9 * df) + z * math.sqrt(2.0 / (9 * df))) ** 3
    if approx <= 0:
        approx = df * 0.5

    # 二分法精化
    lo = max(approx * 0.1, 1e-10)
    hi = max(approx * 10, df * 3)
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        # 卡方 CDF（用正则化不完全伽马函数的近似）
        # 对于大 df，卡方分布近似正态
        if df > 30:
            z_mid = (mid - df) / math.sqrt(2 * df)
            cdf_mid = normal_cdf(z_mid)
        else:
            # 简化：对小 df 也用正态近似（实际应用应查表）
            z_mid = (mid - df) / math.sqrt(2 * df)
            cdf_mid = normal_cdf(z_mid)
        if abs(cdf_mid - p) < tol:
            return mid
        if cdf_mid < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ============================================================
# 第1步：生成模拟阻值数据
# ============================================================

def generate_resistance_data(n=1000, target=100.0, process_sigma=1.8, shift=0.5):
    """
    生成模拟电阻阻值数据。

    真实工艺可能有微小偏移（shift>0 表示工艺均值偏高）。
    process_sigma 控制过程标准差（工艺精度）。
    """
    true_mu = target + shift
    data = generate_normal(n, true_mu, process_sigma)
    return data, true_mu, process_sigma


# ============================================================
# 第2步：任务1 — 分布拟合
# ============================================================

def qq_correlation(data):
    """计算数据与正态分布的 QQ 相关性"""
    n = len(data)
    sorted_data = sorted(data)
    data_mean = statistics.mean(sorted_data)
    data_std = statistics.stdev(sorted_data)
    if data_std == 0:
        return 0

    theoretical = []
    for i in range(1, n):
        p = i / (n + 1)
        theoretical.append(normal_ppf(p))

    standardized = [(x - data_mean) / data_std for x in sorted_data[:-1]]

    n_pts = len(theoretical)
    mean_x = statistics.mean(theoretical)
    mean_y = statistics.mean(standardized)

    cov = sum((theoretical[i] - mean_x) * (standardized[i] - mean_y) for i in range(n_pts))
    std_x = math.sqrt(sum((x - mean_x) ** 2 for x in theoretical))
    std_y = math.sqrt(sum((y - mean_y) ** 2 for y in standardized))

    if std_x * std_y == 0:
        return 0
    return cov / (std_x * std_y)


def assess_distribution(data, target=100.0, usl=105.0, lsl=95.0):
    """评估阻值数据的分布特征"""
    n = len(data)
    mean_d = statistics.mean(data)
    std_d = statistics.stdev(data)
    min_d = min(data)
    max_d = max(data)

    print(f"\n  {'='*50}")
    print(f"  【任务1】分布拟合 —— 阻值分布分析")
    print(f"  {'='*50}")
    print(f"  基本统计量:")
    print(f"    样本量: {n}")
    print(f"    均值: {mean_d:.4f} Ω")
    print(f"    标准差: {std_d:.4f} Ω")
    print(f"    最小值: {min_d:.4f} Ω")
    print(f"    最大值: {max_d:.4f} Ω")
    print(f"    目标值: {target} Ω")
    print(f"    规格上限 (USL): {usl} Ω")
    print(f"    规格下限 (LSL): {lsl} Ω")

    # 偏度
    m3 = sum((x - mean_d) ** 3 for x in data) / n
    skewness = m3 / (std_d ** 3) if std_d > 0 else 0
    # 峰度
    m4 = sum((x - mean_d) ** 4 for x in data) / n
    kurtosis = m4 / (std_d ** 4) if std_d > 0 else 0
    print(f"    偏度: {skewness:.4f} (正态 ≈ 0)")
    print(f"    峰度: {kurtosis:.4f} (正态 ≈ 3)")

    # QQ 相关性
    r_normal = qq_correlation(data)
    print(f"\n  正态 QQ 相关性: {r_normal:.4f}")
    if r_normal > 0.98:
        print(f"  ✅ 数据近似正态分布（QQ 相关性 > 0.98）")
    elif r_normal > 0.95:
        print(f"  ⚠️ 数据大致接近正态分布（QQ 相关性 > 0.95）")
    else:
        print(f"  ❌ 数据偏离正态分布（QQ 相关性 < 0.95）")

    # 不合格率
    defect_rate = sum(1 for x in data if x < lsl or x > usl) / n
    print(f"\n  样本不合格率: {defect_rate:.4f} ({defect_rate*100:.2f}%)")

    return mean_d, std_d


# ============================================================
# 第3步：任务2 — 假设检验（工艺是否偏移）
# ============================================================

def one_sample_t_test(data, mu0=100.0, alpha=0.05):
    """单样本 t 检验：H₀: μ = μ0"""
    n = len(data)
    mean_d = statistics.mean(data)
    std_d = statistics.stdev(data)
    se = std_d / math.sqrt(n)

    t_stat = (mean_d - mu0) / se if se > 0 else 0

    # 用正态近似（n=1000 足够大）
    p_value_two_sided = 2 * (1 - normal_cdf(abs(t_stat)))

    print(f"\n  {'='*50}")
    print(f"  【任务2】假设检验 —— 工艺均值是否偏移？")
    print(f"  {'='*50}")
    print(f"  H₀: μ = {mu0} Ω（工艺正常）")
    print(f"  H₁: μ ≠ {mu0} Ω（工艺偏移）")
    print(f"  α = {alpha}")
    print(f"\n  检验统计量:")
    print(f"    样本均值: {mean_d:.4f} Ω")
    print(f"    样本标准差: {std_d:.4f} Ω")
    print(f"    标准误: {se:.4f} Ω")
    print(f"    t 统计量: {t_stat:.4f}")
    print(f"    p 值 (双侧): {p_value_two_sided:.6f}")

    if p_value_two_sided < alpha:
        print(f"\n  ✅ p = {p_value_two_sided:.6f} < α = {alpha}")
        print(f"  → 拒绝 H₀：工艺均值显著偏离 {mu0} Ω")
        print(f"  → 偏移量: {mean_d - mu0:.4f} Ω")
    else:
        print(f"\n  ⚠️ p = {p_value_two_sided:.6f} >= α = {alpha}")
        print(f"  → 不能拒绝 H₀：没有足够证据表明工艺偏移")

    return t_stat, p_value_two_sided, mean_d - mu0


# ============================================================
# 第4步：任务3 — 过程能力指数 Cp/Cpk
# ============================================================

def process_capability(data, target=100.0, usl=105.0, lsl=95.0, conf_level=0.95):
    """计算过程能力指数 Cp、Cpk 及其置信区间"""
    n = len(data)
    mean_d = statistics.mean(data)
    std_d = statistics.stdev(data)

    # Cp：只考虑离散度
    cp = (usl - lsl) / (6 * std_d) if std_d > 0 else float('inf')

    # Cpk：考虑偏移
    cpu = (usl - mean_d) / (3 * std_d) if std_d > 0 else float('inf')
    cpl = (mean_d - lsl) / (3 * std_d) if std_d > 0 else float('inf')
    cpk = min(cpu, cpl)

    # Cp 的置信区间（卡方分布）
    alpha = 1 - conf_level
    chi2_low = chi2_ppf(alpha / 2, n - 1)
    chi2_high = chi2_ppf(1 - alpha / 2, n - 1)

    cp_low = cp * math.sqrt(chi2_low / (n - 1)) if chi2_low > 0 else 0
    cp_high = cp * math.sqrt(chi2_high / (n - 1))

    print(f"\n  {'='*50}")
    print(f"  【任务3】过程能力分析 —— Cp/Cpk")
    print(f"  {'='*50}")
    print(f"  规格:")
    print(f"    目标值: {target} Ω")
    print(f"    规格上限 (USL): {usl} Ω")
    print(f"    规格下限 (LSL): {lsl} Ω")
    print(f"    公差范围: {usl - lsl} Ω")
    print(f"\n  过程能力指数:")
    print(f"    过程标准差 σ: {std_d:.4f} Ω")
    print(f"    Cp = (USL-LSL)/(6σ) = {cp:.4f}")
    print(f"    CPU = (USL-μ)/(3σ) = {cpu:.4f}")
    print(f"    CPL = (μ-LSL)/(3σ) = {cpl:.4f}")
    print(f"    Cpk = min(CPU, CPL) = {cpk:.4f}")
    print(f"\n  Cp 的 {conf_level:.0%} 置信区间:")
    print(f"    [{cp_low:.4f}, {cp_high:.4f}]")

    # 过程能力等级判定
    print(f"\n  过程能力等级:")
    if cpk >= 2.0:
        grade = "A++（六西格玛级）"
        comment = "世界级质量水平"
    elif cpk >= 1.67:
        grade = "A+（超优）"
        comment = "高质量水平，适合关键部件"
    elif cpk >= 1.33:
        grade = "A（优）"
        comment = "过程能力充足"
    elif cpk >= 1.0:
        grade = "B（良）"
        comment = "过程能力一般，需监控"
    elif cpk >= 0.67:
        grade = "C（差）"
        comment = "过程能力不足，需改进"
    else:
        grade = "D（不合格）"
        comment = "过程严重不足，立即停产整改"

    print(f"    Cpk = {cpk:.4f} → {grade}")
    print(f"    说明: {comment}")

    # 理论不合格率
    z_upper = (usl - mean_d) / std_d if std_d > 0 else float('inf')
    z_lower = (mean_d - lsl) / std_d if std_d > 0 else float('inf')
    p_defect_upper = 1 - normal_cdf(z_upper)
    p_defect_lower = 1 - normal_cdf(z_lower)
    p_total = p_defect_upper + p_defect_lower
    print(f"\n  理论不合格率 (基于正态假设):")
    print(f"    超出 USL: {p_defect_upper:.6f} ({p_defect_upper*1e6:.1f} ppm)")
    print(f"    低于 LSL: {p_defect_lower:.6f} ({p_defect_lower*1e6:.1f} ppm)")
    print(f"    总不合格率: {p_total:.6f} ({p_total*1e6:.1f} ppm)")

    return cp, cpk, cp_low, cp_high


# ============================================================
# 第5步：任务4 — 贝叶斯更新（来料批次品质估计）
# ============================================================

def bayesian_batch_update(prior_mean, prior_std, prior_n,
                          sample_mean, sample_std, sample_n):
    """
    正态分布均值的贝叶斯更新——用于来料批次品质估计。

    先验：来自历史批次的均值和标准差
    数据：当前批次的抽检结果
    返回：后验均值和标准差
    """
    # 先验精度
    prior_precision = prior_n / (prior_std ** 2) if prior_std > 0 else 0
    # 数据精度
    data_precision = sample_n / (sample_std ** 2) if sample_std > 0 else 0

    # 后验均值 = 精度加权平均
    if prior_precision + data_precision > 0:
        posterior_mean = (prior_precision * prior_mean +
                          data_precision * sample_mean) / \
                         (prior_precision + data_precision)
    else:
        posterior_mean = sample_mean

    # 后验方差
    posterior_var = 1.0 / (prior_precision + data_precision)
    posterior_std = math.sqrt(posterior_var) if posterior_var > 0 else 0

    return posterior_mean, posterior_std, prior_precision, data_precision


def question4_bayesian(data):
    """任务4：贝叶斯更新来料批次品质"""
    print(f"\n  {'='*50}")
    print(f"  【任务4】贝叶斯更新 —— 来料批次品质估计")
    print(f"  {'='*50}")

    # 先验：历史批次数据
    # 历史批次均值约 100.3Ω，标准差约 2.0Ω，等效样本量 20
    prior_mean = 100.3
    prior_std = 2.0
    prior_n = 20

    # 抽检当前批次 30 个样品
    random.seed(123)  # 固定的种子用于可复现
    batch_sample = random.sample(data, min(30, len(data)))
    sample_mean = statistics.mean(batch_sample)
    sample_std = statistics.stdev(batch_sample)
    sample_n = len(batch_sample)

    print(f"\n  先验信息（历史批次）:")
    print(f"    先验均值: {prior_mean} Ω")
    print(f"    先验标准差: {prior_std} Ω")
    print(f"    先验等效样本量: {prior_n}")

    print(f"\n  当前批次抽检 (n={sample_n}):")
    print(f"    样本均值: {sample_mean:.4f} Ω")
    print(f"    样本标准差: {sample_std:.4f} Ω")

    # 贝叶斯更新
    post_mean, post_std, prior_prec, data_prec = bayesian_batch_update(
        prior_mean, prior_std, prior_n,
        sample_mean, sample_std, sample_n
    )

    print(f"\n  贝叶斯后验:")
    print(f"    后验均值: {post_mean:.4f} Ω")
    print(f"    后验标准差: {post_std:.4f} Ω")
    print(f"    后验 95% 可信区间: "
          f"[{post_mean - 1.96 * post_std:.4f}, "
          f"{post_mean + 1.96 * post_std:.4f}] Ω")

    print(f"\n  方法对比:")
    print(f"    先验均值: {prior_mean}")
    print(f"    MLE（样本均值）: {sample_mean:.4f}")
    print(f"    后验均值: {post_mean:.4f}")
    print(f"    → 后验是先验和 MLE 的精度加权平均")

    # 权重解释
    total_prec = prior_prec + data_prec
    prior_weight = prior_prec / total_prec * 100 if total_prec > 0 else 0
    data_weight = data_prec / total_prec * 100 if total_prec > 0 else 0
    print(f"\n  权重分析:")
    print(f"    先验权重: {prior_weight:.1f}%")
    print(f"    数据权重: {data_weight:.1f}%")
    print(f"    → 先验权重 = 精度加权，等效样本量越大权重越高")

    # 判断批次是否合格（基于后验均值与规格的比较）
    if post_mean >= 99.5 and post_mean <= 100.5:
        print(f"\n  ✅ 基于贝叶斯更新，该批次品质合格")
        print(f"    后验均值 {post_mean:.2f} Ω 在目标范围 [99.5, 100.5] Ω 内")
    else:
        print(f"\n  ⚠️ 基于贝叶斯更新，该批次品质偏离预期")
        print(f"    后验均值 {post_mean:.2f} Ω 超出目标范围 [99.5, 100.5] Ω")

    return post_mean, post_std


# ============================================================
# 第6步：控制图统计量
# ============================================================

def control_chart_stats(data, subgroup_size=5):
    """
    生成 X̄-R 控制图统计量（文本输出）。
    X̄ 图监控均值偏移，R 图监控离散度变化。
    """
    n = len(data)
    k = n // subgroup_size  # 子组数量
    subgroups = [data[i * subgroup_size:(i + 1) * subgroup_size] for i in range(k)]

    xbar_list = [statistics.mean(sg) for sg in subgroups]
    r_list = [max(sg) - min(sg) for sg in subgroups]

    xbar_bar = statistics.mean(xbar_list)
    r_bar = statistics.mean(r_list)

    # 控制图常数（n=5）
    A2 = 0.577   # X̄ 图控制限系数
    D3 = 0       # R 图下控制限系数
    D4 = 2.114   # R 图上控制限系数

    # X̄ 图控制限
    ucl_x = xbar_bar + A2 * r_bar
    lcl_x = xbar_bar - A2 * r_bar

    # R 图控制限
    ucl_r = D4 * r_bar
    lcl_r = D3 * r_bar

    print(f"\n  {'='*50}")
    print(f"  控制图分析 (X̄-R 图)")
    print(f"  {'='*50}")
    print(f"  子组大小: n={subgroup_size}")
    print(f"  子组数量: k={k}")
    print(f"\n  X̄ 图 (均值控制图):")
    print(f"    中心线 (X̄̿): {xbar_bar:.4f} Ω")
    print(f"    上控制限 (UCL): {ucl_x:.4f} Ω")
    print(f"    下控制限 (LCL): {lcl_x:.4f} Ω")
    print(f"\n  R 图 (极差控制图):")
    print(f"    中心线 (R̄): {r_bar:.4f} Ω")
    print(f"    上控制限 (UCL): {ucl_r:.4f} Ω")
    print(f"    下控制限 (LCL): {lcl_r:.4f} Ω")

    # 检查是否有超出控制限的子组
    out_of_control_x = sum(1 for x in xbar_list if x > ucl_x or x < lcl_x)
    out_of_control_r = sum(1 for r in r_list if r > ucl_r)

    print(f"\n  异常检测:")
    print(f"    X̄ 图超出控制限的子组数: {out_of_control_x}/{k}")
    print(f"    R 图超出控制限的子组数: {out_of_control_r}/{k}")

    if out_of_control_x == 0 and out_of_control_r == 0:
        print(f"  ✅ 过程处于统计受控状态（所有子组在控制限内）")
    else:
        print(f"  ⚠️ 过程存在异常——需要调查特殊原因")

    return xbar_bar, r_bar, ucl_x, lcl_x


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("🏆 毕业项目：质量管控的完整统计分析")
    print("=" * 60)
    print("场景：电阻生产 | 标称值 100Ω | 公差 ±5%")
    print("=" * 60)

    random.seed(42)

    # ========= 生成模拟数据 =========
    print(f"\n{'='*60}")
    print("数据准备：生成 1000 个电阻阻值数据")
    print(f"{'='*60}")

    resistance_data, true_mu, process_sigma = generate_resistance_data(
        n=1000, target=100.0, process_sigma=1.8, shift=0.5
    )
    print(f"  真实工艺均值: {true_mu} Ω")
    print(f"  真实过程标准差: {process_sigma} Ω")
    print(f"  数据范围: [{min(resistance_data):.4f}, {max(resistance_data):.4f}] Ω")

    # ========= 任务1：分布拟合 =========
    mean_d, std_d = assess_distribution(resistance_data)

    # ========= 任务2：假设检验 =========
    t_stat, p_value, shift_est = one_sample_t_test(resistance_data, mu0=100.0)

    # ========= 任务3：过程能力分析 =========
    cp, cpk, cp_low, cp_high = process_capability(resistance_data)

    # ========= 控制图 =========
    control_chart_stats(resistance_data, subgroup_size=5)

    # ========= 任务4：贝叶斯更新 =========
    post_mean, post_std = question4_bayesian(resistance_data)

    # ========= 最终报告 =========
    print(f"\n{'='*60}")
    print("📋 质量控制最终报告")
    print(f"{'='*60}")

    print(f"""
基于以上完整的统计分析，我们得出以下质量控制结论：

1. 阻值分布:
   - 均值 ≈ {mean_d:.2f} Ω（目标 100Ω）
   - 标准差 ≈ {std_d:.2f} Ω
   - 分布类型: 正态分布（QQ 相关性确认）

2. 工艺偏移检验:
   - 实际偏移量: {shift_est:.4f} Ω
   - p 值: {p_value:.6f}
   - 结论: {"工艺存在显著偏移" if p_value < 0.05 else "工艺无明显偏移"}

3. 过程能力:
   - Cp = {cp:.4f}
   - Cpk = {cpk:.4f}
   - Cp 95% CI: [{cp_low:.4f}, {cp_high:.4f}]
   - 过程能力等级: {"充足" if cpk >= 1.33 else "需改进" if cpk >= 1.0 else "不合格"}

4. 贝叶斯批次品质:
   - 后验均值: {post_mean:.4f} Ω
   - 后验标准差: {post_std:.4f} Ω

5. 关键行动建议:
   - Cp > 1.33 说明过程精度够 → 调均值到 100Ω 即可
   - Cp < 1.33 说明过程离散度偏大 → 需要改进工艺
   - 建议频率: 每批次抽检 30 个样品，用贝叶斯更新监控品质趋势
""")
    print("=" * 60)
    print("毕业项目完成！所有统计技能已应用于质量管控。")
    print("=" * 60)


if __name__ == "__main__":
    main()
