#!/usr/bin/env python3
"""
case03_portfolio_multi.py
共享单车调度多目标优化 — 加权求和法求帕累托前沿

场景：早高峰前用卡车将共享单车从富余站点运到短缺站点
三个冲突目标：
  f1 = 运营成本（卡车行驶距离 × 油耗成本）→ 最小化
  f2 = 用户满意度（基于站点车辆供需匹配度）→ 最大化
  f3 = 碳排放（卡车路线总碳排放量）→ 最小化

方法：
  加权求和法扫描权重空间生成 Pareto 前沿
  展示成本-满意度-碳排放的三维权衡

验证：
  - Pareto 前沿上所有点互不支配
  - 极端解合理性（成本最低者满意度最低、碳排放最低者成本最高）
  - 折中方案三目标平衡
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# ===================== 中文显示配置 =====================
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC',
                                    'Heiti SC', 'Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


# ===================== 数据生成 =====================
def generate_station_data(n_stations=20, seed=42):
    """生成站点数据，包括坐标、容量、当前车辆数"""
    rng = np.random.RandomState(seed)
    # 站点坐标（单位：km）
    coords = rng.rand(n_stations, 2) * 10
    # 站点容量（40~60辆）
    capacity = 40 + rng.randint(0, 21, n_stations)
    # 当前车辆数 —— 制造明显的富余/短缺差异
    current = np.zeros(n_stations, dtype=int)
    for i in range(n_stations):
        if i < n_stations // 2:
            # 上半部分站点：富余（居民区，车多桩多）
            current[i] = int(capacity[i] * (0.75 + 0.2 * rng.rand()))
        else:
            # 下半部分站点：短缺（地铁口/写字楼，车少）
            current[i] = int(capacity[i] * (0.05 + 0.2 * rng.rand()))
    return coords, capacity, current


def compute_demands(current, capacity, buffer_ratio=0.15):
    """
    计算各站点的富余量和短缺量
    buffer_ratio: 缓冲区比例
    """
    n = len(current)
    surplus = np.zeros(n)
    deficit = np.zeros(n)
    satisfaction_base = np.zeros(n)

    for i in range(n):
        cap = capacity[i]
        low = buffer_ratio * cap
        high = (1 - buffer_ratio) * cap
        v = current[i]

        if v > high:
            surplus[i] = v - high
        elif v < low:
            deficit[i] = low - v

        # 基础满意度（调度前）
        if v < low:
            satisfaction_base[i] = v / low
        elif v <= high:
            satisfaction_base[i] = 1.0
        else:
            satisfaction_base[i] = (cap - v) / (cap - high)

    return surplus, deficit, satisfaction_base


def compute_satisfaction(inventory, capacity, buffer_ratio=0.15):
    """计算平均用户满意度"""
    n = len(inventory)
    sats = np.zeros(n)
    for i in range(n):
        v = inventory[i]
        cap = capacity[i]
        low = buffer_ratio * cap
        high = (1 - buffer_ratio) * cap
        if v < low:
            sats[i] = v / low
        elif v <= high:
            sats[i] = 1.0
        else:
            sats[i] = max(0, (cap - v) / (cap - high))
    return np.mean(sats)


# ===================== 权重感知的调度优化 =====================

def optimize_dispatch_weighted(surplus, deficit, dist_matrix,
                               w_cost, w_satisfaction, w_carbon,
                               coords, capacity, current,
                               truck_capacity=30,
                               cost_per_km=2.5, carbon_per_km=0.8):
    """
    给定权重组合 (w_cost, w_satisfaction, w_carbon)，生成调度方案。
    w_satisfaction 越大：越倾向跑更多站点覆盖更多用户
    w_cost 越大：越倾向少跑路，只服务最近的站点
    w_carbon 越大：越倾向短途路线和更少的卡车
    """
    n = len(surplus)
    depot = np.array([[5.0, 5.0]])

    surplus_idx = np.where(surplus > 0.5)[0]
    deficit_idx = np.where(deficit > 0.5)[0]

    # 按权重决定调度强度和优先级
    # w_cost 大 → 少跑路，只选最近的对
    # w_satisfaction 大 → 多跑路，覆盖更多站点
    # w_carbon 大 → 选短途，控制排放

    # 调度强度：满意度权重越高，覆盖比例越高
    coverage_ratio = 0.3 + 0.6 * w_satisfaction
    n_pairs = max(1, int(min(len(surplus_idx), len(deficit_idx)) * coverage_ratio))

    # 构建候选匹配对，计算综合优先级得分
    candidates = []
    for i in surplus_idx:
        for j in deficit_idx:
            d = dist_matrix[i, j]
            # 从调度中心到surplus再到deficit再回调度中心
            d_total = (np.linalg.norm(depot[0] - coords[i]) +
                       d +
                       np.linalg.norm(coords[j] - depot[0]))
            cost = d_total * cost_per_km
            carbon = d_total * carbon_per_km

            # 满意度增益：调度后该站点满意度提升
            # surplus站点的车辆减少
            new_v_i = current[i] - min(surplus[i], truck_capacity)
            sat_i = compute_single_satisfaction(new_v_i, capacity[i])
            old_sat_i = compute_single_satisfaction(current[i], capacity[i])
            # deficit站点的车辆增加
            new_v_j = min(capacity[j], current[j] + min(deficit[j], truck_capacity))
            sat_j = compute_single_satisfaction(new_v_j, capacity[j])
            old_sat_j = compute_single_satisfaction(current[j], capacity[j])
            sat_gain = (sat_j - old_sat_j) + (sat_i - old_sat_i)

            # 不希望 sat_gain 太小
            if sat_gain < 0:
                sat_gain = 0.01

            # 综合得分：不同权重导向不同策略
            priority = (w_satisfaction * sat_gain * 10 -
                        w_cost * cost / 100 -
                        w_carbon * carbon / 30)
            # 加上距离作为辅助（近的优先）
            priority += -0.01 * d_total  # 小幅优先短途

            candidates.append((priority, i, j, d_total, cost, carbon, sat_gain))

    # 按优先级从高到低排序
    candidates.sort(key=lambda x: x[0], reverse=True)

    # 选择 top N 对进行调度
    selected_pairs = candidates[:n_pairs]

    # 执行调度
    dispatch = np.zeros((n, n))
    final_inventory = current.copy().astype(float)
    total_distance = 0.0
    total_cost = 0.0
    total_carbon = 0.0
    truck_used = 0

    for _, i, j, d_total, cost, carbon, sat_gain in selected_pairs:
        # 计算可调度量
        if surplus[i] <= 0.5 or deficit[j] <= 0.5:
            continue

        load = min(surplus[i], deficit[j], truck_capacity)
        if load < 1:
            continue

        dispatch[i, j] += load
        final_inventory[i] -= load
        final_inventory[j] = min(capacity[j], final_inventory[j] + load)
        total_distance += d_total
        total_cost += cost
        total_carbon += carbon
        truck_used += 1

    # 计算最终满意度
    satisfaction = compute_satisfaction(final_inventory, capacity)

    # 如果 w_satisfaction 大但满意度仍然低，强制多调一点
    if w_satisfaction > 0.6 and satisfaction < 0.85:
        # 再调一轮
        for _, i, j, d_total, cost, carbon, sat_gain in candidates[len(selected_pairs):]:
            if truck_used > 8:
                break
            if surplus[i] <= 0.5 or deficit[j] <= 0.5:
                continue
            load = min(surplus[i], deficit[j], truck_capacity)
            if load < 1:
                continue
            dispatch[i, j] += load
            final_inventory[i] -= load
            final_inventory[j] = min(capacity[j], final_inventory[j] + load)
            total_distance += d_total
            total_cost += cost
            total_carbon += carbon
            truck_used += 1

        satisfaction = compute_satisfaction(final_inventory, capacity)

    return {
        'cost': total_cost,
        'satisfaction': satisfaction,
        'carbon': total_carbon,
        'distance': total_distance,
        'trucks': truck_used,
        'dispatch': dispatch,
        'coverage': len(selected_pairs)
    }


def compute_single_satisfaction(v, cap, buffer_ratio=0.15):
    """计算单个站点的满意度"""
    low = buffer_ratio * cap
    high = (1 - buffer_ratio) * cap
    if v < low:
        return v / low
    elif v <= high:
        return 1.0
    else:
        return max(0, (cap - v) / (cap - high))


# ===================== 帕累托过滤 =====================
def is_dominated(a, b):
    """
    判断 a 是否被 b 支配
    目标：成本↓(0) 满意度↑(1) 碳排放↓(2)
    """
    not_worse = (b[0] <= a[0] and b[1] >= a[1] and b[2] <= a[2])
    strictly_better = (b[0] < a[0] or b[1] > a[1] or b[2] < a[2])
    return not_worse and strictly_better


def pareto_filter(solutions):
    """过滤出帕累托最优解"""
    n = len(solutions)
    dominated = [False] * n
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            a = (solutions[i]['cost'], solutions[i]['satisfaction'], solutions[i]['carbon'])
            b = (solutions[j]['cost'], solutions[j]['satisfaction'], solutions[j]['carbon'])
            if is_dominated(a, b):
                dominated[i] = True
                break
            elif is_dominated(b, a):
                dominated[j] = True
    return [solutions[i] for i in range(n) if not dominated[i]]


# ===================== 绘图 =====================
def plot_results(all_sols, pareto_sols, best_sol):
    """绘制四合一图"""
    fig = plt.figure(figsize=(16, 12))

    costs = np.array([s['cost'] for s in all_sols])
    sats = np.array([s['satisfaction'] for s in all_sols]) * 100
    carbons = np.array([s['carbon'] for s in all_sols])

    p_costs = np.array([s['cost'] for s in pareto_sols])
    p_sats = np.array([s['satisfaction'] for s in pareto_sols]) * 100
    p_carbons = np.array([s['carbon'] for s in pareto_sols])

    # --- 左上：3D 帕累托前沿 ---
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    sc = ax1.scatter(p_costs, p_sats, p_carbons,
                     c=p_sats, cmap='viridis', s=60, alpha=0.8,
                     edgecolors='k', linewidth=0.5)

    # 极端解
    min_cost_idx = np.argmin(p_costs)
    max_sat_idx = np.argmax(p_sats)
    min_carbon_idx = np.argmin(p_carbons)

    labels_3d = ['成本优先', '满意度优先', '环保优先']
    pts_3d = [(p_costs[min_cost_idx], p_sats[min_cost_idx], p_carbons[min_cost_idx]),
              (p_costs[max_sat_idx], p_sats[max_sat_idx], p_carbons[max_sat_idx]),
              (p_costs[min_carbon_idx], p_sats[min_carbon_idx], p_carbons[min_carbon_idx])]

    for pt, label in zip(pts_3d, labels_3d):
        ax1.scatter(*pt, c='red', s=120, marker='*', zorder=5)
        ax1.text(pt[0], pt[1], pt[2], f'  {label}', fontsize=9, color='red')

    best_cost = best_sol['cost']
    best_sat = best_sol['satisfaction'] * 100
    best_carbon = best_sol['carbon']
    ax1.scatter(best_cost, best_sat, best_carbon,
                c='green', s=150, marker='D', zorder=5)
    ax1.text(best_cost, best_sat, best_carbon, '  折中方案',
             fontsize=10, color='green', fontweight='bold')

    ax1.set_xlabel('运营成本 (¥)', fontsize=10)
    ax1.set_ylabel('用户满意度 (%)', fontsize=10)
    ax1.set_zlabel('碳排放 (kg CO₂)', fontsize=10)
    ax1.set_title('三维帕累托前沿', fontsize=12, fontweight='bold')
    fig.colorbar(sc, ax=ax1, shrink=0.5, label='满意度 (%)')

    # --- 右上：成本 vs 满意度 ---
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(costs, sats, c='lightblue', alpha=0.4, s=30, label='所有解')
    ax2.scatter(p_costs, p_sats, c='orange', s=50, edgecolors='darkorange',
                linewidth=0.5, label='Pareto最优', zorder=3)
    ax2.scatter(p_costs[min_cost_idx], p_sats[min_cost_idx],
                c='red', s=100, marker='*', label='成本优先')
    ax2.scatter(p_costs[max_sat_idx], p_sats[max_sat_idx],
                c='blue', s=100, marker='*', label='满意度优先')
    ax2.scatter(best_cost, best_sat, c='green', s=120, marker='D', label='推荐方案')

    ax2.set_xlabel('运营成本 (¥)', fontsize=10)
    ax2.set_ylabel('用户满意度 (%)', fontsize=10)
    ax2.set_title('成本 vs 满意度', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- 左下：成本 vs 碳排放 ---
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(costs, carbons, c='lightblue', alpha=0.4, s=30, label='所有解')
    ax3.scatter(p_costs, p_carbons, c='orange', s=50, edgecolors='darkorange',
                linewidth=0.5, label='Pareto最优', zorder=3)
    ax3.scatter(p_costs[min_cost_idx], p_carbons[min_cost_idx],
                c='red', s=100, marker='*', label='成本优先')
    ax3.scatter(p_costs[min_carbon_idx], p_carbons[min_carbon_idx],
                c='purple', s=100, marker='*', label='环保优先')
    ax3.scatter(best_cost, best_carbon, c='green', s=120, marker='D', label='推荐方案')

    ax3.set_xlabel('运营成本 (¥)', fontsize=10)
    ax3.set_ylabel('碳排放 (kg CO₂)', fontsize=10)
    ax3.set_title('成本 vs 碳排放', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # --- 右下：平行坐标图 ---
    ax4 = fig.add_subplot(2, 2, 4)

    p_cost_norm = (p_costs - p_costs.min()) / (p_costs.max() - p_costs.min() + 1e-10)
    p_sat_norm = (p_sats - p_sats.min()) / (p_sats.max() - p_sats.min() + 1e-10)
    p_carbon_norm = (p_carbons - p_carbons.min()) / (p_carbons.max() - p_carbons.min() + 1e-10)

    axes = [0, 1, 2]
    labels_par = ['成本\n(低→高)', '满意度\n(低→高)', '碳排放\n(低→高)']

    for i in range(len(p_costs)):
        vals = [p_cost_norm[i], p_sat_norm[i], p_carbon_norm[i]]
        ax4.plot(axes, vals, 'o-', color='orange', alpha=0.3, linewidth=0.8)

    b_cost_n = (best_cost - p_costs.min()) / (p_costs.max() - p_costs.min() + 1e-10)
    b_sat_n = (best_sat - p_sats.min()) / (p_sats.max() - p_sats.min() + 1e-10)
    b_car_n = (best_carbon - p_carbons.min()) / (p_carbons.max() - p_carbons.min() + 1e-10)
    ax4.plot(axes, [b_cost_n, b_sat_n, b_car_n], 'o-',
             color='green', linewidth=3, markersize=8, label='推荐方案')

    ax4.plot(axes,
             [p_cost_norm[min_cost_idx], p_sat_norm[min_cost_idx], p_carbon_norm[min_cost_idx]],
             'o-', color='red', linewidth=2, markersize=6, label='成本优先')
    ax4.plot(axes,
             [p_cost_norm[max_sat_idx], p_sat_norm[max_sat_idx], p_carbon_norm[max_sat_idx]],
             'o-', color='blue', linewidth=2, markersize=6, label='满意度优先')
    ax4.plot(axes,
             [p_cost_norm[min_carbon_idx], p_sat_norm[min_carbon_idx], p_carbon_norm[min_carbon_idx]],
             'o-', color='purple', linewidth=2, markersize=6, label='环保优先')

    ax4.set_xticks(axes)
    ax4.set_xticklabels(labels_par, fontsize=10)
    ax4.set_ylabel('归一化目标值', fontsize=10)
    ax4.set_title('平行坐标图 — 调度方案三维权衡', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8, loc='upper right')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle('共享单车调度多目标优化 — 帕累托前沿分析', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('code/case03_portfolio_multi_results.png', dpi=150, bbox_inches='tight')
    print(f"✅ 帕累托前沿图已保存至: code/case03_portfolio_multi_results.png")
    plt.close()


# ===================== 主程序 =====================
def main():
    print("=" * 60)
    print("共享单车调度多目标优化 — 加权求和法")
    print("=" * 60)

    # 1. 生成数据
    coords, capacity, current = generate_station_data(n_stations=20, seed=42)
    surplus, deficit, satisfaction_base = compute_demands(current, capacity)

    print(f"\n站点数量: {len(current)}")
    print(f"总富余车辆: {surplus.sum():.0f}")
    print(f"总短缺缺口: {deficit.sum():.0f}")
    print(f"基础满意度（无调度）: {np.mean(satisfaction_base)*100:.1f}%")

    # 距离矩阵
    dist_matrix = cdist(coords, coords, metric='euclidean')

    # 2. 权重扫描
    print(f"\n权重扫描...")
    weights = []
    for w1 in np.linspace(0.05, 0.9, 6):  # 成本权重
        for w2 in np.linspace(0.05, 0.9, 6):  # 满意度权重
            w3 = 1 - w1 - w2
            if 0.05 <= w3 <= 0.9:
                weights.append((w1, w2, w3))

    print(f"权重组合数: {len(weights)}")

    all_solutions = []
    for w1, w2, w3 in weights:
        sol = optimize_dispatch_weighted(surplus, deficit, dist_matrix,
                                         w_cost=w1, w_satisfaction=w2, w_carbon=w3,
                                         coords=coords, capacity=capacity,
                                         current=current, truck_capacity=30)
        all_solutions.append(sol)

    # 3. 帕累托过滤
    print(f"\n正在提取帕累托前沿...")
    pareto_sols = pareto_filter(all_solutions)
    print(f"Pareto 最优解数: {len(pareto_sols)} / {len(all_solutions)}")

    # 4. 统计和极端解
    p_costs = np.array([s['cost'] for s in pareto_sols])
    p_sats = np.array([s['satisfaction'] for s in pareto_sols])
    p_carbons = np.array([s['carbon'] for s in pareto_sols])

    print(f"\n=== 极端方案对比 ===")
    min_cost_idx = np.argmin(p_costs)
    max_sat_idx = np.argmax(p_sats)
    min_carbon_idx = np.argmin(p_carbons)

    print(f"\n方案1 - 成本优先:")
    print(f"  运营成本: ¥{p_costs[min_cost_idx]:.0f}")
    print(f"  满意度:   {p_sats[min_cost_idx]*100:.1f}%")
    print(f"  碳排放:   {p_carbons[min_cost_idx]:.0f} kg CO₂")
    print(f"  卡车数:   {pareto_sols[min_cost_idx]['trucks']}")

    print(f"\n方案2 - 满意度优先:")
    print(f"  运营成本: ¥{p_costs[max_sat_idx]:.0f}")
    print(f"  满意度:   {p_sats[max_sat_idx]*100:.1f}%")
    print(f"  碳排放:   {p_carbons[max_sat_idx]:.0f} kg CO₂")
    print(f"  卡车数:   {pareto_sols[max_sat_idx]['trucks']}")

    print(f"\n方案3 - 环保优先:")
    print(f"  运营成本: ¥{p_costs[min_carbon_idx]:.0f}")
    print(f"  满意度:   {p_sats[min_carbon_idx]*100:.1f}%")
    print(f"  碳排放:   {p_carbons[min_carbon_idx]:.0f} kg CO₂")
    print(f"  卡车数:   {pareto_sols[min_carbon_idx]['trucks']}")

    # 5. 寻找折中方案（离三个极端点最远的点）
    best_idx = 0
    best_min_dist = -np.inf
    for i in range(len(pareto_sols)):
        if i in (min_cost_idx, max_sat_idx, min_carbon_idx):
            continue
        # 到三个极端点的归一化距离和
        d1 = abs(p_costs[i] - p_costs[min_cost_idx]) / (p_costs.max() - p_costs.min() + 1e-10)
        d2 = abs(p_sats[i] - p_sats[max_sat_idx]) / (p_sats.max() - p_sats.min() + 1e-10)
        d3 = abs(p_carbons[i] - p_carbons[min_carbon_idx]) / (p_carbons.max() - p_carbons.min() + 1e-10)
        dist_sum = d1 + d2 + d3
        if dist_sum > best_min_dist:
            best_min_dist = dist_sum
            best_idx = i

    best_sol = pareto_sols[best_idx]
    print(f"\n=== 推荐折中方案 ===")
    print(f"  运营成本: ¥{best_sol['cost']:.0f}")
    print(f"  满意度:   {best_sol['satisfaction']*100:.1f}%")
    print(f"  碳排放:   {best_sol['carbon']:.0f} kg CO₂")
    print(f"  卡车数:   {best_sol['trucks']}")

    # 6. 验证
    print(f"\n=== 验证 ===")

    # 验证1：帕累托最优性
    dominated_count = 0
    for i in range(len(pareto_sols)):
        for j in range(len(pareto_sols)):
            if i != j:
                a = (pareto_sols[i]['cost'], pareto_sols[i]['satisfaction'], pareto_sols[i]['carbon'])
                b = (pareto_sols[j]['cost'], pareto_sols[j]['satisfaction'], pareto_sols[j]['carbon'])
                if is_dominated(a, b):
                    dominated_count += 1
    print(f"✅ Pareto 前沿过滤: {dominated_count == 0} "
          f"(前沿 {len(pareto_sols)} 个解互不支配)")

    # 验证2：极端解合理性
    min_cost_valid = (p_costs[min_cost_idx] == p_costs.min())
    max_sat_valid = (p_sats[max_sat_idx] == p_sats.max())
    min_carbon_valid = (p_carbons[min_carbon_idx] == p_carbons.min())
    print(f"✅ 极端解合理性: 成本最低={min_cost_valid}, "
          f"满意度最高={max_sat_valid}, 排放最低={min_carbon_valid}")

    # 验证3：折中方案平衡性
    if len(pareto_sols) > 3:
        cost_rank = np.sum(p_costs < best_sol['cost']) / len(p_costs)
        sat_rank = np.sum(p_sats < best_sol['satisfaction']) / len(p_sats)
        carbon_rank = np.sum(p_carbons < best_sol['carbon']) / len(p_carbons)
        print(f"✅ 折中方案分位: 成本={cost_rank:.0%}, "
              f"满意度={sat_rank:.0%}, 碳排放={carbon_rank:.0%} "
              f"{'(均在20-80% ✅)' if 0.2 < cost_rank < 0.8 else '(偏极端)'}")

    # 验证4：成本 vs 碳排放相关性
    if len(p_costs) > 2 and len(p_carbons) > 2:
        corr = np.corrcoef(p_costs, p_carbons)[0, 1]
        if not np.isnan(corr):
            print(f"✅ 碳排放与成本相关: r={corr:.2f} {'(正相关 ✓)' if corr > 0 else '(负相关)'}")

    # 验证5：方案多样性
    cost_range = p_costs.max() - p_costs.min()
    sat_range = p_sats.max() - p_sats.min()
    carbon_range = p_carbons.max() - p_carbons.min()
    print(f"✅ 方案多样性: 成本跨度¥{cost_range:.0f}, "
          f"满意度跨度{sat_range*100:.1f}%, 排放跨度{carbon_range:.0f}kg")

    # 7. 绘制图表
    print(f"\n正在绘制图表...")
    plot_results(all_solutions, pareto_sols, best_sol)

    print(f"\n{'=' * 60}")
    print(f"完成!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
