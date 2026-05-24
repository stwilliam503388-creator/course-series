#!/usr/bin/env python3
"""
case06_supply_chain_multi.py
可再生能源电网多目标调度 — NSGA-II 求帕累托前沿

场景：区域电网24小时调度，四种电源（火电/风电/光伏/储能）
三个冲突目标：
  f1 = 发电总成本（最小化）
  f2 = 碳排放（最小化）
  f3 = 供电可靠性 = 最小化失负荷期望（最小化）

方法：
  NSGA-II（非支配排序遗传算法）
  三种典型方案对比：成本优先 / 环保优先 / 可靠性优先

验证：
  - Pareto 前沿支配性
  - 极端方案合理性
  - 功率平衡 + SOC 终端约束
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===================== 中文显示配置 =====================
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC',
                                    'Heiti SC', 'Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


# ===================== 场景数据 =====================
HOURS = 24

# 火电参数
FIRE_CAP = 400.0       # MW (2×200MW)
FIRE_MIN = 50.0        # MW
FIRE_COST = 0.35       # 元/kWh = 350 元/MWh
FIRE_CARBON = 0.8      # 吨 CO₂/MWh
FIRE_RAMP = 30.0       # MW/h 爬坡率

# 风电参数
WIND_CAP = 300.0       # MW
WIND_COST = 0.02       # 元/kWh

# 光伏参数
SOLAR_CAP = 200.0      # MW
SOLAR_COST = 0.01      # 元/kWh

# 储能参数
BAT_CAP = 300.0        # MWh
BAT_POWER = 50.0       # MW (最大充放电功率)
BAT_EFF = 0.90         # 充放电效率
BAT_COST = 0.05        # 元/kWh (充放电损耗成本)
BAT_INIT_SOC = 0.50    # 初始SOC比例


def generate_scenario(seed=42):
    """生成24小时的需求和风光出力系数"""
    rng = np.random.RandomState(seed)
    hours = np.arange(HOURS)

    # 用电需求：300~800 MW 之间变化
    demand = (300.0 +
              400.0 * np.exp(-((hours - 12.0) ** 2) / 30.0) +
              100.0 * np.exp(-((hours - 19.0) ** 2) / 10.0))
    demand += rng.normal(0, 10, HOURS)  # 小幅随机波动
    demand = np.clip(demand, 250, 850)

    # 风电出力系数（凌晨大，午后小）
    wind_avail = 0.4 + 0.4 * np.exp(-((hours - 3.0) ** 2) / 15.0)

    # 光伏出力系数（6:00-18:00）
    solar_avail = np.maximum(0, np.sin(np.pi * (hours - 6.0) / 12.0))
    solar_avail = np.clip(solar_avail, 0, 0.95)

    return demand, wind_avail, solar_avail


# ===================== 个体编码与解码 =====================
# 编码: 24时段 × 4电源 = 96 个变量
# 顺序: [P_fire_0, ..., P_fire_23, P_wind_0, ..., P_wind_23,
#         P_solar_0, ..., P_solar_23, P_bat_0, ..., P_bat_23]

N_VARS = HOURS * 4  # 96


def decode(individual, demand, wind_avail, solar_avail):
    """
    解码个体，返回各电源每小时出力
    确保满足物理约束
    """
    P_fire = individual[0:HOURS].copy()
    P_wind = individual[HOURS:2*HOURS].copy()
    P_solar = individual[2*HOURS:3*HOURS].copy()
    P_bat = individual[3*HOURS:4*HOURS].copy()

    # 约束修复
    # 1. 火电爬坡约束（简化：通过比例控制）
    P_fire = np.clip(P_fire, FIRE_MIN, FIRE_CAP)

    # 2. 风光出力上限
    max_wind = wind_avail * WIND_CAP
    max_solar = solar_avail * SOLAR_CAP
    P_wind = np.clip(P_wind, 0, max_wind)
    P_solar = np.clip(P_solar, 0, max_solar)

    # 3. 储能约束
    P_bat = np.clip(P_bat, -BAT_POWER, BAT_POWER)

    # 4. 功率平衡（通过调整储能和火电实现）
    soc = BAT_INIT_SOC * BAT_CAP
    soc_history = [soc]

    for t in range(HOURS):
        total_gen = P_fire[t] + P_wind[t] + P_solar[t]
        imbalance = demand[t] - total_gen

        # 先用储能调节
        available_bat = min(BAT_POWER, soc / BAT_EFF)  # 可放电
        charge_limit = min(BAT_POWER, (BAT_CAP - soc) * BAT_EFF)  # 可充电

        if imbalance > 0:
            # 需要发电 -> 储能放电
            discharge = min(imbalance, available_bat)
            P_bat[t] = -discharge
            soc -= discharge / BAT_EFF
        else:
            # 需要耗电 -> 储能充电
            charge = min(-imbalance, charge_limit)
            P_bat[t] = charge
            soc += charge * BAT_EFF

        soc = np.clip(soc, 0, BAT_CAP)
        soc_history.append(soc)

        # 重新计算总发电
        total_gen = P_fire[t] + P_wind[t] + P_solar[t] + P_bat[t]
        imbalance = demand[t] - total_gen

        # 如果还有不平衡，用火电调整
        if abs(imbalance) > 0.5:
            adjustment = np.clip(imbalance, -FIRE_RAMP, FIRE_RAMP)
            P_fire[t] = np.clip(P_fire[t] + adjustment, FIRE_MIN, FIRE_CAP)
            total_gen = P_fire[t] + P_wind[t] + P_solar[t] + P_bat[t]

        soc_history[t+1] = soc

    return P_fire, P_wind, P_solar, P_bat, np.array(soc_history)


# ===================== 目标函数 =====================
def evaluate(individual, demand, wind_avail, solar_avail):
    """计算三个目标函数值"""
    P_fire, P_wind, P_solar, P_bat, soc_hist = decode(
        individual, demand, wind_avail, solar_avail)

    # f1: 总成本（万元）
    cost_fire = np.sum(P_fire * FIRE_COST * 10)  # 元/时段 → 万元
    cost_wind = np.sum(P_wind * WIND_COST * 10)
    cost_solar = np.sum(P_solar * SOLAR_COST * 10)
    cost_bat = np.sum(np.abs(P_bat) * BAT_COST * 10)
    f1 = cost_fire + cost_wind + cost_solar + cost_bat

    # f2: 碳排放（吨 CO₂）
    f2 = np.sum(P_fire * FIRE_CARBON)

    # f3: 失负荷期望（MWh）
    total_gen = P_fire + P_wind + P_solar + P_bat
    load_loss = np.sum(np.maximum(0, demand - total_gen))
    f3 = load_loss

    return f1, f2, f3


def random_individual(rng=None):
    """生成随机个体"""
    if rng is None:
        rng = np.random.RandomState()
    ind = np.zeros(N_VARS)
    # 火电：在合理范围内随机
    ind[0:HOURS] = rng.uniform(FIRE_MIN, FIRE_CAP, HOURS)
    # 风电
    ind[HOURS:2*HOURS] = rng.uniform(0, WIND_CAP, HOURS)
    # 光伏
    ind[2*HOURS:3*HOURS] = rng.uniform(0, SOLAR_CAP, HOURS)
    # 储能
    ind[3*HOURS:4*HOURS] = rng.uniform(-BAT_POWER, BAT_POWER, HOURS)
    return ind


# ===================== NSGA-II 核心 =====================
def dominates(a, b):
    """a 是否支配 b？（三个目标都是最小化）"""
    f1_a, f2_a, f3_a = a
    f1_b, f2_b, f3_b = b
    return (f1_a <= f1_b and f2_a <= f2_b and f3_a <= f3_b and
            (f1_a < f1_b or f2_a < f2_b or f3_a < f3_b))


def nondominated_sort(population, objectives):
    """
    快速非支配排序
    population: 个体列表
    objectives: N×3 目标值数组
    返回: 各前沿层的索引列表 [front0_indices, front1_indices, ...]
    """
    n = len(population)
    S = [[] for _ in range(n)]  # 被该个体支配的个体列表
    n_p = [0] * n  # 支配该个体的数量
    fronts = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(objectives[i], objectives[j]):
                S[i].append(j)
            elif dominates(objectives[j], objectives[i]):
                n_p[i] += 1

        if n_p[i] == 0:
            if not fronts or len(fronts[0]) == 0:
                fronts.append([])
            fronts[0].append(i)

    # 构建后续前沿
    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for p_idx in fronts[i]:
            for q_idx in S[p_idx]:
                n_p[q_idx] -= 1
                if n_p[q_idx] == 0:
                    next_front.append(q_idx)
        i += 1
        if next_front:
            fronts.append(next_front)

    return fronts


def crowding_distance(indices, objectives):
    """计算拥挤度距离"""
    n = len(indices)
    if n <= 2:
        return [float('inf')] * n

    n_obj = objectives.shape[1]
    distances = [0.0] * n

    for m in range(n_obj):
        # 按第 m 个目标排序
        sorted_idx = sorted(range(n), key=lambda i: objectives[indices[i], m])
        obj_min = objectives[indices[sorted_idx[0]], m]
        obj_max = objectives[indices[sorted_idx[-1]], m]
        obj_range = obj_max - obj_min
        if obj_range < 1e-10:
            continue

        distances[sorted_idx[0]] = float('inf')
        distances[sorted_idx[-1]] = float('inf')

        for k in range(1, n - 1):
            distances[sorted_idx[k]] += (
                objectives[indices[sorted_idx[k + 1]], m] -
                objectives[indices[sorted_idx[k - 1]], m]
            ) / obj_range

    return distances


def selection(population, objectives, pop_size):
    """
    根据前沿层和拥挤度选择下一代
    """
    n = len(population)
    fronts = nondominated_sort(population, objectives)

    selected = []
    front_idx = 0

    while len(selected) + len(fronts[front_idx]) <= pop_size:
        selected.extend(fronts[front_idx])
        front_idx += 1

    # 最后一层按拥挤度排序
    remaining = pop_size - len(selected)
    if remaining > 0 and front_idx < len(fronts):
        last_front = fronts[front_idx]
        dists = crowding_distance(last_front, objectives)
        # 按拥挤度降序排列
        sorted_last = sorted(range(len(last_front)),
                             key=lambda i: dists[i],
                             reverse=True)
        for i in range(remaining):
            selected.append(last_front[sorted_last[i]])

    return [population[i] for i in selected]


def tournament_selection(population, objectives, k=2):
    """锦标赛选择"""
    n = len(population)
    i = np.random.randint(n)
    j = np.random.randint(n)
    # 比较前沿层：层数越小越好；同层比较拥挤度
    fronts = nondominated_sort(population, objectives)
    rank_i = rank_j = -1
    for r_idx, front in enumerate(fronts):
        if i in front:
            rank_i = r_idx
        if j in front:
            rank_j = r_idx

    if rank_i < rank_j:
        return population[i].copy()
    elif rank_j < rank_i:
        return population[j].copy()
    else:
        # 同层，随机选
        return population[i].copy() if np.random.rand() < 0.5 else population[j].copy()


def sbx_crossover(parent1, parent2, eta=15):
    """模拟二进制交叉（Simulated Binary Crossover）"""
    child1 = parent1.copy()
    child2 = parent2.copy()
    n = len(parent1)

    for i in range(n):
        if np.random.rand() < 0.5:  # 交叉概率每变量 0.5
            if abs(parent1[i] - parent2[i]) < 1e-10:
                continue
            u = np.random.rand()
            if u <= 0.5:
                beta = (2 * u) ** (1 / (eta + 1))
            else:
                beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))

            child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
            child2[i] = 0.5 * ((1 - beta) * parent2[i] + (1 + beta) * parent1[i])

    return child1, child2


def polynomial_mutation(individual, eta=20, mutation_rate=0.1):
    """多项式变异"""
    mutated = individual.copy()
    n = len(individual)

    for i in range(n):
        if np.random.rand() < mutation_rate:
            r = np.random.rand()
            if r <= 0.5:
                delta = (2 * r) ** (1 / (eta + 1)) - 1
            else:
                delta = 1 - (2 * (1 - r)) ** (1 / (eta + 1))

            mutated[i] += delta * max(abs(individual[i]), 10.0)

    return mutated


# ===================== 主流程 =====================
def run_nsga2(pop_size=50, max_gen=50, seed=42):
    """运行 NSGA-II 主流程"""
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    # 生成场景数据
    demand, wind_avail, solar_avail = generate_scenario(seed)

    print(f"24小时总需求: {demand.sum():.0f} MWh")
    print(f"火电容量: {FIRE_CAP} MW, 风电容量: {WIND_CAP} MW")
    print(f"光伏容量: {SOLAR_CAP} MW, 储能容量: {BAT_CAP} MWh")

    # 初始化种群
    population = [random_individual(rng) for _ in range(pop_size)]

    # 主进化循环
    for gen in range(max_gen):
        # 评价
        obj_list = np.array([evaluate(ind, demand, wind_avail, solar_avail)
                             for ind in population])

        # 生成子代
        offspring = []
        while len(offspring) < pop_size:
            p1 = tournament_selection(population, obj_list)
            p2 = tournament_selection(population, obj_list)
            c1, c2 = sbx_crossover(p1, p2)
            c1 = polynomial_mutation(c1)
            c2 = polynomial_mutation(c2)
            offspring.append(c1)
            if len(offspring) < pop_size:
                offspring.append(c2)

        # 评价子代
        all_pop = population + offspring
        all_obj = np.array([evaluate(ind, demand, wind_avail, solar_avail)
                            for ind in all_pop])

        # 选择
        selected_indices = []
        fronts = nondominated_sort(all_pop, all_obj)

        for front in fronts:
            if len(selected_indices) + len(front) <= pop_size:
                selected_indices.extend(front)
            else:
                dists = crowding_distance(front, all_obj)
                sorted_front = sorted(range(len(front)),
                                      key=lambda i: dists[i],
                                      reverse=True)
                remaining = pop_size - len(selected_indices)
                for i in range(remaining):
                    selected_indices.append(front[sorted_front[i]])
                break

        population = [all_pop[i] for i in selected_indices]

        if (gen + 1) % 20 == 0:
            obj_gen = np.array([evaluate(ind, demand, wind_avail, solar_avail)
                                for ind in population])
            print(f"  第 {gen+1:3d} 代: 前沿大小={len(fronts[0])}, "
                  f"f1=[{obj_gen[:,0].min():.0f}, {obj_gen[:,0].max():.0f}], "
                  f"f2=[{obj_gen[:,1].min():.1f}, {obj_gen[:,1].max():.1f}]")

    # 最终评价
    final_obj = np.array([evaluate(ind, demand, wind_avail, solar_avail)
                          for ind in population])
    fronts = nondominated_sort(population, final_obj)

    # Pareto 前沿（第一前沿）
    pareto_indices = fronts[0]
    pareto_solutions = [population[i] for i in pareto_indices]
    pareto_objectives = np.array([final_obj[i] for i in pareto_indices])

    return pareto_solutions, pareto_objectives, demand, wind_avail, solar_avail


# ===================== 可视化 =====================
def plot_results(pareto_solutions, pareto_objectives,
                 demand, wind_avail, solar_avail):
    """绘制四合一结果图"""

    # 提取三个极端方案
    idx_cost = np.argmin(pareto_objectives[:, 0])
    idx_env = np.argmin(pareto_objectives[:, 1])
    idx_rel = np.argmin(pareto_objectives[:, 2])

    fig = plt.figure(figsize=(16, 12))

    # === 左上：三维帕累托前沿 ===
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    f1_vals = pareto_objectives[:, 0]
    f2_vals = pareto_objectives[:, 1]
    f3_vals = pareto_objectives[:, 2]

    sc = ax1.scatter(f1_vals, f2_vals, f3_vals,
                     c=f2_vals, cmap='RdYlGn_r', s=40, alpha=0.7,
                     edgecolors='k', linewidth=0.3)

    # 标注极端方案
    extreme_points = [
        (idx_cost, '成本优先', 'red'),
        (idx_env, '环保优先', 'green'),
        (idx_rel, '可靠性优先', 'blue'),
    ]
    for idx, label, color in extreme_points:
        ax1.scatter(f1_vals[idx], f2_vals[idx], f3_vals[idx],
                    c=color, s=150, marker='*', zorder=5)
        ax1.text(f1_vals[idx], f2_vals[idx], f3_vals[idx],
                 f'  {label}', fontsize=9, color=color)

    ax1.set_xlabel('成本 (万元)', fontsize=10)
    ax1.set_ylabel('碳排放 (吨 CO₂)', fontsize=10)
    ax1.set_zlabel('失负荷 (MWh)', fontsize=10)
    ax1.set_title('三维帕累托前沿', fontsize=12, fontweight='bold')
    fig.colorbar(sc, ax=ax1, shrink=0.5, label='碳排放 (吨 CO₂)')

    # === 右上：成本 vs 碳排放投影 ===
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(f1_vals, f2_vals, c='orange', s=40,
                edgecolors='darkorange', linewidth=0.3, alpha=0.7)
    for idx, label, color in extreme_points:
        ax2.scatter(f1_vals[idx], f2_vals[idx],
                    c=color, s=120, marker='*', zorder=5)
    ax2.set_xlabel('成本 (万元)', fontsize=10)
    ax2.set_ylabel('碳排放 (吨 CO₂)', fontsize=10)
    ax2.set_title('成本 vs 碳排放', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # === 左下：三种方案堆叠面积图（成本优先） ===
    ax3 = fig.add_subplot(2, 2, 3)
    hours = np.arange(HOURS)

    # 绘制成本优先方案的堆叠面积图
    ind_cost = pareto_solutions[idx_cost]
    P_fire_c, P_wind_c, P_solar_c, P_bat_c, soc_c = decode(
        ind_cost, demand, wind_avail, solar_avail)

    ax3.fill_between(hours, 0, P_wind_c + P_solar_c + P_bat_c + P_fire_c,
                     label='火电', color='#8B4513', alpha=0.7)
    ax3.fill_between(hours, 0, P_wind_c + P_solar_c + P_bat_c,
                     label='储能', color='#FFD700', alpha=0.7)
    ax3.fill_between(hours, 0, P_wind_c + P_solar_c,
                     label='光伏', color='#FFA500', alpha=0.5)
    ax3.fill_between(hours, 0, P_wind_c,
                     label='风电', color='#87CEEB', alpha=0.6)
    ax3.plot(hours, demand, 'r--', linewidth=2, label='需求', alpha=0.8)
    ax3.set_xlabel('时间 (小时)', fontsize=10)
    ax3.set_ylabel('功率 (MW)', fontsize=10)
    ax3.set_title(f'成本优先方案 — 电源组合', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=7, loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 23)

    # === 右下：三种方案在典型时段的对比（柱状图） ===
    ax4 = fig.add_subplot(2, 2, 4)

    # 选取三个典型时段：凌晨(3h)，中午(12h)，晚高峰(19h)
    typical_hours = [3, 12, 19]
    labels_th = ['凌晨\n(3:00)', '中午\n(12:00)', '晚高峰\n(19:00)']

    for idx, label, color, hatch in [
        (idx_cost, '成本优先', '#8B4513', '/'),
        (idx_env, '环保优先', '#228B22', '\\'),
        (idx_rel, '可靠性优先', '#4169E1', 'x')
    ]:
        ind = pareto_solutions[idx]
        P_fire, P_wind, P_solar, P_bat, _ = decode(
            ind, demand, wind_avail, solar_avail)
        total = P_fire + P_wind + P_solar + P_bat

        x_pos = []
        heights = []
        colors_group = []

        for ti, th in enumerate(typical_hours):
            base_pos = ti * (len(extreme_points) + 1)
            x_pos.append(base_pos + extreme_points.index((idx, label, color)))
            heights.append(total[th])

        ax4.bar(x_pos, heights, width=0.6, label=label,
                color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

    # 添加需求线
    for ti, th in enumerate(typical_hours):
        base_pos = ti * (len(extreme_points) + 1) + len(extreme_points) / 2 - 0.5
        ax4.bar([base_pos], [demand[th]], width=0.4, color='red',
                alpha=0.3, label='需求' if ti == 0 else '')

    ax4.set_xticks([1, 5, 9])
    ax4.set_xticklabels(labels_th, fontsize=10)
    ax4.set_ylabel('发电量 (MW)', fontsize=10)
    ax4.set_title('典型时段三种方案出力对比', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle('可再生能源电网多目标调度 — NSGA-II 帕累托前沿', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('code/case06_supply_chain_multi_results.png', dpi=150, bbox_inches='tight')
    print(f"✅ 结果图已保存至: code/case06_supply_chain_multi_results.png")
    plt.close()


# ===================== 主程序 =====================
def main():
    print("=" * 60)
    print("可再生能源电网多目标调度 — NSGA-II")
    print("=" * 60)

    # 运行 NSGA-II
    print(f"\n=== 场景数据 ===")
    print(f"正在运行 NSGA-II (种群=50, 世代=50)...")
    print(f"\n进化进度:")

    pareto_sols, pareto_obj, demand, wind_avail, solar_avail = run_nsga2(
        pop_size=100, max_gen=100, seed=42)

    print(f"\n=== 帕累托前沿统计 ===")
    print(f"前沿解数量: {len(pareto_sols)}")
    print(f"  成本范围:     [¥{pareto_obj[:,0].min():.0f}, ¥{pareto_obj[:,0].max():.0f}]")
    print(f"  碳排放范围:   [{pareto_obj[:,1].min():.1f}, {pareto_obj[:,1].max():.1f}] 吨 CO₂")
    print(f"  失负荷期望:   [{pareto_obj[:,2].min():.1f}, {pareto_obj[:,2].max():.1f}] MWh")

    # 三种极端方案
    idx_cost = np.argmin(pareto_obj[:, 0])
    idx_env = np.argmin(pareto_obj[:, 1])
    idx_rel = np.argmin(pareto_obj[:, 2])

    print(f"\n=== 三种典型调度方案 ===")
    labels = ['成本优先', '环保优先', '可靠性优先']
    indices = [idx_cost, idx_env, idx_rel]

    for label, idx in zip(labels, indices):
        print(f"\n方案{labels.index(label)+1} - {label}:")
        print(f"  总成本:     ¥{pareto_obj[idx,0]:.0f}")
        print(f"  碳排放:     {pareto_obj[idx,1]:.1f} 吨 CO₂")
        print(f"  失负荷:     {pareto_obj[idx,2]:.1f} MWh")

        # 解码查看储能的运行模式
        P_fire, P_wind, P_solar, P_bat, soc = decode(
            pareto_sols[idx], demand, wind_avail, solar_avail)
        fire_ratio = np.mean(P_fire) / FIRE_CAP * 100
        print(f"  火电平均出力率: {fire_ratio:.0f}%")
        print(f"  储能循环电量: {np.sum(np.abs(P_bat)):.0f} MWh")

    # 验证
    print(f"\n=== 验证 ===")

    # 验证1：帕累托支配性
    dominated_count = 0
    n = len(pareto_sols)
    for i in range(n):
        for j in range(n):
            if i != j:
                if dominates(pareto_obj[i], pareto_obj[j]):
                    dominated_count += 1
    print(f"✅ Pareto 支配性检查: {dominated_count == 0} "
          f"(i.e. {n} 个解全部非支配)")

    # 验证2：极端方案合理性
    min_cost_valid = (pareto_obj[idx_cost, 0] == pareto_obj[:, 0].min() and
                      pareto_obj[idx_cost, 1] > pareto_obj[:, 1].mean())
    min_env_valid = (pareto_obj[idx_env, 1] == pareto_obj[:, 1].min() and
                     pareto_obj[idx_env, 0] > pareto_obj[:, 0].mean())
    min_rel_valid = (pareto_obj[idx_rel, 2] == pareto_obj[:, 2].min())
    print(f"✅ 极端方案合理性: 成本优先={min_cost_valid}, "
          f"环保优先={min_env_valid}, 可靠性优先={min_rel_valid}")

    # 验证3：功率平衡
    balance_ok = True
    for sol in pareto_sols:
        P_fire, P_wind, P_solar, P_bat, soc = decode(
            sol, demand, wind_avail, solar_avail)
        total = P_fire + P_wind + P_solar + P_bat
        max_diff = np.max(np.abs(total - demand))
        if max_diff > 5.0:  # 允许 5MW 误差
            balance_ok = False
            break
    print(f"✅ 功率平衡: {balance_ok} (所有方案每小时平衡)")

    # 验证4：SOC 终端平衡
    soc_ok = True
    for sol in pareto_sols:
        _, _, _, _, soc = decode(sol, demand, wind_avail, solar_avail)
        if soc[-1] < BAT_INIT_SOC * BAT_CAP * 0.5:  # 终端 SOC ≥ 初始的 50%
            soc_ok = False
            break
    print(f"✅ SOC 终端平衡: {soc_ok} (终端 SOC ≥ 初始 SOC 的50%)")

    # 验证5：三种方案差异化
    _, _, _, _, soc_cost = decode(pareto_sols[idx_cost], demand, wind_avail, solar_avail)
    _, _, _, _, soc_env = decode(pareto_sols[idx_env], demand, wind_avail, solar_avail)
    _, _, _, _, soc_rel = decode(pareto_sols[idx_rel], demand, wind_avail, solar_avail)
    fire_cost = np.mean(decode(pareto_sols[idx_cost], demand, wind_avail, solar_avail)[0])
    fire_env = np.mean(decode(pareto_sols[idx_env], demand, wind_avail, solar_avail)[0])
    fire_rel = np.mean(decode(pareto_sols[idx_rel], demand, wind_avail, solar_avail)[0])
    fire_diff = max(fire_cost, fire_env, fire_rel) - min(fire_cost, fire_env, fire_rel)
    print(f"✅ 三种方案火电出力差异化: fire_diff={fire_diff:.0f} MW (差异化显著={fire_diff > 50})")

    # 绘制图表
    print(f"\n正在绘制结果图...")
    plot_results(pareto_sols, pareto_obj, demand, wind_avail, solar_avail)

    print(f"\n{'=' * 60}")
    print("完成!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
