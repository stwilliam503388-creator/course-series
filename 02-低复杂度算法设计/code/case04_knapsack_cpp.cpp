/**
 * 案例 4：0/1 背包问题（C++ 实现）
 * =====================================
 *
 * 核心算法：0/1 背包动态规划
 *
 * 功能：
 *   - generate_data(n, max_weight)           生成随机物品数据
 *   - solve_bruteforce(weights, values, W)   回溯枚举所有子集 O(2ⁿ)
 *   - solve_dp(weights, values, W)           二维 DP O(n·W)
 *   - space_optimized(weights, values, W)    一维滚动数组 DP
 *   - greedy_density(weights, values, W)     贪心（按单位价值，展示反例）
 *   - benchmark()                            生成性能对比表
 *   - verify()                               验证各方法结果一致
 *
 * 编译: g++ -O2 -std=c++11 case04_knapsack_cpp.cpp -o case04_knapsack_cpp
 * 运行: ./case04_knapsack_cpp [benchmark|table]
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <iomanip>
#include <string>
#include <sstream>
#include <cassert>

using namespace std;

static mt19937 rng(42);

// ============================================================
// 1. 数据生成
// ============================================================

/**
 * 生成背包测试数据。
 *
 * @param n          物品数量
 * @param max_weight 物品最大重量（1 ~ max_weight）
 * @return (weights, values) 通过引用参数返回
 */
void generate_data(vector<int> &weights, vector<int> &values, int n, int max_weight = 10) {
    weights.resize(n);
    values.resize(n);
    for (int i = 0; i < n; ++i) {
        weights[i] = uniform_int_distribution<int>(1, max_weight)(rng);
        values[i]  = uniform_int_distribution<int>(1, max_weight * 2)(rng);
    }
}

// ============================================================
// 2. 暴力解法：回溯枚举所有子集
// ============================================================

/**
 * 暴力枚举所有物品组合（位运算）。
 *
 * 每个物品「选或不选」，共 2ⁿ 种组合。
 * 时间复杂度: O(n · 2ⁿ)
 */
int solve_bruteforce(const vector<int> &weights, const vector<int> &values, int W) {
    int n = weights.size();
    int best_value = 0;

    // 用迭代法枚举所有子集（位运算）
    for (int mask = 0; mask < (1 << n); ++mask) {
        int cur_weight = 0, cur_value = 0;
        for (int i = 0; i < n; ++i) {
            if (mask & (1 << i)) {
                cur_weight += weights[i];
                cur_value  += values[i];
            }
        }
        if (cur_weight <= W && cur_value > best_value) {
            best_value = cur_value;
        }
    }
    return best_value;
}

// ============================================================
// 3. 二维 DP
// ============================================================

/**
 * 二维动态规划求解 0/1 背包。
 *
 * 状态定义：
 *   dp[i][j] = 前 i 个物品中选，总重量不超过 j 的最大总价值
 *
 * 时间复杂度: O(n·W)
 * 空间复杂度: O(n·W)
 */
int solve_dp(const vector<int> &weights, const vector<int> &values, int W) {
    int n = weights.size();
    // 创建 (n+1) × (W+1) 的二维数组，初始化为 0
    vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));

    for (int i = 1; i <= n; ++i) {
        int w_i = weights[i - 1];
        int v_i = values[i - 1];
        for (int j = 0; j <= W; ++j) {
            if (j < w_i) {
                dp[i][j] = dp[i - 1][j];  // 装不下，只能不选
            } else {
                dp[i][j] = max(dp[i - 1][j],            // 不选
                               dp[i - 1][j - w_i] + v_i); // 选
            }
        }
    }
    return dp[n][W];
}

// ============================================================
// 4. 空间优化版本：一维滚动数组
// ============================================================

/**
 * 一维滚动数组 DP（空间优化版）。
 *
 * 核心观察：dp[i][j] 只依赖于 dp[i-1][...]（上一行）
 *
 * 关键细节：j 必须从 W 到 0 倒序遍历！
 *
 * 时间复杂度: O(n·W)
 * 空间复杂度: O(W)
 */
int space_optimized(const vector<int> &weights, const vector<int> &values, int W) {
    int n = weights.size();
    vector<int> dp(W + 1, 0);

    for (int i = 0; i < n; ++i) {
        int w_i = weights[i];
        int v_i = values[i];
        // 倒序遍历：保证 dp[j - w_i] 还是上一轮的旧值
        for (int j = W; j >= w_i; --j) {
            dp[j] = max(dp[j], dp[j - w_i] + v_i);
        }
    }
    return dp[W];
}

// ============================================================
// 5. 贪心解法（用于展示反例）
// ============================================================

/**
 * 贪心算法：按单位价值（价值/重量）从高到低选物品。
 *
 * 重要：这个解法是「错误」的——它不一定给出最优解。
 *
 * 时间复杂度: O(n log n)
 * 空间复杂度: O(n)
 */
int greedy_density(const vector<int> &weights, const vector<int> &values, int W) {
    int n = weights.size();

    // 计算每个物品的单位价值
    // 使用 vector<pair<double, pair<int,int>>> 储值 (密度, (重量, 价值))
    vector<pair<double, pair<int,int>>> items(n);
    for (int i = 0; i < n; ++i) {
        double density = static_cast<double>(values[i]) / weights[i];
        items[i] = make_pair(density, make_pair(weights[i], values[i]));
    }

    // 按单位价值从高到低排序
    sort(items.begin(), items.end(),
         [](const pair<double, pair<int,int>> &a,
            const pair<double, pair<int,int>> &b) {
             return a.first > b.first;
         });

    int total_weight = 0;
    int total_value = 0;

    for (int i = 0; i < n; ++i) {
        int w = items[i].second.first;
        int v = items[i].second.second;
        if (total_weight + w <= W) {
            total_weight += w;
            total_value += v;
        }
    }
    return total_value;
}

// ============================================================
// 6. 打印 DP 表（教学演示）
// ============================================================

/**
 * 打印二维 DP 表的填表过程。
 */
void dp_print_table(const vector<int> &weights, const vector<int> &values, int W) {
    int n = weights.size();
    vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));

    cout << endl;
    cout << "物品列表：" << endl;
    cout << "  " << setw(4) << "物品" << " | " << setw(4) << "重量" << " | " << setw(4) << "价值" << endl;
    cout << "  " << string(18, '-') << endl;
    for (int i = 0; i < n; ++i) {
        cout << "  " << setw(4) << i + 1 << " | " << setw(4) << weights[i]
             << " | " << setw(4) << values[i] << endl;
    }

    cout << endl;
    cout << "背包容量 W = " << W << endl;
    cout << endl;
    cout << "DP 填表过程：" << endl;

    // 打印表头
    cout << "  " << setw(5) << "i/j";
    for (int j = 0; j <= W; ++j) cout << setw(4) << j;
    cout << endl;

    for (int i = 0; i <= n; ++i) {
        if (i == 0)
            cout << "  " << setw(5) << "0(空)";
        else
            cout << "  " << setw(4) << i << "  ";

        for (int j = 0; j <= W; ++j) {
            int val;
            if (i == 0) {
                val = 0;
            } else {
                int w_i = weights[i - 1];
                int v_i = values[i - 1];
                if (j < w_i)
                    val = dp[i - 1][j];
                else
                    val = max(dp[i - 1][j], dp[i - 1][j - w_i] + v_i);
                dp[i][j] = val;
            }
            cout << setw(4) << val;
        }
        cout << endl;
    }

    cout << endl;
    cout << "最优解: dp[" << n << "][" << W << "] = " << dp[n][W] << endl;
}

// ============================================================
// 7. 基准测试
// ============================================================

/**
 * 基准测试：对比不同规模的背包求解性能。
 */
void benchmark() {
    cout << "========================================================================" << endl;
    cout << "案例 4：0/1 背包基准测试（C++）" << endl;
    cout << "========================================================================" << endl;

    struct Scenario {
        int n, W;
        const char *desc;
    };
    Scenario scenarios[] = {
        {5,  10,  "小规模 n=5"},
        {10, 20,  "中规模 n=10"},
        {20, 50,  "较大规模 n=20"},
        {30, 100, "大规模 n=30"},
        {40, 200, "超大规模 n=40"},
    };

    cout << endl;
    cout << right << setw(4) << "n" << " | " << setw(4) << "W"
         << " | " << setw(14) << "暴力回溯"
         << " | " << setw(14) << "二维 DP"
         << " | " << setw(14) << "滚动数组"
         << " | " << setw(10) << "DP加速比" << endl;
    cout << string(72, '-') << endl;

    for (int si = 0; si < 5; ++si) {
        int n = scenarios[si].n;
        int W = scenarios[si].W;

        vector<int> weights, values;
        generate_data(weights, values, n, W / 4);

        // 调整 W 使其合理
        int total_w = 0;
        for (int i = 0; i < n; ++i) total_w += weights[i];
        int W_actual = min(W, total_w / 2);

        // 暴力回溯（n <= 20 才跑，否则跳过）
        string brute_str;
        double time_brute = 1e9;

        if (n <= 20) {
            auto t1 = chrono::high_resolution_clock::now();
            /* int brute_val = */ solve_bruteforce(weights, values, W_actual);
            auto t2 = chrono::high_resolution_clock::now();
            time_brute = chrono::duration<double>(t2 - t1).count();
            ostringstream oss;
            oss << fixed << setprecision(6) << time_brute << "s";
            brute_str = oss.str();
        } else {
            brute_str = "   ❌ 超时  ";
        }

        // 二维 DP
        auto t1 = chrono::high_resolution_clock::now();
        int dp_val = solve_dp(weights, values, W_actual);
        auto t2 = chrono::high_resolution_clock::now();
        double time_dp = chrono::duration<double>(t2 - t1).count();

        // 滚动数组
        t1 = chrono::high_resolution_clock::now();
        int roll_val = space_optimized(weights, values, W_actual);
        t2 = chrono::high_resolution_clock::now();
        double time_roll = chrono::duration<double>(t2 - t1).count();

        // 验证 DP 和滚动数组结果一致
        assert(dp_val == roll_val);

        // 加速比（暴力 / DP）
        string ratio_str;
        if (time_brute < 1e8) {
            double ratio = time_brute / time_dp;
            ostringstream oss;
            oss << fixed << setprecision(1) << ratio << "x";
            ratio_str = oss.str();
        } else {
            ratio_str = "    ∞   ";
        }

        cout << right << setw(4) << n << " | " << setw(4) << W_actual
             << " | " << setw(14) << brute_str
             << " | " << setw(10) << fixed << setprecision(6) << time_dp << "s"
             << " | " << setw(10) << fixed << setprecision(6) << time_roll << "s"
             << " | " << setw(10) << ratio_str << endl;
    }

    // ---- 贪心反例演示 ----
    cout << endl;
    cout << "========================================================================" << endl;
    cout << "贪心反例演示" << endl;
    cout << "========================================================================" << endl;

    // 构造一个经典的让贪心失败的例子
    // 物品 1: w=4, v=12, 单价=3.0  ← 单价最高
    // 物品 2: w=3, v=8,  单价≈2.67
    // 物品 3: w=3, v=7,  单价≈2.33
    // W = 6
    // 贪心: 选物品1(单价3.0) → 容量剩2 → 结束 → 总价值=12 ❌
    // 最优: 选物品2+物品3 = 总重6 → 总价值=15 ✅
    int weights_ex[] = {4, 3, 3};
    int values_ex[]  = {12, 8, 7};
    int W_ex = 6;
    vector<int> wv(weights_ex, weights_ex + 3);
    vector<int> vv(values_ex, values_ex + 3);

    int greedy_val = greedy_density(wv, vv, W_ex);
    int optimal_val = solve_dp(wv, vv, W_ex);

    cout << endl;
    cout << "经典反例：W=" << W_ex << endl;
    cout << "  物品 1: 重量=4, 价值=12, 单价=3.0  ← 单价最高" << endl;
    cout << "  物品 2: 重量=3, 价值=8,  单价≈2.67" << endl;
    cout << "  物品 3: 重量=3, 价值=7,  单价≈2.33" << endl;
    cout << endl;
    cout << "  贪心结果: " << greedy_val << " (选了物品 1, 容量剩 2, 装不下其他)" << endl;
    cout << "  DP 最优:  " << optimal_val << " (选物品 2 + 物品 3 = 总重 6)" << endl;
    cout << endl;
    cout << "  " << setw(12) << "贪心正确? "
         << (greedy_val != optimal_val ? "❌ 不是最优！" : "✅ 正确") << endl;

    cout << endl;
    cout << "结论：" << endl;
    cout << "  1. 暴力回溯在 n=20 时勉强可用，n=30 以上直接爆炸" << endl;
    cout << "  2. DP 在 n=40, W=200 时仍瞬间完成" << endl;
    cout << "  3. 贪心算法在某些情况下会失败（局部最优 ≠ 全局最优）" << endl;
    cout << "========================================================================" << endl;
}

// ============================================================
// 8. 正确性验证
// ============================================================

/**
 * 验证所有方法在多种输入下结果一致。
 */
void verify() {
    cout << "正在验证背包 DP 正确性..." << endl;

    struct TestCase {
        vector<int> weights, values;
        int W;
        string desc;
    };
    vector<TestCase> test_cases;

    // 边界测试
    {
        vector<int> w1(1, 1), v1(1, 5);
        test_cases.push_back({w1, v1, 0, "容量为 0"});
        test_cases.push_back({w1, v1, 1, "一个物品，刚好装下"});
        vector<int> w2(1, 5), v2(1, 5);
        test_cases.push_back({w2, v2, 3, "一个物品，装不下"});
    }

    // 小规模随机
    for (int n : {3, 5, 8, 10, 12}) {
        vector<int> w, v;
        generate_data(w, v, n, 8);
        int total_w = 0;
        for (int i = 0; i < n; ++i) total_w += w[i];
        int W = total_w / 2;
        test_cases.push_back({w, v, W, "随机 n=" + to_string(n)});
    }

    bool all_pass = true;
    for (size_t ti = 0; ti < test_cases.size(); ++ti) {
        const TestCase &tc = test_cases[ti];

        int brute_val = -1;
        if ((int)tc.weights.size() <= 15) {
            brute_val = solve_bruteforce(tc.weights, tc.values, tc.W);
        }

        int dp_val = solve_dp(tc.weights, tc.values, tc.W);
        int roll_val = space_optimized(tc.weights, tc.values, tc.W);

        // DP 和滚动数组必须一致
        if (dp_val != roll_val) {
            cout << "  ❌ DP 与滚动数组不一致: " << tc.desc << endl;
            all_pass = false;
        } else if (brute_val != -1 && dp_val != brute_val) {
            cout << "  ❌ DP 与暴力结果不一致: " << tc.desc
                 << " (DP=" << dp_val << ", brute=" << brute_val << ")" << endl;
            all_pass = false;
        } else {
            if (brute_val != -1) {
                cout << "  ✅ 通过: " << tc.desc << " (最优值=" << dp_val << ")" << endl;
            } else {
                cout << "  ✅ 通过: " << tc.desc << " (最优值=" << dp_val
                     << ", 暴力跳过 n=" << tc.weights.size() << ")" << endl;
            }
        }
    }

    if (all_pass)
        cout << "🎉 所有测试用例通过！" << endl;
    else
        cout << "⚠️  存在失败的测试用例！" << endl;
}

// ============================================================
// 9. 主入口
// ============================================================

int main(int argc, char *argv[]) {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc > 1) {
        string arg = argv[1];
        if (arg == "benchmark") {
            benchmark();
        } else if (arg == "table") {
            int w_arr[] = {4, 3, 5, 2, 1};
            int v_arr[] = {7, 5, 8, 3, 2};
            vector<int> weights(w_arr, w_arr + 5);
            vector<int> values(v_arr, v_arr + 5);
            dp_print_table(weights, values, 10);
        } else {
            cout << "用法: ./case04_knapsack_cpp [benchmark|table]" << endl;
        }
    } else {
        verify();
        cout << endl;
        cout << "提示：运行 './case04_knapsack_cpp benchmark' 查看性能对比" << endl;
        cout << "     运行 './case04_knapsack_cpp table' 打印 DP 填表过程" << endl;
    }
    return 0;
}
