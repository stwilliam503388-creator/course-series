/**
 * 案例 3: 1D 热传导方程 FDM 求解 — 芯片散热模拟（C++ 实现）
 * ============================================================
 *
 * 物理方程: ∂u/∂t = α ∂²u/∂x²
 * 数值方法: FTCS 显式格式
 * 边界条件: 左边界 Dirichlet (固定温度), 右边界 Neumann (绝热)
 *
 * 编译: g++ -O2 -std=c++11 case03_heat_cpp.cpp -o case03_heat_cpp
 * 运行: ./case03_heat_cpp
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <algorithm>
#include <chrono>

using namespace std;

// ============================================================
// 物理参数结构体
// ============================================================

struct Parameters {
    double L;           // 芯片长度 [m] = 10 mm
    double alpha;       // 热扩散率 [m²/s]
    double T_left;      // 左边界温度 [°C] (Dirichlet)
    double T_initial;   // 初始温度 [°C]
    int nx;             // 空间网格数
    int nt;             // 时间步数
};

/**
 * 设置物理和数值参数。
 *
 * 场景: 手机芯片满负荷运行
 *   - 芯片长度 L = 10 mm
 *   - 热扩散率 α = 1.0 × 10⁻⁴ m²/s (硅的典型值)
 *   - 初始温度 u(x,0) = 25 °C (环境温度)
 *   - 左边界温度 u(0,t) = 85 °C (芯片热点)
 *   - 右边界绝热 ∂u/∂x(L,t) = 0
 */
Parameters set_parameters() {
    return {
        0.01,         // L = 10 mm
        1.0e-4,       // alpha
        85.0,         // T_left
        25.0,         // T_initial
        50,           // nx
        2000          // nt
    };
}

// ============================================================
// 网格构建
// ============================================================

struct Mesh {
    vector<double> x;   // 坐标数组
    double dx;          // 网格间距
    double dt;          // 时间步长
    double dt_cfl;      // CFL 限制下的最大稳定时间步长
    int nx;             // 网格点数
};

/**
 * 构建一维网格。
 *
 * CFL 稳定性条件: α·dt/dx² ≤ 0.5
 */
Mesh build_mesh(const Parameters &params) {
    double L = params.L;
    int nx = params.nx;
    double alpha = params.alpha;

    double dx = L / (nx - 1);           // 空间步长
    double dt_cfl = 0.5 * dx * dx / alpha;
    double dt = dt_cfl * 0.9;           // 取 90% 以保证安全

    vector<double> x(nx);
    for (int i = 0; i < nx; ++i) {
        x[i] = i * dx;
    }

    return {x, dx, dt, dt_cfl, nx};
}

// ============================================================
// 初值条件
// ============================================================

/**
 * 设置初始温度分布 u(x, 0)。
 * 芯片初始温度均匀，等于环境温度。
 */
vector<double> initial_condition(const vector<double> &x, const Parameters &params) {
    vector<double> u(x.size(), params.T_initial);
    return u;
}

// ============================================================
// 显式 FTCS 求解
// ============================================================

/**
 * FTCS 显式格式求解 1D 热传导方程。
 *
 * u^{n+1}_i = u^n_i + r · (u^n_{i-1} - 2u^n_i + u^n_{i+1})
 * 其中 r = α·Δt/Δx²
 *
 * 边界条件:
 *   - 左边界 (x=0): Dirichlet, u = T_left
 *   - 右边界 (x=L): Neumann, ∂u/∂x = 0 (绝热)
 *
 * @return {u_final: 最后时刻温度场, u_all: 完整时间演化}
 */
struct HeatResult {
    vector<vector<double>> u_all;   // [nt+1][nx]
    vector<double> u_final;         // 最后时刻温度场
};

HeatResult solve_explicit(const Parameters &params, const Mesh &mesh) {
    double alpha = params.alpha;
    double T_left = params.T_left;
    int nx = mesh.nx;
    int nt = params.nt;
    double dx = mesh.dx;
    double dt = mesh.dt;
    const auto &x = mesh.x;

    double r = alpha * dt / (dx * dx);  // CFL 数
    cout << "[显式] r = α·Δt/Δx² = " << fixed << setprecision(4) << r
         << "  (稳定条件: r ≤ 0.5)" << endl;

    // 初始化温度场
    vector<vector<double>> u(nt + 1, vector<double>(nx));
    u[0] = initial_condition(x, params);

    // 时间推进
    for (int n = 0; n < nt; ++n) {
        // 内部点: FTCS 更新
        for (int i = 1; i < nx - 1; ++i) {
            u[n + 1][i] = u[n][i] + r * (u[n][i - 1] - 2.0 * u[n][i] + u[n][i + 1]);
        }

        // 左边界: Dirichlet (固定温度)
        u[n + 1][0] = T_left;

        // 右边界: Neumann (绝热, ∂u/∂x=0 → u_{nx-1} = u_{nx-2})
        u[n + 1][nx - 1] = u[n + 1][nx - 2];
    }

    return {u, u[nt]};
}

// ============================================================
// 稳态解析解
// ============================================================

/**
 * 稳态解析解。
 *
 * 对 1D 热传导方程 ∂u/∂t = α ∂²u/∂x² 的稳态 (∂u/∂t=0):
 *   d²u/dx² = 0 → u(x) = Ax + B
 *
 * 边界条件:
 *   u(0) = T_left
 *   ∂u/∂x(L) = 0 (绝热)
 *
 * 解: u(x) = T_left (因为绝热边界下, 整根棒温度均匀化到 T_left)
 */
vector<double> analytical_steady_state(const vector<double> &x, const Parameters &params) {
    return vector<double>(x.size(), params.T_left);
}

// ============================================================
// 可视化 (文本输出)
// ============================================================

/**
 * 以文本形式输出温度分布。
 */
void visualize(const HeatResult &result, const Mesh &mesh, const Parameters &params,
               const string &title) {
    const auto &x = mesh.x;
    const auto &u_final = result.u_final;
    int nx = mesh.nx;

    cout << endl;
    cout << "============================================================" << endl;
    cout << "  " << title << endl;
    cout << "============================================================" << endl;
    cout << "  网格点数: " << mesh.nx << ", "
         << "dx = " << fixed << setprecision(3) << mesh.dx * 1000 << " mm, "
         << "dt = " << fixed << setprecision(1) << mesh.dt * 1e6 << " μs" << endl;
    cout << endl;

    // 文本条形图
    cout << "  位置(mm) | 温度(°C) | 分布" << endl;
    cout << "  " << string(50, '-') << endl;
    int step = max(1, nx / 10);
    for (int i = 0; i < nx; i += step) {
        double xi = x[i] * 1000;  // 转 mm
        int bar_len = static_cast<int>((u_final[i] - params.T_initial) / 1.5);
        bar_len = max(0, min(bar_len, 40));
        cout << "  " << setw(8) << fixed << setprecision(2) << xi
             << "  | " << setw(8) << fixed << setprecision(2) << u_final[i]
             << " | " << string(bar_len, '#') << endl;
    }

    // 关键数据
    double max_temp = *max_element(u_final.begin(), u_final.end());
    double min_temp = *min_element(u_final.begin(), u_final.end());
    double sum_temp = 0;
    for (double t : u_final) sum_temp += t;
    double avg_temp = sum_temp / u_final.size();

    cout << endl;
    cout << "  最高温度: " << fixed << setprecision(2) << max_temp << " °C" << endl;
    cout << "  最低温度: " << fixed << setprecision(2) << min_temp << " °C" << endl;
    cout << "  平均温度: " << fixed << setprecision(2) << avg_temp << " °C" << endl;
    cout << endl;
}

// ============================================================
// 与解析解对比验证
// ============================================================

/**
 * 将数值解与解析解对比，验证代码正确性。
 * 对 Dirichlet-Dirichlet 边界（左右都固定温度）进行比较。
 */
void verify_with_analytical(const Parameters &params, const Mesh &mesh) {
    cout << "============================================================" << endl;
    cout << "解析验证: 与解析解对比" << endl;
    cout << "============================================================" << endl;

    double T_left = params.T_left;
    double T_initial = params.T_initial;
    double T_right = T_initial;  // 假设右边界固定为 T_initial

    int nx = mesh.nx;
    double dx = mesh.dx;
    const auto &x = mesh.x;
    double dt = mesh.dt;
    double alpha = params.alpha;
    double r = alpha * dt / (dx * dx);

    // 用显式 FTCS 计算到 t=0.5s
    int nt_verify = static_cast<int>(0.5 / dt);
    vector<vector<double>> u_num(nt_verify + 1, vector<double>(nx));
    u_num[0] = initial_condition(x, params);

    for (int n = 0; n < nt_verify; ++n) {
        for (int i = 1; i < nx - 1; ++i) {
            u_num[n + 1][i] = u_num[n][i]
                + r * (u_num[n][i - 1] - 2.0 * u_num[n][i] + u_num[n][i + 1]);
        }
        u_num[n + 1][0] = T_left;
        u_num[n + 1][nx - 1] = T_right;  // Dirichlet-Dirichlet
    }

    double t_final = nt_verify * dt;

    // 稳态解析解：线性分布 u(x) = T_left + (T_right - T_left) * x / L
    vector<double> u_analytical(nx);
    for (int i = 0; i < nx; ++i) {
        u_analytical[i] = T_left + (T_right - T_left) * x[i] / params.L;
    }

    // 计算误差
    double max_error = 0.0;
    for (int i = 0; i < nx; ++i) {
        double err = fabs(u_num[nt_verify][i] - u_analytical[i]);
        if (err > max_error) max_error = err;
    }

    cout << "  时间: t = " << fixed << setprecision(3) << t_final << "s" << endl;
    cout << "  最大绝对误差: " << fixed << setprecision(6) << max_error << " °C" << endl;
    cout << "  相对误差: " << fixed << setprecision(2)
         << max_error / (T_left - T_initial) * 100 << "%" << endl;
    cout << endl;

    // 在几个点抽样对比
    vector<int> sample_indices = {0, nx / 4, nx / 2, 3 * nx / 4, nx - 1};
    cout << "  " << setw(8) << "位置(mm)"
         << " " << setw(12) << "数值解(°C)"
         << " " << setw(12) << "解析解(°C)"
         << " " << setw(10) << "误差" << endl;
    cout << "  " << string(44, '-') << endl;
    for (int i : sample_indices) {
        double xi = x[i] * 1000;
        cout << "  " << setw(8) << fixed << setprecision(2) << xi
             << " " << setw(12) << fixed << setprecision(4) << u_num[nt_verify][i]
             << " " << setw(12) << fixed << setprecision(4) << u_analytical[i]
             << " " << setw(10) << fixed << setprecision(4)
             << fabs(u_num[nt_verify][i] - u_analytical[i]) << endl;
    }
}

// ============================================================
// 主入口
// ============================================================

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    cout << "╔══════════════════════════════════════════════════╗" << endl;
    cout << "║     1D 热传导方程 FDM 求解 — 芯片散热模拟       ║" << endl;
    cout << "╚══════════════════════════════════════════════════╝" << endl;
    cout << endl;

    // 1. 设置参数
    Parameters params = set_parameters();
    cout << "物理参数:" << endl;
    cout << "  芯片长度: " << params.L * 1000 << " mm" << endl;
    cout << "  热扩散率: " << scientific << setprecision(2) << params.alpha << " m²/s" << endl;
    cout << fixed;
    cout << "  左边界温度: " << params.T_left << " °C (Dirichlet)" << endl;
    cout << "  右边界: 绝热 (Neumann, ∂u/∂x=0)" << endl;
    cout << "  初始温度: " << params.T_initial << " °C" << endl;
    cout << endl;

    // 2. 构建网格
    Mesh mesh = build_mesh(params);
    cout << "空间网格: nx = " << mesh.nx << ", dx = "
         << fixed << setprecision(3) << mesh.dx * 1000 << " mm" << endl;
    cout << "时间步长: dt = " << fixed << setprecision(1) << mesh.dt * 1e6 << " μs" << endl;
    cout << "CFL 限制: dt ≤ " << fixed << setprecision(1) << mesh.dt_cfl * 1e6 << " μs"
         << " (α·dt/dx² ≤ 0.5)" << endl;
    cout << endl;

    // 3. 显式求解
    HeatResult result_explicit = solve_explicit(params, mesh);
    visualize(result_explicit, mesh, params, "显式 FTCS 格式结果");

    // 4. 解析验证
    verify_with_analytical(params, mesh);

    cout << endl;
    cout << "✓ 热传导模拟完成" << endl;

    return 0;
}
