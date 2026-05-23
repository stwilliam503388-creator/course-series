/**
 * 案例 3：归并排序（C++ 实现）
 * ==============================
 *
 * 功能：
 *   - generate_data(n, near_sorted=false)   生成测试数据
 *   - merge_sort(arr)                       归并排序
 *   - insertion_sort(arr)                   插入排序（接近有序对比）
 *   - benchmark()                           对比不同 n 的耗时
 *   - verify()                              正确性验证
 *
 * 编译: g++ -O2 -std=c++11 case03_merge_cpp.cpp -o case03_merge_cpp
 * 运行: ./case03_merge_cpp [benchmark]
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <iomanip>
#include <string>
#include <cassert>

// ============================================================
// 类型别名 & 全局随机数引擎
// ============================================================
using namespace std;
using Data = vector<int>;

static mt19937 rng(42);

// ============================================================
// 1. 数据生成
// ============================================================

/**
 * 生成测试数据。
 *
 * @param n            数据规模（元素个数）
 * @param near_sorted  是否生成「接近有序」的数据
 * @return 包含 n 个整数的 vector
 *
 * 说明：
 *   - near_sorted=false: 完全随机排列
 *   - near_sorted=true:  先有序，再随机挑 1% 的元素打乱
 */
Data generate_data(int n, bool near_sorted = false) {
    Data arr(n);

    if (near_sorted) {
        // 先生成一个有序数组
        for (int i = 0; i < n; ++i) arr[i] = i + 1;
        // 随机挑 1% 的位置打乱（至少 1 个）
        int swap_count = max(1, n / 100);
        for (int k = 0; k < swap_count; ++k) {
            int a = uniform_int_distribution<int>(0, n - 1)(rng);
            int b = uniform_int_distribution<int>(0, n - 1)(rng);
            swap(arr[a], arr[b]);
        }
    } else {
        // 完全随机：生成 1 ~ n*10 范围内的不重复随机数
        // 用 shuffle 实现随机排列
        for (int i = 0; i < n; ++i) arr[i] = i + 1;
        shuffle(arr.begin(), arr.end(), rng);
        // 随机缩放使数据范围变大
        for (int i = 0; i < n; ++i) arr[i] *= (1 + uniform_int_distribution<int>(0, 5)(rng));
    }
    return arr;
}

// ============================================================
// 2. 归并排序（分治）
// ============================================================

/**
 * 合并两个有序区间 [l, mid) 和 [mid, r) 到原数组。
 *
 * @param arr  待排序数组（引用，原地修改）
 * @param l    左边界（包含）
 * @param mid  中间分界
 * @param r    右边界（不包含）
 * @param tmp  临时缓冲区
 */
void merge(Data &arr, int l, int mid, int r, Data &tmp) {
    int i = l, j = mid, k = l;

    while (i < mid && j < r) {
        if (arr[i] <= arr[j])
            tmp[k++] = arr[i++];
        else
            tmp[k++] = arr[j++];
    }
    while (i < mid) tmp[k++] = arr[i++];
    while (j < r)   tmp[k++] = arr[j++];

    // 拷贝回原数组
    for (k = l; k < r; ++k) arr[k] = tmp[k];
}

/**
 * 归并排序（递归实现）。
 *
 * 分治三步曲：
 * 1. 分（Divide）：从中间切两半
 * 2. 解（Conquer）：递归排序两半
 * 3. 合（Combine）：合并两个有序子数组
 *
 * 时间复杂度: O(n log n)
 * 空间复杂度: O(n) —— 需要临时数组
 */
void merge_sort(Data &arr, int l, int r, Data &tmp) {
    if (r - l <= 1) return;

    int mid = l + (r - l) / 2;
    merge_sort(arr, l, mid, tmp);
    merge_sort(arr, mid, r, tmp);
    merge(arr, l, mid, r, tmp);
}

/**
 * 归并排序入口。
 */
Data merge_sort(Data arr) {
    Data tmp(arr.size());
    merge_sort(arr, 0, arr.size(), tmp);
    return arr;
}

// ============================================================
// 3. 插入排序（用于接近有序数据的对比）
// ============================================================

/**
 * 插入排序。
 *
 * 工作原理：像整理扑克牌——新拿到的牌插入到手中已排好序的牌的正确位置。
 *
 * 时间复杂度:
 *   - 平均/最差: O(n²)
 *   - 最好: O(n) —— 数据已经有序
 * 空间复杂度: O(1) —— 原地排序
 */
Data insertion_sort(Data arr) {
    int n = arr.size();
    for (int i = 1; i < n; ++i) {
        int key = arr[i];
        int j = i - 1;
        // 把比 key 大的元素向右移动一位
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            --j;
        }
        arr[j + 1] = key;
    }
    return arr;
}

// ============================================================
// 4. 基准测试
// ============================================================

/**
 * 基准测试：对比不同 n 下各排序算法的耗时。
 *
 * 场景 1: 随机完全无序数据
 * 场景 2: 接近有序数据
 */
void benchmark() {
    cout << "========================================================================" << endl;
    cout << "案例 3：归并排序基准测试（C++）" << endl;
    cout << "========================================================================" << endl;

    // ---- 场景 1：随机完全无序 ----
    cout << endl;
    cout << "[场景 1] 随机数据（完全无序）" << endl;
    cout << "------------------------------------------------------------------------" << endl;
    cout << right << setw(8) << "n"
         << " | " << setw(14) << "std::sort"
         << " | " << setw(14) << "归并排序"
         << " | " << setw(10) << "加速比" << endl;
    cout << "------------------------------------------------------------------------" << endl;

    vector<int> ns = {100, 1000, 10000, 100000};

    for (int n : ns) {
        Data data = generate_data(n, false);

        // std::sort (C++ 内置)
        Data d1 = data;
        auto t1 = chrono::high_resolution_clock::now();
        sort(d1.begin(), d1.end());
        auto t2 = chrono::high_resolution_clock::now();
        double time_builtin = chrono::duration<double>(t2 - t1).count();

        // 手写归并排序
        d1 = data;
        t1 = chrono::high_resolution_clock::now();
        Data res_merge = merge_sort(std::move(d1));
        t2 = chrono::high_resolution_clock::now();
        double time_merge = chrono::duration<double>(t2 - t1).count();

        // 验证结果一致
        {
            Data d_sorted = data;
            sort(d_sorted.begin(), d_sorted.end());
            assert(res_merge == d_sorted);
        }

        double ratio = time_builtin / time_merge;
        cout << right << setw(8) << n
             << " | " << setw(10) << fixed << setprecision(6) << time_builtin << "s"
             << " | " << setw(10) << fixed << setprecision(6) << time_merge << "s"
             << " | " << setw(8) << fixed << setprecision(2) << ratio << "x" << endl;
    }

    // ---- 场景 2：接近有序数据 ----
    cout << endl;
    cout << "[场景 2] 接近有序数据（打破思维定势）" << endl;
    cout << "------------------------------------------------------------------------" << endl;
    cout << right << setw(8) << "n"
         << " | " << setw(14) << "归并排序"
         << " | " << setw(14) << "插入排序"
         << " | " << setw(10) << "加速比" << endl;
    cout << "------------------------------------------------------------------------" << endl;

    for (int n : ns) {
        Data data = generate_data(n, true);

        // 归并排序
        Data d1 = data;
        auto t1 = chrono::high_resolution_clock::now();
        Data res_merge = merge_sort(std::move(d1));
        auto t2 = chrono::high_resolution_clock::now();
        double time_merge = chrono::duration<double>(t2 - t1).count();

        // 插入排序
        d1 = data;
        t1 = chrono::high_resolution_clock::now();
        Data res_insert = insertion_sort(std::move(d1));
        t2 = chrono::high_resolution_clock::now();
        double time_insert = chrono::duration<double>(t2 - t1).count();

        // 验证结果一致
        {
            Data d_sorted = data;
            sort(d_sorted.begin(), d_sorted.end());
            assert(res_merge == d_sorted);
            assert(res_insert == d_sorted);
        }

        double ratio = time_merge / time_insert;
        cout << right << setw(8) << n
             << " | " << setw(10) << fixed << setprecision(6) << time_merge << "s"
             << " | " << setw(10) << fixed << setprecision(6) << time_insert << "s"
             << " | " << setw(8) << fixed << setprecision(2) << ratio << "x" << endl;
    }

    cout << endl;
    cout << "结论：" << endl;
    cout << "  1. 对完全随机数据，std::sort（内省排序）通常快于手写归并排序" << endl;
    cout << "  2. 对接近有序数据，插入排序 O(n) 碾压归并排序 O(n log n)" << endl;
    cout << "  3. 算法没有绝对好坏，取决于数据特征！" << endl;
    cout << "========================================================================" << endl;
}

// ============================================================
// 5. 正确性验证
// ============================================================

/**
 * 验证所有排序方法在多种输入下结果一致。
 */
void verify() {
    cout << "正在验证排序正确性..." << endl;

    vector<pair<Data, string>> test_cases;

    // 预定义的边界情况
    test_cases.emplace_back(Data{}, "空数组");
    test_cases.emplace_back(Data{1}, "单元素");
    test_cases.emplace_back(Data{5, 5, 5, 5}, "全部相同");
    test_cases.emplace_back(Data{3, 1, 2}, "小规模随机");
    test_cases.emplace_back(Data{10, 9, 8, 7, 6, 5}, "完全逆序");
    test_cases.emplace_back(Data{1, 2, 3, 4, 5}, "已经有序");

    // 随机生成更多测试用例
    for (int n : {5, 10, 20, 50, 100}) {
        test_cases.emplace_back(generate_data(n, false), "随机 n=" + to_string(n));
        test_cases.emplace_back(generate_data(n, true),  "接近有序 n=" + to_string(n));
    }

    bool all_pass = true;
    for (auto &p : test_cases) {
        Data &arr = p.first;
        string desc = p.second;

        // 标准答案
        Data expected = arr;
        sort(expected.begin(), expected.end());

        Data res_merge = merge_sort(arr);
        Data res_insert = insertion_sort(arr);

        if (res_merge != expected || res_insert != expected) {
            cout << "  ❌ 失败: " << desc << endl;
            all_pass = false;
        } else {
            cout << "  ✅ 通过: " << desc << endl;
        }
    }

    if (all_pass)
        cout << "🎉 所有测试用例通过！" << endl;
    else
        cout << "⚠️  存在失败的测试用例！" << endl;
}

// ============================================================
// 6. 主入口
// ============================================================

int main(int argc, char *argv[]) {
    if (argc > 1 && string(argv[1]) == "benchmark") {
        benchmark();
    } else {
        verify();
        cout << endl;
        cout << "提示：运行 './case03_merge_cpp benchmark' 查看性能对比" << endl;
    }
    return 0;
}
