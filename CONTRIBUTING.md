# 🤝 贡献指南

感谢你有兴趣为 **运筹学与优化方法课程系列** 做出贡献！

## 📋 贡献方式

### 1. 报告问题
- 代码运行错误
- 文档中的笔误或不准确之处
- 建议新的案例或课程主题

### 2. 提交代码
- 修复 Bug
- 新增教学案例
- 改进现有代码的可读性或性能
- 添加 C++ 版本的算法实现

### 3. 改进文档
- 修正术语翻译
- 补充知识点说明
- 添加参考文献

## 🔧 开发流程

### 环境准备

```bash
# 克隆你的 Fork
git clone https://github.com/<your-username>/course-series.git
cd course-series

# 安装依赖
pip install -r requirements.txt
```

### 提交规范

提交信息请遵循以下格式：

```
<type>(<scope>): <description>

# 示例
feat(04-运筹学): 新增网络流案例
fix(01-概率论): 修复蒙特卡洛仿真中的随机种子问题
docs(README): 更新课程统计数据
```

**Type 类型：**
- `feat` — 新功能/新案例
- `fix` — Bug 修复
- `docs` — 文档更新
- `refactor` — 代码重构
- `test` — 测试相关
- `chore` — 构建/工具变更

### 代码规范

1. **Python 代码**
   - 遵循 PEP 8 风格
   - 所有案例代码必须可独立运行
   - 核心案例仅依赖 `numpy`，进阶案例可用 `scipy`/`pyomo`/`simpy`
   - 添加必要的中文注释说明算法步骤

2. **C++ 代码**
   - 使用 C++11 标准
   - 文件命名：`caseXX_<name>_cpp.cpp`
   - 包含完整的测试用例
   - 编译命令：`g++ -O2 -std=c++11 -o output file.cpp`

3. **文档**
   - 使用 Markdown 格式
   - 数学公式使用 LaTeX 语法
   - 遵循现有课程模板结构

### 案例模板

每个新案例应遵循 7 步结构：

```
1. 场景描述 — 用实际问题引入
2. 数学建模 — 定义变量、目标、约束
3. 方法选择 — 说明为什么选这个方法
4. 代码实现 — 可运行的完整代码
5. 结果分析 — 解释输出含义
6. 验证标准 — 如何确认结果正确
7. 延伸思考 — 拓展问题和改进方向
```

### 验证

提交前请确保：

```bash
# 运行所有 Python 代码
make python

# 如果修改了 C++ 代码
make cpp
```

## 📝 Pull Request 流程

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/your-feature`
3. 编写代码并确保通过验证
4. 提交更改并推送
5. 发起 Pull Request，描述你的改动内容

## 💬 交流

如有问题，欢迎通过 Issues 与我们交流。
