# 运筹学与优化方法课程系列 — 一键验证
# 用法:
#   make          → 运行所有 Python 代码验证
#   make cpp      → 编译并运行所有 C++ 代码
#   make python   → 仅运行 Python 代码
#   make list     → 列出所有可运行代码

.PHONY: all python cpp list clean

PYTHON := python3
CXX := g++
CXXFLAGS := -O2 -std=c++11

# 查找所有 Python 代码（排除 venv 和 cache）
PY_FILES := $(shell find . -name '*.py' -not -path '*/venv/*' -not -path '*/.venv/*' -not -path '*/__pycache__/*' | sort)

# 查找所有 C++ 代码
CPP_FILES := $(shell find . -name '*_cpp.cpp' | sort)

# 默认: 列出所有代码
all: list

list:
	@echo "========================================"
	@echo "  运筹学与优化方法课程系列 — 代码清单"
	@echo "========================================"
	@echo ""
	@echo "Python 代码 ($(words $(PY_FILES)) 个):"
	@for f in $(PY_FILES); do \
		dir=$$(dirname $$f | sed 's|./||'); \
		name=$$(basename $$f); \
		printf "  %-50s %s\n" "$$dir/$$name"; \
	done
	@echo ""
	@echo "C++ 代码 ($(words $(CPP_FILES)) 个):"
	@for f in $(CPP_FILES); do \
		printf "  %s\n" "$$f"; \
	done
	@echo ""
	@echo "用法:"
	@echo "  make python  — 运行所有 Python 代码（验证）"
	@echo "  make cpp     — 编译并运行所有 C++ 代码"
	@echo "  make clean   — 清理编译产物"

python:
	@echo "========================================"
	@echo "  运行所有 Python 代码"
	@echo "========================================"
	@errors=0; count=0; \
	for f in $(PY_FILES); do \
		echo "--- $$f ---"; \
		if $(PYTHON) "$$f" > /tmp/$$(basename $$f).log 2>&1; then \
			echo "  ✅"; \
		else \
			echo "  ❌ (see /tmp/$$(basename $$f).log)"; \
			errors=$$((errors+1)); \
		fi; \
		count=$$((count+1)); \
	done; \
	echo ""; \
	echo "$$count 个运行完成, $$errors 个失败"

cpp:
	@echo "========================================"
	@echo "  编译并运行所有 C++ 代码"
	@echo "========================================"
	@errors=0; \
	for f in $(CPP_FILES); do \
		name=$$(basename $$f _cpp.cpp); \
		bin=/tmp/$$name; \
		echo "--- $$f ---"; \
		if $(CXX) $(CXXFLAGS) "$$f" -o "$$bin" 2>/tmp/$$name-build.log; then \
			echo "  编译 ✅"; \
			if "$$bin" > /tmp/$$name-run.log 2>&1; then \
				echo "  运行 ✅"; \
			else \
				echo "  运行 ❌ (see /tmp/$$name-run.log)"; \
				errors=$$((errors+1)); \
			fi; \
		else \
			echo "  编译 ❌ (see /tmp/$$name-build.log)"; \
			errors=$$((errors+1)); \
		fi; \
	done; \
	echo ""; \
	echo "$(words $(CPP_FILES)) 个编译完成, $$errors 个失败"

clean:
	find /tmp -maxdepth 1 -type f -name 'case*' -delete
	@echo "已清理编译产物"
