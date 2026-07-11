#!/bin/bash
# check_env.sh —— CuTe 开发环境依赖检测
# 用法: ./check_env.sh [--install]
#   无参数: 只检测，输出报告
#   --install: 检测到缺失时尝试自动安装（仅 git clone CUTLASS；系统级依赖给出指令）

set -u
INSTALL=0
[ "${1:-}" = "--install" ] && INSTALL=1

green() { printf "\033[32m%s\033[0m\n" "$*"; }
red()   { printf "\033[31m%s\033[0m\n" "$*"; }
yel()   { printf "\033[33m%s\033[0m\n" "$*"; }

pass() { green "[OK]   $*"; }
fail() { red "[FAIL] $*"; }
warn() { yel "[WARN] $*"; }

PROBLEMS=0

echo "================ CuTe 环境检测 ================"
echo

# ---- 1. CUDA Toolkit / nvcc ----
echo "## 1. CUDA Toolkit"
# /usr/bin/nvcc 已知有 bug（invalid option: --orig_src_file_name），必须用完整路径
NVCC=""
for cand in /usr/local/cuda-12.8/bin/nvcc /usr/local/cuda/bin/nvcc; do
    if [ -x "$cand" ]; then NVCC="$cand"; break; fi
done
if [ -n "$NVCC" ]; then
    VER=$("$NVCC" --version | grep -oP 'release \K[0-9.]+')
    pass "nvcc: $NVCC (CUDA $VER)"
    if [ "$NVCC" = "/usr/bin/nvcc" ]; then
        warn "  /usr/bin/nvcc 已知有 bug，已改用 $NVCC"
    fi
else
    fail "未找到 nvcc（检查 /usr/local/cuda*/bin/nvcc）"
    PROBLEMS=$((PROBLEMS+1))
fi
echo

# ---- 2. GCC 完整性（最易踩坑：cc1plus 残缺） ----
echo "## 2. 宿主编译器 GCC"
BEST_GCC=""
for v in 9 10 11 12 13; do
    G="/usr/bin/g++-$v"
    [ -x "$G" ] || continue
    if [ -f "/usr/lib/gcc/x86_64-linux-gnu/$v/cc1plus" ]; then
        pass "g++-$v 完整 (cc1plus 存在)"
        [ -z "$BEST_GCC" ] && BEST_GCC="g++-$v"
    else
        fail "g++-$v 残缺: 缺 cc1plus，不可用"
        PROBLEMS=$((PROBLEMS+1))
    fi
done
if [ -z "$BEST_GCC" ] && [ -x /usr/bin/g++ ]; then
    warn "未找到带版本号的完整 g++，nvcc 可能编译失败"
fi
echo

# ---- 3. GPU 架构（决定 sm_xx） ----
echo "## 3. GPU 架构"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | while read -r line; do
        cc=$(echo "$line" | grep -oP '\d+\.\d+' | tail -1)
        case "$cc" in
            9.0)    arch="sm_90  (Hopper)" ;;
            10.0|10.1) arch="sm_100/sm_101 (Blackwell)" ;;
            12.0)   arch="sm_120 (Blackwell)" ;;
            8.0|8.6|8.7|8.9) arch="sm_80/86/87/89 (Ampere/Ada)" ;;
            7.0|7.5) arch="sm_70/75 (Volta/Turing)" ;;
            *)      arch="sm_??" ;;
        esac
        echo "    $line  ->  $arch"
    done
else
    warn "nvidia-smi 不可用，无法检测 GPU 架构"
fi
echo

# ---- 4. CUTLASS（header-only） ----
echo "## 4. CUTLASS / CuTe 头文件"
CUTLASS_DIR=""
# 优先当前目录，再查常见位置
for d in ./cutlass ../cutlass "$HOME/code/cutlass" /opt/cutlass; do
    if [ -f "$d/include/cute/tensor.hpp" ]; then
        CUTLASS_DIR="$(cd "$d" && pwd)"
        break
    fi
done
if [ -n "$CUTLASS_DIR" ]; then
    pass "CUTLASS: $CUTLASS_DIR/include/cute"
else
    fail "未找到 CUTLASS（需要 include/cute/tensor.hpp）"
    if [ "$INSTALL" -eq 1 ]; then
        echo "    -> 自动 clone 到 ./cutlass ..."
        git clone --depth 1 https://github.com/NVIDIA/cutlass.git ./cutlass \
            && pass "CUTLASS 已安装到 ./cutlass" \
            || fail "git clone 失败"
    else
        echo "    修复: git clone --depth 1 https://github.com/NVIDIA/cutlass.git"
        PROBLEMS=$((PROBLEMS+1))
    fi
fi
echo

# ---- 5. CUDA include（libcu++ 依赖） ----
echo "## 5. libcu++ (cuda/std/*)"
CUDA_INC="/usr/local/cuda-12.8/include"
if [ -f "$CUDA_INC/cuda/std/utility" ]; then
    pass "libcu++: $CUDA_INC/cuda/std"
else
    fail "未找到 cuda/std/utility（检查 CUDA toolkit 安装）"
    PROBLEMS=$((PROBLEMS+1))
fi
echo

# ---- 总结 ----
echo "==============================================="
if [ "$PROBLEMS" -eq 0 ]; then
    green "全部通过。"
    echo
    echo "建议编译命令:"
    echo "  nvcc -arch=sm_XX --allow-unsupported-compiler -ccbin $BEST_GCC \\"
    echo "       -std=c++17 -extended-lambda -I $CUTLASS_DIR/include -o out src.cu"
else
    red "发现 $PROBLEMS 个问题，见上方 [FAIL]。"
fi
