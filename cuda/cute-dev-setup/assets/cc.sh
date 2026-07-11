#!/bin/bash
# cc.sh —— CuTe 编译脚本（由 cute-dev-setup skill 生成）
# 模板说明：CUDA_PREFIX / NVCC / CUTLASS_INC / GCC_BIN 在生成时按 check_env.sh 结果填好。
#
# 用法:
#   ./cc.sh foo.cu                  # 默认 GPU0，输出 foo，编译后运行
#   ./cc.sh foo.cu -g 1             # 指定 GPU 号（影响 arch 选择与运行设备）
#   ./cc.sh foo.cu -o bar           # 指定输出名
#   ./cc.sh foo.cu -n               # 只编译不运行
#
# 架构映射：GPU0 -> sm_120 (Blackwell) ; GPU1/2 -> sm_80 (A800)
# 按实际机器改 ARCH_MAP；arch 与 GPU 号的对应写在 case 里。

set -e

# ---- 环境占位符（生成时替换；若未替换则用合理默认） ----
NVCC="${NVCC:-/usr/local/cuda-12.8/bin/nvcc}"
GCC_BIN="${GCC_BIN:-g++-9}"
CUTLASS_INC="${CUTLASS_INC:-$(cd "$(dirname "$0")" && pwd)/cutlass/include}"

GPU=0; OUTPUT=""; RUN=1; SRC=""
while [ $# -gt 0 ]; do
    case "$1" in
        -g) GPU="$2"; shift 2;;
        -o) OUTPUT="$2"; shift 2;;
        -n) RUN=0; shift;;
        -h|--help) sed -n '2,11p' "$0"; exit 0;;
        *)  if [ -z "$SRC" ]; then SRC="$1";
            elif [ -z "$OUTPUT" ]; then OUTPUT="$1"; fi
            shift;;
    esac
done

[ -z "$SRC" ] && { echo "错误：未指定 .cu 源文件"; exit 1; }
[ -z "$OUTPUT" ] && OUTPUT="${SRC%.cu}"

# GPU 号 -> sm_xx（按你的机器改这里）
case "$GPU" in
    0)   ARCH=sm_120;;   # Blackwell
    1|2) ARCH=sm_80;;    # A800
    *)   echo "警告：未知 GPU $GPU，默认 sm_120"; ARCH=sm_120;;
esac

echo "==> 编译 $SRC (GPU $GPU, $ARCH) -> $OUTPUT"
"$NVCC" -arch="$ARCH" --allow-unsupported-compiler -ccbin "$GCC_BIN" \
    -std=c++17 -extended-lambda -O2 \
    -I "$CUTLASS_INC" \
    -o "$OUTPUT" "$SRC"
echo "==> 编译完成"

if [ "$RUN" -eq 1 ]; then
    echo "==> 运行 (CUDA_VISIBLE_DEVICES=$GPU)"
    echo "----------------------------------------"
    CUDA_VISIBLE_DEVICES="$GPU" "./$OUTPUT"
fi
