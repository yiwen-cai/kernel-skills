# CuTe 环境故障排查手册

按报错关键词查表。每条给: 原因 / 验证方法 / 修复。

## 目录
- [1. nvcc: `invalid option: --orig_src_file_name`](#1)
- [2. gcc: `cannot execute 'cc1plus'`](#2)
- [3. nvcc: `__device__ annotation on lambda requires --extended-lambda`](#3)
- [4. VSCode: `无法打开源文件 "cuda/std/utility"`](#4)
- [5. VSCode: `无法打开源文件 "stddef.h"`](#5)
- [6. VSCode: `__global__` / `__device__` 下波浪线](#6)
- [7. 编译慢 / 模板报错难以定位](#7)
- [8. cpptools 补不出 CuTe 符号（make_tensor/make_layout）](#8)
- [9. clangd `unsupported CUDA gpu architecture: sm_120`](#9)
- [10. clangd `driver clang not found in PATH`](#10)
- [11. clangd CUDA 头假阳性（__half 等）无穷尽](#11)
- [12. clangd 装哪个版本（Ubuntu 20.04）](#12)
- [13. clangd `System include extraction: driver execution failed`](#13)

---

<a id="1"></a>
## 1. nvcc: `invalid option: --orig_src_file_name`

**原因**: `/usr/bin/nvcc` 有 bug（部分 CUDA 12.x 安装的问题），它给底层编译器传了不支持的选项。

**验证**: `which nvcc` 若是 `/usr/bin/nvcc` 即中招。

**修复**: 永远用完整路径 `/usr/local/cuda-12.8/bin/nvcc`。在 `.zshrc`/`.bashrc` 加 alias:
```bash
alias nvcc='/usr/local/cuda-12.8/bin/nvcc'
```

---

<a id="2"></a>
## 2. gcc: `cannot execute 'cc1plus': execvp: No such file or directory`

**原因**: nvcc 默认调用的 gcc 版本安装残缺 —— `gcc` 可执行文件在，但 `cc1plus`（C++ 前端）缺失。常见于系统装了多个 gcc，默认指向的那个不完整。

**验证**: 逐一检查每个版本的完整性:
```bash
for v in 9 10 11 12; do
    printf "g++-%s: " $v
    [ -f /usr/lib/gcc/x86_64-linux-gnu/$v/cc1plus ] && echo OK || echo "缺 cc1plus"
done
```
找到带 `OK` 的最低版本（通常 g++-9 最稳）。

**修复**: 编译时显式指定宿主编译器:
```bash
nvcc ... -ccbin g++-9 ...
```
**注意**: `--allow-unsupported-compiler` 也需要（当 GCC 版本超过 CUDA 官方支持的上限时）。

---

<a id="3"></a>
## 3. nvcc: `__device__ annotation on lambda requires --extended-lambda`

**原因**: 在 lambda 上加 `__device__`/`__host__` 需要 nvcc 的扩展 lambda 支持，默认关闭。

**修复**: 编译加 `-extended-lambda`。CuTe kernel 常用 device lambda，建议默认开启。

---

<a id="4"></a>
## 4. VSCode: `无法打开源文件 "cuda/std/utility"`

**原因**: IntelliSense (cpptools) 找不到 libcu++ 头文件。`cuda/std/*` 是 NVIDIA 的 libcu++，随 CUDA toolkit 安装。

**验证**:
```bash
ls /usr/local/cuda-12.8/include/cuda/std/utility  # 应存在
```

**修复**: 在 `.vscode/c_cpp_properties.json` 的 `includePath` 加:
```
/usr/local/cuda-12.8/include
/usr/local/cuda-12.8/include/cuda/std
```

---

<a id="5"></a>
## 5. VSCode: `无法打开源文件 "stddef.h"` (或 `cstddef`、`stdlib.h`)

**原因**: cpptools 没从 GCC 推导出系统头文件路径。`stddef.h` 是 GCC 固定安装头文件。

**验证**: 查 g++ 的真实搜索路径:
```bash
echo | g++-9 -E -Wp,-v -xc++ - 2>&1 | grep -A15 'search starts here'
```

**修复**: 把上述输出的所有路径加进 `includePath`，关键是:
```
/usr/lib/gcc/x86_64-linux-gnu/9/include      # stddef.h 在这
/usr/include/c++/9                           # C++ 标准库
/usr/include/x86_64-linux-gnu/c++/9
/usr/include/c++/9/backward
```
同时 `compilerPath` 指向完整的那个 g++（如 `/usr/bin/g++-9`），不要指向残缺的 `/usr/bin/g++`。

---

<a id="6"></a>
## 6. VSCode: `__global__` / `__device__` 等关键字报红

**原因**: `.cu` 文件没被识别为 CUDA C++，或 cpptools 不知道处于 CUDA 编译环境（缺 `__CUDACC__` 宏），导致它跳过 `__device__` 分支代码。

**修复**:
1. `.vscode/settings.json`:
   ```json
   { "files.associations": { "*.cu": "cuda-cpp", "*.cuh": "cuda-cpp" } }
   ```
2. `c_cpp_properties.json` 的 `defines` 加 `"__CUDACC__"`，并设 `__CUDACC_VER_MAJOR__/MINOR__/BUILD__`。

---

<a id="7"></a>
## 7. 编译慢 / 模板报错难以定位

- 编译慢: CuTe 重模板，首次编译几十秒正常。开发循环中用 `-O0` 替代 `-O2` 可加速；调试用 `-G`（很慢，仅定位时用）。
- 模板报错长: 在报错栈里找**第一个**指向**你自己代码**的行，那才是真因；上面的 `cute/...` 是模板展开链。
- 确认 arch: Blackwell 用 `sm_120`，Ampere(如 A100/A800) 用 `sm_80`。arch 不匹配会报 `unsupported ...` 或运行时 illegal memory access。

---

<a id="8"></a>
## 8. cpptools 补不出 CuTe 符号（make_tensor/make_layout）

**症状**: 输入 `make_lay` 候选里没有 `make_layout`；但 cmd+点击能跳转到定义。

**根因**: cpptools 的已确认 bug [#5261](https://github.com/microsoft/vscode-cpptools/issues/5261)（维护者明确 "No plans to fix"）。它的预处理器对**宏参数化的 `#include`**（如 CuTe 头里的 `#include CUDA_STD_HEADER(tuple)`）识别不了，且对**非活跃 `#if` 分支**也做语法检查，导致 cute 头解析链在 `#if defined(__CUDACC_RTC__)` 块里中断，符号不进表。

**验证**: cpptools 日志里搜 "应输入文件名" 或 "Expected filename"，命中 `array_subbyte.hpp`、`type_traits.hpp` 等即此 bug。

**修复**: 换 clangd（见 SKILL.md 步骤 4）。cpptools 这条路无解。

---

<a id="9"></a>
## 9. clangd: `unsupported CUDA gpu architecture: sm_120`

**根因**: sm_120 (Blackwell) 的 clang 支持首发于 LLVM 20（[PR #127187](https://github.com/llvm/llvm-project/pull/127187)，2025-02 合入）。clangd-18 及更早不认 sm_120。

**修复**（按优先级）:
1. **根治**: 装 clangd-22（见 #12），含完整 sm_120 支持。
2. **兜底**（无法升级时）: 在 `.clangd` 的 `Remove` 加 `"--cuda-gpu-arch=*"`，让 frontend 完全不感知 arch，纯靠 `-D__CUDA_ARCH__=1200` 驱动 CUTLASS 的 `#if __CUDA_ARCH__` 派发。社区验证可行，但个别 SM120 专属 MMA 分支跳转可能不准。

---

<a id="10"></a>
## 10. clangd: `driver clang not found in PATH`

**根因**: clangd 做 system include 提取时要调独立的 `clang` 可执行文件，但 `apt install clangd-22` **不带** clang frontend 二进制（clangd 用内嵌 libclang，但探测步骤要独立 clang）。PATH 里只有 `clang-22`，没有无版本号的 `clang`。

**验证**: `which clang` → not found；`which clang-22` → `/usr/bin/clang-22`。

**修复**: 装 clang-22 并建软链：
```bash
sudo apt install clang-22
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-22 100
```

---

<a id="11"></a>
## 11. clangd CUDA 头假阳性（__half / __CUDA_ARCH_HAS_FEATURE__ / CUtensorMapDataType ...）无穷尽

**症状**: clangd 跑起来后，编辑器里对 `cuda_fp16.hpp`/`cuda_bf16.h`/`cuda_fp8.hpp`/`copy_sm90_desc.hpp` 等 SDK 头报各种 `member_decl_does_not_match`、`pp_expr_bad_token_lparen`、`unknown_typename`。补全和跳转正常，只是红波浪线噪音。

**根因**: clang 对 CUDA SDK 头的 host/device 双定义、`NV_IF_TARGET` 宏包裹、跨头 out-of-line 定义等结构的语义分析和 nvcc 不一致，产生大量假阳性。逐条根因不一（`__half` 是 Sema 阶段签名匹配、`__CUDA_ARCH_HAS_FEATURE__` 是预处理宏展开、`CUtensorMapDataType` 是 arch 派发没走到），**没有统一根治**。

**为什么不逐条 Suppress**: 实测追完一条（如 `__half`）会冒下一条（`__CUDA_ARCH_HAS_FEATURE__`），再追完又冒下一条（`CUtensorMapDataType`）…… 无底洞。每条单独 Suppress 性价比极低。

**修复**: 全局关诊断，只留跳转补全。`.clangd` 加：
```yaml
Diagnostics:
  Suppress:
    - "*"
```
同时在 `settings.json` 加 `"clangd.diagnostics": { "unusedIncludes": "none", "missingIncludes": "none" }`。

**为什么这样 OK**: clangd 的符号索引（补全/跳转用）和诊断（红波浪线）是**独立的**，关诊断不影响补全。这些假阳性本就不影响真编译（nvcc 编译没问题）。若用户坚持要诊断，明确告知"CUDA + clangd 的诊断治不干净，社区共识是关掉"。

---

<a id="12"></a>
## 12. clangd 装哪个版本（Ubuntu 20.04 focal）

**事实**: focal 默认仓库（apt 镜像）最高只到 **clangd-18**。19/20 已退出 LLVM 维护窗口，当前 apt.llvm.org 只为 focal 提供 21/22。

**选型**:
- **sm_120 (Blackwell)**: 必须 ≥ 19。装 clangd-22（LLVM PPA，见下）。
- **sm_80 (Ampere) / sm_90 (Hopper)**: clangd-18 够用，`sudo apt install clangd-18`，无需 PPA。

**装 clangd-22（focal + PPA）**:
```bash
wget -qO- https://apt.llvm.org/llvm.sh | sudo bash -s -- 22 focal
sudo apt update && sudo apt install -y clangd-22 clang-22
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-22 100
```
（apt.llvm.org 在国内通，慢的话换清华镜像 `mirrors.tuna.tsinghua.edu.cn/llvm-apt`。）

**踩过的坑**: `--background-index-storage=<path>` 这个 flag **在 clangd-18/22 里都不存在**（很多旧博客/LLVM review 提过，但实际没合入或被改名）。控制索引存储位置要靠软链 `~/.cache/clangd` → 其他盘，不要瞎传这个 flag（clangd 会启动失败 exit 1）。

---

<a id="13"></a>
## 13. clangd: `System include extraction: driver execution failed`

**症状**: clangd 日志报 `System include extraction: driver execution failed with return code: 1. Args: [/usr/local/cuda/bin/nvcc -E -v -x cuda -]`。

**根因**: clangd 用 nvcc 跑 `-E -v -x cuda` 探测系统 include 路径，但 **nvcc 的 `-x` 参数要 `cu` 不是 `cuda`**（`-x cuda` 是 clang 语法）。这是 clangd 对 nvcc 的已知 flag 误传，**不影响补全**（clangd 用 fallback 命令拿 include，preamble 照常建）。

**修复**: 通常不需要修（无害噪音）。若强迫症，把 `settings.json` 的 `--query-driver` 改成 `/usr/bin/g++-9,/usr/bin/gcc-9`（用 g++ 提取系统 include，绕开 nvcc 探测），`.clangd` 的 `Compiler: nvcc` 保留即可。
