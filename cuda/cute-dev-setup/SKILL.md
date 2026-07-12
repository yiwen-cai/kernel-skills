---
name: cute-dev-setup
description: "在标准 VSCode 环境下配置 CUTLASS/CuTe 开发环境，包含依赖检测与安装。Use when the user wants to set up, configure, or troubleshoot a CuTe/CUTLASS CUDA development environment — triggers include 配置 cute 环境、CUTLASS 环境搭建、CuTe 编译报错、VSCode 对 .cu/cute 头文件报红、nvcc cc1plus 错误、cuda/std 头文件找不到、设置 CUDA 开发环境。"
---

# CuTe 开发环境配置

配置 CUTLASS/CuTe 的 CUDA 开发环境（header-only，只需 include 路径）。核心是三件事：装 CUTLASS、配编译命令、配 VSCode 代码智能（clangd 为主，cpptools 降级为备选）。已踩过的坑见 [references/troubleshooting.md](references/troubleshooting.md)。

**关键事实**：cpptools 对 CuTe 这种深度模板库补全很弱（[已确认 bug #5261](https://github.com/microsoft/vscode-cpptools/issues/5261)，不修）—— 它认不出宏参数化的 `#include CUDA_STD_HEADER(...)`，导致 cute 头解析链中断、`make_tensor` 等符号补不出来。**社区与 NVIDIA 官方推荐改用 clangd**（[NVIDIA IDE Setup 文档](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/ide_setup.html)）。本 skill 默认配 clangd；cpptools 仅保留扩展用于调试，IntelliSense 关掉。

## 工作流

### 1. 先跑依赖检测（必做）

把 [scripts/check_env.sh](scripts/check_env.sh) 拷到目标目录执行。它检测：CUDA toolkit/nvcc 路径、GCC 完整性（`cc1plus` 是否残缺）、GPU 架构、CUTLASS 是否就位、libcu++ 头文件。加 `--install` 可自动 git clone CUTLASS。

```bash
bash check_env.sh            # 只检测
bash check_env.sh --install  # 检测 + 自动装 CUTLASS
```

**依据检测结果确定三个关键变量**（后面所有配置都用）：
- `NVCC`：必须用 `/usr/local/cuda-12.8/bin/nvcc` 这种完整路径，**不要用 `/usr/bin/nvcc`**（有 bug，见 troubleshooting #1）
- `GCC_BIN`：选检测报告里第一个 `cc1plus 完整` 的版本（常是 `g++-9`）。残缺版本会让 nvcc 报 `cc1plus: No such file`
- `ARCH`：按 GPU compute capability 选。Blackwell(cc 12.0)→`sm_120`，Ampere(cc 8.0 如 A800/A100)→`sm_80`，Hopper(cc 9.0)→`sm_90`

### 2. 装 CUTLASS（header-only）

```bash
git clone --depth 1 https://github.com/NVIDIA/cutlass.git
# 验证: include/cute/tensor.hpp 应存在
```

### 3. 编译命令（标准模板）

```bash
$NVCC -arch=$ARCH --allow-unsupported-compiler -ccbin $GCC_BIN \
      -std=c++17 -extended-lambda -O2 \
      -I ./cutlass/include -o out src.cu
```

三个 flag 缺一不可：
- `--allow-unsupported-compiler`：GCC 版本超 CUDA 官方支持上限时必需
- `-ccbin g++-9`：绕开残缺的默认 gcc
- `-extended-lambda`：CuTe kernel 常用 device lambda，默认开

**推荐**: 用 [assets/cc.sh](assets/cc.sh) 封装（拷过去后按检测结果填顶部 `NVCC/GCC_BIN/CUTLASS_INC` 占位符，并改 `ARCH_MAP`）。用法 `./cc.sh foo.cu -g 1`。

### 4. VSCode 代码智能配置（clangd 为主）

**为什么用 clangd 不用 cpptools**：cpptools 对 CuTe 模板补全弱（bug #5261 不修），clangd 是 NVIDIA 官方推荐。但 clangd 对 CUDA SDK 头有大量假阳性诊断（`__half`、`__CUDA_ARCH_HAS_FEATURE__`、`CUtensorMapDataType` 等无穷无尽），这些**不影响跳转/补全/真编译**，所以默认全局关诊断（`Suppress: ["*"]`），只留跳转补全。

#### 4a. 装 clangd（需 sudo）

Ubuntu 20.04 默认仓库只到 clangd-18，**sm_120 (Blackwell) 需要 clangd ≥ 19**（sm_120 支持首发于 LLVM 20）。必须加 LLVM 官方 PPA 装 22：

```bash
# 加 LLVM 源（官方脚本，会写 /etc/apt/sources.list）
wget -qO- https://apt.llvm.org/llvm.sh | sudo bash -s -- 22 focal
sudo apt update && sudo apt install -y clangd-22 clang-22
# 建 clang 软链（clangd 内部探测要找无版本号的 clang）
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-22 100
```

**sm_80 (Ampere) 场景**可以用 clangd-18（`sudo apt install clangd-18`），不必加 PPA。但 Blackwell 必须 ≥ 19。

#### 4b. 装 vscode-clangd 扩展

Cursor/VSCode 扩展市场搜 `clangd`（作者 LLVM），装到 **Remote 侧**。它 package.json 含 `cuda-cpp` 激活条件，现 settings 已把 `.cu/.cuh`→`cuda-cpp`，无需额外 file association。

#### 4c. 放配置文件

**`.vscode/settings.json`** —— 直接拷 [assets/settings.json](assets/settings.json)。关键项：
- `C_Cpp.intelliSenseEngine: disabled`（关 cpptools IntelliSense，避免和 clangd 打架；保留 cpptools 扩展用于调试）
- `clangd.path: /usr/bin/clangd-22`
- `clangd.arguments` 含 `--query-driver=/usr/local/cuda/bin/nvcc`（**必须**，否则 `.clangd` 里 `Compiler: nvcc` 被静默忽略）

**`.clangd`**（项目根）—— 基于 [assets/clangd.yaml.tmpl](assets/clangd.yaml.tmpl)，替换占位符：
- `__PROJECT__` → 项目绝对路径（CUTLASS 三条 include）
- `--cuda-gpu-arch` + `-D__CUDA_ARCH__` → 按卡（sm_120/1200, sm_80/800, sm_90/900）
- `-D__CUDACC_VER_MINOR__` → CUDA minor 版本

#### 4d. reload + 验证

Close Remote Connection 重连，打开 [assets/hello_cute.cu](assets/hello_cute.cu)。看输出面板 `clangd` 频道：
- ✅ 看到 `Built preamble of size ... for file hello_cute.cu`
- ✅ 删掉 `auto layout = make_layout(...)` 那行重敲 `make_lay` → 候选出 `make_layout` = 成功
- ⚠️ 若报 `unsupported CUDA gpu architecture: sm_120` → clangd < 19，在 `.clangd` 的 `Remove` 加 `--cuda-gpu-arch=*` 兜底（纯靠 `-D__CUDA_ARCH__` 驱动），或升级到 clangd-22

#### cpptools 备选路线（仅在无 clangd 时）

若环境无法装 clangd，用 [assets/c_cpp_properties.json.tmpl](assets/c_cpp_properties.json.tmpl) 配 cpptools。但 **CuTe 符号补全会残缺**（`make_tensor`/`make_layout` 等模板工厂补不出），只能靠 cmd+点击跳转读源码。详见 troubleshooting #4-#6。

### 5. 验证

拷 [assets/hello_cute.cu](assets/hello_cute.cu) 到目标目录，用步骤 3 的命令编译运行。期望输出 `[OK] CuTe 环境正常!`。若失败，对照 [references/troubleshooting.md](references/troubleshooting.md) 按报错关键词查。

## 常见坑速查（详见 troubleshooting.md）

| 报错 | 根因 | 修复 |
|---|---|---|
| `invalid option: --orig_src_file_name` | 用了 `/usr/bin/nvcc` | 改用完整路径 nvcc |
| `cannot execute 'cc1plus'` | 默认 gcc 残缺 | `-ccbin g++-9` 或 `update-alternatives gcc → gcc-9` |
| `unsupported CUDA gpu architecture: sm_120` | clangd < 19 不认 Blackwell | 装 clangd-22（LLVM PPA）；或在 `.clangd` Remove 加 `--cuda-gpu-arch=*` 兜底 |
| `driver clang not found in PATH` | clangd 找不到 frontend | 装 `clang-22` + `update-alternatives --install /usr/bin/clang clang /usr/bin/clang-22 100` |
| cpptools 补不出 `make_layout`/`make_tensor` | cpptools bug #5261（不修） | 换 clangd |
| clangd 一堆 CUDA 头假阳性（`__half` 等） | clang 对 SDK 头假阳性，无根治 | `.clangd` 加 `Diagnostics.Suppress: ["*"]`，只留跳转补全 |
| `cuda/std/utility` 找不到 | VSCode 缺 CUDA includePath | 加 `/usr/local/cuda-12.8/include` |
| `stddef.h` 找不到 | VSCode 缺 GCC 系统头路径 | 查 `g++-9 -E -Wp,-v` 补全 includePath |

## 远程开发提醒

用户通过 SSH 在远程服务器开发时（常见于实验室 GPU 服务器）：
- VSCode 配置文件（`.vscode/`）必须落在**服务器端**项目目录，不是本地
- cpptools 扩展需装在 **Remote** 侧（VSCode 会提示 "Install in SSH: ..."）
- 若用户用本地编辑器同步远程文件而非 Remote-SSH，includePath 要用本地能访问的路径——优先建议改用 Remote-SSH
