---
name: cute-dev-setup
description: "在标准 VSCode 环境下配置 CUTLASS/CuTe 开发环境，包含依赖检测与安装。Use when the user wants to set up, configure, or troubleshoot a CuTe/CUTLASS CUDA development environment — triggers include 配置 cute 环境、CUTLASS 环境搭建、CuTe 编译报错、VSCode 对 .cu/cute 头文件报红、nvcc cc1plus 错误、cuda/std 头文件找不到、设置 CUDA 开发环境。"
---

# CuTe 开发环境配置

配置 CUTLASS/CuTe 的 CUDA 开发环境（header-only，只需 include 路径）。核心是三件事：装 CUTLASS、配编译命令、配 VSCode IntelliSense。已踩过的坑见 [references/troubleshooting.md](references/troubleshooting.md)。

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

### 4. VSCode IntelliSense 配置

用户用 VSCode 时，在工作区 `.vscode/` 放两个文件：

**`.vscode/c_cpp_properties.json`** —— 基于 [assets/c_cpp_properties.json.tmpl](assets/c_cpp_properties.json.tmpl)，把占位符替换为检测结果：
- `__CUDA_INCLUDE__` → `/usr/local/cuda-12.8/include`（解决 `cuda/std/utility` 找不到）
- `__GCC_BIN__` → 完整的 g++（如 `/usr/bin/g++-9`），**不要指向残缺的 `/usr/bin/g++`**
- `__GCC_*_INCLUDE__` → `g++-9 -E -Wp,-v -xc++ -` 输出的系统头路径（解决 `stddef.h` 找不到）
- `__CUDA_MAJOR__/MINOR__/BUILD__` → 如 12/8/61

`defines` 里的 `__CUDACC__` 必须保留 —— 否则 cpptools 跳过 `__device__` 分支，CuTe 设备代码全报红。

**`.vscode/settings.json`** —— 直接拷 [assets/settings.json](assets/settings.json)，把 `.cu/.cuh` 关联到 `cuda-cpp`。

**配置后**: 让用户在 VSCode 执行 `C/C++: Reset IntelliSense Database`；仍有红波浪线则 `Developer: Reload Window`。

### 5. 验证

拷 [assets/hello_cute.cu](assets/hello_cute.cu) 到目标目录，用步骤 3 的命令编译运行。期望输出 `[OK] CuTe 环境正常!`。若失败，对照 [references/troubleshooting.md](references/troubleshooting.md) 按报错关键词查。

## 常见坑速查（详见 troubleshooting.md）

| 报错 | 根因 | 修复 |
|---|---|---|
| `invalid option: --orig_src_file_name` | 用了 `/usr/bin/nvcc` | 改用完整路径 nvcc |
| `cannot execute 'cc1plus'` | 默认 gcc 残缺 | `-ccbin g++-9` |
| `cuda/std/utility` 找不到 | VSCode 缺 CUDA includePath | 加 `/usr/local/cuda-12.8/include` |
| `stddef.h` 找不到 | VSCode 缺 GCC 系统头路径 | 查 `g++-9 -E -Wp,-v` 补全 includePath |
| `__global__` 下波浪线 | 缺 `__CUDACC__` 宏 | c_cpp_properties 的 defines 加上 |

## 远程开发提醒

用户通过 SSH 在远程服务器开发时（常见于实验室 GPU 服务器）：
- VSCode 配置文件（`.vscode/`）必须落在**服务器端**项目目录，不是本地
- cpptools 扩展需装在 **Remote** 侧（VSCode 会提示 "Install in SSH: ..."）
- 若用户用本地编辑器同步远程文件而非 Remote-SSH，includePath 要用本地能访问的路径——优先建议改用 Remote-SSH
