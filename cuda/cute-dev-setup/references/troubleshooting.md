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
