---
name: operator-accuracy-test
description: "为手写 CUDA/CuTe 算子设计并落地精度测试（数值对拍）标准流程。提供 SOP 模板、测试脚手架、.cuh 范式样板，覆盖 reference 选型、多 dtype（fp32/bf16/fp16/fp8）容差表、TF32 污染规避、NaN 处理、退出码语义、报告格式。Use when the user wants to verify a CUDA/CuTe kernel's numerical correctness, write an accuracy/correctness test for an operator, compare GPU output against a reference, pick atol/rtol tolerances, set up operator precision testing, or troubleshoot kernel correctness — triggers include 算子精度测试、数值对拍、验证 kernel 正确性、reference 对比、atol/rtol 怎么定、softmax/gemm/attention 测试、TF32 污染 cuBLAS、bit-exact、fp8/bf16 容差。"
---

# 算子精度测试（数值对拍）

为手写 CUDA/CuTe 算子落地**精度测试标准流程**：SOP 文档 + 脚手架脚本 + .cuh 范式样板。解决三个常见痛点：(1) kernel 靠复制粘贴进测试文件，源改了测试过期、假绿；(2) 不知道 reference 用什么、容差定多少；(3) gemm 用 cuBLAS 当 reference 被 TF32 静默污染。

适用范围：sm_120（Blackwell）。已随 vecAdd 算子端到端验证过（三 case PASS + bug 注入 FAIL 检测）。

## 何时用

用户写完/改了一个 CUDA/CuTe 算子，要验证数值正确性。典型信号：新算子写好了怎么测、kernel 结果对不对、精度误差多少算正常、reference 用什么、阈值怎么定。

## 工作流

### 1. 落 SOP 到项目（一次性）

把 [assets/SOP.md.tmpl](assets/SOP.md.tmpl) 复制到项目根改名 `SOP.md`，按项目实际算子调整第 6 节边界 case 清单。SOP 是项目的单一事实来源（reference 选型、容差表、NaN/退出码/报告格式）。

### 2. 写算子（.cu + kernel_<op>.cuh 范式）

每算子拆两个文件，**禁止复制粘贴 kernel**：

```cpp
// kernel_<op>.cuh —— 范式对齐 cutlass/examples/51_hopper_gett/gett_kernel.cuh
#pragma once
#include <cuda_runtime.h>
#include "cute/tensor.hpp"
namespace <op> {
template <...>
__global__ void kernelName(...) {
  using namespace cute;          // 函数体内，不放文件作用域
  ...
}
inline __global__ void nonTpl(...) { ... }  // 非模板必须 inline
}
```

要点：自包含、`#pragma once`、namespace 包裹、函数内 `using namespace`、模板跨 TU 安全/非模板用 `inline`。测试文件 `#include "kernel_<op>.cuh"` 调真 kernel，源改了测试自动同步。完整范例见 [assets/kernel_vecAdd.example.cuh](assets/kernel_vecAdd.example.cuh)。

### 3. 生成测试骨架

把 [assets/new_accuracy_test.sh](assets/new_accuracy_test.sh) 拷到项目根，`./new_accuracy_test.sh <op>` 生成 `test_<op>.cu`。骨架已实现共用部分：

- `compareArrays<T>` 模板（内部 `(float)got[i]` upcast，cutlass bf16/fp8 类型均可隐式转 float）
- `tolerance_for(dtype, op_kind)` 查表（fp32/bf16/fp16/e4m3/e5m2 × reduction/bitexact）
- `printResult`（对齐 PyTorch 格式：max_abs/max_rel/mismatch/argmax + 前 5 mismatch）
- `main` 退出码（fail>0 ? 1 : 0，禁止恒 0）

只留 reference / 输入生成 / case 的 TODO 占位要填。填充范例见 [assets/test_vecAdd.example.cu](assets/test_vecAdd.example.cu)。

### 4. 填 TODO + 跑

填 `<op>_cpu`（手写 host fp32 reference）、三类输入（均匀/正态/边界，固定 `srand(42)`）、各 case 的 `run_case`（alloc→launch→D2H→reference→compare→print）。编译运行 `./cc.sh test_<op>.cu`，全 PASS 且退出码 0。

## 关键约束（必读）

- **reference 必须 fp32**。BF16/FP8 算子的 reference 用 fp32 upcast 计算（先 `(float)x`），不要用 cuBLAS 的 bf16/fp8 接口当 reference
- **gemm 走 cuBLAS 必须强制 PEDANTIC_MATH**。sm_120 默认 `CUBLAS_DEFAULT_MATH`（TF32 on），不显式禁用 reference 走 TF32 污染基准。完整片段见 SOP 第 2 节
- **bit-exact 算子（vecAdd/transpose）用 atol=rtol=0**，任何偏差都是 bug；reduction 类（softmax/gemm/reduce）用 PyTorch 默认（fp32: 1.3e-6/1e-5）。判定依据：**算子是否含浮点运算**（纯搬运/重排/逐元素 → bit-exact；含 max/exp/sum/累加 → reduction）
- **退出码 fail>0→1**，禁止恒 0（CI 靠退出码判定成败）
- **2D/多维算子（transpose/gemm/attention）**：buffer 拍平成一维传入 `compareArrays`，reference 也输出**同 layout 的拍平结果**（如 row-major），逐元素比对即可——维度信息只用在 kernel launch 和 reference 计算里，比对阶段退化成 1D
- **NaN 搬运类**：transpose 这类纯搬运算子搬运含 NaN 的输入时，输出也应是 NaN（正确行为）。但 `equal_nan=false`（默认）下 NaN!=NaN 会误判 FAIL。这类 case 要么用 `equal_nan=true`，要么约定测试输入不含 NaN

## 容差速查

判定公式：`|got-exp| ≤ atol + rtol·|exp|`（分母 expected，对齐 PyTorch 2.12）

| dtype | rtol | atol |
|---|---|---|
| float32 reduction | 1.3e-6 | 1e-5 |
| float32 bit-exact | 0 | 0 |
| bfloat16 | 1.6e-2 | 1e-5 |
| float16 | 1e-3 | 1e-5 |
| float_e4m3_t | 0 | 0.125 |
| float_e5m2_t | 0 | 0.2 |

事实依据（容差来源、TF32 文档出处、cutlass 类型精度）见 [references/tolerance-and-tf32.md](references/tolerance-and-tf32.md)。

## 验证流程可用性

落地后做两项检查确认流程真的生效（不是假绿）：
1. **正常实现全 PASS**：三 case PASS、退出码 0
2. **注入 bug 必 FAIL**：临时改 kernel（如 `+` 改 `-`），重编重跑，三 case 全 FAIL + 打印 mismatch + 退出码 1。bit-exact 容差（0/0）下任何偏差都必须被检测到

## 不做什么

- 不做可复用 C++ 测试库（产物是 SOP + 脚手架，流程优先于抽象）
- 不做数值稳定性遍历 / 跨硬件一致性（SOP 第 12 节留扩展点，第一阶段只做数值误差对拍）
- 不支持 A800 sm_80（仅 Blackwell；跨硬件是阶段三）
- 不引入 PyTorch 跨语言对拍（reference 用 cuBLAS + 手写）
