# 容差表与 TF32 事实依据

skill 里容差表和 TF32 处理的权威依据，按需加载。所有事实附官方文档出处。

## 判定公式

`|actual - expected| ≤ atol + rtol · |expected|`，分母是 `expected`（参考值）。

来源：[PyTorch 2.12 `torch.testing.assert_close`](https://docs.pytorch.org/docs/stable/testing.html)、[`torch.allclose`](https://docs.pytorch.org/docs/stable/generated/torch.allclose.html)。

## 各 dtype 默认容差（PyTorch 2.12）

| dtype | rtol | atol |
|---|---|---|
| float64 | 1e-7 | 1e-7 |
| float32 | 1.3e-6 | 1e-5 |
| float16 | 1e-3 | 1e-5 |
| bfloat16 | 1.6e-2 | 1e-5 |
| 整数/bool | 0 | 0 |

dtype 不匹配时取两者容差的最大值。

**注意默认值分歧**（容易踩坑）：
- `torch.allclose`：固定默认 rtol=1e-5, atol=1e-8（不分 dtype）
- `torch.testing.assert_close`：dtype-aware（上表）
- `numpy.allclose`：1e-5 / 1e-8
- `numpy.testing.assert_allclose`：1e-7 / 0
- `triton.testing.assert_close`：atol=1e-2, rtol=0

混用会让同一份误差时过时不过。SOP 的容差表统一对齐 `assert_close`，FP8 用 Triton tutorial 量级。

## FP8 容差

PyTorch 无 FP8 默认（非 IEEE 标准类型）。业界参考：
- Triton tutorial 03：fp8 `atol=0.125, rtol=0`
- xformers fp8 量级类似

float_e4m3_t（1+4+3，max 448，**无 Inf**，0x7f=NaN）精度最低；float_e5m2_t（1+5+2，有 Inf）动态范围大但精度更低。容差随算子校准后更新。

## equal_nan 与 Inf

- `equal_nan` 默认 False。False 时 NaN vs NaN → fail（IEEE 754：NaN != NaN）
- Inf/-Inf 必须**精确相等**（同符号同位置），不受 tolerance 和 equal_nan 影响

来源：同 PyTorch 2.12 文档。

## TF32 污染（gemm 必读）

**核心事实**：sm_120（Blackwell）上 cuBLAS 默认 `CUBLAS_DEFAULT_MATH`，FP32 GEMM 静默走 TF32 Tensor Core（10-bit mantissa 而非 23-bit）。架构无关，Blackwell 不例外。

来源（cuBLAS 12.8 文档）：
- [§2.4.22 cublasSetMathMode](https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html#cublassetmathmode)："the default math mode is `CUBLAS_DEFAULT_MATH`"
- [§2.2.10 cublasMath_t](https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html#cublasmath-t)：`CUBLAS_DEFAULT_MATH` = "Tensor Cores will be used whenever possible"
- [§2.1.11 Tensor Core Usage](https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html#tensor-core-usage)：自 cuBLAS 11.0.0 "automatically make use of Tensor Core capabilities wherever possible, unless they are explicitly disabled by selecting pedantic compute modes"

**后果**：若用 cuBLAS 当 gemm reference 而不显式禁用 TF32，reference 自己降了精度，被测的真 FP32 kernel 反而显得"误差为 0"，基准被污染，整个 gemm 精度测试失效。

**禁用方式**：
- 代码内：`cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH)`（§2.4.22）—— 推荐
- 环境变量：`NVIDIA_TF32_OVERRIDE=0`（§2.2.11），全局覆盖，优先级高于程序配置

`CUBLAS_PEDANTIC_MATH` 至少在 cuBLAS 11.3.1 已存在，12.8 仍为有效枚举值。

**linking**：`nvcc app.cu -lcublas -o app`（§2.1.9），无需 `-lcublasLt`、无需 `-L` 路径。

## .cuh 跨 TU include

- 模板 `__global__` kernel 定义放 .cuh 被多 TU include：默认 `-static-global-template-stub=true` 给每个 TU 的 stub 强制 static 链接，**不冲突**（nvcc 文档 [§4.2.5.1](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvpc/index.html#static-global-template-stub)）
- **非模板** `__global__` 定义放 .cuh 被多 TU include：**ODR 冲突**，需 `inline __global__`（§6.6.5）
- `#pragma once` 只防单 TU 内重复展开，不防跨 TU 冲突

## cutlass 4.6.0 浮点类型

- `cutlass::float_e4m3_t`（`cutlass/float8.h`）：1+4+3，max 448，无 Inf
- `cutlass::float_e5m2_t`（`cutlass/float8.h`）：1+5+2，有 Inf
- `cutlass::bfloat16_t`（`cutlass/bfloat16.h`）：1+8+7，与 fp32 同指数位
- **无 BF8 类型**（8-bit 只有标准 FP8）
- 互转：从 float 构造 `explicit`，转 float `operator float()` 非 explicit（可隐式）→ `compareArrays<T>` 内 `(float)got[i]` 直接 upcast
