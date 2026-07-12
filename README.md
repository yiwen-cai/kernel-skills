# kernel-skills

GPU kernel 开发相关的 Claude Code / Codex skill 集合。按编程模型分目录组织，每个子目录是一个可独立使用的 skill。

## 目录结构

```
kernel-skills/
├── cuda/                            # NVIDIA CUDA 相关 skill
│   ├── cute-dev-setup/              # 配置 CUTLASS/CuTe 开发环境
│   ├── operator-accuracy-test/      # 算子精度测试（数值对拍）标准流程
│   └── operator-performance-test/   # 算子 Quick/Full 性能测试标准流程
├── triton/                          # Triton 相关 skill（占位）
└── npu/                             # 国产 NPU（昇腾等）相关 skill（占位）
```

## 已有 skill

### [cuda/cute-dev-setup](cuda/cute-dev-setup)

在标准 VSCode 环境下配置 CUTLASS/CuTe 的 CUDA 开发环境。CUTLASS 是 header-only，这个 skill 解决的是「怎么把它接进你的项目和 IDE」。

包含：
- **依赖检测脚本** — 自动探测 nvcc 路径（避开 `/usr/bin/nvcc` 的 bug）、GCC 完整性（检查 `cc1plus` 是否残缺）、GPU 架构、CUTLASS 是否就位、libcu++ 头文件
- **编译脚本模板** — 封装 nvcc 调用，按 GPU 号自动选 `sm_120`/`sm_80`，正确处理 `-ccbin`/`--allow-unsupported-compiler`/`-extended-lambda`
- **VSCode IntelliSense 配置模板** — 解决 `cuda/std/utility`、`stddef.h` 找不到、`__global__` 报红等常见问题
- **故障排查手册** — 按 nvcc/cc1plus/VSCode 报错关键词查表的诊断指南
- **最小验证程序** — 确认环境能编译能跑的 hello world

### [cuda/operator-accuracy-test](cuda/operator-accuracy-test)

为手写 CUDA/CuTe 算子落地**精度测试标准流程**（数值对拍）。解决三个痛点：kernel 复制粘贴进测试导致源改测试过期假绿、不知 reference 用什么/容差定多少、gemm 用 cuBLAS 当 reference 被 TF32 静默污染。

包含：
- **SOP 模板** — 12 节标准流程（reference 选型、多 dtype 容差表、TF32 污染规避、NaN 处理、退出码语义、报告格式、kernel 同步机制、扩展点），复制到项目根即用
- **测试脚手架脚本** — `./new_accuracy_test.sh <op>` 生成 `test_<op>.cu` 骨架，模板化 `compareArrays<T>`（支持 fp32/bf16/fp16/fp8）+ `tolerance_for` 容差查表 + 对齐 PyTorch 的报告格式 + 退出码语义
- **.cuh 范式样板** — 对齐 CUTLASS 官方的 kernel 头文件范式（`#pragma once` + namespace + 函数内 using namespace），通过 `#include` 机制杜绝复制粘贴分叉
- **vecAdd 端到端范例** — 可直接编译运行的三 case 测试（含 .cuh + test.cu），照着改新算子
- **容差与 TF32 事实依据** — 每个容差值和 TF32 处理附 PyTorch/cuBLAS/nvcc 官方文档出处

适用范围：sm_120（Blackwell），第一阶段数值误差对拍（稳定性/跨硬件留扩展点）。

### [cuda/operator-performance-test](cuda/operator-performance-test)

为 CUDA/CuTe 算子建立可持续的 **Quick/Full 性能测试标准流程**。Quick 使用 CUDA Event 提供延迟统计、MFU/MBU、analytical Roofline、reference speedup 和命名基线回归判断；Full 运行固定分层 case，并用 NSYS/NCU 深入分析一个预先配置的代表 case。

包含：
- **Quick/Full 执行契约** — 正确性 gate、warmup/自动 batching、median/P10/P90/min、固定 case 分层与代表 case profiling
- **指标语义** — 区分 MFU/MBU/analytical Roofline 派生估算和 NCU 硬件计数器实测结论
- **命名基线与回归门禁** — 显式 create/update、Git 可追踪，默认 `>5%` 且 `>0.2 us`，连续两次命中才判定回归
- **安全的 NCU 权限流程** — Full 只交互认证一次，不保存密码，结束后清除 sudo ticket
- **状态与报告规范** — `PASS`、`REGRESSION`、`UNSTABLE`、`PROFILE_PARTIAL`、`INVALID`，以及 JSON/Markdown/manifest 产物约定
- **SOP 与算子配置模板** — 可复制到项目中扩展新算子

当前经 GPU 0、Blackwell CC 12.0、vecAdd 端到端验证；其他 GPU/架构需要显式扩展和重新校准。

## 使用方式

这些 skill 是给 [Claude Code](https://claude.com/claude-code) 用的。两种用法：

**作为 skill 安装**（自动触发）：

把对应 skill 目录拷到 `~/.claude/skills/`，Claude Code 会自动发现。之后说「配置 cute 环境」「.cu 文件报红」「给算子写精度测试」等就会触发。

```bash
git clone https://github.com/yiwen-cai/kernel-skills.git
cp -r kernel-skills/cuda/cute-dev-setup ~/.claude/skills/
cp -r kernel-skills/cuda/operator-accuracy-test ~/.claude/skills/
cp -r kernel-skills/cuda/operator-performance-test ~/.claude/skills/
```

**手动用脚本/模板**：skill 目录里的 `assets/` 可以脱离 Claude Code 单独使用，比如直接拷 `new_accuracy_test.sh` 生成测试骨架、拷 `SOP.md.tmpl` 落地项目 SOP。

## License

[MIT](LICENSE)
