---
name: operator-performance-test
description: 为 CUDA/CuTe 算子建立并执行固定的 Quick/Full 性能测试工作流，覆盖 CUDA Event 延迟、MFU/MBU、analytical Roofline、显式命名基线、性能回归门禁，以及代表 case 的 NSYS/NCU 深度分析和 JSON/Markdown 报告。Use when the user wants to benchmark or profile a CUDA kernel/operator, measure operator performance, compare against a named baseline or reference implementation, diagnose memory-bound versus compute-bound behavior, generate an Nsight performance report, or standardize a repeatable operator performance workflow; triggers include 算子性能测试、kernel benchmark、quick/full profile、MFU、MBU、roofline、nsys、ncu、性能基线、性能回归、性能报告。
---

# CUDA 算子性能测试

建立或使用项目内固定的 Quick/Full 流程。当前经端到端验证的默认目标是物理 GPU 0、Blackwell CC 12.0；不要默默扩展到其他 GPU 或架构。

## 先检查项目

1. 读取项目的 `AGENTS.md`、`CLAUDE.md` 和已有性能 SOP。
2. 优先使用项目已有入口，例如 `./perf.sh`；不要另造平行 runner。
3. 检查 GPU 0、CUDA、NSYS、NCU、编译器、Git SHA、活动 GPU 进程和温度。
4. 若项目没有流程，把 [assets/performance-testing-sop.md.tmpl](assets/performance-testing-sop.md.tmpl) 落到项目，并以 [assets/operator-config.example.json](assets/operator-config.example.json) 建立算子配置；实现细节遵循 [references/workflow-contract.md](references/workflow-contract.md)。

## 固定执行顺序

### Quick

运行：

```bash
./perf.sh quick <op> [--baseline <name>]
```

依次执行环境校验、正确性 gate、warmup、自动 batching、CUDA Event 多次采样和报告。延迟使用 median/P10/P90/min；Quick 延迟是回归判断的唯一来源。报告算法 FLOPs/bytes、有效吞吐、MFU/MBU、算术强度和 analytical Roofline，并明确这些是基于算法量与配置峰值的派生指标，不是硬件计数器。

### Full

运行：

```bash
./perf.sh full <op> [--baseline <name>]
```

先完整执行 Quick，再运行固定分层 case 矩阵；只对配置中的代表 case 执行 NSYS/NCU。Profiler duration 仅用于诊断，不可替代 CUDA Event 延迟。

NCU 需要权限时，只允许 `sudo -v` 交互认证一次、后台续期、`sudo -n` 执行 NCU/ownership 修正、最后 `sudo -k`。绝不读取、传递或保存 sudo 密码，尤其不要写入 `.env`。

## 基线与判定

- 基线必须显式命名、显式 create/update，并纳入 Git 历史；普通测试不得自动覆盖。
- 默认 regression 条件：相对变慢 `>5%` 且绝对变慢 `>0.2 us`，两项同时满足；第一次命中后再测一次，两次都命中才判定 `REGRESSION`。
- reference 实现存在时报告 speedup；不存在时仍与命名历史基线比较。
- 活动 compute process、GPU 状态查询失败或温度达到 85°C 时标记 `UNSTABLE`。
- NSYS/NCU/权限失败时保留普通 benchmark 与报告，标记 `PROFILE_PARTIAL` 并记录原因；不得把该结果更新为正式基线。

## 报告和产物

同时生成机器可读 JSON、manifest 和 Markdown。记录 GPU/架构、工具版本、源码 SHA、case/variant、统计值、公式、环境状态、基线差异、profiler 状态和复现命令。

保留可审查的 JSON/Markdown；忽略大型 `.ncu-rep`、`.nsys-rep`、SQLite 和 build 产物。原始 profiler 文件不得进入基线。

## 扩展新算子

显式定义 quick cases、固定分层 full cases、一个代表 case、variant/reference、dtype、计算模式、算法 FLOPs/bytes 和阈值。先验证正常实现 PASS，再注入可控性能退化确认回归 gate 真能失败。不要把第一个算子的硬编码公式复制到其他算子。

## 交付门禁

提交前运行语法/单元测试、Quick、至少一次真实 Full，并检查报告与 raw ignore 策略。若测试代码或配置再变化，重新执行受影响验证。只有自动化验证与用户验收都通过后才允许 commit/merge。
