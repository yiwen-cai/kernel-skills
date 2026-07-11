# kernel-skills

GPU kernel 开发相关的 Claude Code / Codex skill 集合。按编程模型分目录组织，每个子目录是一个可独立使用的 skill。

## 目录结构

```
kernel-skills/
├── cuda/                     # NVIDIA CUDA 相关 skill
│   └── cute-dev-setup/       # 配置 CUTLASS/CuTe 开发环境
├── triton/                   # Triton 相关 skill（占位）
└── npu/                      # 国产 NPU（昇腾等）相关 skill（占位）
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

## 使用方式

这些 skill 是给 [Claude Code](https://claude.com/claude-code) 用的。两种用法：

**作为 skill 安装**（自动触发）：

把对应 skill 目录拷到 `~/.claude/skills/`，Claude Code 会自动发现。之后说「配置 cute 环境」「.cu 文件报红」等就会触发。

```bash
git clone https://github.com/yiwen-cai/kernel-skills.git
cp -r kernel-skills/cuda/cute-dev-setup ~/.claude/skills/
```

**手动用脚本**：skill 目录里的 `scripts/` 和 `assets/` 可以脱离 Claude Code 单独使用，比如直接跑 `check_env.sh` 检测环境。

## License

[MIT](LICENSE)
