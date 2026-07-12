#!/bin/bash
# new_accuracy_test.sh —— 算子精度测试脚手架
# 用法：./new_accuracy_test.sh <算子名>
# 产物：tests/test_<算子名>.cu，从项目根 ./scripts/cc.sh tests/test_<算子名>.cu 编译运行
# 按 SOP.md 生成：模板化 compareArrays + tolerance_for 查表 + 占位 TODO
set -e

OP="${1:?用法: $0 <算子名>}"
OUT="tests/test_${OP}.cu"

if [ -f "$OUT" ]; then
  echo "已存在 $OUT，放弃覆盖（如需重新生成请先删除）"; exit 1
fi

if [ ! -f "src/${OP}/kernel_${OP}.cuh" ]; then
  echo "警告：src/${OP}/kernel_${OP}.cuh 不存在，生成的测试文件编译会失败。请先按 SOP 第 1 节创建该文件。"
fi

cat > "$OUT" << 'CU_EOF'
// 精度测试: __OP__
// 按 SOP.md 生成：reference + 三类输入 + compareArrays + 退出码
// 填充所有 TODO 后 ./scripts/cc.sh tests/test___OP__.cu 编译运行
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

#include "__OP__/kernel___OP__.cuh"

// ============ TestResult ============
struct TestResult {
  const char* name;
  float max_abs_err;
  float max_rel_err;
  int   mismatch_count;
  int   total;
  int   argmax_idx;   // 最大绝对误差出现位置，-1 表示无
  bool  passed;
};

// ============ 容差查表（对齐 SOP 第 3 节，单一事实来源）============
// op_kind: "reduction" | "bitexact"
// dtype:   "float" | "bf16" | "fp16" | "e4m3" | "e5m2"
struct Tol { float atol; float rtol; };
Tol tolerance_for(const char* dtype, const char* op_kind) {
  // bit-exact 类一律 0/0
  if (op_kind[0] == 'b') return {0.0f, 0.0f};  // "bitexact"
  // reduction 类按 dtype 分档
  if (dtype[0] == 'f' && dtype[1] == 'l') return {1e-5f, 1.3e-6f};   // "float"
  if (dtype[0] == 'b')                  return {1e-5f, 1.6e-2f};      // "bf16"
  if (dtype[0] == 'f' && dtype[1] == 'p') return {1e-5f, 1e-3f};     // "fp16"
  if (dtype[0] == 'e' && dtype[2] == '3') return {0.125f, 0.0f};     // "e4m3"
  if (dtype[0] == 'e' && dtype[2] == '2') return {0.2f, 0.0f};       // "e5m2"
  return {1e-5f, 1.3e-6f};  // 默认 fp32 reduction
}

// ============ compareArrays 模板（内部 upcast float 比对）============
// 公式（对齐 PyTorch 2.12）: |got-exp| <= atol + rtol*|exp|
// cutlass 类型（bfloat16_t/float_e4m3_t 等）均可隐式 operator float()
template <class T>
TestResult compareArrays(const char* name,
                         const T* got, const T* expected, int n,
                         float atol, float rtol, bool equal_nan) {
  TestResult r{name, 0.0f, 0.0f, 0, n, -1, true};
  for (int i = 0; i < n; ++i) {
    float g = (float)got[i];
    float e = (float)expected[i];
    if (std::isnan(g) || std::isnan(e)) {
      if (equal_nan && std::isnan(g) && std::isnan(e)) continue;
      r.mismatch_count++; r.passed = false;
      if (r.argmax_idx < 0) r.argmax_idx = i;
      continue;
    }
    float diff = std::fabs(g - e);
    if (diff > r.max_abs_err) { r.max_abs_err = diff; r.argmax_idx = i; }
    float rel = diff / (std::fabs(e) + 1e-12f);
    if (rel > r.max_rel_err) r.max_rel_err = rel;
    if (diff > atol + rtol * std::fabs(e)) { r.mismatch_count++; r.passed = false; }
  }
  return r;
}

// ============ 打印结果（对齐 SOP 第 10 节 / PyTorch 格式）============
void printResult(const TestResult& r, const float* got, const float* expected) {
  const char* tag = r.passed ? "PASS" : "FAIL";
  printf("[%s] %s  max_abs=%.3e max_rel=%.3e mismatch=%d/%d (at idx %d)\n",
         tag, r.name, r.max_abs_err, r.max_rel_err, r.mismatch_count, r.total, r.argmax_idx);
  if (!r.passed && got && expected) {
    int shown = 0;
    for (int i = 0; i < r.total && shown < 5; ++i) {
      float g = got[i], e = expected[i];
      float diff = std::fabs(g - e);
      if (diff > 0.0f) {
        printf("  mismatch[%d]: idx=%d  got=%.4f  expected=%.4f\n", shown, i, g, e);
        shown++;
      }
    }
  }
}

// ============ 输入生成 ============
void fillUniform(float* a, int n, unsigned seed=42) {
  srand(seed);
  for (int i = 0; i < n; ++i) a[i] = (rand()/(float)RAND_MAX)*2.0f - 1.0f;  // [-1,1)
}
// TODO: fillNormal（Box-Muller 正态分布）
// TODO: 按算子的边界 case 生成器（见 SOP 第 6 节）

// ============ reference 实现（按算子填）============
// TODO: __OP___cpu 的手写 host fp32 reference
// void __OP___cpu(const float* in, float* out, int n) { ... }

// ============ 单个 case 的运行框架（按算子填）============
// 返回 0=PASS, 1=FAIL
int run_case(const char* case_name, const float* hA, const float* hB, int n,
             float atol, float rtol) {
  // TODO: 按 SOP 流程填充
  // 1. cudaMalloc + cudaMemcpy H2D
  // 2. launch kernel: __OP__::...<<<GRID, BLOCK>>>(...)
  // 3. cudaMemcpy D2H 回 hC
  // 4. 跑 __OP___cpu 得到 expected
  // 5. compareArrays<float>(case_name, hC, expected, n, atol, rtol, false)
  // 6. printResult(r, hC, expected)
  // 7. return r.passed ? 0 : 1
  (void)case_name; (void)hA; (void)hB; (void)n; (void)atol; (void)rtol;
  printf("[SKIP] %s  (run_case 未实现)\n", case_name);
  return 0;  // 占位：未实现时不算失败
}

int main() {
  printf("===== 精度测试: __OP__ =====\n");
  int fail = 0;

  // TODO: 按 SOP 第 6 节填入各 case
  // 例：
  // Tol t = tolerance_for("float", "bitexact");  // vecAdd 是 bit-exact 类
  // float *hA = (float*)malloc(N*sizeof(float));
  // float *hB = (float*)malloc(N*sizeof(float));
  // fillUniform(hA, N); fillUniform(hB, N, 43);
  // fail += run_case("random_uniform", hA, hB, N, t.atol, t.rtol);
  // ... 其他 case

  printf("\nTotal: %d passed, %d failed\n", 0 /*TODO: pass 计数*/, fail);
  return fail > 0 ? 1 : 0;
}
CU_EOF

# 替换占位符 __OP__ 为实际算子名（heredoc 用引号无法变量展开，事后 sed）
sed -i "s/__OP__/${OP}/g" "$OUT"

echo "生成 $OUT"
echo "下一步："
echo "  1. 填充 TODO（reference、run_case、各 case）"
echo "  2. ./scripts/cc.sh $OUT 编译运行"
