// 精度测试: vecAdd
// 按 SOP.md 生成：reference + 三类输入 + compareArrays + 退出码
// 填充所有 TODO 后 ./cc.sh test_vecAdd.cu 编译运行
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

#include "vecAdd/kernel_vecAdd.cuh"

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

// ============ reference：手写 host fp32（vecAdd 是 bit-exact，但仍提供 fp32 reference）============
void vecAdd_cpu(const float* A, const float* B, float* C, int n) {
  for (int i = 0; i < n; ++i) C[i] = A[i] + B[i];
}

// ============ 单个 case 的运行框架 ============
// 返回 0=PASS, 1=FAIL
int run_case(const char* case_name, const float* hA, const float* hB, int n,
             float atol, float rtol) {
  const int BLOCK_SIZE = 256;
  const int GRID_SIZE  = 4;

  // 1. alloc + H2D
  float *dA, *dB, *dC;
  cudaMalloc(&dA, n * sizeof(float));
  cudaMalloc(&dB, n * sizeof(float));
  cudaMalloc(&dC, n * sizeof(float));
  cudaMemcpy(dA, hA, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, n * sizeof(float), cudaMemcpyHostToDevice);

  // 2. launch kernel（fp32 实例化）
  vecadd::vecAdd<float, float, float, BLOCK_SIZE, GRID_SIZE><<<GRID_SIZE, BLOCK_SIZE>>>(dA, dB, dC, n);
  cudaDeviceSynchronize();

  // 3. D2H 回 hC
  float* hC = (float*)malloc(n * sizeof(float));
  cudaMemcpy(hC, dC, n * sizeof(float), cudaMemcpyDeviceToHost);

  // 4. 跑 reference 得到 expected
  float* expected = (float*)malloc(n * sizeof(float));
  vecAdd_cpu(hA, hB, expected, n);

  // 5. 比对（vecAdd 是 bit-exact，equal_nan=false）
  TestResult r = compareArrays<float>(case_name, hC, expected, n, atol, rtol, false);

  // 6. 打印
  printResult(r, hC, expected);

  // 7. 清理
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  free(hC); free(expected);
  return r.passed ? 0 : 1;
}

int main() {
  printf("===== 精度测试: vecAdd =====\n");
  int fail = 0;
  int pass = 0;

  // vecAdd 是 bit-exact 类，容差 0/0
  Tol t = tolerance_for("float", "bitexact");

  // case 1: 随机均匀，N=1000（非 2 的幂）
  {
    int N = 1000;
    float* hA = (float*)malloc(N * sizeof(float));
    float* hB = (float*)malloc(N * sizeof(float));
    fillUniform(hA, N, 42);
    fillUniform(hB, N, 43);
    fail += run_case("random_uniform", hA, hB, N, t.atol, t.rtol);
    free(hA); free(hB);
  }

  // case 2: N < BLOCK_SIZE，N=100
  {
    int N = 100;
    float* hA = (float*)malloc(N * sizeof(float));
    float* hB = (float*)malloc(N * sizeof(float));
    fillUniform(hA, N, 142);
    fillUniform(hB, N, 143);
    fail += run_case("n_lt_blocksize", hA, hB, N, t.atol, t.rtol);
    free(hA); free(hB);
  }

  // case 3: N = GRID_SIZE * BLOCK_SIZE 边界，N=1024（恰好整除）
  {
    int N = 1024;
    float* hA = (float*)malloc(N * sizeof(float));
    float* hB = (float*)malloc(N * sizeof(float));
    fillUniform(hA, N, 242);
    fillUniform(hB, N, 243);
    fail += run_case("n_eq_grid_block", hA, hB, N, t.atol, t.rtol);
    free(hA); free(hB);
  }

  pass = 3 - fail;
  printf("\nTotal: %d passed, %d failed\n", pass, fail);
  return fail > 0 ? 1 : 0;
}
