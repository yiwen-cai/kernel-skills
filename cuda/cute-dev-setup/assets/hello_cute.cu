// hello_cute.cu —— CuTe 环境验证程序
// 验证 CUTLASS/CuTe 头文件能 include、能编译、能在 GPU 上跑。
// 涉及核心概念: Layout (索引->偏移的函数), make_tensor (指针+Layout->Tensor)。

#include <cstdio>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/util/print.hpp>

using namespace cute;

int main() {
    // CPU 上: 2x4 row-major, shape=(2,4) stride=(4,1)
    auto layout = make_layout(make_shape(2, 4), make_stride(4, 1));
    printf("[CPU] layout: ");
    print(layout);
    printf("\n[CPU] layout(1,2) = %d\n", (int)layout(1, 2));  // 1*4 + 2*1 = 6

    // GPU 上: 指针 + Layout = Tensor
    int *d_ptr = nullptr;
    cudaMalloc(&d_ptr, sizeof(int) * 8);
    auto d_tensor = make_tensor(make_gmem_ptr(d_ptr), layout);
    (void)d_tensor;

    int h[8]; for (int i = 0; i < 8; ++i) h[i] = i;
    cudaMemcpy(d_ptr, h, sizeof(int) * 8, cudaMemcpyHostToDevice);
    int out[8];
    cudaMemcpy(out, d_ptr, sizeof(int) * 8, cudaMemcpyDeviceToHost);
    printf("[GPU] 取回: ");
    for (int i = 0; i < 8; ++i) printf("%d ", out[i]);
    printf("\n\n[OK] CuTe 环境正常!\n");
    cudaFree(d_ptr);
    return 0;
}
