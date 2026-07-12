// kernel_vecAdd.cuh —— vecAdd 算子的 kernel 定义
// 范式对齐 cutlass/examples/51_hopper_gett/gett_kernel.cuh：
//   #pragma once + 自包含 include + namespace 包裹 + 函数内 using namespace cute
// 测试文件通过 #include "kernel_vecAdd.cuh" 调用 kernel，禁止复制粘贴。
#pragma once

#include <cuda_runtime.h>
#include "cute/tensor.hpp"

namespace vecadd {

// CuTe 风格 vecAdd：用 make_tensor/local_tile 描述数据访问
template <class TA, class TB, class TC, int BLOCK_SIZE, int GRID_SIZE>
__global__ void vecAdd(TA *A, TB *B, TC *C, int M)
{
  using namespace cute;
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  // global Tensor
  auto vectorLayout = make_layout(make_shape(M), make_stride(Int<1>{}));
  Tensor mA = make_tensor(make_gmem_ptr(A), vectorLayout);
  Tensor mB = make_tensor(make_gmem_ptr(B), vectorLayout);
  Tensor mC = make_tensor(make_gmem_ptr(C), vectorLayout);

  // CTA
  auto cta_tiler = make_shape(Int<BLOCK_SIZE>{});
  int cta_tile_num = ceil_div(M, BLOCK_SIZE);

  for (int cta_tile_id = bid; cta_tile_id < cta_tile_num; cta_tile_id += GRID_SIZE)
  {
    int global_i = cta_tile_id * BLOCK_SIZE + tid;
    if (global_i < M)
    {
      Tensor gA = local_tile(mA, cta_tiler, cta_tile_id);
      Tensor gB = local_tile(mB, cta_tiler, cta_tile_id);
      Tensor gC = local_tile(mC, cta_tiler, cta_tile_id);
      gC(tid) = gA(tid) + gB(tid);
    }
  }
}

// 裸 C 风格 vecAdd：对照学习用
template <class TA, class TB, class TC, int BLOCK_SIZE, int GRID_SIZE>
__global__ void cudaVecAdd(TA *A, TB *B, TC *C, int M)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int tile_size = BLOCK_SIZE * GRID_SIZE;
  int num_tiles = (M + tile_size - 1) / tile_size;

  for (int tile_id = 0; tile_id < num_tiles; tile_id++)
  {
    int i = tile_id * tile_size + bid * BLOCK_SIZE + tid;
    if (i < M)
    {
      C[i] = A[i] + B[i];
    }
  }
}

}  // namespace vecadd
