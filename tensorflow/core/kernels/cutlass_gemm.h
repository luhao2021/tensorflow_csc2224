#ifndef TENSORFLOW_CORE_KERNELS_CUTLASS_GEMM_H_
#define TENSORFLOW_CORE_KERNELS_CUTLASS_GEMM_H_

#include "cuda.h"
#include "cuda_runtime.h"
#include "cutlass/half.h"
#include "tensorflow/core/platform/cus.h"

using tensorflow::cus;

cudaError_t cutlassCusGemm(CUstream stream, bool transpose_a, bool transpose_b,
                           int M, int N, int K, cus alpha, cus const* A,
                           int lda, cus const* B, int ldb, cus beta, cus* C,
                           int ldc);
#endif  // TENSORFLOW_CORE_KERNELS_CUTLASS_GEMM_H_