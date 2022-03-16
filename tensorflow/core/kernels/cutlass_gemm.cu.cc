#include "tensorflow/core/kernels/cutlass_gemm.h"

#include <vector>

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "tensorflow/core/platform/default/logging.h"

#define CUTLASSGEMM(LayoutA, LayoutB)                                  \
  using CutlassGemm =                                                  \
      cutlass::gemm::device::Gemm<cus, LayoutA, cus, LayoutB, cus,     \
                                  cutlass::layout::ColumnMajor>;       \
  CutlassGemm gemm_operator;                                           \
  CutlassGemm::Arguments args({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, \
                              {C, ldc}, {alpha, beta});                \
  cutlass::Status status = gemm_operator(args, nullptr, stream);       \
  if (status != cutlass::Status::kSuccess) {                           \
    return cudaErrorUnknown;                                           \
  }                                                                    \

cudaError_t cutlassCusGemm(CUstream stream, bool transpose_a,
                           bool transpose_b, int M, int N, int K,
                           cus alpha, cus const* A, int lda, cus const* B,
                           int ldb, cus beta, cus* C, int ldc) {
  using ColumnMajor = cutlass::layout::ColumnMajor;
  using RowMajor = cutlass::layout::RowMajor;

  if (transpose_a) {
    if (transpose_b) {
      CUTLASSGEMM(RowMajor, RowMajor)
    } else {
      CUTLASSGEMM(RowMajor, ColumnMajor)
    } 
  } else {
    if (transpose_b) {
      CUTLASSGEMM(ColumnMajor, RowMajor)
    } else {
      CUTLASSGEMM(ColumnMajor, ColumnMajor)
    }
  }
  return cudaSuccess;
}