#include "tensorflow/core/kernels/cutlass_gemm.h"

#include <vector>

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"

using ColumnMajor = cutlass::layout::ColumnMajor;
using RowMajor = cutlass::layout::RowMajor;
using AccumulationType = float;

#define CUTLASS_GEMM(LayoutA, LayoutB, type)                           \
  using CutlassGemm =                                                  \
      cutlass::gemm::device::Gemm<type, LayoutA, type, LayoutB, type,  \
                                  cutlass::layout::ColumnMajor,        \
                                  AccumulationType>;                   \
  CutlassGemm gemm_operator;                                           \
  CutlassGemm::Arguments args({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, \
                              {C, ldc}, {alpha, beta});                \
  cutlass::Status status = gemm_operator(args, nullptr, stream);       \
  if (status != cutlass::Status::kSuccess) {                           \
    return cudaErrorUnknown;                                           \
  }                                                                    \
  return cudaSuccess;

#define CUTLASS_STRIDED_BATCH(LayourA, LayoutB, type)                        \
  using Gemm =                                                               \
      cutlass::gemm::device::GemmBatched<type, LayourA, type, LayoutB, type, \
                                         cutlass::layout::ColumnMajor,       \
                                         AccumulationType>;                  \
  Gemm gemm_op;                                                              \
  cutlass::Status status = gemm_op({{m, n, k},                               \
                                    {A, lda},                                \
                                    batch_stride_A,                          \
                                    {B, ldb},                                \
                                    batch_stride_B,                          \
                                    {C, ldc},                                \
                                    batch_stride_C,                          \
                                    {C, ldc},                                \
                                    batch_stride_C,                          \
                                    {alpha, beta},                           \
                                    batch_count},                            \
                                   nullptr, stream);                         \
  if (status != cutlass::Status::kSuccess) {                                 \
    return cudaErrorUnknown;                                                 \
  }                                                                          \
  return cudaSuccess;

cudaError_t cutlassCusGemm(CUstream stream, bool transpose_a, bool transpose_b,
                           int M, int N, int K, cus alpha, cus const* A,
                           int lda, cus const* B, int ldb, cus beta, cus* C,
                           int ldc) {
  if (transpose_a) {
    if (transpose_b) {
      CUTLASS_GEMM(RowMajor, RowMajor, cus)
    } else {
      CUTLASS_GEMM(RowMajor, ColumnMajor, cus)
    }
  } else {
    if (transpose_b) {
      CUTLASS_GEMM(ColumnMajor, RowMajor, cus)
    } else {
      CUTLASS_GEMM(ColumnMajor, ColumnMajor, cus)
    }
  }
}

cudaError_t cutlassCusStridedBatchedGemm(
    CUstream stream, bool transpose_a, bool transpose_b, int m, int n, int k,
    cus alpha, cus const* A, int lda, long long int batch_stride_A,
    cus const* B, int ldb, long long int batch_stride_B, cus* C, int ldc,
    long long int batch_stride_C, cus beta, int batch_count) {
  if (transpose_a) {
    if (transpose_b) {
      CUTLASS_STRIDED_BATCH(RowMajor, RowMajor, cus)
    } else {
      CUTLASS_STRIDED_BATCH(RowMajor, ColumnMajor, cus)
    }
  } else {
    if (transpose_b) {
      CUTLASS_STRIDED_BATCH(ColumnMajor, RowMajor, cus)
    } else {
      CUTLASS_STRIDED_BATCH(ColumnMajor, ColumnMajor, cus)
    }
  }
}
