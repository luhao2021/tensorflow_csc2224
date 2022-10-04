#include "tensorflow/core/kernels/cutlass_conv.h"

#include <vector>

#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_dgrad.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/kernel/default_conv2d_wgrad.h"
#include "cutlass/conv/threadblock/threadblock_swizzle.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/convolution.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "tensorflow/stream_executor/platform/logging.h"

using ElementAccumulator = cus;

cudaError_t cutlassCusConvForward(CUstream stream, MatrixCoord stride,
                                  Tensor4DCoord padding, MatrixCoord dilation,
                                  Tensor4DCoord input_size, void* input_data,
                                  Tensor4DCoord filter_size, void* filter_data,
                                  Tensor4DCoord output_size, void* output_data,
                                  float alpha, float beta) {
  using ElementA = cus;
  using ElementB = cus;
  using ElementC = cus;
  using ElementCompute = cus;
  using SmArch = cutlass::arch::Sm75;

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementC,            // Data type of output matrix.
      1,                   // The number of elements per vectorized.
                           // memory access. This becomes the vector width of
                           // math instructions in the epilogue too.
      ElementAccumulator,  // Data type of accumulator
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::Default>;  // Data type for
                                                       // alpha/beta in linear
                                                       // combination

  using Conv2d = typename cutlass::conv::kernel::DefaultConv2dFprop<
      ElementA, cutlass::layout::TensorNHWC, ElementB,
      cutlass::layout::TensorNHWC, ElementC, cutlass::layout::TensorNHWC,
      ElementAccumulator, cutlass::arch::OpClassSimt, SmArch, ThreadblockShape,
      WarpShape, InstructionShape, EpilogueOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2,
      cutlass::arch::OpMultiplyAddSaturate,
      cutlass::conv::IteratorAlgorithm::kAnalytic>::Kernel;

  using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2d>;

  cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  const cutlass::layout::TensorNHWC layout_x =
      cutlass::layout::TensorNHWC::packed(input_size);
  const cutlass::layout::TensorNHWC layout_w =
      cutlass::layout::TensorNHWC::packed(filter_size);
  const cutlass::layout::TensorNHWC layout_c =
      cutlass::layout::TensorNHWC::packed(output_size);

  cutlass::conv::Conv2dProblemSize problem_size(
      input_size, filter_size, padding, stride, dilation, output_size, mode,
      split_k_slices);

  cutlass::TensorRef<cus, cutlass::layout::TensorNHWC> tensor_x(
      static_cast<cus*>(input_data), layout_x);
  cutlass::TensorRef<cus, cutlass::layout::TensorNHWC> tensor_w(
      static_cast<cus*>(filter_data), layout_w);
  cutlass::TensorRef<cus, cutlass::layout::TensorNHWC> tensor_c(
      static_cast<cus*>(output_data), layout_c);

  typename ImplicitGemm::Arguments arguments{
      problem_size, tensor_x,
      tensor_w,     tensor_c,
      tensor_c,     {static_cast<cus>(alpha), static_cast<cus>(beta)}};

  ImplicitGemm implicitGemmOp;
  size_t workspaceSize = implicitGemmOp.get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workSpace(workspaceSize);

  cutlass::Status status = implicitGemmOp.can_implement(arguments);
  if (status != cutlass::Status::kSuccess)
    VLOG(3) << "operation not possible" << cutlassGetStatusString(status);
  status = implicitGemmOp(arguments, workSpace.get(), stream);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}

cudaError_t cutlassCusConvBackwardData(
    CUstream stream, MatrixCoord stride, Tensor4DCoord padding,
    MatrixCoord dilation, Tensor4DCoord input_size, void* input_data,
    Tensor4DCoord filter_size, void* filter_data, Tensor4DCoord output_size,
    void* output_data, float alpha, float beta) {
  using ElementA = cus;
  using ElementB = cus;
  using ElementC = cus;
  using ElementCompute = cus;
  using SmArch = cutlass::arch::Sm75;

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementC,            // Data type of output matrix.
      1,                   // The number of elements per vectorized.
                           // memory access. This becomes the vector width of
                           // math instructions in the epilogue too.
      ElementAccumulator,  // Data type of accumulator
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::Default>;  // Data type for
                                                       // alpha/beta in linear
                                                       // combination

  using Conv2d = typename cutlass::conv::kernel::DefaultConv2dDgrad<
      ElementA, cutlass::layout::TensorNHWC, ElementB,
      cutlass::layout::TensorNHWC, ElementC, cutlass::layout::TensorNHWC,
      ElementAccumulator, cutlass::arch::OpClassSimt, SmArch, ThreadblockShape,
      WarpShape, InstructionShape, EpilogueOp,
      cutlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<1>, 2,
      cutlass::arch::OpMultiplyAddSaturate,
      cutlass::conv::IteratorAlgorithm::kAnalytic,
      cutlass::conv::StrideSupport::kStrided>::Kernel;

  using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2d>;

  cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  const cutlass::layout::TensorNHWC layout_x =
      cutlass::layout::TensorNHWC::packed(input_size);
  const cutlass::layout::TensorNHWC layout_w =
      cutlass::layout::TensorNHWC::packed(filter_size);
  const cutlass::layout::TensorNHWC layout_c =
      cutlass::layout::TensorNHWC::packed(output_size);

  cutlass::conv::Conv2dProblemSize problem_size(
      input_size, filter_size, padding, stride, dilation, output_size, mode,
      split_k_slices);

  cutlass::TensorRef<cus, cutlass::layout::TensorNHWC> tensor_x(
      static_cast<cus*>(input_data), layout_x);
  cutlass::TensorRef<cus, cutlass::layout::TensorNHWC> tensor_w(
      static_cast<cus*>(filter_data), layout_w);
  cutlass::TensorRef<cus, cutlass::layout::TensorNHWC> tensor_c(
      static_cast<cus*>(output_data), layout_c);

  typename ImplicitGemm::Arguments arguments{
      problem_size, tensor_c,
      tensor_w,     tensor_x,
      tensor_x,     {static_cast<cus>(alpha), static_cast<cus>(beta)}};

  ImplicitGemm implicitGemmOp;
  size_t workspaceSize = implicitGemmOp.get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workSpace(workspaceSize);

  cutlass::Status status = implicitGemmOp.can_implement(arguments);
  if (status != cutlass::Status::kSuccess)
    VLOG(3) << "operation not possible" << cutlassGetStatusString(status);
  status = implicitGemmOp(arguments, workSpace.get(), stream);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}

cudaError_t cutlassCusConvBackwardFilter(
    CUstream stream, MatrixCoord stride, Tensor4DCoord padding,
    MatrixCoord dilation, Tensor4DCoord input_size, void* input_data,
    Tensor4DCoord filter_size, void* filter_data, Tensor4DCoord output_size,
    void* output_data, float alpha, float beta) {
  using ElementA = cus;
  using ElementB = cus;
  using ElementC = cus;
  using ElementCompute = cus;
  using SmArch = cutlass::arch::Sm75;

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementC,            // Data type of output matrix.
      1,                   // The number of elements per vectorized.
                           // memory access. This becomes the vector width of
                           // math instructions in the epilogue too.
      ElementAccumulator,  // Data type of accumulator
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::Default>;  // Data type for
                                                       // alpha/beta in linear
                                                       // combination

  using Conv2d = typename cutlass::conv::kernel::DefaultConv2dWgrad<
      ElementA, cutlass::layout::TensorNHWC, ElementB,
      cutlass::layout::TensorNHWC, ElementC, cutlass::layout::TensorNHWC,
      ElementAccumulator, cutlass::arch::OpClassSimt, SmArch, ThreadblockShape,
      WarpShape, InstructionShape, EpilogueOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2,
      cutlass::arch::OpMultiplyAdd,
      cutlass::conv::IteratorAlgorithm::kAnalytic>::Kernel;

  using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2d>;

  cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  const cutlass::layout::TensorNHWC layout_x =
      cutlass::layout::TensorNHWC::packed(input_size);
  const cutlass::layout::TensorNHWC layout_w =
      cutlass::layout::TensorNHWC::packed(filter_size);
  const cutlass::layout::TensorNHWC layout_c =
      cutlass::layout::TensorNHWC::packed(output_size);

  cutlass::conv::Conv2dProblemSize problem_size(
      input_size, filter_size, padding, stride, dilation, output_size, mode,
      split_k_slices);

  cutlass::TensorRef<cus, cutlass::layout::TensorNHWC> tensor_x(
      static_cast<cus*>(input_data), layout_x);
  cutlass::TensorRef<cus, cutlass::layout::TensorNHWC> tensor_w(
      static_cast<cus*>(filter_data), layout_w);
  cutlass::TensorRef<cus, cutlass::layout::TensorNHWC> tensor_c(
      static_cast<cus*>(output_data), layout_c);

  //   typename ImplicitGemm::Arguments arguments{
  //       problem_size, tensor_x,
  //       tensor_w,     tensor_c,
  //       tensor_c,     {static_cast<cus>(alpha), static_cast<cus>(beta)}};

  typename ImplicitGemm::Arguments arguments{
      problem_size, tensor_c,
      tensor_x,     tensor_w,
      tensor_w,     {static_cast<cus>(alpha), static_cast<cus>(beta)}};

  ImplicitGemm implicitGemmOp;
  size_t workspaceSize = implicitGemmOp.get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workSpace(workspaceSize);

  cutlass::Status status = implicitGemmOp.can_implement(arguments);
  if (status != cutlass::Status::kSuccess)
    VLOG(3) << "operation not possible" << cutlassGetStatusString(status);
  status = implicitGemmOp(arguments, workSpace.get(), stream);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}

cudaError_t cutlassCusBiasActivationConv(
    CUstream stream, MatrixCoord stride, Tensor4DCoord padding,
    MatrixCoord dilation, Tensor4DCoord input_size, const void* input_data,
    Tensor4DCoord filter_size, const void* filter_data,
    Tensor4DCoord output_size, void* output_data, float alpha, float beta) {
  return cudaErrorUnknown;
}
