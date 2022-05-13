#ifndef TENSORFLOW_CORE_KERNELS_CUTLASS_CONV_H_
#define TENSORFLOW_CORE_KERNELS_CUTLASS_CONV_H_

#include "tensorflow/core/platform/cus.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/matrix_coord.h"

using cutlass::MatrixCoord;
using cutlass::Tensor4DCoord;

using tensorflow::cus;

cudaError_t cutlassCusConvForward(MatrixCoord stride, Tensor4DCoord padding,
                           MatrixCoord dilation, Tensor4DCoord input_size,
                           void* input_data, Tensor4DCoord filter_size,
                           void* filter_data, Tensor4DCoord output_size,
                           void* output_data, float alpha, float beta);

cudaError_t cutlassCusConvBackwardData(MatrixCoord stride, Tensor4DCoord padding,
                           MatrixCoord dilation, Tensor4DCoord input_size,
                           void* input_data, Tensor4DCoord filter_size,
                           void* filter_data, Tensor4DCoord output_size,
                           void* output_data, float alpha, float beta);

cudaError_t cutlassCusConvBackwardFilter(MatrixCoord stride, Tensor4DCoord padding,
                           MatrixCoord dilation, Tensor4DCoord input_size,
                           void* input_data, Tensor4DCoord filter_size,
                           void* filter_data, Tensor4DCoord output_size,
                           void* output_data, float alpha, float beta);

cudaError_t cutlassCusBiasActivationConv(
    MatrixCoord stride, Tensor4DCoord padding, MatrixCoord dilation,
    Tensor4DCoord input_size, const void* input_data, Tensor4DCoord filter_size,
    const void* filter_data, Tensor4DCoord output_size, void* output_data,
    void* alpha, void* beta);

#endif  // TENSORFLOW_CORE_KERNELS_CUTLASS_CONV_H_