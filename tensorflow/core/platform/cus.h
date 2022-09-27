/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_PLATFORM_CUS_H_
#define TENSORFLOW_CORE_PLATFORM_CUS_H_

// #include "tensorflow/core/platform/emulation/fp16.h"
// #include "tensorflow/core/platform/emulation/bf16.h"
// #include "tensorflow/core/platform/emulation/exp8.h"
#include "tensorflow/core/platform/emulation/general_float.h"


namespace tensorflow {
CUS_DEVICE_FUNC uint32_t CastF32ToValue(const float f) {
  // uint32_t v = static_cast<uint32_t>(fp16::fp16_ieee_from_fp32_value(f));
  // uint32_t v = exp8::float_to_type(f, /*truncate=*/true);
  return generalfloat::float_to_type(f);
}

CUS_DEVICE_FUNC float CastValueToF32(const uint32_t u) {
  // return fp16::fp16_ieee_to_fp32_value(v);
  // return bf16::bfloat16_to_float(u);
  return generalfloat::type_to_float(u);
}

CUS_DEVICE_FUNC cus CusAdd(const cus a, const cus b) {
  return cus(static_cast<float>(a) + static_cast<float>(b));
}

CUS_DEVICE_FUNC cus CusSub(const cus a, const cus b) {
  return cus(static_cast<float>(a) - static_cast<float>(b));
}

CUS_DEVICE_FUNC cus CusMul(const cus a, const cus b) {
  return cus(static_cast<float>(a) * static_cast<float>(b));
}

CUS_DEVICE_FUNC cus CusDiv(const cus a, const cus b) {
  return cus(static_cast<float>(a) / static_cast<float>(b));
}

CUS_DEVICE_FUNC cus CusMax(const cus a, const cus b) {
  return a > b ? a : b;
}

CUS_DEVICE_FUNC bool CusEq(const cus a, const cus b) {
  return static_cast<float>(a) == static_cast<float>(b);
}
CUS_DEVICE_FUNC bool CusNe(const cus a, const cus b) {
  return static_cast<float>(a) != static_cast<float>(b);
}
CUS_DEVICE_FUNC bool CusLt(const cus a, const cus b) {
  return static_cast<float>(a) < static_cast<float>(b);
}
CUS_DEVICE_FUNC bool CusLe(const cus a, const cus b) {
  return static_cast<float>(a) <= static_cast<float>(b);
}
CUS_DEVICE_FUNC bool CusGt(const cus a, const cus b) {
  return static_cast<float>(a) > static_cast<float>(b);
}
CUS_DEVICE_FUNC bool CusGe(const cus a, const cus b) {
  return static_cast<float>(a) >= static_cast<float>(b);
}

CUS_DEVICE_FUNC cus CusNeg(const cus a) {
  return cus(-static_cast<float>(a));
}

CUS_DEVICE_FUNC cus CusLog(const cus a) {
  return cus(__nv_logf(static_cast<float>(a)));
}

CUS_DEVICE_FUNC cus CusExp(const cus a) {
  return cus(__nv_expf(static_cast<float>(a)));
}

CUS_DEVICE_FUNC cus CusSqrt(const cus a) {
  return cus(__nv_sqrtf(static_cast<float>(a)));
}

CUS_DEVICE_FUNC cus CusRsqrt(const cus a) {
  return cus(__nv_rsqrtf(static_cast<float>(a)));
}

}  // namespace tensorflow



#endif  // TENSORFLOW_CORE_PLATFORM_CUS_H_
