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

// This type only supports conversion back and forth with float.

#include <complex>
#include <cmath>

#ifdef __CUDACC__
// All functions callable from CUDA code must be qualified with __device__
#define CUSTOM_DEVICE_FUNC __host__ __device__

#else
#define CUSTOM_DEVICE_FUNC

#endif

namespace tensorflow {


typedef std::complex<float> complex64;
typedef std::complex<double> complex128;

extern "C" inline CUSTOM_DEVICE_FUNC uint32_t CastF32ToValue(const float f);
extern "C" inline CUSTOM_DEVICE_FUNC float CastValueToF32(const uint32_t u);

struct cus {
  uint32_t value;

  CUSTOM_DEVICE_FUNC constexpr cus() : value(0) {}
  explicit CUSTOM_DEVICE_FUNC cus(const float& f) : value(CastF32ToValue(f)) {}
  explicit CUSTOM_DEVICE_FUNC cus(const double& d) : cus(static_cast<float>(d)) {}
  explicit CUSTOM_DEVICE_FUNC cus(const complex64& c64)
      : cus(static_cast<float>(c64.real())) {}
  explicit CUSTOM_DEVICE_FUNC cus(const complex128& c128)
      : cus(static_cast<float>(c128.real())) {}

  template <class T>
  explicit CUSTOM_DEVICE_FUNC cus(const T& value)
      : cus(static_cast<float>(value)) {}

  CUSTOM_DEVICE_FUNC operator float() const { return CastValueToF32(value); }

  explicit CUSTOM_DEVICE_FUNC operator double() const {
    float f = static_cast<float>(*this);
    return static_cast<double>(f);
  }

};


extern "C" {
inline CUSTOM_DEVICE_FUNC cus CusAdd(const cus a, const cus b);
inline CUSTOM_DEVICE_FUNC cus CusSub(const cus a, const cus b);
inline CUSTOM_DEVICE_FUNC cus CusMul(const cus a, const cus b);
inline CUSTOM_DEVICE_FUNC cus CusDiv(const cus a, const cus b);

inline CUSTOM_DEVICE_FUNC bool CusEq(const cus a, const cus b);
inline CUSTOM_DEVICE_FUNC bool CusNe(const cus a, const cus b);
inline CUSTOM_DEVICE_FUNC bool CusLt(const cus a, const cus b);
inline CUSTOM_DEVICE_FUNC bool CusLe(const cus a, const cus b);
inline CUSTOM_DEVICE_FUNC bool CusGt(const cus a, const cus b);
inline CUSTOM_DEVICE_FUNC bool CusGe(const cus a, const cus b);
inline CUSTOM_DEVICE_FUNC cus CusMax(const cus a, const cus b);

inline CUSTOM_DEVICE_FUNC cus CusNeg(const cus a);
inline CUSTOM_DEVICE_FUNC cus CusLog(const cus a);
inline CUSTOM_DEVICE_FUNC cus CusExp(const cus a);
inline CUSTOM_DEVICE_FUNC cus CusExpm1(const cus a);
inline CUSTOM_DEVICE_FUNC cus CusLog1p(const cus a);
inline CUSTOM_DEVICE_FUNC cus CusCos(const cus a);
inline CUSTOM_DEVICE_FUNC cus CusSin(const cus a);
inline CUSTOM_DEVICE_FUNC cus CusTanh(const cus a);
inline CUSTOM_DEVICE_FUNC cus CusSqrt(const cus a);
inline CUSTOM_DEVICE_FUNC cus CusRsqrt(const cus a);
inline CUSTOM_DEVICE_FUNC cus CusCbrt(const cus a);
inline CUSTOM_DEVICE_FUNC cus CusFloor(const cus a);
inline CUSTOM_DEVICE_FUNC cus CusCeil(const cus a);
inline CUSTOM_DEVICE_FUNC cus CusAbs(const cus a);
inline CUSTOM_DEVICE_FUNC cus CusRoundNearestAfz(const cus a);
inline CUSTOM_DEVICE_FUNC cus CusSign(const cus a);
inline CUSTOM_DEVICE_FUNC cus CusIsFinite(const cus a);
}

inline CUSTOM_DEVICE_FUNC cus operator+(const cus& a, const cus& b){ return CusAdd(a,b);}
inline CUSTOM_DEVICE_FUNC cus operator-(const cus& a){ return CusNeg(a);}
inline CUSTOM_DEVICE_FUNC cus operator-(const cus& a, const cus& b){ return CusSub(a,b);}
inline CUSTOM_DEVICE_FUNC cus operator*(const cus& a, const cus& b){ return CusMul(a,b);}
inline CUSTOM_DEVICE_FUNC cus operator/(const cus& a, const cus& b){ return CusDiv(a,b);}
inline CUSTOM_DEVICE_FUNC cus& operator+=(cus& a, const cus& b) { 
  a = a + b;
  return a;
}
inline CUSTOM_DEVICE_FUNC cus& operator-=(cus& a, const cus& b) { 
  a = a - b;
  return a;
}
inline CUSTOM_DEVICE_FUNC cus& operator*=(cus& a, const cus& b) { 
  a = a * b;
  return a;
}
inline CUSTOM_DEVICE_FUNC cus& operator/=(cus& a, const cus& b){
  a = a / b;
  return a;
}
inline CUSTOM_DEVICE_FUNC bool operator==(const cus& a, const cus& b){ return CusEq(a,b);}
inline CUSTOM_DEVICE_FUNC bool operator<(const cus& a, const cus& b){ return CusLt(a,b);}
inline CUSTOM_DEVICE_FUNC bool operator<=(const cus& a, const cus& b){ return CusLe(a,b);}
inline CUSTOM_DEVICE_FUNC bool operator!=(const cus& a, const cus& b){ return CusNe(a,b);}
inline CUSTOM_DEVICE_FUNC bool operator>(const cus& a, const cus& b){ return CusGt(a,b);}
inline CUSTOM_DEVICE_FUNC bool operator>=(const cus& a, const cus& b){ return CusGe(a,b);}

}  // namespace tensorflow

namespace std {
template <>
struct hash<tensorflow::cus> {
  std::size_t operator()(tensorflow::cus const& c) const noexcept {
    std::size_t h1 = std::hash<uint32_t>{}(c.value);
    return h1;
  }
};
}  // namespace std

namespace tensorflow {

CUSTOM_DEVICE_FUNC uint32_t CastF32ToValue(const float f){
  return  *(uint32_t*) &f;
}

CUSTOM_DEVICE_FUNC float CastValueToF32(const uint32_t u){
  return *(float*)&u;
}


CUSTOM_DEVICE_FUNC cus CusAdd(const cus a, const cus b) {
  return cus(static_cast<float>(a) + static_cast<float>(b));
}

CUSTOM_DEVICE_FUNC cus CusSub(const cus a, const cus b) {
  return cus(static_cast<float>(a) - static_cast<float>(b));
}

CUSTOM_DEVICE_FUNC cus CusMul(const cus a, const cus b) {
  return cus(static_cast<float>(a) * static_cast<float>(b));
}

CUSTOM_DEVICE_FUNC cus CusDiv(const cus a, const cus b) {
  return cus(static_cast<float>(a) / static_cast<float>(b));
}

CUSTOM_DEVICE_FUNC cus CusMax(const cus a, const cus b) {
  return a > b ? a : b;
}


CUSTOM_DEVICE_FUNC bool CusEq(const cus a, const cus b) { return static_cast<float>(a) == static_cast<float>(b); }
CUSTOM_DEVICE_FUNC bool CusNe(const cus a, const cus b) { return static_cast<float>(a) != static_cast<float>(b); }
CUSTOM_DEVICE_FUNC bool CusLt(const cus a, const cus b) { return static_cast<float>(a) < static_cast<float>(b);}
CUSTOM_DEVICE_FUNC bool CusLe(const cus a, const cus b) { return static_cast<float>(a) <= static_cast<float>(b);}
CUSTOM_DEVICE_FUNC bool CusGt(const cus a, const cus b) { return static_cast<float>(a) > static_cast<float>(b);}
CUSTOM_DEVICE_FUNC bool CusGe(const cus a, const cus b) { return static_cast<float>(a) >= static_cast<float>(b); }


CUSTOM_DEVICE_FUNC cus CusNeg(const cus a) {
  return cus(-static_cast<float>(a));
}

CUSTOM_DEVICE_FUNC cus CusLog(const cus a){
  return cus(std::log(static_cast<float>(a)));
}

CUSTOM_DEVICE_FUNC cus CusExp(const cus a){
  return cus(std::exp(static_cast<float>(a)));
}

CUSTOM_DEVICE_FUNC cus CusSqrt(const cus a){
  return cus(std::sqrt(static_cast<float>(a)));
}

CUSTOM_DEVICE_FUNC cus CusRsqrt(const cus a){
  return cus(1.0f / std::sqrt(static_cast<float>(a)));
}

}  // namespace tensorflow


#endif  // TENSORFLOW_CORE_PLATFORM_CUS_H_