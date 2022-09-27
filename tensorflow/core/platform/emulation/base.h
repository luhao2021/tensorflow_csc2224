#ifndef TENSORFLOW_CORE_PLATFORM_EMULATION_BASE_H_
#define TENSORFLOW_CORE_PLATFORM_EMULATION_BASE_H_

#include<complex>
#include<cmath>

#ifdef __CUDACC__
// All functions callable from CUDA code must be qualified with __device__
#define CUS_DEVICE_FUNC __host__ __device__
// CUDA requires forceinline or inlining is undefined
#define CUS_INLINE __forceinline__
#else
#define CUS_DEVICE_FUNC
#define CUS_INLINE inline
#endif

extern "C" {
  float __nv_logf(float);
  float __nv_expf(float);
  float __nv_sqrtf(float);
  float __nv_rsqrtf(float);
}


namespace tensorflow {

typedef std::complex<float> complex64;
typedef std::complex<double> complex128;

extern "C" CUS_INLINE CUS_DEVICE_FUNC uint32_t CastF32ToValue(const float f);
extern "C" CUS_INLINE CUS_DEVICE_FUNC float CastValueToF32(const uint32_t u);

struct cus {
  uint32_t value;

  CUS_DEVICE_FUNC constexpr cus() : value(0) {}
  CUS_INLINE explicit CUS_DEVICE_FUNC cus(const float& f) : value(CastF32ToValue(f)) {}
  CUS_INLINE explicit CUS_DEVICE_FUNC cus(const double& d) : cus(static_cast<float>(d)) {}
  explicit CUS_DEVICE_FUNC cus(const complex64& c64)
      : cus(static_cast<float>(c64.real())) {}
  explicit CUS_DEVICE_FUNC cus(const complex128& c128)
      : cus(static_cast<float>(c128.real())) {}

  template <class T>
  explicit CUS_DEVICE_FUNC cus(const T& value)
      : cus(static_cast<float>(value)) {}

  CUS_INLINE CUS_DEVICE_FUNC operator float() const { return CastValueToF32(value); }

  explicit CUS_DEVICE_FUNC operator double() const {
    float f = static_cast<float>(*this);
    return static_cast<double>(f);
  }

};


extern "C" {
CUS_INLINE CUS_DEVICE_FUNC cus CusAdd(const cus a, const cus b);
CUS_INLINE CUS_DEVICE_FUNC cus CusSub(const cus a, const cus b);
CUS_INLINE CUS_DEVICE_FUNC cus CusMul(const cus a, const cus b);
CUS_INLINE CUS_DEVICE_FUNC cus CusDiv(const cus a, const cus b);

CUS_INLINE CUS_DEVICE_FUNC bool CusEq(const cus a, const cus b);
CUS_INLINE CUS_DEVICE_FUNC bool CusNe(const cus a, const cus b);
CUS_INLINE CUS_DEVICE_FUNC bool CusLt(const cus a, const cus b);
CUS_INLINE CUS_DEVICE_FUNC bool CusLe(const cus a, const cus b);
CUS_INLINE CUS_DEVICE_FUNC bool CusGt(const cus a, const cus b);
CUS_INLINE CUS_DEVICE_FUNC bool CusGe(const cus a, const cus b);
CUS_INLINE CUS_DEVICE_FUNC cus CusMax(const cus a, const cus b);

CUS_INLINE CUS_DEVICE_FUNC cus CusNeg(const cus a);
CUS_INLINE CUS_DEVICE_FUNC cus CusLog(const cus a);
CUS_INLINE CUS_DEVICE_FUNC cus CusExp(const cus a);
CUS_INLINE CUS_DEVICE_FUNC cus CusExpm1(const cus a);
CUS_INLINE CUS_DEVICE_FUNC cus CusLog1p(const cus a);
CUS_INLINE CUS_DEVICE_FUNC cus CusCos(const cus a);
CUS_INLINE CUS_DEVICE_FUNC cus CusSin(const cus a);
CUS_INLINE CUS_DEVICE_FUNC cus CusTanh(const cus a);
CUS_INLINE CUS_DEVICE_FUNC cus CusSqrt(const cus a);
CUS_INLINE CUS_DEVICE_FUNC cus CusRsqrt(const cus a);
CUS_INLINE CUS_DEVICE_FUNC cus CusCbrt(const cus a);
CUS_INLINE CUS_DEVICE_FUNC cus CusFloor(const cus a);
CUS_INLINE CUS_DEVICE_FUNC cus CusCeil(const cus a);
CUS_INLINE CUS_DEVICE_FUNC cus CusAbs(const cus a);
CUS_INLINE CUS_DEVICE_FUNC cus CusRoundNearestAfz(const cus a);
CUS_INLINE CUS_DEVICE_FUNC cus CusSign(const cus a);
CUS_INLINE CUS_DEVICE_FUNC cus CusIsFinite(const cus a);
}

CUS_INLINE CUS_DEVICE_FUNC cus operator+(const cus& a, const cus& b){ return CusAdd(a,b);}
CUS_INLINE CUS_DEVICE_FUNC cus operator-(const cus& a){ return CusNeg(a);}
CUS_INLINE CUS_DEVICE_FUNC cus operator-(const cus& a, const cus& b){ return CusSub(a,b);}
CUS_INLINE CUS_DEVICE_FUNC cus operator*(const cus& a, const cus& b){ return CusMul(a,b);}
CUS_INLINE CUS_DEVICE_FUNC cus operator/(const cus& a, const cus& b){ return CusDiv(a,b);}
CUS_INLINE CUS_DEVICE_FUNC cus& operator+=(cus& a, const cus& b) { 
  a = a + b;
  return a;
}
CUS_INLINE CUS_DEVICE_FUNC cus& operator-=(cus& a, const cus& b) { 
  a = a - b;
  return a;
}
CUS_INLINE CUS_DEVICE_FUNC cus& operator*=(cus& a, const cus& b) { 
  a = a * b;
  return a;
}
CUS_INLINE CUS_DEVICE_FUNC cus& operator/=(cus& a, const cus& b){
  a = a / b;
  return a;
}
CUS_INLINE CUS_DEVICE_FUNC bool operator==(const cus& a, const cus& b){ return CusEq(a,b);}
CUS_INLINE CUS_DEVICE_FUNC bool operator<(const cus& a, const cus& b){ return CusLt(a,b);}
CUS_INLINE CUS_DEVICE_FUNC bool operator<=(const cus& a, const cus& b){ return CusLe(a,b);}
CUS_INLINE CUS_DEVICE_FUNC bool operator!=(const cus& a, const cus& b){ return CusNe(a,b);}
CUS_INLINE CUS_DEVICE_FUNC bool operator>(const cus& a, const cus& b){ return CusGt(a,b);}
CUS_INLINE CUS_DEVICE_FUNC bool operator>=(const cus& a, const cus& b){ return CusGe(a,b);}

} // namespace tensorflow

namespace std {
template <>
struct hash<tensorflow::cus> {
  std::size_t operator()(tensorflow::cus const& c) const noexcept {
    std::size_t h1 = std::hash<uint32_t>{}(c.value);
    return h1;
  }
};
}  // namespace std


#endif  // TENSORFLOW_CORE_PLATFORM_EMULATION_BASE_H_
