#ifndef TENSORFLOW_CORE_PLATFORM_EMULATION_generalfloat_H_
#define TENSORFLOW_CORE_PLATFORM_EMULATION_generalfloat_H_

#include "tensorflow/core/platform/emulation/base.h"


namespace generalfloat {
#if __clang__
#define CONST_OPT 
#else 
#define CONST_OPT constexpr
#endif

// This works with all floating point with sign = 1, exp < 8 and mantissa <= 23
// todo(chenhao) add a temporary check for exp==8, maybe could change to cast float to double
// and then back to general float to support exp >= 8
#define F32_WIDTH 32
#define F32_EXP 8
#define F32_MANTISSA 23
#define F32_EXP_MANT 31
#define F32_BIAS 127
#define F32_EXP_MASK 0x7F800000
#define F32_MANT_MASK 0x007FFFFF
#define F32_SIGN_MASK 0x80000000

#define WIDTH 15
#define SIGN 1

#define EXP 7

#define MANTISSA (WIDTH - SIGN - EXP)

#define EXP_BIAS ((1 << (EXP - 1)) - 1)
#define EXP_MAX EXP_BIAS
#define EXP_DENORM (1 - EXP_BIAS)

#define RTNE_BIT (F32_MANTISSA - MANTISSA)
#define RTNE_VALUE (1 << (RTNE_BIT - 1))

#define ONES(n) ((1 << (n)) - 1)

#define EXP_MANT_MASK ONES(WIDTH - 1)

#define EXP_MASK (ONES(EXP) << MANTISSA)
// #define EXP_MASK ((EXP_MANT_MASK >> MANTISSA) << MANTISSA)
#define MANT_MASK (ONES(MANTISSA))
// #define MANT_MASK (EXP_MANT_MASK - EXP_MASK)
#define SIGN_MASK (1 << (WIDTH - 1))
#define BIAS_DIFF (F32_BIAS - EXP_BIAS)

#define MANT_DIFF (F32_MANTISSA - MANTISSA)
#define MANT_MASK_ON_F32 (MANT_MASK << MANT_DIFF)

CONST_OPT CUS_DEVICE_FUNC CUS_INLINE uint32_t float_as_uint32(float f) {
  return *(uint32_t*)&f;
}

CONST_OPT CUS_DEVICE_FUNC CUS_INLINE float uint32_as_float(uint32_t v) {
  return *(float*)&v;
}

CONST_OPT CUS_DEVICE_FUNC CUS_INLINE float type_to_float(uint32_t x) {
#if EXP == 8
  uint32_t f_b = x << MANT_DIFF;
  return uint32_as_float(f_b);
#else
  const uint32_t e = (x & EXP_MASK) >> MANTISSA;    // exponent
  const uint32_t m = (x & MANT_MASK) << MANT_DIFF;  // mantissa
  const float m_f = static_cast<float>(m);
  // 31 - (v - 127) represents the clz
  const uint32_t v =  float_as_uint32(m_f) >> F32_MANTISSA;
  // sign : normalized : denormalized
  const uint32_t f_bit =
      (x & SIGN_MASK) << (F32_WIDTH - WIDTH) |
      (e != 0) *
          ((e + BIAS_DIFF + (e == ONES(EXP)) * BIAS_DIFF) << F32_MANTISSA | m) |
      ((e == 0) & (m != 0)) *
          ((EXP_DENORM - (F32_EXP_MANT - v - F32_EXP)) << F32_MANTISSA |
           ((m << (F32_EXP_MANT - (v - F32_BIAS) - F32_EXP)) &
            MANT_MASK_ON_F32));
  return uint32_as_float(f_bit);
#endif
}

// https://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion
CONST_OPT CUS_DEVICE_FUNC CUS_INLINE uint32_t float_to_type(float f) {
#if EXP == 8
  uint32_t v = *(uint32_t*)&f;
  // uint32_t lsb = (v >> MANT_DIFF)&1;
  // uint32_t rounding_bias = ONES(MANT_DIFF-1) + lsb;
  // v += rounding_bias;
  v += RTNE_VALUE;
  return v >> MANT_DIFF;
#else
  const uint32_t denorm_min = float_as_uint32(type_to_float(1U));
  const uint32_t denorm_min_exp = (denorm_min & F32_EXP_MASK) >> F32_MANTISSA;

  const uint32_t v = *(uint32_t*)&f + RTNE_VALUE;
  const uint32_t e = (v & F32_EXP_MASK) >> F32_MANTISSA;
  const uint32_t m = v & F32_MANT_MASK;
  // sign : normalized : denormalized : saturate
  return (v & F32_SIGN_MASK) >> (F32_WIDTH - WIDTH) |
         ((e > BIAS_DIFF) & (e <= (F32_BIAS + EXP_MAX))) *
             ((((e - BIAS_DIFF) << MANTISSA) & EXP_MASK) | m >> RTNE_BIT) |
         ((e <= BIAS_DIFF) & (e >= denorm_min_exp)) *
             (((((1 << F32_MANTISSA) - RTNE_VALUE + m) >> (EXP_DENORM - (e - F32_BIAS) + MANT_DIFF - 1)) + 1) >> 1) |
         (e > (F32_BIAS + EXP_MAX)) * EXP_MASK;
#endif
}

}  // namespace generalfloat

#endif  // TENSORFLOW_CORE_PLATFORM_EMULATION_BF16_H_
