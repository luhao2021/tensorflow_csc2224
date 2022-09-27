#ifndef TENSORFLOW_CORE_PLATFORM_EMULATION_EXP8_H_
#define TENSORFLOW_CORE_PLATFORM_EMULATION_EXP8_H_

#include "tensorflow/core/platform/emulation/base.h"

// #define SHIFT_BIT 23 // mantissa = 0 bit
// #define SHIFT_BIT 19
// #define SHIFT_BIT 18
// #define SHIFT_BIT 17
#define SHIFT_BIT 16 // bfloat16
// #define SHIFT_BIT 15
// #define SHIFT_BIT 14
// #define SHIFT_BIT 13 // tfloat32
// #define SHIFT_BIT 12 

namespace exp8 {
  // CUS_DEVICE_FUNC static inline float bfloat16ToFloat(uint16_t v){
CUS_DEVICE_FUNC CUS_INLINE static uint32_t float_to_type(float f, bool truncate) {
  uint32_t v = *(uint32_t*)&f;
  if (truncate) {
    return v >> SHIFT_BIT;
  }
  // todo(chenhao) probably try rounding version later 
  return v >> SHIFT_BIT;
}

CUS_DEVICE_FUNC CUS_INLINE static float type_to_float(uint32_t v) {
  uint32_t v_32 = v << SHIFT_BIT;
  return *(float*)&v_32; 
}

}

#undef SHIFT_BIT

#endif // TENSORFLOW_CORE_PLATFORM_EMULATION_BF16_H_
