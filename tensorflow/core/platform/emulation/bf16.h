#ifndef TENSORFLOW_CORE_PLATFORM_EMULATION_BF16_H_
#define TENSORFLOW_CORE_PLATFORM_EMULATION_BF16_H_

#include "tensorflow/core/platform/emulation/base.h"

namespace bf16 {
  // CUS_DEVICE_FUNC static inline float bfloat16ToFloat(uint16_t v){
CUS_DEVICE_FUNC CUS_INLINE static uint32_t float_to_bfloat16(float f, bool truncate) {
  uint32_t v = *(uint32_t*)&f;
  if (truncate) {
    return v >> 16;
  }
  // todo(chenhao) probably try rounding version later 
  return v >> 16;
}

CUS_DEVICE_FUNC CUS_INLINE static float bfloat16_to_float(uint32_t v) {
  uint32_t v_32 = v << 16;
  return *(float*)&v_32; 
}

}

#endif // TENSORFLOW_CORE_PLATFORM_EMULATION_BF16_H_