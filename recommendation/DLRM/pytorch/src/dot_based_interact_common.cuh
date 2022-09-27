#ifndef DOT_BASED_INTERACT_COMMON_H
#define DOT_BASED_INTERACT_COMMON_H
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#ifndef __HIP_PLATFORM_HCC__
#include <device_launch_parameters.h>
#include <cuda_fp16.hpp>
#endif

#include <math.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

namespace dlrm_dot {
  template <uint x>
    struct Log2 {
      static constexpr uint value = 1 + Log2<x / 2>::value;
    };
  template <>
    struct Log2<1> {
      static constexpr uint value = 0;
    };

  struct __align__(8) half4 {
    half2 vals[2];
  };
}
#endif // DOT_BASED_INTERACT_COMMON_H
