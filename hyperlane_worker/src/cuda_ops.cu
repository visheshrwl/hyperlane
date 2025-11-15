#include "cuda_ops.h"
#include <iostream>

namespace hyperlane::worker::cuda {

// INT4 quantization kernel
__global__ void quantize_fp16_to_int4_kernel(const half* input, int4* output,
                                              size_t num_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    // Convert 4x FP16 values to a single INT4 (2-bit per element)
    // Placeholder: simple scaling to [0, 15] range
    half v1 = input[idx * 4];
    half v2 = input[idx * 4 + 1];
    half v3 = input[idx * 4 + 2];
    half v4 = input[idx * 4 + 3];

    // Quantize to 4-bit (0-15 range)
    uint8_t q1 = (uint8_t)(__half2float(v1) * 7.5f + 7.5f);
    uint8_t q2 = (uint8_t)(__half2float(v2) * 7.5f + 7.5f);
    uint8_t q3 = (uint8_t)(__half2float(v3) * 7.5f + 7.5f);
    uint8_t q4 = (uint8_t)(__half2float(v4) * 7.5f + 7.5f);

    // Pack 4x 4-bit into 2 bytes
    uint16_t packed = (q1 << 12) | (q2 << 8) | (q3 << 4) | q4;
    output[idx].x = (int)(packed & 0xFFFF);
  }
}

// INT4 dequantization kernel
__global__ void dequantize_int4_to_fp16_kernel(const int4* input, half* output,
                                                size_t num_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    uint16_t packed = (uint16_t)(input[idx].x & 0xFFFF);

    uint8_t q1 = (packed >> 12) & 0xF;
    uint8_t q2 = (packed >> 8) & 0xF;
    uint8_t q3 = (packed >> 4) & 0xF;
    uint8_t q4 = packed & 0xF;

    // Dequantize back to FP16
    output[idx * 4] = __float2half((q1 / 7.5f) - 1.0f);
    output[idx * 4 + 1] = __float2half((q2 / 7.5f) - 1.0f);
    output[idx * 4 + 2] = __float2half((q3 / 7.5f) - 1.0f);
    output[idx * 4 + 3] = __float2half((q4 / 7.5f) - 1.0f);
  }
}

void quantize_fp16_to_int4(const half* input, int4* output, size_t num_elements,
                           cudaStream_t stream) {
  size_t block_size = 256;
  size_t grid_size = (num_elements + block_size - 1) / block_size;
  quantize_fp16_to_int4_kernel<<<grid_size, block_size, 0, stream>>>(
      input, output, num_elements);
}

void dequantize_int4_to_fp16(const int4* input, half* output, size_t num_elements,
                             cudaStream_t stream) {
  size_t block_size = 256;
  size_t grid_size = (num_elements + block_size - 1) / block_size;
  dequantize_int4_to_fp16_kernel<<<grid_size, block_size, 0, stream>>>(
      input, output, num_elements);
}

}  // namespace hyperlane::worker::cuda
