#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace hyperlane::worker::cuda {

/**
 * Quantize FP16 tensor to INT4 in VRAM using a custom CUDA kernel.
 * Reduces tensor size 4x and enables DMA-friendly transport.
 */
void quantize_fp16_to_int4(const half* input, int4* output, size_t num_elements,
                           cudaStream_t stream);

/**
 * Dequantize INT4 tensor back to FP16 in VRAM.
 */
void dequantize_int4_to_fp16(const int4* input, half* output, size_t num_elements,
                             cudaStream_t stream);

/**
 * Async memcpy wrapper for explicit stream management.
 */
inline cudaError_t async_memcpy_h2d(void* dst, const void* src, size_t size,
                                    cudaStream_t stream) {
  return cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
}

inline cudaError_t async_memcpy_d2h(void* dst, const void* src, size_t size,
                                    cudaStream_t stream) {
  return cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream);
}

}  // namespace hyperlane::worker::cuda
