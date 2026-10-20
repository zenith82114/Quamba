/******************************************************************************
 * Quantized causal conv1d (+ SiLU) whose output u is written twice, once per consumer:
 *   out  : int8 at the static per-tensor scale scale_out (the x_proj GEMM input), the same codes as
 *          quant_causal_conv1d_fwd_kernel;
 *   out2 : int8 for the selective scan, at a second static scale scale_out2, or (kDynamic) at a
 *          per-token scale over each group of kChannels channels, absmax / 127, written to scales.
 * Sequence-last layout only. A block computes kChannels channels x kNThreads tokens (one token per
 * thread), so the per-token absmax over a channel group stays inside the block: the thread runs over
 * its token's channels once for the absmax and once more to quantize, recomputing the cheap conv.
 ******************************************************************************/

#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include "causal_conv1d.h"
#include "common/static_switch.h"

template<int kNThreads, int kWidth, int kChannels, bool kDynamic, typename input_t, typename weight_t>
__global__ __launch_bounds__(kNThreads)
void quant_causal_conv1d_dual_fwd_kernel(QuantConvDualParams params) {
    constexpr int kL = kNThreads;
    constexpr int kHalo = kWidth - 1;
    __shared__ input_t x_smem[kChannels][kL + kHalo];
    __shared__ float w_smem[kChannels][kWidth];
    __shared__ float b_smem[kChannels];

    const float scale_wx = params.scale_w * params.scale_x;
    const int tid = threadIdx.x;
    const int batch_id = blockIdx.x;
    const int c0 = blockIdx.y * kChannels;
    const int l0 = blockIdx.z * kL;

    // the input tile and the kWidth - 1 tokens before it (zeros before the sequence start)
    const input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride;
    for (int idx = tid; idx < kChannels * (kL + kHalo); idx += kNThreads) {
        const int c = idx / (kL + kHalo), j = idx % (kL + kHalo);
        const int l = l0 - kHalo + j;
        x_smem[c][j] = (l >= 0 && l < params.seqlen) ? x[(c0 + c) * params.x_c_stride + l * params.x_l_stride] : 0;
    }
    const weight_t *weight = reinterpret_cast<weight_t *>(params.weight_ptr);
    for (int idx = tid; idx < kChannels * kWidth; idx += kNThreads) {
        const int c = idx / kWidth, w = idx % kWidth;
        w_smem[c][w] = float(weight[(c0 + c) * params.weight_c_stride + w * params.weight_width_stride]);
    }
    for (int c = tid; c < kChannels; c += kNThreads) {
        b_smem[c] = params.bias_ptr == nullptr ? 0.f
            : float(reinterpret_cast<weight_t *>(params.bias_ptr)[c0 + c]) * params.scale_b;
    }
    __syncthreads();

    const int l = l0 + tid;
    if (l >= params.seqlen) { return; }

    // same arithmetic as quant_causal_conv1d_fwd_kernel, so out matches its codes
    auto conv = [&](int c) {
        float acc = 0.f;
        #pragma unroll
        for (int w = 0; w < kWidth; ++w) { acc += w_smem[c][w] * float(x_smem[c][tid + w]); }
        float v = scale_wx * acc + b_smem[c];
        if (params.silu_activation) { v = v / (1 + expf(-v)); }
        return v;
    };
    auto quant = [](float v, float scale) {
        const int q = int(roundf(v / scale));
        return static_cast<input_t>(q > 127 ? 127 : q < -128 ? -128 : q);
    };

    float scale2 = params.scale_out2;
    if constexpr (kDynamic) {
        float amax = 0.f;
        for (int c = 0; c < kChannels; ++c) { amax = fmaxf(amax, fabsf(conv(c))); }
        scale2 = fmaxf(amax, 1e-5f) / 127.f;
        reinterpret_cast<float *>(params.scales_ptr)[batch_id * params.scales_batch_stride
            + blockIdx.y * params.scales_g_stride + l] = scale2;
    }
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride + l * params.out_l_stride;
    input_t *out2 = reinterpret_cast<input_t *>(params.out2_ptr) + batch_id * params.out2_batch_stride + l * params.out2_l_stride;
    for (int c = 0; c < kChannels; ++c) {
        const float v = conv(c);
        out[(c0 + c) * params.out_c_stride] = quant(v, params.scale_out);
        out2[(c0 + c) * params.out2_c_stride] = quant(v, scale2);
    }
}

template<int kWidth, int kChannels, bool kDynamic, typename input_t, typename weight_t>
void quant_causal_conv1d_dual_fwd_launch(QuantConvDualParams &params, cudaStream_t stream) {
    constexpr int kNThreads = 128;
    dim3 grid(params.batch, params.dim / kChannels, (params.seqlen + kNThreads - 1) / kNThreads);
    auto kernel = &quant_causal_conv1d_dual_fwd_kernel<kNThreads, kWidth, kChannels, kDynamic, input_t, weight_t>;
    kernel<<<grid, kNThreads, 0, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int kWidth, typename input_t, typename weight_t>
void quant_causal_conv1d_dual_fwd_width(QuantConvDualParams &params, cudaStream_t stream) {
    // static second scale: any channel tile works; dynamic: the tile is the scale group
    if (params.group_size == 0) {
        quant_causal_conv1d_dual_fwd_launch<kWidth, 64, false, input_t, weight_t>(params, stream);
    } else if (params.group_size == 32) {
        quant_causal_conv1d_dual_fwd_launch<kWidth, 32, true, input_t, weight_t>(params, stream);
    } else if (params.group_size == 64) {
        quant_causal_conv1d_dual_fwd_launch<kWidth, 64, true, input_t, weight_t>(params, stream);
    } else if (params.group_size == 128) {
        quant_causal_conv1d_dual_fwd_launch<kWidth, 128, true, input_t, weight_t>(params, stream);
    }
}

template<typename input_t, typename weight_t>
void quant_causal_conv1d_dual_fwd_cuda(QuantConvDualParams &params, cudaStream_t stream) {
    if (params.width == 2) {
        quant_causal_conv1d_dual_fwd_width<2, input_t, weight_t>(params, stream);
    } else if (params.width == 3) {
        quant_causal_conv1d_dual_fwd_width<3, input_t, weight_t>(params, stream);
    } else if (params.width == 4) {
        quant_causal_conv1d_dual_fwd_width<4, input_t, weight_t>(params, stream);
    }
}

template void quant_causal_conv1d_dual_fwd_cuda<int8_t, int8_t>(QuantConvDualParams &params, cudaStream_t stream);
