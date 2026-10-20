/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

#include "causal_conv1d.h"
#include "common/type_shim.h"

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")


// CUDA forward declarations
template<typename input_t, typename weight_t>
void quant_causal_conv1d_fwd_cuda(QuantConvParamsBase &params, cudaStream_t stream);

template<typename input_t, typename weight_t>
void quant_causal_conv1d_channellast_fwd_cuda(QuantConvParamsBase &params, cudaStream_t stream);

template<typename input_t, typename weight_t>
void quant_causal_conv1d_update_cuda(QuantConvParamsBase &params, cudaStream_t stream);

template<typename input_t, typename weight_t>
void quant_causal_conv1d_dual_fwd_cuda(QuantConvDualParams &params, cudaStream_t stream);



void set_conv_params_fwd(QuantConvParamsBase &params,
                         // sizes
                         const size_t batch,
                         const size_t dim,
                         const size_t seqlen,
                         const size_t width,
                         // device pointers
                         const at::Tensor x,
                         const float scale_x,
                         const at::Tensor weight,
                         const float scale_w,
                         const at::Tensor out,
                         const float scale_out,
                         const float scale_b,
                         void* bias_ptr,
                         bool silu_activation) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.seqlen = seqlen;
    params.width = width;

    params.silu_activation = silu_activation;

    // Set the pointers and strides.
    params.x_ptr = x.data_ptr();
    params.scale_x = scale_x;
    params.weight_ptr = weight.data_ptr();
    params.scale_w = scale_w;
    params.bias_ptr = bias_ptr;
    params.scale_b = scale_b;
    params.out_ptr = out.data_ptr();
    params.scale_out = scale_out;
    // All stride are in elements, not bytes.
    params.x_batch_stride = x.stride(0);
    params.x_c_stride = x.stride(1);
    params.x_l_stride = x.stride(-1);
    params.weight_c_stride = weight.stride(0);
    params.weight_width_stride = weight.stride(1);
    params.out_batch_stride = out.stride(0);
    params.out_c_stride = out.stride(1);
    params.out_l_stride = out.stride(-1);
}


at::Tensor
quant_causal_conv1d_fwd(
    const at::Tensor &x, const float &scale_x,
    const at::Tensor &weight, const float &scale_w,
    const float &scale_out, const float &scale_b, 
    const c10::optional<at::Tensor> &bias_,
    const c10::optional<at::Tensor> &seq_idx_,
    const c10::optional<at::Tensor> &initial_states_,
    c10::optional<at::Tensor> &final_states_out_,
    bool silu_activation
) {
    auto input_type = x.scalar_type();
    auto weight_type = weight.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Char);
    TORCH_CHECK(weight_type == at::ScalarType::Char);

    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(weight.is_cuda());

    const auto sizes = x.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];
    const int width = weight.size(-1);

    CHECK_SHAPE(x, batch_size, dim, seqlen);
    CHECK_SHAPE(weight, dim, width);

    TORCH_CHECK(x.stride(2) == 1 || x.stride(1) == 1);
    const bool is_channel_last = x.stride(1) == 1 && x.stride(2) > 1;

    if (is_channel_last) {
        TORCH_CHECK(dim % 8 == 0, "causal_conv1d only supports channel dimension divisible by 8 for now");
        TORCH_CHECK(x.stride(2) % 8 == 0 and x.stride(0) % 8 == 0, "causal_conv1d with channel last layout requires strides (x.stride(0) and x.stride(2)) to be multiples of 8");
    }
    TORCH_CHECK(width >= 2 && width <= 4, "causal_conv1d only supports width between 2 and 4");

    if (bias_.has_value()) {
        auto bias = bias_.value();
        TORCH_CHECK(bias.scalar_type() == weight_type);
        TORCH_CHECK(bias.is_cuda());
        TORCH_CHECK(bias.stride(-1) == 1);
        CHECK_SHAPE(bias, dim);
    }

    if (seq_idx_.has_value()) {
        throw std::logic_error("No implementation for channel last");
    }

    at::Tensor out = torch::empty_like(x);

    QuantConvParamsBase params;
    set_conv_params_fwd(params, batch_size, dim, seqlen, width, x, scale_x, weight, scale_w, out, scale_out, scale_b,
                        bias_.has_value() ? bias_.value().data_ptr() : nullptr,
                        silu_activation);

    if (seq_idx_.has_value()) {
        throw std::logic_error("No implementation for seq_idx");
    } else {
        params.seq_idx_ptr = nullptr;
    }

    if (initial_states_.has_value()) {
        throw std::logic_error("No implementation for initial_states");
    } else {
        params.initial_states_ptr = nullptr;
    }

    if (final_states_out_.has_value()) {
        throw std::logic_error("No implementation for final_states_out");
    } else {
        params.final_states_ptr = nullptr;
    }

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_INTEGRAL(x.scalar_type(), "quant_causal_conv1d_fwd", [&] {
        DISPATCH_WTYPE_INTEGRAL(weight.scalar_type(), "quant_causal_conv1d_fwd", [&] {
            if (!is_channel_last) {
                quant_causal_conv1d_fwd_cuda<input_t, weight_t>(params, stream);
            } else {
                quant_causal_conv1d_channellast_fwd_cuda<input_t, weight_t>(params, stream);
            }
        });
    });
    return out;
}


// Returns {out, out2, scales}: out at scale_out (the x_proj input), out2 for the selective scan at
// scale_out2 (group_size == 0, scales is empty) or at per-token scales over groups of group_size
// channels (32, 64 or 128), scales of shape (batch, dim / group_size, seqlen).
std::vector<at::Tensor>
quant_causal_conv1d_dual_fwd(
    const at::Tensor &x, const float &scale_x,
    const at::Tensor &weight, const float &scale_w,
    const float &scale_out, const float &scale_b,
    const c10::optional<at::Tensor> &bias_,
    const float &scale_out2, const int64_t &group_size,
    bool silu_activation
) {
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Char);
    TORCH_CHECK(weight.scalar_type() == at::ScalarType::Char);
    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(weight.is_cuda());

    const auto sizes = x.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];
    const int width = weight.size(-1);

    CHECK_SHAPE(x, batch_size, dim, seqlen);
    CHECK_SHAPE(weight, dim, width);
    TORCH_CHECK(x.stride(2) == 1, "quant_causal_conv1d_dual_fwd only supports the sequence-last layout");
    TORCH_CHECK(width >= 2 && width <= 4, "causal_conv1d only supports width between 2 and 4");
    TORCH_CHECK(group_size == 0 || group_size == 32 || group_size == 64 || group_size == 128,
                "group_size must be 0 (static second scale), 32, 64 or 128");
    const int tile = group_size == 0 ? 64 : group_size;
    TORCH_CHECK(dim % tile == 0, "dim must be divisible by ", tile);
    if (bias_.has_value()) {
        auto bias = bias_.value();
        TORCH_CHECK(bias.scalar_type() == weight.scalar_type());
        TORCH_CHECK(bias.is_cuda());
        TORCH_CHECK(bias.stride(-1) == 1);
        CHECK_SHAPE(bias, dim);
    }

    at::Tensor out = torch::empty({batch_size, dim, seqlen}, x.options());
    at::Tensor out2 = torch::empty({batch_size, dim, seqlen}, x.options());
    at::Tensor scales = group_size == 0 ? torch::empty({0}, x.options().dtype(at::ScalarType::Float))
        : torch::empty({batch_size, dim / group_size, seqlen}, x.options().dtype(at::ScalarType::Float));

    QuantConvDualParams params;
    set_conv_params_fwd(params, batch_size, dim, seqlen, width, x, scale_x, weight, scale_w, out, scale_out, scale_b,
                        bias_.has_value() ? bias_.value().data_ptr() : nullptr,
                        silu_activation);
    params.out2_ptr = out2.data_ptr();
    params.out2_batch_stride = out2.stride(0);
    params.out2_c_stride = out2.stride(1);
    params.out2_l_stride = out2.stride(2);
    params.scale_out2 = scale_out2;
    params.group_size = group_size;
    params.scales_ptr = group_size == 0 ? nullptr : scales.data_ptr();
    params.scales_batch_stride = group_size == 0 ? 0 : scales.stride(0);
    params.scales_g_stride = group_size == 0 ? 0 : scales.stride(1);

    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_INTEGRAL(x.scalar_type(), "quant_causal_conv1d_dual_fwd", [&] {
        DISPATCH_WTYPE_INTEGRAL(weight.scalar_type(), "quant_causal_conv1d_dual_fwd", [&] {
            quant_causal_conv1d_dual_fwd_cuda<input_t, weight_t>(params, stream);
        });
    });
    return {out, out2, scales};
}


at::Tensor
quant_causal_conv1d_update(const at::Tensor &x,
                     const at::Tensor &conv_state, const float &scale_x,
                     const at::Tensor &weight, const float &scale_w,
                     const float &scale_out, const float &scale_b,
                     const c10::optional<at::Tensor> &bias_, 
                     bool silu_activation) {
    auto input_type = x.scalar_type();
    auto weight_type = weight.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Char);
    TORCH_CHECK(weight_type == at::ScalarType::Char);
    TORCH_CHECK(conv_state.scalar_type() == input_type);

    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(conv_state.is_cuda());
    TORCH_CHECK(weight.is_cuda());

    const auto sizes = x.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int width = weight.size(-1);

    CHECK_SHAPE(x, batch_size, dim);
    CHECK_SHAPE(conv_state, batch_size, dim, width);
    CHECK_SHAPE(weight, dim, width);

    TORCH_CHECK(width >= 2 && width <= 4, "causal_conv1d only supports width between 2 and 4");

    if (bias_.has_value()) {
        auto bias = bias_.value();
        TORCH_CHECK(bias.scalar_type() == weight_type);
        TORCH_CHECK(bias.is_cuda());
        TORCH_CHECK(bias.stride(-1) == 1);
        CHECK_SHAPE(bias, dim);
    }

    at::Tensor out = torch::empty_like(x);

    QuantConvParamsBase params;
    set_conv_params_fwd(params, batch_size, dim, /*seqlen=*/1, width, x, scale_x, weight, scale_w, out, scale_out, scale_b,
                        bias_.has_value() ? bias_.value().data_ptr() : nullptr,
                        silu_activation);

    params.conv_state_ptr = conv_state.data_ptr();
    // All stride are in elements, not bytes.
    params.conv_state_batch_stride = conv_state.stride(0);
    params.conv_state_c_stride = conv_state.stride(1);
    params.conv_state_l_stride = conv_state.stride(2);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_INTEGRAL(x.scalar_type(), "quant_causal_conv1d_update_cuda", [&] {
        DISPATCH_WTYPE_INTEGRAL(weight.scalar_type(), "quant_causal_conv1d_update_cuda", [&] {
            quant_causal_conv1d_update_cuda<input_t, weight_t>(params, stream);
        });
    });
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &quant_causal_conv1d_fwd, "Quantized causal conv1d forward");
    m.def("update", &quant_causal_conv1d_update, "Quantized causal conv1d update");
    m.def("fwd_dual", &quant_causal_conv1d_dual_fwd, "Quantized causal conv1d forward, u written once per consumer");
}