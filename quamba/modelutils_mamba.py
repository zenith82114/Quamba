import os
import gc
import copy
import logging
from tqdm import tqdm
from functools import partial
import json

import torch
import torch.nn as nn
from datasets import load_dataset

from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.utils.generation import InferenceParams
from mamba_ssm.ops.triton.layer_norm import layer_norm_fn, RMSNorm

from quamba.quamba_mixer_seq import QuambaLMHeadModel
from .qEmbedding import W4O16Embedding, W8O16Embedding
from .qLinearLayer import HadLinear, W4A16B16O16Linear, W4A8B16O16Linear, W8A8B16O16Linear
from .qActLayer import ActIdentity
from .qMamba2 import Mamba2Simple, W4A8QMamba2, W4A16QMamba2, W8A8QMamba2
from .qMambaLayer import MambaSimple, W4A8QMamba, W4A16QMamba, W8A8QMamba, FakeQMamba
from .fake_quant import FakeQuantAct, set_fake_quant_state
from .quant_utils import fake_quant_weight
from .qHadamard import Hadamard
from .qBlock import HybridQuambaBlock
# from .fusedNorm import FusedRMSNorm
from .qNorm import QRMSNorm
from .observer import PerTensorMinmaxObserver, PerTensorPercentileObserver
from .observer import PerSSDGroupObserver, CrossHeadMinmaxObserver
from .observer import CachedStatesCrossHeadMinmaxObserver
from .gptq_utils import GPTQ
from .reorder_utils import get_reorder_params, reorder_mamba
from .hadamard_utils import had_transform
from .data_loaders import get_loaders

logger = logging.getLogger(__name__)

@torch.no_grad()
def fuse_ln_linear(norm, linear) -> None:
    """
    fuse the layernorm weight to the adjacent linear layer.
    """
    linear_dtype = linear.weight.dtype

    # Calculating new weight and bias
    W_ = linear.weight.data.double()
    linear.weight.data = (W_ * norm.weight.double()).to(linear_dtype)
    if hasattr(norm, 'bias') and norm.bias is not None:
        if linear.bias is None:
            linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float32))
        linear.bias.data = linear.bias.data.double() + torch.matmul(W_, norm.bias.to(torch.float32))
        linear.bias.data = linear.bias.data.to(linear_dtype)
    # Reset the learnable weight in RMSNorm to 1
    norm.weight.data = torch.ones_like(norm.weight).to(norm.weight.dtype) # Reset the weight to 1

@torch.no_grad()
def configure_model(model, model_type, use_had_transform=True):
    device = next(model.parameters()).device
    if model_type == "mamba":
        # process embedding and lm_head
        if use_had_transform:
            # Sometimes, lm_head is tied to embedding, we make a clone for lm_head first
            lm_head_clone = model.lm_head.weight.data.clone()
            # transform embedding first
            model.backbone.embedding.weight.data = had_transform(model.backbone.embedding.weight.data)
            # do layernorm fusion to lm_head and then transform
            model.lm_head.weight = torch.nn.Parameter(lm_head_clone * model.backbone.norm_f.weight.view(1, -1)).to("cuda") # must re-initialize it with nn.Parameter to untie lm_head and embedding, otherwise, it will not work
            model.backbone.norm_f.weight.data = torch.ones_like(model.backbone.norm_f.weight)
            model.lm_head.weight.data = had_transform(model.lm_head.weight.data)
            torch.cuda.empty_cache()
        layers = model.backbone.layers
        for i in range(len(layers)):
            if isinstance(layers[i], Block):
                # fuse norm to in_proj first
                fuse_ln_linear(layers[i].norm, layers[i].mixer.in_proj)
                # use simplied mamba block to get the scaling factors
                # from linear layers without pain
                m = MambaSimple(originalLayer=layers[i].mixer, use_had_transform=use_had_transform).to(device)
                layers[i].mixer = m
                torch.cuda.empty_cache()
    elif model_type == "mamba2":
        # process embedding and lm_head
        if use_had_transform:
            # Sometimes, lm_head is tied to embedding, we make a clone for lm_head first
            lm_head_clone = model.lm_head.weight.data.clone()
            # transform embedding first
            model.backbone.embedding.weight.data = had_transform(model.backbone.embedding.weight.data)
            # do layernorm fusion to lm_head and then transform
            model.lm_head.weight = torch.nn.Parameter(lm_head_clone * model.backbone.norm_f.weight.view(1, -1)).to("cuda") # must re-initialize it with nn.Parameter to untie lm_head and embedding, otherwise, it will not work
            model.backbone.norm_f.weight.data = torch.ones_like(model.backbone.norm_f.weight)
            model.lm_head.weight.data = had_transform(model.lm_head.weight.data)
            torch.cuda.empty_cache()
        # process layers
        layers = model.backbone.layers
        for i in range(len(layers)):
            if isinstance(layers[i], Block):
                # fuse norm to in_proj first
                fuse_ln_linear(layers[i].norm, layers[i].mixer.in_proj)
                # use simplied mamba block to get the scaling factors
                # from linear layers without pain
                m = Mamba2Simple(originalLayer=layers[i].mixer, use_had_transform=use_had_transform).to(device)
                layers[i].mixer = m
                torch.cuda.empty_cache()
    else:
        raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba' and 'mamba2'")
    model.config.use_cache = False
    model.eval()
    return model

@torch.no_grad()
def run_calibration_forward(model, tokenizer, num_samples=512, seq_len=512,
                            calibration_dataset=None, preprocess_fn=None):
    device = next(model.parameters()).device
    if calibration_dataset is None:
        logger.info("Calibrate with monology/pile-uncopyrighted")
        calibration_dataset = load_dataset("monology/pile-uncopyrighted", data_files="val.jsonl.zst", split="train")

        def preprocess(data, tokenizer, max_tokens, device):
            input_ids = tokenizer(data["text"], return_tensors="pt",
                    max_length=max_tokens, truncation=True).input_ids.to(device)
            return input_ids
        preprocess_fn = partial(preprocess, tokenizer=tokenizer, max_tokens=seq_len, device=device)

    logger.info("Run calibration")
    for i in tqdm(range(num_samples)):
        input_ids = preprocess_fn(calibration_dataset[i])
        # prepare inference cache for getting ssm_state scales
        prompt_len = input_ids.size(1)
        inf_cache = model.allocate_inference_cache(1, prompt_len)
        lengths_per_sample = torch.full((1,), prompt_len, dtype=torch.int32, device=device)
        inference_params = InferenceParams(
            max_seqlen=prompt_len,
            max_batch_size=1,
            seqlen_offset=0,
            key_value_memory_dict=inf_cache,
            lengths_per_sample=lengths_per_sample,
        )
        # do not set num_last_tokens because we want all activations to lm_head
        model(input_ids, inference_params=inference_params)
        # clean up the cache
        del inf_cache

@torch.no_grad()
def run_quamba_calibration(
        model, model_type, tokenizer, num_samples=512, seq_len=512,
        calibration_dataset=None, preprocess_fn=None
    ):

    if model_type == "mamba":
        layers = model.backbone.layers
        is_traget_block = lambda block: isinstance(block, Block)
        get_mamba = lambda block: block.mixer
        is_calib_ops = lambda op: isinstance(op, (torch.nn.Linear, ActIdentity))
        is_x = lambda op: op == "x_proj"
        is_ssm_state = lambda op: op == "ssm_state_act"
        percentile_alpha=0.9995 # for smaller model like 130m, use 0.99999
    elif model_type == "mamba2":
        layers = model.backbone.layers
        is_traget_block = lambda block: isinstance(block, Block)
        get_mamba = lambda block: block.mixer
        is_calib_ops = lambda op: isinstance(op, (torch.nn.Linear, ActIdentity))
        is_x = lambda op: op == "x_conv_out"
        is_ssm_state = lambda op: op == "ssm_state_act"
        percentile_alpha=0.9995  # for smaller model like 130m, use 0.99999
    else:
        raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba' and 'mamba2'")

    # register min/max observers, num_layer + lm_head
    observers = [{} for _ in range(len(layers) + 1)]

    def stat_hook(m, inputs, outputs, op, block_idx):
        # register the new information to observer
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        observers[block_idx][op + ":input"].update(inputs.clone().detach())

        if isinstance(outputs, tuple):
            outputs = outputs[0]
        observers[block_idx][op + ":output"].update(outputs.clone().detach())

    hooks = []
    for i in range(len(layers)):
        if not is_traget_block(layers[i]):
            continue

        mixer = get_mamba(layers[i])
        for name, m in mixer.named_modules():
            if is_calib_ops(m):
                # FIXME(HY): hardcode everything for now
                a_bits = 8
                a_clip_ratio = 1.0
                op = name.split(".")[0]
                if is_x(op) or is_ssm_state(op):
                    observers[i][op + ":input"] = PerTensorPercentileObserver(
                        n_bits=a_bits,
                        clip_ratio=a_clip_ratio,
                        sym=True,
                        percentile_alpha=percentile_alpha
                    )
                else:
                    observers[i][op + ":input"] = PerTensorMinmaxObserver(
                        n_bits=a_bits,
                        clip_ratio=a_clip_ratio,
                        sym=True
                    )
                observers[i][op + ":output"] = PerTensorMinmaxObserver(
                    n_bits=a_bits,
                    clip_ratio=a_clip_ratio,
                    sym=True
                )
                hooks.append(
                    m.register_forward_hook(partial(stat_hook, op=op, block_idx=i))
                )
    # add observer hook for lm_head
    observers[-1]["lm_head:input"] = PerTensorMinmaxObserver(
        n_bits=a_bits, clip_ratio=a_clip_ratio, sym=True)
    observers[-1]["lm_head:output"] = PerTensorMinmaxObserver(
        n_bits=a_bits, clip_ratio=a_clip_ratio, sym=True)
    hooks.append(
        model.lm_head.register_forward_hook(partial(stat_hook, op="lm_head", block_idx=-1))
    )

    run_calibration_forward(model, tokenizer, num_samples, seq_len, calibration_dataset, preprocess_fn)

    for h in hooks:
        h.remove()

    # collect in/output scaling factors for layers, num_layer + lm_head
    act_scales = [{} for _ in range(len(layers) + 1)]
    for i in range(len(layers) + 1):
        for name, observer in observers[i].items():
            scale, base = observer.get_quantization_parameters()
            # FIXME (HY): hardcode to not use base now
            act_scales[i][name] = scale.to(torch.float32)
    del observers
    return act_scales

@torch.no_grad()
def run_quamba2_calibration(
        model, model_type, tokenizer, reorder_params,
        num_samples=512, seq_len=512, calibration_dataset=None, preprocess_fn=None
    ):

    if model_type == "mamba":
        raise NotImplementedError("Not support for mamba")
    elif model_type == "mamba2":
        layers = model.backbone.layers
        is_traget_block = lambda block: isinstance(block, Block)
        get_mamba = lambda block: block.mixer
        is_x = lambda op: op == "x_conv_out"
        is_BC = lambda op: op == "B_conv_out" or op == "C_conv_out"
        is_ssm_state = lambda op: op == "ssm_state_act"
        is_calib_ops = lambda op: isinstance(op, (torch.nn.Linear, ActIdentity))
        percentile_alpha=0.99999
    else:
        raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba2'")

    # register min/max observers, num_layer + lm_head
    observers = [{} for _ in range(len(layers) + 1)]

    def stat_hook(m, inputs, outputs, op, block_idx):
        # register the new information to observer
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        observers[block_idx][op + ":input"].update(inputs.clone().detach())

        if isinstance(outputs, tuple):
            outputs = outputs[0]
        observers[block_idx][op + ":output"].update(outputs.clone().detach())

    hooks = []
    for i in range(len(layers)):
        if not is_traget_block(layers[i]):
            continue
        head_groups = reorder_params["head_groups"][i]
        channel_group = reorder_params["channel_group"][i]
        mixer = get_mamba(layers[i])
        for name, m in mixer.named_modules():
            if is_calib_ops(m):
                # FIXME(HY): hardcode everything for now
                a_bits = 8
                a_clip_ratio = 1.0
                op = name.split(".")[0]
                if is_x(op):
                    observers[i][op + ":input"] = CrossHeadMinmaxObserver(
                        n_bits=a_bits,
                        clip_ratio=a_clip_ratio,
                        sym=True,
                        ngroups=mixer.ngroups,
                        headdim=mixer.headdim,
                        head_groups=head_groups,
                        channel_group=channel_group,
                    )
                elif is_BC(op):
                    observers[i][op + ":input"] = PerSSDGroupObserver(
                        n_bits=a_bits,
                        clip_ratio=a_clip_ratio,
                        sym=True,
                        dstate=mixer.d_state,
                    )
                elif is_ssm_state(op):
                    observers[i][op + ":input"] = CachedStatesCrossHeadMinmaxObserver(
                        n_bits=a_bits,
                        clip_ratio=a_clip_ratio,
                        sym=True,
                        ngroups=mixer.ngroups,
                        headdim=mixer.headdim,
                        dstate=mixer.d_state,
                        head_groups=head_groups,
                        channel_group=channel_group,
                    )
                else:
                    observers[i][op + ":input"] = PerTensorMinmaxObserver(
                        n_bits=a_bits,
                        clip_ratio=a_clip_ratio,
                        sym=True
                    )
                observers[i][op + ":output"] = PerTensorMinmaxObserver(
                    n_bits=a_bits,
                    clip_ratio=a_clip_ratio,
                    sym=True
                )
                hooks.append(
                    m.register_forward_hook(partial(stat_hook, op=op, block_idx=i))
                )
    # add observer hook for lm_head
    observers[-1]["lm_head:input"] = PerTensorMinmaxObserver(
        n_bits=a_bits, clip_ratio=a_clip_ratio, sym=True)
    observers[-1]["lm_head:output"] = PerTensorMinmaxObserver(
        n_bits=a_bits, clip_ratio=a_clip_ratio, sym=True)
    hooks.append(
        model.lm_head.register_forward_hook(partial(stat_hook, op="lm_head", block_idx=-1))
    )
    if calibration_dataset is None:
        logger.info("Calibrate with monology/pile-uncopyrighted")
        calibration_dataset = load_dataset("monology/pile-uncopyrighted", data_files="val.jsonl.zst", split="train")
        calibration_dataset.shuffle(seed=42)

        device = next(model.parameters()).device
        def preprocess(data, tokenizer, max_tokens, device):
            input_ids = tokenizer(data["text"], return_tensors="pt",
                    max_length=max_tokens, truncation=True).input_ids.to(device)
            return input_ids
        preprocess_fn = partial(preprocess, tokenizer=tokenizer, max_tokens=seq_len, device=device)

    logger.info("Run calibration")
    for i in tqdm(range(num_samples)):
        input_ids = preprocess_fn(calibration_dataset[i])
        # prepare inference cache for getting ssm_state scales
        prompt_len = input_ids.size(1)
        inf_cache = model.allocate_inference_cache(1, prompt_len)
        lengths_per_sample = torch.full((1,), prompt_len, dtype=torch.int32, device=device)
        inference_params = InferenceParams(
            max_seqlen=prompt_len,
            max_batch_size=1,
            seqlen_offset=0,
            key_value_memory_dict=inf_cache,
            lengths_per_sample=lengths_per_sample,
        )
        # do not set num_last_tokens because we want all activations to lm_head
        model(input_ids, inference_params=inference_params)
        # clean up the cache
        del inf_cache

    for h in hooks:
        h.remove()

    # collect in/output scaling factors for layers, num_layer + lm_head
    act_scales = [{} for _ in range(len(layers) + 1)]
    for i in range(len(layers) + 1):
        for name, observer in observers[i].items():
            scale, base = observer.get_quantization_parameters()
            # FIXME (HY): hardcode to not use base now
            act_scales[i][name] = scale

    return act_scales

@torch.no_grad()
def fuse_had_matrices(model):
    # fuse the had matrices with the weight matrices in linear layers.
    # Do this after reordering and before applying gptq
    layers = model.backbone.layers
    for i in range(len(layers)):
        # in_proj: fuse had matrices with weight matrices
        if isinstance(layers[i].mixer.in_proj, HadLinear):
            layers[i].mixer.in_proj.fuse_hadamard()
        # out_proj: fuse had matrices with weight matrices
        if isinstance(layers[i].mixer.out_proj, HadLinear):
            layers[i].mixer.out_proj.fuse_hadamard()
    return model

@torch.no_grad()
def apply_gptq(model, tokenizer, device, w_bits=4, group_size=128, quantize_lm_head=False, lm_head_bits=4):
    # Hardcode gptq hyper-parameters for now
    nsamples = 128
    seqlen = 1024
    bits = w_bits
    assert bits in [4, 8], "Only support 4 or 8 bits weights for now"
    logging.info("Start Quantized Linear Layers with GPTQ")
    logging.info("* Number of samples: %d" % nsamples)
    logging.info("* Sequence length: %d" % seqlen)
    logging.info("* Target bit-width for weights: %d" % bits)
    logging.info("* Group size: %d" % group_size)
    logging.info("Build calibration loader for GPTQ")
    #build dataloader
    dataloader, _ = get_loaders("wikitext2", tokenizer, nsamples=nsamples, seqlen=seqlen)
    layers = model.backbone.layers
    model.backbone.embedding = model.backbone.embedding.to(device)
    layers[0] = layers[0].to(device)
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, seqlen, model.config.d_model), dtype=dtype, device=device
    )
    residual = torch.zeros(
        (nsamples, seqlen, model.config.d_model), dtype=dtype, device=device
    )

    cache = {"i": 0}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, res = None, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass

    # the hook to collect inputs for in_proj, out_proj, and lm_head
    def add_batch(module, inp, out, gptq):
        gptq.add_batch(inp[0].data, out.data)

    layers[0] = layers[0].module # remove Catcher
    layers[0] = layers[0].cpu()
    model.backbone.embedding = model.backbone.embedding.cpu()
    torch.cuda.empty_cache()
    for i in tqdm(range(len(layers))):
        # get layer
        layer = layers[i].to(device)

        # create GPTQ objects for in_proj and out_proj
        gptq = {
            "in_proj": GPTQ(layer.mixer.in_proj),
            "out_proj": GPTQ(layer.mixer.out_proj),
        }
        handles = [
            layer.mixer.in_proj.register_forward_hook(partial(add_batch, gptq=gptq["in_proj"])),
            layer.mixer.out_proj.register_forward_hook(partial(add_batch, gptq=gptq["out_proj"]))
        ]
        for j in range(nsamples):
            layer(
                inps[j].unsqueeze(0),
                residual=residual[j].unsqueeze(0)
            )
        for h in handles:
            h.remove()

        # start running GPTQ
        for name in gptq.keys():
            logging.debug(f"Performing GPTQ on layer.{i}.mixer.{name} with {bits} bits")
            gptq[name].fasterquant(
                percdamp=0.01, group_size=group_size, w_bits=bits
            )
            gptq[name].free()
        del gptq

        # collect the outputs for the next layer
        for j in range(nsamples):
            inps[j], residual[j] = layer(inps[j].unsqueeze(0), residual=residual[j].unsqueeze(0))

        # garbage collection and clean cache
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        gc.collect()

    if not quantize_lm_head:
        model = model.to(device)
        return model

    model = model.to("cpu") # move model to cpu to save memory
    model.lm_head = model.lm_head.to(device)
    model.backbone.norm_f = model.backbone.norm_f.to(device)
    logging.info("Quantizing lm_head with GPTQ")
    gptq_lm_head = GPTQ(model.lm_head)
    handle = model.lm_head.register_forward_hook(partial(add_batch, gptq=gptq_lm_head))

    assert model.backbone.fused_add_norm, "Only support fused_add_norm=True for now"
    #Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L202
    final_hidden_states = layer_norm_fn(
        x=inps,
        weight=model.backbone.norm_f.weight,
        bias=model.backbone.norm_f.bias,
        eps=model.backbone.norm_f.eps,
        residual=residual,
        prenorm=False,
        residual_in_fp32=model.backbone.residual_in_fp32,
        is_rms_norm=isinstance(model.backbone.norm_f, RMSNorm),
    )

    for j in range(nsamples):
        model.lm_head(final_hidden_states[j].unsqueeze(0))

    handle.remove()
    # compute with fp16 to save memory
    gptq_lm_head.fasterquant(
        percdamp=0.01, group_size=group_size, w_bits=lm_head_bits, dtype=torch.float16
    )
    gptq_lm_head.free()
    del gptq_lm_head

    torch.cuda.empty_cache()
    gc.collect()

    model = model.to(device)
    return model


def quantize_norm_a8(block_type, norm, layer_idx, act_scales, device):
    norm = QRMSNorm.from_fp16(
        originalLayer=norm,
        output_scale=act_scales[layer_idx]["in_proj:input"].item())
    return norm.to(device)

def quantize_mixer_w8a8(block_type, mixer, layer_idx, act_scales, device):
    W8A8Mixers = {
        "Mamba": W8A8QMamba,
        "Mamba2": W8A8QMamba2,
    }
    if block_type not in W8A8Mixers.keys():
        raise ValueError(f"Not find {block_type} in W8A8 Mixer")
    if W8A8Mixers[block_type] is None:
        raise ValueError(f"Not support {block_type} with W8A8")
    mixer = W8A8Mixers[block_type].from_fp16(
                originalLayer=mixer,
                act_scales=act_scales[layer_idx],
                use_had_transform=True)
    return mixer.to(device)

def quantize_mixer_w4a16(block_type, mixer, layer_idx, act_scales, device):
    W4A16Mixers = {
        "Mamba": W4A16QMamba,
        "Mamba2": W4A16QMamba2,
    }
    if block_type not in W4A16Mixers.keys():
        raise ValueError(f"Not find {block_type} in W4A16 Mixer")
    if W4A16Mixers[block_type] is None:
        raise ValueError(f"Not support {block_type} with W4A16")
    mixer = W4A16Mixers[block_type].from_fp16(originalLayer=mixer, use_had_transform=True)
    return mixer.to(device)

def quantize_mixer_w4a8(block_type, mixer, layer_idx, act_scales, device):
    W4A8Mixers = {
        "Mamba": W4A8QMamba,
        "Mamba2": W4A8QMamba2,
    }
    if block_type not in W4A8Mixers.keys():
        raise ValueError(f"Not find {block_type} in W4A8 Mixer")
    if W4A8Mixers[block_type] is None:
        raise ValueError(f"Not support {block_type} with W4A8")
    mixer = W4A8Mixers[block_type].from_fp16(
                originalLayer=mixer,
                act_scales=act_scales[layer_idx],
                use_had_transform=True)
    return mixer.to(device)

def get_quantize_block_fn(act_scales, w_bits, a_bits, device):
    if w_bits == 4 and a_bits == 8:
        quantize_norm_fn = partial(quantize_norm_a8, act_scales=act_scales, device=device)
        quantize_mixer_fn = partial(quantize_mixer_w4a8, act_scales=act_scales, device=device)
    elif w_bits == 4 and a_bits == 16:
        quantize_norm_fn = lambda block_type, norm, layer_idx: norm # just return the original layer
        quantize_mixer_fn = partial(quantize_mixer_w4a16, act_scales=act_scales, device=device)
    elif w_bits == 8 and a_bits == 8:
        quantize_norm_fn = partial(quantize_norm_a8, act_scales=act_scales, device=device)
        quantize_mixer_fn = partial(quantize_mixer_w8a8, act_scales=act_scales, device=device)
    else:
        raise ValueError(f"Unsupport w{w_bits}a{a_bits}, only w8a8, w4a8, and w4a16 are supported")
    return quantize_norm_fn, quantize_mixer_fn

@torch.no_grad()
def quantize_fp16_model(model, model_type, act_scales, device, w_bits=4, a_bits=8, quantize_embedding=True, quantize_lm_head=True):
    assert w_bits in [4, 8], "Only support 4 or 8 bits weights for now"
    assert a_bits in [8, 16], "Only support 8 or 16 bits activations for now"
    quantize_norm_fn, quantize_mixer_fn = get_quantize_block_fn(act_scales, w_bits, a_bits, device)

    model.config.use_cache = False
    if model_type == "mamba":
        if quantize_embedding:
            # replace embedding layer
            logging.info(f'Applying quantized embedding')
            if w_bits == 4:
                model.backbone.embedding = W4O16Embedding.from_fp16(model.backbone.embedding)
            elif w_bits == 8:
                model.backbone.embedding = W8O16Embedding.from_fp16(model.backbone.embedding)
            else:
                raise ValueError(f"Unsupport w{w_bits}a{a_bits}, only w8a8, w4a8, and w4a16 are supported")
            gc.collect()
            torch.cuda.empty_cache()
        # replace layers
        logging.info(f'Applying quantized layers')
        layers = model.backbone.layers
        for i in tqdm(range(len(layers))):
            if isinstance(layers[i], Block):
                # replace with fused RMSNorm
                layers[i].fused_add_norm = True
                layers[i].norm = quantize_norm_fn(
                    block_type="Mamba",
                    norm=layers[i].norm,
                    layer_idx=i)
                layers[i].mixer = quantize_mixer_fn(
                    block_type="Mamba",
                    mixer=layers[i].mixer,
                    layer_idx=i)
                # garbage collection and clean cache
                gc.collect()
                torch.cuda.empty_cache()
    elif model_type == "mamba2":
        if quantize_embedding:
            # replace embedding layer
            logging.info(f'Applying quantized embedding')
            if w_bits == 4:
                model.backbone.embedding = W4O16Embedding.from_fp16(model.backbone.embedding)
            elif w_bits == 8:
                model.backbone.embedding = W8O16Embedding.from_fp16(model.backbone.embedding)
            else:
                raise ValueError(f"Unsupport w{w_bits}a{a_bits}, only w8a8, w4a8, and w4a16 are supported")
            gc.collect()
            torch.cuda.empty_cache()
        # replace layers
        logging.info(f'Applying quantized layers')
        layers = model.backbone.layers
        for i in tqdm(range(len(layers))):
            if isinstance(layers[i], Block):
                layers[i].fused_add_norm = True
                layers[i].norm = quantize_norm_fn(
                    block_type="Mamba2",
                    norm=layers[i].norm,
                    layer_idx=i)
                layers[i].mixer = quantize_mixer_fn(
                    block_type="Mamba2",
                    mixer=layers[i].mixer,
                    layer_idx=i)
                # garbage collection and clean cache
                gc.collect()
                torch.cuda.empty_cache()
    else:
        raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba' and 'mamba2'")

    if quantize_lm_head:
        logging.info(f'Applying quantized lm_head')
        # replace lm_head and norm_f with quantized version
        if w_bits == 4 and a_bits == 16:
            # do nothing to w4a16 norm_f
            model.lm_head = W4A16B16O16Linear.from_fp16(model.lm_head)
        elif w_bits == 4 and a_bits == 8:
            # model.backbone.norm_f = FusedRMSNorm.from_fp16(
            model.backbone.norm_f = QRMSNorm.from_fp16(
                model.backbone.norm_f, output_scale=act_scales[-1]["lm_head:input"].item())
            model.lm_head = W4A8B16O16Linear.from_fp16(model.lm_head, act_scales[-1]["lm_head:input"])
        elif w_bits == 8 and a_bits == 8:
            # model.backbone.norm_f = FusedRMSNorm.from_fp16(
            model.backbone.norm_f = QRMSNorm.from_fp16(
                model.backbone.norm_f, output_scale=act_scales[-1]["lm_head:input"].item())
            model.lm_head = W8A8B16O16Linear.from_fp16(model.lm_head, act_scales[-1]["lm_head:input"].item())
        else:
            raise ValueError(f"Unsupport w{w_bits}a{a_bits}, only w8a8, w4a8, and w4a16 are supported")

    gc.collect()
    torch.cuda.empty_cache()
    return model

def quantize_fp16_model_act_hybrid(model, model_type, act_scales, device, w_bits=4,
                                   layer_wise_hybrid_config=None #this is expected to be a list
                                   ):
    assert w_bits in [4], "Only support 4 bits weights for now"
    a_bits = [8, 16]

    logging.info(f"Quantizing model with w{w_bits} and a{a_bits}. HybridBlock will be create.")

    # for each a_bits get the correcponding quantization function
    quant_function_pairs = {}
    for a in a_bits:
        quant_function_pairs[f'W{w_bits}A{a}'] = get_quantize_block_fn(act_scales, w_bits, a, device) # quantize_norm_fn, quantize_mixer_fn

    model.config.use_cache = False
    if model_type == "mamba2":
        # replace embedding layer
        logging.info(f'Applying quantized embedding')
        if w_bits == 4:
            model.backbone.embedding = W4O16Embedding.from_fp16(model.backbone.embedding)
        elif w_bits == 8:
            model.backbone.embedding = W8O16Embedding.from_fp16(model.backbone.embedding)
        else:
            raise ValueError(f"Unsupport w{w_bits}a{a_bits}, only w8a8, w4a8, and w4a16 are supported")
        # replace layers
        logging.info(f'Applying quantized layers')
        layers = model.backbone.layers
        for i in tqdm(range(len(layers))):
            if isinstance(layers[i], Block):
                layers[i].fused_add_norm = True

                mixers = {}
                norms = {}

                if layer_wise_hybrid_config:
                    #Case 1: bitwidth of each layer is specified, each layers only create the block/norm with the specified bitwidth
                    bit_config = layer_wise_hybrid_config[i]
                    (quantize_norm_fn, quantize_mixer_fn) = quant_function_pairs[bit_config]
                    mixers[bit_config] = quantize_mixer_fn(
                        block_type="Mamba2",
                        mixer=layers[i].mixer,
                        layer_idx=i)
                    norms[bit_config] = quantize_norm_fn(
                        block_type="Mamba2",
                        norm=layers[i].norm,
                        layer_idx=i)
                else:
                    #Case 2: bitwidth of each layer is not specified, each layers create the block/norm with all possible bitwidth
                    for bits, (quantize_norm_fn, quantize_mixer_fn) in quant_function_pairs.items():
                        mixers[f"W{w_bits}A{bits}"] = quantize_mixer_fn(
                            block_type="Mamba2",
                            mixer=layers[i].mixer,
                            layer_idx=i)
                        norms[f"W{w_bits}A{bits}"] = quantize_norm_fn(
                            block_type="Mamba2",
                            norm=layers[i].norm,
                            layer_idx=i)

                layers[i] = HybridQuambaBlock.from_block_and_mixer_norm_dict(
                    block=layers[i],
                    mixers_dict=mixers,
                    norms_dict=norms,
                ).to(device)
                layers[i].set_mixer(next(iter(mixers))) #default
                gc.collect()
                torch.cuda.empty_cache()
    else:
        raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba2' for hybrid mode")

    logging.info(f'Applying quantized lm_head')

    #FIXME(brian1009): Hard-fix for now, but we may need to make it configurable as well.
    model.backbone.norm_f = QRMSNorm.from_fp16(
        model.backbone.norm_f, output_scale=act_scales[-1]["lm_head:input"].item())
    model.lm_head = W4A8B16O16Linear.from_fp16(model.lm_head, act_scales[-1]["lm_head:input"])

    gc.collect()
    torch.cuda.empty_cache()
    return model


def get_model_size(model, model_name, w_bits, a_bits):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    model_mb = (param_size + buffer_size) / 1024**2
    logging.info(f'W{w_bits}A{a_bits} {model_name} size: {model_mb:.3f} MB')


def resolve_fake_quant_config(args):
    # w_group_size=None follows Quamba: per-tensor W8, group-128 W4
    w_group_size = args.w_group_size
    if w_group_size is None:
        w_group_size = 128 if args.w_bits < 8 else 0
    return {
        "w_bits": args.w_bits,
        "w_group_size": w_group_size,
        # Quamba keeps dt_proj at W8 for W4 models (K=dt_rank is too small for the W4 kernels)
        "dt_proj_bits": max(args.w_bits, 8),
        "dt_proj_group_size": 0 if args.w_group_size is None else w_group_size,
        "conv_group_size": w_group_size if args.fq_conv_group else 0,
        "a_bits": args.a_bits,
        "a_group_size": args.a_group_size,
        "a_group_mode": args.a_group_mode,
    }


@torch.no_grad()
def fake_quantize_model_mamba(model, model_type, tokenizer, device, args, calibration_dataset=None, calib_preprocess_fn=None):
    """
    Simulated quantization: keep the fp16 mixers and quantize-dequantize weights and activations,
    with optional channel groups. Fake-quantized models are never stored to or loaded from --pretrained_dir.
    """
    if model_type != "mamba":
        raise NotImplementedError("--fake_quant only supports Mamba1 for now")
    cfg = resolve_fake_quant_config(args)
    logging.info(f"Fake quantization: {cfg}")
    if args.pretrained_dir:
        logging.warning("--pretrained_dir is ignored with --fake_quant")
    quantize_act = cfg["a_bits"] < 16

    model = configure_model(model, model_type, use_had_transform=True)
    act_cfg = dict(n_bits=cfg["a_bits"], group_size=cfg["a_group_size"], mode=cfg["a_group_mode"])
    layers = model.backbone.layers
    for i in range(len(layers)):
        layers[i].mixer = FakeQMamba(layers[i].mixer, **act_cfg)
    if args.quantize_lm_head:
        # the norm_f output in Quamba, i.e., the lm_head input
        model.lm_head_act = FakeQuantAct("lm_head:input", model.lm_head.in_features, device=device, **act_cfg)
        model.lm_head.register_forward_pre_hook(lambda m, inputs: (model.lm_head_act(inputs[0]),))

    # calibrate static activation scales on fp16 weights, as Quamba does
    if quantize_act and cfg["a_group_mode"] == "static":
        set_fake_quant_state(model, "observe")
        run_calibration_forward(model, tokenizer, args.calib_data_num, args.calib_seqlen,
                                calibration_dataset, calib_preprocess_fn)
        set_fake_quant_state(model, "finalize")

    model = fuse_had_matrices(model)
    if args.apply_gptq:
        if cfg["w_group_size"] == 0:
            raise ValueError("GPTQ needs per-channel or group weight scales, try --w_group_size -1 or --w_group_size 128")
        model = apply_gptq(model, tokenizer, device, w_bits=cfg["w_bits"], group_size=cfg["w_group_size"],
                           quantize_lm_head=args.quantize_lm_head, lm_head_bits=cfg["w_bits"])
    for i in range(len(layers)):
        layers[i].mixer.quantize_weights_(
            w_bits=cfg["w_bits"], w_group_size=cfg["w_group_size"],
            dt_proj_bits=cfg["dt_proj_bits"], dt_proj_group_size=cfg["dt_proj_group_size"],
            conv_group_size=cfg["conv_group_size"])
    if args.quantize_lm_head and not getattr(model.lm_head, "apply_gptq", False):
        model.lm_head.weight.data = fake_quant_weight(model.lm_head.weight.data, cfg["w_bits"], cfg["w_group_size"])
    if args.quantize_embedding:
        # Quamba uses one scale per token embedding (row)
        emb_group_size = cfg["w_group_size"] if cfg["w_group_size"] > 0 else -1
        emb = model.backbone.embedding
        emb.weight.data = fake_quant_weight(emb.weight.data, cfg["w_bits"], emb_group_size)

    set_fake_quant_state(model, "quant" if quantize_act else "off")
    gc.collect()
    torch.cuda.empty_cache()
    return model.to(device)


def get_quantized_model_name(args):
    """
    Directory name of the quantized model under <pretrained_dir>/ut-enyac/.
    The standard recipes keep the names of the released models, quamba[2]-<size>-w{W}a{A} (or -w4aX-<config>):
    GPTQ for sub-8-bit weights, quantized embedding and lm_head, and head grouping for Mamba2 with 8-bit
    activations. Every deviation that changes the stored model adds a suffix, so it never reuses another cache.
    """
    model_name = args.model.lower().split('/')[-1]
    is_mamba2 = model_name.startswith("mamba2")
    model_name = model_name.replace("mamba", "quamba") # replace mamba with quamba
    if args.hybrid_blocks:
        model_name = model_name + f"-w{args.w_bits}aX"
        if args.hybrid_blocks_config:
            config_name = args.hybrid_blocks_config.split("/")[-1].replace(".json", "")
            model_name = model_name + f"-{config_name}"
    else:
        model_name = model_name + f"-w{args.w_bits}a{args.a_bits}"
    if args.w_bits < 8 and not args.apply_gptq:
        model_name += "-nogptq"
    if args.w_bits >= 8 and args.apply_gptq:
        model_name += "-gptq"
    if not args.hybrid_blocks: # hybrid models always quantize the embedding and lm_head
        if not args.quantize_embedding:
            model_name += "-noemb"
        if not args.quantize_lm_head:
            model_name += "-nohead"
    if is_mamba2 and (args.hybrid_blocks or args.a_bits == 8) and not args.group_heads:
        model_name += "-nogh"
    return model_name


def check_quantized_model_config(quantized_model_path, model_type, args):
    """
    Refuse to load a cached model whose layers do not match the requested settings, e.g., one stored
    by an older version whose directory names did not encode --quantize_embedding / --quantize_lm_head.
    (GPTQ and head grouping leave no trace in config.json, so they are only distinguished by the name.)
    """
    if args.hybrid_blocks:
        return
    with open(os.path.join(quantized_model_path, "config.json"), "r") as file:
        config = json.load(file)
    w, a = args.w_bits, args.a_bits
    expected = {
        "mixer": f"W{w}A{a}QMamba" + ("2" if model_type == "mamba2" else ""),
        "embedding": f"W{w}O16Embedding" if args.quantize_embedding else "Embedding",
        "lm_head": f"W{w}A{a}B16O16Linear" if args.quantize_lm_head else "Linear",
    }
    found = {
        "mixer": config.get("ssm_cfg", {}).get("layer"),
        "embedding": config.get("embedding_cfg", {}).get("layer", "Embedding"),
        "lm_head": config.get("lm_head_cfg", {}).get("layer", "Linear"),
    }
    mismatches = [f"{k}: stored {found[k]}, requested {expected[k]}" for k in expected if found[k] != expected[k]]
    if mismatches:
        raise ValueError(
            f"The cached model {quantized_model_path} does not match the requested quantization settings "
            f"({'; '.join(mismatches)}). It was probably stored by an older version that did not encode "
            f"these settings in the directory name. Delete or rename it to quantize again.")


def quantize_model_mamba(model, model_type, tokenizer, device, args, calibration_dataset=None, calib_preprocess_fn=None):
    if getattr(args, "fake_quant", False):
        return fake_quantize_model_mamba(model, model_type, tokenizer, device, args,
                                         calibration_dataset, calib_preprocess_fn)
    # restore the quantized model from pretrained_dir
    if args.pretrained_dir:
        # change the name to lookup the model in the pretrained_dir
        model_name = args.model.lower().split('/')[-1]

        # already loaded the quantized model in build_mamba_and_tokenizer in utils.py
        if model_name.startswith("quamba"): # ut-enyac/quamba or ut-enyac/quamba2
            get_model_size(model, args.model, args.w_bits, args.a_bits)
            return model

        # load quantied model in args.pretrained_dir to replace fp16 mamba
        # This will require much more memory, since we will have
        # fp16 mamba, qumaba, and quamba weight in the memory at the same time
        if model_name.startswith("mamba"): # mamba or mamba2
            quantized_model_path = os.path.join(args.pretrained_dir, "ut-enyac", get_quantized_model_name(args))
            # load the quantized model if it exists
            if os.path.isdir(quantized_model_path):
                check_quantized_model_config(quantized_model_path, model_type, args)
                logging.info(f"Loading quantized model from {quantized_model_path}")
                model = QuambaLMHeadModel.from_pretrained(quantized_model_path, device="cuda")
                get_model_size(model, args.model, args.w_bits, args.a_bits)
                return model
            else:
                logging.warning(f"{quantized_model_path} does not exist.")
                logging.warning("Runing calibration and quantization from scratch")
        else:
            logging.warning(f"Unsupported model {args.model} in ut-enyac/ model registry")

    # replace the mamba blocks with simple blocks to get the scaling factors
    # we hardcode use_had_transform=True to fix the configuration, so it is easier for users
    model = configure_model(model, model_type, use_had_transform=True) # W4A16 needs had_transform as well
    logging.info(f"Target bit-width W{args.w_bits}A{args.a_bits}")
    if args.a_bits == 8:
        # Run calibration to get scale and reorder params
        if args.group_heads:
            logging.info(f"Reordering weights and activations for head grouping")
            reorder_params = get_reorder_params(model, model_type, tokenizer, num_samples=512, seq_len=512)
            reorder_mamba(model, reorder_params)
            # collect 8-bit activation scales
            act_scales = run_quamba2_calibration(model, model_type, tokenizer, reorder_params,
                                                num_samples=args.calib_data_num,
                                                seq_len=args.calib_seqlen,
                                                calibration_dataset=calibration_dataset,
                                                preprocess_fn=calib_preprocess_fn)
        else:
            # collect 8-bit activation scales
            act_scales = run_quamba_calibration(model, model_type, tokenizer,
                                                num_samples=args.calib_data_num,
                                                seq_len=args.calib_seqlen,
                                                calibration_dataset=calibration_dataset,
                                                preprocess_fn=calib_preprocess_fn)
    elif args.a_bits == 16:
        # not doing anything for activations
        act_scales = {}
        if args.group_heads:
            logging.info(f"Activation bit-width is set to 16, skip weights reordering and head grouping")
    else:
        raise ValueError(f"Unsupported activation bit-width: {args.a_bits}, try --a_bits 8 or --a_bits 16?")

    # fuse the had matrices with the weight matrices in linear layers.
    # Do this after reordering and before applying gptq
    model = fuse_had_matrices(model)
    # Apply GPTQ to quantize linear
    if args.apply_gptq:
        # hybrid models always quantize lm_head (W4A8), the others only with --quantize_lm_head
        model = apply_gptq(model, tokenizer, device, w_bits=args.w_bits,
                           quantize_lm_head=args.quantize_lm_head or args.hybrid_blocks,
                           lm_head_bits=args.w_bits)
    # Replace (reordered, fused, and GPTQ quantized) modules with quantized version

    if args.hybrid_blocks: # create hybrid block
        if args.hybrid_blocks_config:
            logging.info(f"Loading hybrid blocks config from {args.hybrid_blocks_config}")
            with open(args.hybrid_blocks_config, "r") as file:
                hybrid_blocks_configs = json.load(file)
        else:
            hybrid_blocks_configs = None
        model = quantize_fp16_model_act_hybrid(model, model_type, act_scales, device, w_bits=args.w_bits,
                                               layer_wise_hybrid_config=hybrid_blocks_configs)
    else:
        model = quantize_fp16_model(
            model, model_type, act_scales, device,
            w_bits=args.w_bits, a_bits=args.a_bits,
            quantize_embedding=args.quantize_embedding,
            quantize_lm_head=args.quantize_lm_head
        )
    # store the state_dict if not quamba
    model_name = args.model.lower().split('/')[-1]
    if args.pretrained_dir is not None and not model_name.startswith("quamba"):
        # change the name to store the model in the pretrained_dir
        model_name = get_quantized_model_name(args)
        quantized_model_path = os.path.join(args.pretrained_dir, "ut-enyac", model_name)
        if not args.hybrid_blocks:
            # we slightly hack the api: we use MambaLMHeadModel instead of QuambaLMHeadModel to store the model here
            model.config.ssm_cfg['layer'] = model.backbone.layers[0].mixer.__class__.__name__
            model.config.norm_cfg = {"norm": model.backbone.layers[0].norm.__class__.__name__}
            model.config.embedding_cfg = {"layer": model.backbone.embedding.__class__.__name__}
            model.config.lm_head_cfg = {"layer": model.lm_head.__class__.__name__}
            # We apply Hadamard transforms so we cannot tie embeddings and lm_head
            model.config.tie_embeddings = False # no used in QuambaLMHeadModel
            if hasattr(model.config, "use_cache"):
                delattr(model.config, "use_cache")
            model.save_pretrained(quantized_model_path)
            logging.info(f"The quantized model is stored at {quantized_model_path}")
        else:
            ssm_layer_infos = []
            norm_infos = []

            for i in range(len(model.backbone.layers)):
                mixer_norms_info = model.backbone.layers[i].get_class_info_dict_mixers_norms()
                ssm_layer_infos.append(mixer_norms_info["mixers"])
                norm_infos.append(mixer_norms_info["norms"])

            mixer_norms_info = model.backbone.layers[0].get_class_info_dict_mixers_norms()
            model.config.ssm_cfg['layer'] = ssm_layer_infos
            model.config.norm_cfg = {"norm": norm_infos}
            model.config.embedding_cfg = {"layer": model.backbone.embedding.__class__.__name__}
            model.config.lm_head_cfg = {"layer": model.lm_head.__class__.__name__}
            # We apply Hadamard transforms so we cannot tie embeddings and lm_head
            model.config.tie_embeddings = False # no used in QuambaLMHeadModel
            if hasattr(model.config, "use_cache"):
                delattr(model.config, "use_cache")
            model.config.is_hybrid = True
            model.save_pretrained(quantized_model_path)
            logging.info(f"The quantized model is stored at {quantized_model_path}")

        # store tokenizer for mamba2-8b
        if "mamba2-8b" in args.model:
            # model.save_pretrained should already create the saved dir
            saved_dir = quantized_model_path
            tokenizer.save(saved_dir)
            logging.info(f"Tokenizer is stored at {saved_dir}")
    # quantized model
    get_model_size(model, args.model, args.w_bits, args.a_bits)
    return model.to(device)




