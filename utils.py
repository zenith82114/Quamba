import os
import json
import random
import argparse
import numpy as np

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from quamba.megatron_utils import _GPTSentencePieceTokenizer
from quamba.quamba_mixer_seq import QuambaLMHeadModel


def build_mamba_and_tokenizer(args, model_type="mamba"):
    is_quamba = False
    device = "cuda"
    dtype = torch.float16 # use half, otherwise real quant won't run
    if model_type == "mamba" or model_type == "mamba2":
        if "mamba2-8b" not in args.model:
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", resume_download=None)
        else:
            # NOTE(hychiang): Special handle for mamba2-8b's tokenizer from NVIDIA Megatron
            # FIXME: hardcode the tokenizer file name for now
            tokenizer_ckpt = os.path.join(args.model, "mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model")
            tokenizer = _GPTSentencePieceTokenizer(tokenizer_ckpt)
        model = MambaLMHeadModel.from_pretrained(args.model, device=device, dtype=dtype)
    elif model_type == "quamba" or model_type == "quamba2":
        assert args.pretrained_dir, "Please specify the --pretrained_dir for quamba models"
        quantized_model_path = os.path.join(args.pretrained_dir, args.model)
        assert os.path.exists(quantized_model_path), f"Quantized model {quantized_model_path} not found"
        if "quamba2-8b" not in args.model:
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", resume_download=None)
        else:
            # NOTE(hychiang): Special handle for mamba2-8b's tokenizer from NVIDIA Megatron
            # FIXME: the model and tokenizer will be initizlied again in modelutils_mamba.py
            tokenizer_ckpt = os.path.join(args.pretrained_dir, args.model, "mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model")
            tokenizer = _GPTSentencePieceTokenizer(tokenizer_ckpt)
        model = QuambaLMHeadModel.from_pretrained(quantized_model_path, device="cuda")
        is_quamba = True
    else:
        raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba', 'mamba2', 'quamba' and 'quamba2'")
    return model, tokenizer, is_quamba



def set_deterministic(seed):
    # Fix all possible random seef for reproduce
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_quantize_options(parser):
    # quantization parameters
    parser.add_argument(
        '--quantize', action='store_true', default=False,
    )
    # calibration parameters
    parser.add_argument(
        '--calib_data_num', type=int, default=512,
        help='Number of calibration data (default: 512)'
    )
    parser.add_argument(
        '--calib_seqlen', type=int, default=512,
        help='Maximum sequence length for calibration data (default: 512)'
    )
    parser.add_argument(
        '--calib_dataset', type=str, default='pile', choices=['pile', 'hellaswag'],
        help='Activation calibration data. pile: the first --calib_data_num val docs of the Pile, truncated '
        'to --calib_seqlen tokens (Quamba). hellaswag: --calib_data_num random --calib_seqlen-token windows '
        'of the HellaSwag test contexts (MambaQuant uses 64 x 258). (default: pile)'
    )
    parser.add_argument(
        '--calib_seed', type=int, default=1,
        help='[--calib_dataset hellaswag] Seed of the window draw. Seed N gives the same windows as '
        'MambaQuant\'s --calib_seed N. (default: 1)'
    )
    parser.add_argument(
        '--act_observer', type=str, default='quamba', choices=['quamba', 'hist', 'hist_xpct'],
        help='Static activation observers. quamba: running absmax, with an EMA of the per-sample 0.9995 '
        'percentile at the conv output and the SSM state. hist: MambaQuant\'s histogram pooled over all '
        'calibration data, clipped at the --hist_percentile quantile of each side, for every activation. '
        'hist_xpct: hist in place of the running absmax, but Quamba\'s percentile at the conv output and '
        'the SSM state. (default: quamba)'
    )
    parser.add_argument(
        '--hist_percentile', type=float, default=0.999999,
        help='[--act_observer hist] Quantile of each side kept unclipped. (default: 0.999999, as in MambaQuant)'
    )
    parser.add_argument(
        '--asym_act', action='store_true', default=False,
        help='[--fake_quant, static per-tensor activations] Asymmetric activation scales with zero points, '
        'as in MambaQuant. The int8 kernels are symmetric only. (default: False)'
    )
    # load/store model
    parser.add_argument(
        '--pretrained_dir', type=str, default=None,
        help='The path to store both the quantized model and its act_scales_cache.'
        'Not storing if not provided. (default: None)'
    )
    # quantization parameters
    parser.add_argument(
        '--group_heads',  action='store_true', default=False,
        help='Whether to group heads during the reordering (default: False)'
    )
    parser.add_argument(
        "--quantize_embedding", action='store_true', default=False,
        help="Whether to quantize the embedding layer (default: False)"
    )
    parser.add_argument(
        "--quantize_lm_head", action='store_true', default=False,
        help="Whether to quantize the lm_head layer (default: False)"
    )
    parser.add_argument(
        '--apply_gptq',  action='store_true', default=False,
        help='Whether to apply the GPTQ quantizer (default: False)'
    )
    parser.add_argument(
        '--w_bits', type=int, default=8,
        help='The bit-width for weights applied in the real quantization (default: 8)'
    )
    parser.add_argument(
        '--a_bits', type=int, default=8,
        help='The bit-width for activations applied in the real quantization (default: 8)'
    )
    # fake (simulated) quantization with channel groups, Mamba1 only
    parser.add_argument(
        '--fake_quant', action='store_true', default=False,
        help='Simulate quantization (quantize-dequantize in fp16) instead of using the real quantized kernels. '
        'Any --w_bits/--a_bits is allowed (--a_bits 16 disables activation quantization). '
        'The model is not stored to --pretrained_dir. (default: False)'
    )
    parser.add_argument(
        '--w_group_size', type=int, default=None,
        help='[--fake_quant] Weight group size along the input dim: 0 = per-tensor, -1 = per-output-channel, '
        '>0 = group size. (default: None, i.e., Quamba: per-tensor for W8, 128 for W4)'
    )
    parser.add_argument(
        '--a_group_size', type=int, default=0,
        help='[--fake_quant] Activation group size along the channel dim: 0 = per-tensor, -1 = per-channel, '
        '>0 = group size. (default: 0)'
    )
    parser.add_argument(
        '--a_group_mode', type=str, default='static', choices=['static', 'dynamic'],
        help='[--fake_quant] static: calibrated scale per group shared by all tokens; '
        'dynamic: absmax scale per token and per group at runtime. (default: static)'
    )
    parser.add_argument(
        '--fq_conv_group', action='store_true', default=False,
        help='[--fake_quant] Also group the conv1d weight along channels with --w_group_size '
        '(default: False, i.e., int8 per-tensor as in Quamba)'
    )
    parser.add_argument(
        '--hybrid_blocks', action='store_true', default=False,
        help='Whether to create hybrid blocks for configuring act_bits of blocks in a dynamic fashion.'
    )
    parser.add_argument(
        '--hybrid_blocks_config', type=str, default=None,
        help='Path to the the configuration for hybrid blocks'
    )
    
def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model', type=str,
        help='Mamba to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Whether to print the debug level information'
    )
    ##### General Evaluation Settings #####
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--task_list', type=lambda s: [item for item in s.split(',')], default=["lambada_openai"],
        help='Task to be evaled, e.g., --task_list lambada_openai,hellaswag,arc_easy,arc_challenge,piqa,winogrande'
    )
    parser.add_argument(
        '--eval_zero_shot', action='store_true', default=False,
        help='Whether to evaluate the zero-shot performance Task(s) specified by `--tasks_list`, e.g, --tasks_list lambada_openai,hellaswag,arc_easy,arc_challenge,piqa,winogrande'
    )
    parser.add_argument(
        '--eval_few_shot', action='store_true', default=False,
        help='Whether to evaluate the few-shot performance. Task(s) specified by `--tasks_list` and `--fewshot`'
    )
    parser.add_argument(
        '--fewshot', type=int, default=0,
        help='Number of shots for few-shot evaluation (0 for zero-shot)'
    )
    parser.add_argument(
        '--eval_generation', action='store_true', default=False,
        help='Whether to evaluate the performance of the generation tasks. Task(s) specified by `--tasks_list`, e.g, --tasks_list nq_open,squadv2'
    )
    parser.add_argument(
        '--testing', action='store_true',
        help='testing with decreased sample count'
    )
    parser.add_argument(
        '--eval_ppl', action='store_true', default=False,
        help='Whether to evaluate the wikitext2 ppl'
    )
    parser.add_argument(
        '--ppl_dataset', type=str, default='wikitext2',
        help='Dataset(s) for ppl evaluation, comma-separated, e.g., wikitext2,pile'
    )
    parser.add_argument(
        '--log_dir', type=str,
        help='path to the json log file storing the result of lm_evan and quantization settingarg'
    )
    get_quantize_options(parser)
    args = parser.parse_args()
    if args.asym_act and not (args.fake_quant and args.a_group_size == 0 and args.a_group_mode == "static"):
        parser.error("--asym_act needs --fake_quant with static per-tensor activations")
    if args.act_observer != "quamba" and args.fake_quant and args.a_group_size != 0:
        parser.error("--act_observer hist only supports per-tensor activations (--a_group_size 0)")
    return args
