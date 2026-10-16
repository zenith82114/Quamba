#!/bin/bash

usage() {
  echo "Usage: $0 <model> [precision] [group quantization options]"
  echo "Model options: state-spaces/mamba-2.8b, state-spaces/mamba2-2.7b"
  echo "Precision options: fp16, w8a8, w4a8, w4a16"
  echo "                   any wXaY (e.g., w4a4) with fake quantization (Mamba1 only)"
  echo "Note: If precision is omitted, model name must contain 'quamba'"
  echo ""
  echo "Group quantization options (simulated with --fake_quant, Mamba1 only):"
  echo "  --fake            fake quantization with Quamba's default granularity"
  echo "  --wg <size>       weight group size: 0 = per-tensor, -1 = per-output-channel, >0 = group size"
  echo "                    (default: per-tensor for W8, 128 for W4)"
  echo "  --ag <size>       activation group size: 0 = per-tensor, -1 = per-channel, >0 = group size (default: 0)"
  echo "  --amode <mode>    activation scales: static (calibrated) or dynamic (per token) (default: static)"
  echo "  --conv_group      also group the conv1d weight along channels with --wg"
  echo "Any of these options enables --fake_quant."
  echo ""
  echo "Examples:"
  echo "  $0 state-spaces/mamba-2.8b w8a8 --ag 128 --amode dynamic"
  echo "  $0 state-spaces/mamba-2.8b w4a4 --wg 64 --ag 64 --amode dynamic"
}

MODEL=$1
shift
if [[ -z "$MODEL" ]]; then
  usage
  exit 1
fi

PRECISION=""
if [[ -n "$1" && "$1" != --* ]]; then
  PRECISION=$1
  shift
fi

FAKE_QUANT=0
GROUP_FLAGS=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --fake)
      FAKE_QUANT=1; shift ;;
    --wg)
      [[ -z "$2" ]] && { echo "Error: --wg needs a value"; exit 1; }
      GROUP_FLAGS+=" --w_group_size $2"; FAKE_QUANT=1; shift 2 ;;
    --ag)
      [[ -z "$2" ]] && { echo "Error: --ag needs a value"; exit 1; }
      GROUP_FLAGS+=" --a_group_size $2"; FAKE_QUANT=1; shift 2 ;;
    --amode)
      [[ "$2" != "static" && "$2" != "dynamic" ]] && { echo "Error: --amode must be static or dynamic"; exit 1; }
      GROUP_FLAGS+=" --a_group_mode $2"; FAKE_QUANT=1; shift 2 ;;
    --conv_group)
      GROUP_FLAGS+=" --fq_conv_group"; FAKE_QUANT=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

# Check if PRECISION is missing and MODEL is not "quamba"
if [[ -z "$PRECISION" && "$MODEL" != *"quamba"* ]]; then
  echo "Error: Precision is required unless model contains 'quamba'"
  exit 1
fi

if [[ $FAKE_QUANT == 1 ]]; then
  if [[ "$MODEL" == *"quamba"* || "$PRECISION" == "fp16" ]]; then
    echo "Error: group quantization options need an fp16 mamba model and a wXaY precision"
    exit 1
  fi
  if [[ "$MODEL" == *"mamba2"* ]]; then
    echo "Error: group (fake) quantization only supports Mamba1 for now"
    exit 1
  fi
fi


if [[ $FAKE_QUANT == 1 ]]; then
  # fake-quantized models are never stored to or loaded from --pretrained_dir
  CMD="python main.py $MODEL --batch_size 16 --eval_ppl --ppl_dataset pile --log_dir ./logs"
else
  CMD="python main.py $MODEL --batch_size 16 --eval_ppl --ppl_dataset pile --pretrained_dir ./pretrained_models --log_dir ./logs"
fi

# Append group_heads flag if model is mamba2
if [[ "$MODEL" == *"mamba2"* ]]; then
  CMD+=" --group_heads"
fi

if [[ "$MODEL" != *"quamba"* ]]; then
    case $PRECISION in
    fp16)
        ;;
    w8a8)
        # default --w_bits 8 --a_bits 8
        CMD+=" --quantize"
        ;;
    w4a8)
        CMD+=" --quantize --w_bits 4 --a_bits 8"
        ;;
    w4a16)
        CMD+=" --quantize --w_bits 4 --a_bits 16"
        ;;
    *)
        # other bit-widths only run with fake quantization; GPTQ for sub-8-bit weights as in w4a8
        if [[ $FAKE_QUANT == 1 && $PRECISION =~ ^w([0-9]+)a([0-9]+)$ ]]; then
            W_BITS=${BASH_REMATCH[1]}
            A_BITS=${BASH_REMATCH[2]}
            CMD+=" --quantize --w_bits $W_BITS --a_bits $A_BITS"
        else
            echo "Unsupported precision: $PRECISION (other wXaY precisions need group quantization options, e.g., --fake)"
            exit 1
        fi
        ;;
    esac
fi

if [[ $FAKE_QUANT == 1 ]]; then
  CMD+=" --fake_quant$GROUP_FLAGS"
fi

echo "Running: $CMD"
eval $CMD
