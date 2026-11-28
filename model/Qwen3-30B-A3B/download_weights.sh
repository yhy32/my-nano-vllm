#!/usr/bin/env bash
set -euo pipefail

case "${1:-bf16}" in
    bf16) variant=bf16; model_id=Qwen/Qwen3-30B-A3B ;;
    fp8) variant=fp8; model_id=Qwen/Qwen3-30B-A3B-FP8 ;;
    *)
        printf '用法: %s [bf16|fp8] [--endpoint URL] [hf download 参数]\n' "$0" >&2
        exit 2
        ;;
esac
if (( $# > 0 )); then
    shift
fi

if [[ "${1:-}" == "--endpoint" ]]; then
    if [[ -z "${2:-}" || "$2" == -* ]]; then
        printf '%s\n' '--endpoint 需要提供镜像 URL。' >&2
        exit 2
    fi
    export HF_ENDPOINT="$2"
    shift 2
fi

if ! command -v hf >/dev/null 2>&1; then
    printf '%s\n' '找不到 hf，请先激活 nanovllm 环境，或运行 python -m pip install -U huggingface_hub。' >&2
    exit 127
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec hf download "$model_id" --local-dir "$script_dir/$variant" "$@"

