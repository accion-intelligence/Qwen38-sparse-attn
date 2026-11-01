"""One long prefill (single request, 1 output token) for kernel-level profiling with nsys.

    VLLM_ENABLE_V1_MULTIPROCESSING=0 nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
        --trace=cuda,nvtx --cuda-graph-trace=node --sample=none -o prefill-256k \
        python scripts/profile_prefill.py --model $MODEL --prompts prompts.json --length 260096 \
        --chunk 16384 --block-size 64
    nsys stats --report cuda_gpu_kern_sum --format csv -o prefill-256k prefill-256k.nsys-rep
    python scripts/nsys_breakdown.py prefill-256k_cuda_gpu_kern_sum.csv

A short warmup prefill runs first; only the measured prefill is inside the cudaProfilerStart/Stop range.
"""
import argparse
import json
import time

import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--prompts', required=True, help='build_prompts.py output')
    ap.add_argument('--length', type=int, required=True)
    ap.add_argument('--chunk', type=int, default=8192)
    ap.add_argument('--block-size', type=int, default=None)
    args = ap.parse_args()
    rows = [r for r in json.load(open(args.prompts)) if r['length'] == args.length]
    ids = rows[0]['prompt_token_ids']
    llm = LLM(args.model, dtype='bfloat16', max_model_len=262144, max_num_seqs=1,
              max_num_batched_tokens=args.chunk, enable_chunked_prefill=True, enable_prefix_caching=False,
              gpu_memory_utilization=0.92, language_model_only=True, seed=20261015, block_size=args.block_size)
    sp = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
    # warmup: one chunk-sized prefill so lazy init / autotune is out of the capture
    llm.generate([TokensPrompt(prompt_token_ids=ids[:2 * args.chunk])], sp, use_tqdm=False)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    t0 = time.perf_counter()
    out = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    torch.cuda.profiler.stop()
    print(f'PREFILL length={args.length} chunk={args.chunk} wall={dt:.2f}s first_token={out[0].outputs[0].token_ids}', flush=True)


if __name__ == '__main__':
    main()
