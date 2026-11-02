"""Where does one long-prompt prefill spend its GPU time? Per-kernel-category breakdown with torch.profiler,
stock vLLM engine in-process (same flags as scripts/_engine.py: chunked prefill, no prefix caching, eager).

    source scripts/h200/env.sh && python scripts/h200/profile_prefill.py --model $QWEN38_MODEL \
        --prompts $H200_WORK/prompts.json --lengths 65536,131072,260096 --output $H200_WORK/profile-stock

Writes <output>/<length>.json (category totals, top kernels) and prints a table. Categories are regex buckets
over CUDA kernel names; unmatched names land in 'other' and are listed so the regexes can be extended.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _engine import add_engine_args, make_llm, load_prompts  # noqa: E402

CATEGORIES = [
    ('attention', r'flash_fwd|flash::|fmha|FlashAttn|attn_fwd|attention|BatchPrefill|SinglePrefill|paged_attn|reshape_and_cache|concat_and_cache'),
    ('gdn/linear-attn', r'gdn|gated_delta|delta_rule|chunk_|recurrent|causal_conv|conv1d|solve_tril|cumsum|mamba|ssm|fla_|wy_|kkt|gating'),
    ('gemm', r'gemm|Gemm|cutlass|nvjet|xmma|cublas|matmul|wgmma|sm90_|Sm90'),
    ('norm/rope/act', r'rms_norm|rmsnorm|layer_norm|layernorm|silu|gelu|act_and_mul|rotary|rope|embedding'),
    ('memory/elementwise', r'elementwise|vectorized|copy|Copy|fill|index|gather|scatter|cat_|CatArray|convert|cast|unrolled|reduce|Reduce|softmax|cumsum|sort|topk|argmax|triton_'),
]


def categorize(name: str) -> str:
    for cat, pat in CATEGORIES:
        if re.search(pat, name):
            return cat
    return 'other'


def main():
    ap = argparse.ArgumentParser()
    add_engine_args(ap)
    ap.add_argument('--prompts', type=Path, required=True)
    ap.add_argument('--kinds', default='docs_needle')
    ap.add_argument('--lengths', default='65536,131072,260096')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--top', type=int, default=30)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    from torch.profiler import ProfilerActivity, profile
    from vllm import SamplingParams

    llm = make_llm(args)
    sp = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
    prompts = load_prompts(args.prompts, args.kinds, args.lengths)
    # warm-up: JIT kernels (FlashInfer GDN prefill etc.) on a short request
    short = {'prompt_token_ids': prompts[0]['prompt_token_ids'][:4096]}
    llm.generate([short], sp, use_tqdm=False)
    torch.cuda.synchronize()

    summary = {}
    for length in (int(x) for x in args.lengths.split(',')):
        req = next(r for r in prompts if r['length'] == length)
        inp = {'prompt_token_ids': req['prompt_token_ids']}
        torch.cuda.synchronize(); t0 = time.perf_counter()
        llm.generate([inp], sp, use_tqdm=False)
        torch.cuda.synchronize(); wall = time.perf_counter() - t0
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
            llm.generate([inp], sp, use_tqdm=False)
            torch.cuda.synchronize()
        rows = []
        for ev in prof.key_averages():
            dt = getattr(ev, 'device_time_total', None)
            if dt is None:
                dt = getattr(ev, 'cuda_time_total', 0.0)
            if ev.device_type.name != 'CUDA' or dt <= 0:
                continue
            rows.append({'name': ev.key, 'us': dt, 'count': ev.count, 'category': categorize(ev.key)})
        rows.sort(key=lambda r: -r['us'])
        total = sum(r['us'] for r in rows) or 1.0
        cats = defaultdict(float)
        for r in rows:
            cats[r['category']] += r['us']
        res = {'length': length, 'wall_s_unprofiled': wall, 'gpu_kernel_total_s': total / 1e6,
               'categories': {c: {'s': v / 1e6, 'pct': 100 * v / total} for c, v in sorted(cats.items(), key=lambda kv: -kv[1])},
               'top_kernels': [{**r, 'pct': 100 * r['us'] / total} for r in rows[:args.top]],
               'other_kernels': [r['name'] for r in rows if r['category'] == 'other'][:40]}
        (args.output / f'{length}.json').write_text(json.dumps(res, indent=1))
        summary[length] = res
        print(f'\n=== length {length}: wall {wall:.2f} s (unprofiled), GPU kernel time {total / 1e6:.2f} s ===', flush=True)
        for c, v in res['categories'].items():
            print(f'  {c:22s} {v["s"]:8.2f} s  {v["pct"]:5.1f}%')
        print('  top kernels:')
        for r in res['top_kernels'][:15]:
            print(f'    {r["pct"]:5.1f}%  {r["us"] / 1e3:9.1f} ms  x{r["count"]:<6d} [{r["category"]}] {r["name"][:110]}')
        if res['other_kernels']:
            print('  uncategorised:', res['other_kernels'][:12])
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=1))
    print('\nwrote', args.output)


if __name__ == '__main__':
    main()
