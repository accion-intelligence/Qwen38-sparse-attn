"""Latency client for a running vLLM endpoint (single instance or PD proxy): fires `--concurrency` streaming
/v1/completions requests at once (prompt token ids from prompts.json), records per-request TTFT, per-token
latencies (TPOT), end-to-end time, and the first generated tokens; optional repetitions.

    source scripts/h200/env.sh && python scripts/h200/pd_client.py --url http://127.0.0.1:30010 --model model \
        --prompts $H200_WORK/prompts.json --lengths 65536 --concurrency 2 --max-tokens 256 --output pd-64k.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _engine import load_prompts  # noqa: E402


async def one(client, url, model, req, max_tokens, gate):
    import httpx  # noqa: F401
    payload = {'model': model, 'prompt': req['prompt_token_ids'], 'max_tokens': max_tokens, 'temperature': 0.0, 'ignore_eos': True,
               'stream': True, 'stream_interval': 1, 'return_token_ids': True, 'skip_special_tokens': False,
               'stream_options': {'include_usage': True}}
    await gate.wait()
    t0 = time.perf_counter()
    times, toks = [], []
    async with client.stream('POST', url + '/v1/completions', json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith('data: ') or line[6:] == '[DONE]':
                continue
            ev = json.loads(line[6:])
            if ev.get('error'):
                raise RuntimeError(str(ev['error']))
            for ch in ev.get('choices', []):
                ids = ch.get('token_ids') or []
                if ids:
                    now = time.perf_counter()
                    times.extend([now] * len(ids))
                    toks.extend(ids)
    if not times:
        return {'request_id': req['request_id'], 'error': 'no tokens'}
    ttft = (times[0] - t0) * 1000
    tpot = [(times[i] - times[i - 1]) * 1000 for i in range(1, len(times))]
    return {'request_id': req['request_id'], 'length': req['length'], 'ttft_ms': ttft, 'e2e_ms': (times[-1] - t0) * 1000,
            'tokens': len(toks), 'tpot_mean_ms': statistics.fmean(tpot) if tpot else None,
            'tpot_p50_ms': statistics.median(tpot) if tpot else None, 'tpot_p99_ms': sorted(tpot)[int(0.99 * (len(tpot) - 1))] if tpot else None,
            'first_tokens': toks[:8]}


async def burst(url, model, reqs, max_tokens, timeout):
    import httpx
    gate = asyncio.Event()
    async with httpx.AsyncClient(timeout=timeout, trust_env=False, limits=httpx.Limits(max_connections=len(reqs) + 2)) as client:
        tasks = [asyncio.create_task(one(client, url, model, r, max_tokens, gate)) for r in reqs]
        gate.set()
        return await asyncio.gather(*tasks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--model', default='model')
    ap.add_argument('--prompts', type=Path, required=True)
    ap.add_argument('--kinds', default='docs_needle')
    ap.add_argument('--lengths', default='65536')
    ap.add_argument('--concurrency', type=int, default=1)
    ap.add_argument('--max-tokens', type=int, default=256)
    ap.add_argument('--repetitions', type=int, default=1)
    ap.add_argument('--warmup', type=int, default=1)
    ap.add_argument('--timeout', type=float, default=3600)
    ap.add_argument('--output', type=Path, default=None)
    args = ap.parse_args()
    prompts = load_prompts(args.prompts, args.kinds, args.lengths)
    results = []
    for length in (int(x) for x in args.lengths.split(',')):
        pool = sorted([r for r in prompts if r['length'] == length], key=lambda r: r.get('index', 0))
        reqs = [pool[i % len(pool)] for i in range(args.concurrency)]
        for rep in range(args.warmup + args.repetitions):
            t0 = time.perf_counter()
            rows = asyncio.run(burst(args.url, args.model, reqs, args.max_tokens, args.timeout))
            wall = (time.perf_counter() - t0) * 1000
            ok = [r for r in rows if 'ttft_ms' in r]
            rec = {'length': length, 'concurrency': args.concurrency, 'max_tokens': args.max_tokens, 'warmup': rep < args.warmup,
                   'wall_ms': wall, 'rows': rows,
                   'ttft_mean_ms': statistics.fmean(r['ttft_ms'] for r in ok) if ok else None,
                   'ttft_max_ms': max(r['ttft_ms'] for r in ok) if ok else None,
                   'tpot_mean_ms': statistics.fmean(r['tpot_mean_ms'] for r in ok if r['tpot_mean_ms']) if ok else None,
                   'tpot_p99_ms': max(r['tpot_p99_ms'] for r in ok if r['tpot_p99_ms']) if ok else None}
            results.append(rec)
            tag = 'warmup' if rec['warmup'] else f'rep{rep - args.warmup + 1}'
            print(f"len {length} x{args.concurrency} {tag}: TTFT mean {rec['ttft_mean_ms']:.0f} ms (max {rec['ttft_max_ms']:.0f}) | "
                  f"TPOT mean {rec['tpot_mean_ms']:.1f} ms p99 {rec['tpot_p99_ms']:.1f} | wall {wall:.0f} ms | first tokens {ok[0]['first_tokens'][:4] if ok else '-'}", flush=True)
            if args.output:
                args.output.write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
