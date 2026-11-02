# NCU: FA4 block-sparse forward vs dense FA3 / FA4 (H200, 8192 x 262144, head_dim 256, GQA 24/4, bf16)

Run directory: `/local1/li004074/exp/ncu/20261101-fa4sparse-vs-fa3/` (harness/ncu_attn_driver.py, run_ncu.sh,
reports/, analysis/). ncu 2025.2.0, `--set full --clock-control base` (SM clock locked at 1.36 GHz, so absolute
times are not the unlocked numbers; ratios and utilisations are). GPU 4 (idle).

## Sparse kernel (55% of key blocks kept, random lists; `full_sparse_kv262144.ncu-rep`)

| metric | value | reading |
|---|---|---|
| `gpu__time_duration` | 46.7 ms | unlocked run of the same case: 48.2 ms -> the GPU runs near base clock under this load anyway |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 86.3% (min 81.2, max 89.7 across SMs) | tensor-pipe bound; 14% of active cycles have no MMA in flight |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | 83.3% | |
| `sm__cycles_active` min / avg / max | 57.8M / 61.3M / 63.6M | tail imbalance ~4% (max/avg 1.04): the static LPT order copes with uniform random lists |
| occupancy | 1 CTA/SM: 384 threads (1 producer + 2 MMA warpgroups), 168 regs, 194 KB smem | by design (FA3 structure) |
| `smsp__issue_active` | 44.7% of cycles; 0.67 eligible warps / cycle | expected for a wgmma kernel (few, long instructions) |
| stall reasons per issue (sum 5.03) | wait 1.30 (26%), selected 1.00, dispatch_stall 0.87 (17%), long_scoreboard 0.59 (12%), not_selected 0.49, gmma 0.41 (8%), short_scoreboard 0.18 | fixed-latency dependency chains (softmax / rescale between wgmma groups) and dispatch interlocks dominate; memory latency (long_scoreboard) is second |
| `lts__t_sector_hit_rate` | 60-62% | each CTA streams its own K/V blocks; L2 (50 MB) cannot hold a kv-head's 256 MB of K+V |
| `gpu__dram_throughput` | 26% of peak; 59 GB read | 227 GB of K/V requests from 1536 CTAs, 62% served by L2 |
| shared-memory bank conflicts | 225 K of 5.36 G wavefronts | none |
| FP32 instructions | 4.48 G non-fused vs 0.93 G fused | NCU rule: up to 41% more FP32 throughput if fused (softmax exp/scale chain) |
| NCU rule "L1TEX global load pattern 4 of 32 B/sector" (est. 56%) | 34 M global loads, 98.7% L1 hit | the per-thread int32 block-list / page-table reads; negligible volume, the estimate is not credible here |

## Dense references (same shape, same clock lock)

| metric | FA4 sparse (55% kept) | FA3 dense (vLLM C++) | FA4 dense (CuTe) |
|---|---:|---:|---:|
| time (base clock) | 46.7 ms | 72.8 ms | 72.8 ms |
| tensor pipe active (avg / min over SMs) | 86.3% / 81.2% | 99.7% / 94% | 99.9% / 94% |
| SM throughput % of peak | 83.3 | 96.7 | 96.9 |
| grid | 1536 CTAs | 132 CTAs (persistent) | 1536 CTAs |
| L2 hit rate | 62% | 93.6% | 93.6% |
| DRAM read | 59 GB (26% of peak BW) | 4.1 GB (1.2%) | 13.4 GB (3.8%) |
| stall long_scoreboard / issue | 0.59 | 0.31 | 0.73 |
| stall wait / dispatch / gmma | 1.30 / 0.87 / 0.41 | 1.64 / 1.14 / 0.59 | 1.49 / 1.13 / 0.37 |
| smem bank conflicts | 0.2 M | 1.6 M | 0.002 M |

## Kernel experiments (same shape, unlocked clocks unless noted)

| change | 8K x 256K, 55% kept | 40% kept | correctness |
|---|---:|---:|---|
| baseline FA4 sparse (b37) | 48.2 ms (593 TF on kept) | 35.3 ms | - |
| structured lists (35% blocks shared by all rows/heads): L2 hit 62% -> 75%, DRAM 59 -> 37 GB | 46.8 ms at base clock = unchanged | | n/a |
| patch 2: producer software-pipelines the list-index loads | 48.65 ms = unchanged | 35.8 ms | identical lists/outputs |
| patch 3: no mask function on 'full' blocks (dense passes None on interior blocks; the sparse consumer called the causal mask on every block) | **45.4 ms (629 TF)**, -6.6% | **33.6 ms**, -4.9% | max rel err unchanged (2.4-3.7e-3), paged == gathered |

Removing the causal-mask function on fully visible blocks saves ~6% kernel time. Raising the L2 hit rate
and prefetching list entries gave no measurable gain in these cases.

## NCU of the patched kernel (patches 1-3, `full_sparse_kv262144_patched3.ncu-rep`, base clock)

| metric | unpatched | patched | FA4 dense |
|---|---:|---:|---:|
| time | 46.7 ms | 44.2 ms (-5.4%) | 72.8 ms (all blocks) |
| tensor pipe active avg / min | 86.3 / 81.2% | **90.7 / 85.4%** | 99.9 / 94% |
| instructions executed | 14.5 G | 12.8 G (-12%) | 19.4 G |
| L2 hit / DRAM throughput | 62% / 26% | 71% / 17% | 94% / 4% |
| stalls per issue: wait / dispatch / long_sb / gmma / barrier | 1.30 / 0.87 / 0.59 / 0.41 / 0.03 | 1.32 / 0.82 / 0.62 / 0.59 / 0.13 | 1.49 / 1.13 / 0.73 / 0.37 / 0.11 |

Removing the mask function cuts 12% of instructions and lifts tensor-pipe activity by 4.4 points.
80-token KV blocks improved kernel time by 1–7% but break 1024-token page alignment and are not enabled.
[Block-size measurements](h200-fa4-kb80-vs-kb64.log).

## Persistent grid

`SortedPersistentTileScheduler` (patch 4) with a persistent grid of 132 CTAs produced correct results only after
delaying the Q-pipeline release past the epilogue (sO aliases sQ), and ran at 105 TFLOPS instead of 700. NCU
(`--metrics l1tex__t_sectors_pipe_lsu_mem_local_op_{ld,st}.sum`, 8K x 64K, 55% kept):

| variant | time | local loads / stores (sectors) | SASS local ld / st |
|---|---:|---:|---:|
| static grid (1536 CTAs) | 10.7 ms | 0 / 0 | 0 / 0 |
| persistent sorted (132 CTAs, delayed Q release) | 69.0 ms | 7.50 G / 7.51 G | 1.36 G / 1.36 G |

The persistent loop keeps extra state live across tiles and spills at the consumer's 240-register ceiling.
The default sorted scheduler therefore launches one CTA per tile, longest lists first.

## Sorted dispatch

With compile-time tile validity (like `SingleTileScheduler`) and the grid launched in list-length order, no
spills and correct output. Synthetic lists with per-head density spread (list max/mean 1.6-1.7, close to the
real prompts' 1.6-1.9), GPU 5, same run:

| case | static FA4 order | sorted order | gain |
|---|---:|---:|---:|
| 8K x 256K, 54% kept | 50.9 ms | 43.4 ms | -15% |
| 8K x 256K, 37% kept | 32.1 ms | 29.8 ms | -7% |
| 8K x 64K, 52% / 39% kept | 8.85 / 7.43 ms | 8.79 / 7.40 ms | ~0 |

At 64K a layer-chunk is ~1536 CTAs of short lists (few waves of imbalance to fix); at 256K the longest-first
order removes the tail. Cost: one `argsort` of 1536 list lengths per call (~20 us).
