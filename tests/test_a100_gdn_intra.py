"""a100_attn.gdn_intra.fused_kkt_solve vs vLLM's chunk_scaled_dot_kkt_fwd + solve_tril (Qwen3.8 GDN shapes)."""
import torch

from a100_attn.gdn_intra import fused_kkt_solve
from vllm.third_party.flash_linear_attention.ops.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from vllm.third_party.flash_linear_attention.ops.cumsum import chunk_local_cumsum
from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices
from vllm.third_party.flash_linear_attention.ops.solve_tril import solve_tril


def timed(fn, it=20):
    for _ in range(3):
        fn()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); s.record()
    for _ in range(it):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it


def run(lens, Hg=16, H=48, K=128, seed=0):
    torch.manual_seed(seed)
    T = sum(lens)
    dev = 'cuda'
    k = torch.nn.functional.normalize(torch.randn(1, T, Hg, K, device=dev), dim=-1).to(torch.bfloat16)  # l2-normed
    beta = torch.rand(1, T, H, device=dev).to(torch.bfloat16)
    g = -torch.rand(1, T, H, device=dev) * 0.5                                 # log decay per token (<= 0)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), device=dev, dtype=torch.int32)
    ci = prepare_chunk_indices(cu, 64)
    gc = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu, chunk_indices=ci)
    ref = lambda: solve_tril(A=chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g=gc, cu_seqlens=cu, chunk_indices=ci,  # noqa: E731
                                                       output_dtype=torch.float32),
                             cu_seqlens=cu, chunk_indices=ci, output_dtype=k.dtype)
    new = lambda: fused_kkt_solve(k, gc, beta, cu, ci)  # noqa: E731
    a0, a1 = ref().float(), new().float()
    err = ((a0 - a1).abs().max() / a0.abs().max()).item()
    t0, t1 = timed(ref), timed(new)
    ok = err < 2e-2
    print(f'lens={lens}: max rel err {err:.2e} | vLLM kkt+solve {t0:.3f} ms  fused {t1:.3f} ms ({t0 / t1:.2f}x) '
          f'{"OK" if ok else "FAIL"}')
    return ok


if __name__ == '__main__':
    ok = True
    for lens in ([8192], [1000, 7192], [8191], [77, 8115]):
        ok &= run(lens)
    print('ALL OK' if ok else 'SOME FAILED')
