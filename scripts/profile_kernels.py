import argparse
import time

import torch
import tinyserve_ext


def build_paged_attention_inputs(args):
    device = "cuda"
    dtype = torch.float32

    # Allocate a fragmented paged KV cache similar to decode-time serving.
    max_blocks_per_seq = (args.max_seq_len + args.block_size - 1) // args.block_size
    max_num_blocks = args.num_seqs * max_blocks_per_seq + args.extra_blocks

    k_cache = torch.randn(
        max_num_blocks,
        args.block_size,
        args.num_kv_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    v_cache = torch.randn_like(k_cache)
    query = torch.randn(args.num_seqs, args.num_q_heads, args.head_dim, device=device, dtype=dtype)
    out = torch.empty_like(query)

    context_lens = torch.randint(
        args.min_seq_len,
        args.max_seq_len + 1,
        (args.num_seqs,),
        dtype=torch.int32,
        device=device,
    )
    block_tables = torch.full(
        (args.num_seqs, max_blocks_per_seq),
        -1,
        dtype=torch.int32,
        device=device,
    )

    physical_blocks = torch.randperm(max_num_blocks, device=device, dtype=torch.int32)
    pool_idx = 0
    for seq_idx in range(args.num_seqs):
        seq_len = int(context_lens[seq_idx].item())
        num_blocks = (seq_len + args.block_size - 1) // args.block_size
        block_tables[seq_idx, :num_blocks] = physical_blocks[pool_idx : pool_idx + num_blocks]
        pool_idx += num_blocks

    return out, query, k_cache, v_cache, block_tables, context_lens


def build_reshape_inputs(args):
    device = "cuda"
    dtype = torch.float32

    # Build random slot mappings to profile scattered cache writes.
    max_num_blocks = (args.num_tokens + args.block_size - 1) // args.block_size + args.extra_blocks
    key = torch.randn(args.num_tokens, args.num_kv_heads, args.head_dim, device=device, dtype=dtype)
    value = torch.randn_like(key)
    key_cache = torch.empty(
        max_num_blocks,
        args.block_size,
        args.num_kv_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    value_cache = torch.empty_like(key_cache)

    slots = torch.randperm(max_num_blocks * args.block_size, device=device, dtype=torch.long)
    slot_mapping = slots[: args.num_tokens].contiguous()

    return key, value, key_cache, value_cache, slot_mapping


def profile_paged_attention(args):
    inputs = build_paged_attention_inputs(args)
    kernel_func = tinyserve_ext.paged_attention_v7 if args.kernel == "paged_attention_v7" else tinyserve_ext.paged_attention_v6

    # Warm up before Nsight's selected launch to avoid first-run overheads.
    for _ in range(args.warmup):
        kernel_func(*inputs)
    torch.cuda.synchronize()

    # NVTX makes the measured region easy to find in timeline tools.
    torch.cuda.nvtx.range_push(f"tinyserve_{args.kernel}")
    start = time.perf_counter()
    for _ in range(args.iterations):
        kernel_func(*inputs)
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - start
    torch.cuda.nvtx.range_pop()

    context_lens = inputs[-1]
    total_tokens = int(context_lens.sum().item())
    bytes_per_iter = (
        total_tokens * args.num_kv_heads * args.head_dim * 4 * 2
        + args.num_seqs * args.num_q_heads * args.head_dim * 4
    )
    avg_ms = elapsed_s * 1000.0 / args.iterations
    effective_gbps = (bytes_per_iter / 1e9) / (avg_ms / 1000.0)

    print(f"kernel={args.kernel}")
    print(f"iterations={args.iterations}")
    print(f"num_seqs={args.num_seqs}")
    print(f"context_tokens_total={total_tokens}")
    print(f"avg_latency_ms={avg_ms:.4f}")
    print(f"manual_effective_bandwidth_gbps={effective_gbps:.2f}")


def profile_reshape_and_cache(args):
    inputs = build_reshape_inputs(args)

    # Warm up before collecting timing/profiling data.
    for _ in range(args.warmup):
        tinyserve_ext.reshape_and_cache(*inputs)
    torch.cuda.synchronize()

    # Keep a named range around the repeated kernel launches.
    torch.cuda.nvtx.range_push("tinyserve_reshape_and_cache")
    start = time.perf_counter()
    for _ in range(args.iterations):
        tinyserve_ext.reshape_and_cache(*inputs)
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - start
    torch.cuda.nvtx.range_pop()

    bytes_per_iter = args.num_tokens * args.num_kv_heads * args.head_dim * 4 * 4
    avg_ms = elapsed_s * 1000.0 / args.iterations
    effective_gbps = (bytes_per_iter / 1e9) / (avg_ms / 1000.0)

    print("kernel=reshape_and_cache")
    print(f"iterations={args.iterations}")
    print(f"num_tokens={args.num_tokens}")
    print(f"avg_latency_ms={avg_ms:.4f}")
    print(f"manual_effective_bandwidth_gbps={effective_gbps:.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Stable Nsight Compute targets for TinyServe kernels.")
    parser.add_argument(
        "--kernel",
        choices=["paged_attention_v6", "paged_attention_v7", "reshape_and_cache"],
        required=True,
    )
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--num-seqs", type=int, default=64)
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--min-seq-len", type=int, default=1024)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--num-q-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--extra-blocks", type=int, default=4096)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for profiling.")

    torch.manual_seed(67)
    torch.cuda.manual_seed(67)

    if args.kernel in {"paged_attention_v6", "paged_attention_v7"}:
        profile_paged_attention(args)
    else:
        profile_reshape_and_cache(args)


if __name__ == "__main__":
    main()
