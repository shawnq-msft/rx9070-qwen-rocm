# Qwen3.6-27B Q4_K_S validation session — 2026-05

## HF model inventory (Qwen3.6-27B, Q4-class)

Official: only BF16 (`Qwen/Qwen3.6-27B`, ~55 GB) and FP8 (`Qwen/Qwen3.6-27B-FP8`). **No official Q4 GGUF.**

Third-party Q4 sizes (verified from HF tree API):

unsloth/Qwen3.6-27B-GGUF (most complete, has mmproj for VL):
- IQ4_XS 15.44 GB · IQ4_NL 16.07 GB
- Q4_0 15.79 · Q4_K_S 15.86 · Q4_K_M 16.82
- UD-Q4_K_XL 17.61 (Unsloth Dynamic, top quality at 4-bit)

froggeric/Qwen3.6-27B-MTP-GGUF (preserves MTP head):
- IQ4_XS-mtp 15.53 · Q4_K_M-mtp 17.00

havenoammo/Qwen3.6-27B-MTP-UD-GGUF: UD-Q4_K_XL only, 18.06

RDson/Qwen3.6-27B-MTP-Q4_K_M-GGUF: 16.49 (single file)

Lorbus/Qwen3.6-27B-int4-AutoRound (safetensors for vLLM): ~21 GB total, AutoRound INT4

## Critical arch finding from llama.cpp loader

```
arch                  = qwen35
n_layer               = 64
n_head                = 24
n_head_kv             = 4         # only 16 of 64 layers are attention
n_embd                = 5120
n_ctx_train           = 262144
file size             = 14.76 GiB (4.71 BPW)   # Q4_K_S
```

Loader also prints `llama_memory_recurrent: ROCm0 RS buffer size = 149.62 MiB` and `sched_reserve: fused Gated Delta Net (autoregressive/chunked) enabled` — confirming hybrid SSM+Attn architecture, not dense.

## Failing-config stderr excerpts

### -fa on, KV q8_0/q8_0
```
slot update_slots: id  0 | task 0 | prompt processing progress, n_tokens = 33
ROCm error: invalid device function
  current device: 0, in function launch_fattn at .../fattn-common.cuh:1053
  hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, fattn_kernel,
    block_dim.x * block_dim.y * block_dim.z, nbytes_shared)
.../ggml-cuda.cu:97: ROCm error
Aborted (core dumped)
```
Loader showed `n_embd_head_k_all = 256` — the missing fattn instance is for head_dim=256 on RDNA4.

### -fa off, KV q8_0/q8_0
```
llama_init_from_model: V cache quantization requires flash_attn
common_init_result: failed to create context
Segmentation fault (core dumped)
```

### -fa on, KV f16/f16, ctx 16384, ub 512 b 2048
```
llama_kv_cache: ROCm0 KV buffer size = 1024.00 MiB
llama_memory_recurrent: ROCm0 RS buffer size = 149.62 MiB
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 495.00 MiB on device 0:
  cudaMalloc failed: out of memory
graph_reserve: failed to allocate compute buffers
```
Reducing to `-ub 128 -b 512` only dropped the failed alloc to ~124 MiB, still OOM.

### Default parallel (4) at ctx 32k with K/V q8_0
```
llama_kv_cache: size = 1088.00 MiB ( 32768 cells, 16 layers, 4/1 seqs)
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 598.50 MiB ... out of memory
llama_init_from_model: failed to allocate buffer for rs cache
```
Note "rs cache" failure — RS scales with parallel slots.

## Working config (the only one)

```
PORT=8820 CTX=12288
-ngl 999 -fa off -ctk q8_0 -ctv f16 -c 12288 -ub 256 -b 1024 --parallel 1
```
Memory layout reported by loader:
```
load_tensors: ROCm0 model buffer size = 14429.10 MiB
load_tensors: CPU_Mapped model buffer size = 682.03 MiB   (embed/output not on GPU)
llama_kv_cache: ROCm0 KV buffer size = 588.00 MiB
                K (q8_0): 204.00 MiB, V (f16): 384.00 MiB
llama_memory_recurrent: ROCm0 RS buffer size = 149.62 MiB
```
Perf: 37 prompt tokens → 275 completion tokens in 16.01 s → **17.18 tok/s TG**.

## TurboQuant TQ4 / K4V3 verdict for Qwen3.6-27B

- vLLM TurboQuant route: doesn't load GGUF; needs GPTQ/AWQ/FP8 safetensors. Lorbus AutoRound INT4 (~21 GB) won't fit on 16 GB. Hybrid-arch vLLM support on this box only validated for Qwopus 9B GPTQ.
- llama.cpp TurboQuant fork (Agustin / turbo-tan / CarapaceUDE): turbo4 / TQ3_1S / TQ4_1S KV types only activate against TurboQuant-quantized weights. **No TurboQuant Qwen3.6-27B weights on HF as of 2026-05.**
- Even if available, KV is <1 GB on this hybrid arch — squeezing it doesn't move the ctx ceiling because compute buffer + weights are the binding constraint.

## KV-to-RAM offload verdict

`--no-kv-offload` works but cuts TG from ~17 to single digits. RS cache stays on GPU regardless. Not worth it for 27B.

## Future-unblock signal

Watch llama.cpp PRs that add fattn instances for head_dim=256 on RDNA4 / gfx1201 / wave64. That single change lets `-fa on` work, which unlocks `-ctv q8_0`, which roughly doubles the practical ctx ceiling for any qwen35-arch model on this box.
