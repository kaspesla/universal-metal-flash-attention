# UMFA → ai-toolkit integration (Mac training)

Goal: ai-toolkit training on Apple Silicon runs attention (forward + backward)
through Universal Metal Flash Attention, selected by config, with automatic
fallback to PyTorch SDPA for anything UMFA can't handle.

## Findings (2026-11-01)

- Machine: M3 Max 128 GB, **macOS 14.8.9**, Xcode 16.2, Swift 6.0.3.
- UMFA + its kernel submodule both declare `.macOS(.v15)` and compile every
  kernel with `MTLLanguageVersion.version3_2` (~12 call sites). MSL 3.2 is
  macOS 15+, so **the library cannot load on this Mac as-is**. MSL 3.1
  (macOS 14) still supports `bfloat`, so a backport looks feasible but is unproven.
- Python entry points (`metal_sdpa_extension`): `metal_scaled_dot_product_attention`
  (SDPA-shaped, routes to autograd when inputs require grad) and
  `metal_flash_attention_autograd` (fwd+bwd, no mask). Masked or GQA calls
  with grad take the eager path inside the extension.
- ai-toolkit already has `train.attention_backend` (default `native`), applied in
  `BaseSDTrainProcess.py:1806` via diffusers `set_attention_backend`. Many
  extension models call `F.scaled_dot_product_attention` directly, so a
  diffusers-only backend would miss them.
- ai-toolkit has no venv yet (`run_mac.zsh` → `python -m manager` creates `.venv`).
  `~/miniconda3` has torch 2.14.1 (MPS ok) and diffusers 0.41.0.dev0, but it's
  not the pinned ai-toolkit env.
- All three repos are upstream clones (ostris/ai-toolkit, bghira/*), so work goes
  on local branches.
- Expectation: README shows BF16 forward roughly at parity with MPS SDPA
  (0.93–1.03x on FLUX). For training, the likely payoff is **memory** (no
  materialized S×S attention matrix in backward) more than raw speed. We
  measure rather than assume.

## Plan

### Phase 0: feasibility (go/no-go)
- [ ] Spike: on macOS 14, compile one bf16 attention kernel at MSL 3.1 (runtime JIT + `xcrun metal -std=macos-metal3.1`)
- [ ] If it works: add one `mfaLanguageVersion()` helper (3.2 when available, else 3.1), route all `.version3_2` sites through it, lower `platforms` to `.macOS(.v14)` in both Package.swift files
- [ ] `swift build -c release`, then `swift test` for the core attention tests
- [ ] If MSL 3.1 can't compile the kernels → stop and fall back to the macOS 15 upgrade

### Phase 1: build the Python extension into ai-toolkit's env
- [ ] Create the ai-toolkit venv via its manager, add `pybind11`
- [ ] `python setup.py build_ext --inplace`, install `pytorch_custom_op_ffi`
- [ ] Run `pytest tests/ -m "not slow"` (GPU tests included, since we have the GPU)

### Phase 2: ai-toolkit integration (small, opt-in)
- [ ] New `toolkit/util/metal_flash_attention.py`: `install()` wraps
      `F.scaled_dot_product_attention`; routes MPS fp16/bf16/fp32 calls with
      supported shapes to UMFA, falls back to the original for dropout>0,
      unsupported masks/dtypes/devices, or any UMFA error (warn once)
- [ ] Config: `train.attention_backend: metal_flash` → call `install()` and skip
      diffusers `set_attention_backend` (diffusers' `native` path then hits the
      patched SDPA too, so one hook covers diffusers and custom models)
- [ ] Optional env var `AITK_METAL_FLASH_ATTENTION=1` for UI-launched jobs

### Phase 3: verification
- [ ] Unit check: forward and grads vs PyTorch SDPA (bf16 & fp32, causal/non-causal, with/without mask) within tolerance
- [ ] Short real training run (target model TBD) for N steps, both backends: loss curves, s/it, peak memory (`torch.mps.driver_allocated_memory`)
- [ ] Write results below

## Open questions
1. macOS 14 backport (Phase 0) or upgrade to macOS 15/26?
2. Which model(s) do you train? This sets the benchmark and mask/GQA paths.
3. Env: ai-toolkit's own `.venv` (recommended) or `~/miniconda3`?

## Decisions (2026-11-01)
- No macOS 14 backport for now. Ken will test on macOS 15+ (other machine or after upgrading).
- Target models: Wan 2.1/2.2, Minimax H3. Env: ai-toolkit's `.venv`.
- No env var: the UI writes the same YAML, so `train.attention_backend` covers it.

## Review
Done on ai-toolkit branch `feature/metal-flash-attention` (uncommitted):
- `toolkit/util/metal_flash_attention.py`: `install_metal_flash_attention()` imports
  `metal_sdpa_extension`. The import itself registers UMFA for
  `aten::scaled_dot_product_attention` on the MPS dispatch key
  (`metal_sdpa_backend.cpp` `TORCH_LIBRARY_IMPL(aten, MPS, ...)`), so every MPS SDPA
  call in the process goes to UMFA. Not reversible in-process. (An earlier version also
  monkeypatched F.sdpa, which was redundant.) UMFA's C++ does its own fallback
  (dropout, masks while training, unsupported dtype/shape).
- Lesson (2026-11-01): the first check script imported the extension before taking the
  torch baseline, so it compared UMFA to itself (exactly 0 error, identical timings).
  Now it takes the baseline before importing and scores both against an fp32 explicit-math reference.
  `aten::_scaled_dot_product_attention_math_for_mps` has no derivative, so it can't be the reference.
- `BaseSDTrainProcess.py`: `attention_backend == 'metal_flash'` → install, skip diffusers `set_attention_backend`.
- `testing/check_metal_flash_attention.py`: fwd + grads vs torch SDPA, timing, dispatch counters.

Backward: implemented in UMFA (`MetalFlashAttentionFn::backward` →
`mfa_attention_backward`, recomputes from saved LSE, FP32 grads cast back).
Used whenever q/k/v require grad and there's no real mask.

Verified here (macOS 14, stub extension): MPS fwd+bwd routing, kwargs/positional
args, CPU passthrough, diffusers `dispatch_attention_fn` (Wan path), uninstall.
**Not verified: real UMFA kernels** (they need macOS 15+).

Known gaps:
- Minimax H3 batches with padding pass a real bool mask → training falls back to torch SDPA for those calls.
- Minimax H3 VSA sparse attention uses flex_attention, not SDPA, so it isn't affected (it needs triton anyway).
- Masked calls cost one GPU sync for UMFA's all-true-mask check.

### ComfyUI node (2026-11-01)
`examples/comfyui-umfa/`: "UMFA Attention (Metal)" MODEL→MODEL node sets
`optimized_attention_override` → ComfyUI's `attention_pytorch`, whose SDPA is served by UMFA
once the extension is imported (lazily, on first use). ComfyUI on Mac defaults to
sub-quadratic attention. Verified against ComfyUI 0.12.3 code with a counting SDPA stand-in;
real-kernel test pending on the macOS 15+ Mac. Install steps in its README.

Results from the macOS 15+ Mac (check script, bf16, S=4096): forward error matches torch;
grads ~2x torch error (~1 bf16 ulp); fwd+bwd 0.86x (12 heads), 1.04x (24), 1.19x (40), 1.43x causal.

UMFA bugs found and fixed (2026-11-01, verified on the macOS 15+ Mac):
- `55fd867` in-stream path committed `torch::mps::commit()` (default stream) off the
  stream's queue → `_dispatch_assert_queue_fail` crash on ComfyUI's worker thread.
- `fc44179` `mfa_attention_encode_mtl` dropped offsets/strides (regression from `0206bba`)
  → every no-grad call with transposed or chunked q/k/v returned garbage (H3 samples were a
  grid of blocks). After the fix, H3 samples look right.
- (2026-11-02) training samples went black after step 0 (weights NaN). Upstream `1f0f44c`
  removed every `torch::mps::synchronize()` before UMFA's own-queue calls on the theory that
  shared storage gives cross-queue ordering. It doesn't: torch's queued producers (the
  `.contiguous()` copies, d_output, the grad zero-fills, the LSE CPU→MPS copy) can run after
  UMFA reads or writes. Restored the syncs via a per-thread `synchronize_current_stream()`,
  moved the backward sync after the zero-fills, and allocate LSE/D scratch on the device.
  Not yet verified on the macOS 15+ Mac. The check script gained a model-shaped training case
  (fused qkv projection, no syncs, worst of 3 steps) that should catch this.

Measured: ComfyUI Minimax H3, 5 s video, fresh process per run: 57 → 49 s/it (~1.16x
end to end) vs ComfyUI's default attention. ai-toolkit native vs metal_flash training s/it
and memory on the same config are not yet measured.

### Runbook on macOS 15+
Forks: `kaspesla/ai-toolkit` and `kaspesla/universal-metal-flash-attention`, both on
branch `feature/metal-flash-attention` (`upstream` remote = original repo).
0. `git clone -b feature/metal-flash-attention --recursive https://github.com/kaspesla/universal-metal-flash-attention`
   and `git clone -b feature/metal-flash-attention https://github.com/kaspesla/ai-toolkit`, then `./run_mac.zsh` once in ai-toolkit to create `.venv`
1. `cd universal-metal-flash-attention && git submodule update --init && swift build -c release`
2. `cd examples/pytorch-custom-op-ffi && ~/Sandbox/ai-toolkit/.venv/bin/pip install pybind11 && ~/Sandbox/ai-toolkit/.venv/bin/python setup.py install`
3. `cd ~/Sandbox/ai-toolkit && .venv/bin/python testing/check_metal_flash_attention.py`
   — expect rel err ≲1e-2 (bf16), `pytorch_fallback=0`
4. Add `attention_backend: metal_flash` under `train:` in a Wan/Minimax config; compare s/it and memory against `native` for ~50 steps.
