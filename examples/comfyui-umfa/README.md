# UMFA Attention for ComfyUI

A ComfyUI custom node that runs a model's attention on Universal Metal Flash
Attention (UMFA) during sampling on Apple Silicon.

```
Load Diffusion Model ─► UMFA Attention (Metal) ─► KSampler
```

## How it works

- ComfyUI on Apple Silicon defaults to **sub-quadratic** attention (chunked,
  memory-saving, usually slower), not PyTorch's fused SDPA.
- The node clones the model and sets
  `transformer_options["optimized_attention_override"]`, so that model's
  attention uses ComfyUI's PyTorch path. Models without the node keep
  ComfyUI's default.
- Importing UMFA's `metal_sdpa_extension` registers UMFA for
  `aten::scaled_dot_product_attention` on MPS, so those SDPA calls run on the
  Metal kernels. Inference has no gradients, so attention masks stay on UMFA too.
- The node also registers `umfa` in ComfyUI's attention registry
  (`comfy.ldm.modules.attention.get_attention_function("umfa")`).

Caveats:

- Once the node has run, **every** MPS SDPA call in ComfyUI goes to UMFA until
  restart, including other nodes or encoders that call SDPA directly.
- Main diffusion models (Wan, Flux, ...) pass `transformer_options` to
  attention, so the override reaches them. Some VAEs and auxiliary modules don't,
  and they keep ComfyUI's default attention.

## Requirements

- macOS 15+ (UMFA compiles its kernels with Metal Shading Language 3.2)
- Xcode 16+
- UMFA's Python extension built against **ComfyUI's own Python**. An extension
  built for another venv (e.g. ai-toolkit's) won't load if the torch versions
  differ.

## Install (ComfyUI desktop app)

The desktop app's Python is `~/Documents/ComfyUI/.venv` (check `basePath` in
`~/Library/Application Support/ComfyUI/config.json` if you moved it).

1. Build UMFA's Swift library (skip if already built):
   ```bash
   cd ~/Sandbox/universal-metal-flash-attention && swift build -c release
   ```
2. Check ComfyUI's torch version (UMFA's docs list torch >= 2.12):
   ```bash
   ~/Documents/ComfyUI/.venv/bin/python -c "import torch; print(torch.__version__)"
   ```
3. Build the extension into ComfyUI's Python:
   ```bash
   ~/Documents/ComfyUI/.venv/bin/python -m pip install pybind11
   ```
   ```bash
   cd ~/Sandbox/universal-metal-flash-attention/examples/pytorch-custom-op-ffi && ~/Documents/ComfyUI/.venv/bin/python -m pip install --no-build-isolation .
   ```
4. Link the node into ComfyUI and restart ComfyUI:
   ```bash
   ln -s ~/Sandbox/universal-metal-flash-attention/examples/comfyui-umfa ~/Documents/ComfyUI/custom_nodes/comfyui-umfa
   ```

The extension loads the Swift library from `universal-metal-flash-attention/.build`,
so don't move or delete that folder.

## Use and compare

Add **UMFA Attention (Metal)** (category `advanced/attention`) between the model
loader and KSampler. The ComfyUI log shows
`UMFA: scaled_dot_product_attention on MPS now runs on Metal Flash Attention`
when it loads.

To compare against ComfyUI's default, keep the same workflow and seed, toggle the
node's `enabled` input, and compare s/it and the images. `enabled: false` puts
that model back on sub-quadratic attention. To compare against plain PyTorch
SDPA instead, restart ComfyUI with `--use-pytorch-cross-attention` and leave the
node out (UMFA isn't loaded until a node runs).

Expect the largest gains on long sequences: high-resolution images and video
(e.g. Wan at real resolutions and frame counts).
