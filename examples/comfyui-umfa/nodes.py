"""ComfyUI node that runs a model's attention on Universal Metal Flash Attention.

Importing UMFA's `metal_sdpa_extension` registers its kernel for
aten::scaled_dot_product_attention on the MPS dispatch key, so every MPS SDPA
call in the process goes to UMFA from then on (until ComfyUI restarts). ComfyUI
on Apple Silicon defaults to sub-quadratic attention, which never calls SDPA,
so the node switches the patched model to ComfyUI's PyTorch attention path via
transformer_options["optimized_attention_override"]. Unpatched models keep
ComfyUI's default.
"""
import logging

import torch

from comfy.ldm.modules.attention import attention_pytorch

_ext = None


def _load_umfa():
    global _ext
    if _ext is not None:
        return _ext
    if not torch.backends.mps.is_available():
        raise RuntimeError("UMFA Attention needs an MPS device (Apple Silicon)")
    try:
        import metal_sdpa_extension
    except ImportError as e:
        raise ImportError(
            "UMFA Attention needs UMFA's metal_sdpa_extension built into ComfyUI's Python. "
            "See examples/comfyui-umfa/README.md in universal-metal-flash-attention."
        ) from e
    if not metal_sdpa_extension.is_metal_available():
        raise RuntimeError("UMFA reports Metal is not supported on this device")
    _ext = metal_sdpa_extension
    logging.info("UMFA: scaled_dot_product_attention on MPS now runs on Metal Flash Attention")
    return _ext


def umfa_attention(q, k, v, heads, **kwargs):
    """ComfyUI attention function: the PyTorch path, with SDPA served by UMFA."""
    _load_umfa()
    return attention_pytorch(q, k, v, heads, **kwargs)


def umfa_attention_override(func, *args, **kwargs):
    # kwargs carries _inside_attn_wrapper, so attention_pytorch won't re-apply the override
    return umfa_attention(*args, **kwargs)


try:
    from comfy.ldm.modules.attention import register_attention_function

    register_attention_function("umfa", umfa_attention)
except ImportError:
    pass  # ComfyUI before the attention registry; the node below still works


class UMFAAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/attention"
    DESCRIPTION = (
        "Run this model's attention on Universal Metal Flash Attention (Apple Silicon). "
        "Once loaded, UMFA serves every MPS scaled_dot_product_attention call until ComfyUI restarts."
    )

    def patch(self, model, enabled):
        if not enabled:
            return (model,)
        _load_umfa()
        m = model.clone()
        m.model_options["transformer_options"]["optimized_attention_override"] = umfa_attention_override
        return (m,)


NODE_CLASS_MAPPINGS = {"UMFAAttention": UMFAAttention}
NODE_DISPLAY_NAME_MAPPINGS = {"UMFAAttention": "UMFA Attention (Metal)"}
