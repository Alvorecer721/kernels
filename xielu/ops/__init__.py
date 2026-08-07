"""Inference-side entry point, imported by vLLM to detect the CUDA kernel.

vLLM's activation layer does ``import xielu.ops`` and then constructs
``torch.classes.xielu.XIELU()``; importing the extension here registers both the
operators and that class.
"""

import torch

import _xielu  # noqa: F401  (registers the xielu namespace with torch)

__all__ = ["XIELU"]


def XIELU():  # noqa: N802 - mirrors the torchbind class name vLLM expects
    """Construct the torchbind XIELU handle backing vLLM's fused path."""
    return torch.classes.xielu.XIELU()
