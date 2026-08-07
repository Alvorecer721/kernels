"""Inference-side entry point, imported by vLLM to detect the CUDA kernel.

vLLM's activation layer does ``import xielu.ops`` and then constructs
``torch.classes.xielu.XIELU()``; importing the extension here registers both the
operators and that class.

Importing this module also repairs vLLM's XIELU for torch.compile: upstream's
``forward_native`` branches on ``torch._dynamo.is_compiling()`` and calls
``logger.warning_once`` in the compiled branch — a fatal graph break under
fullgraph/AOT capture — and the raw torchbind handle held by the module cannot
be serialized into vLLM's compile cache. The fused kernel is therefore exposed
as a torch custom op (traceable as an opaque node, cacheable by reference) and
``XIELU.forward_native`` is replaced with a branch-free version that uses it,
so the kernel runs inside compiled graphs instead of falling back to Python.
"""

import torch

import _xielu  # noqa: F401  (registers the xielu namespace with torch)

__all__ = ["XIELU"]


def XIELU():  # noqa: N802 - mirrors the torchbind class name vLLM expects
    """Construct the torchbind XIELU handle backing vLLM's fused path."""
    return torch.classes.xielu.XIELU()


_KERNEL_HANDLE = None


def _kernel():
    global _KERNEL_HANDLE
    if _KERNEL_HANDLE is None:
        _KERNEL_HANDLE = torch.classes.xielu.XIELU()
    return _KERNEL_HANDLE


@torch.library.custom_op("xielu_shim::fused_forward", mutates_args=())
def _fused_forward(
    x: torch.Tensor,
    alpha_p: torch.Tensor,
    alpha_n: torch.Tensor,
    beta: float,
    eps: float,
    with_vector_loads: bool,
) -> torch.Tensor:
    original_shape = x.shape
    while x.dim() < 3:
        x = x.unsqueeze(0)
    if x.dim() > 3:
        x = x.reshape(-1, 1, x.size(-1))
    out = _kernel().forward(
        x.contiguous(), alpha_p, alpha_n, beta, eps, with_vector_loads
    )
    return out.reshape(original_shape)


@_fused_forward.register_fake
def _(x, alpha_p, alpha_n, beta, eps, with_vector_loads):
    return torch.empty_like(x)


def _patch_vllm_xielu():
    try:
        from vllm.model_executor.layers import activation
    except Exception:
        return
    cls = getattr(activation, "XIELU", None)
    if cls is None or getattr(cls, "_xielu_shim_patched", False):
        return

    def forward_native(self, input: torch.Tensor) -> torch.Tensor:
        if input.is_cuda:
            return torch.ops.xielu_shim.fused_forward(
                input,
                self.alpha_p,
                self.alpha_n,
                self._beta_scalar,
                self._eps_scalar,
                self.with_vector_loads,
            )
        return self._xielu_python(input)

    def forward(self, *args, **kwargs):
        return forward_native(self, *args, **kwargs)

    cls.forward_native = forward_native
    cls.forward_cuda = forward_native
    # CustomOp.__init__ snapshots self._forward_method BEFORE this module is
    # imported (the import happens inside XIELU.__init__), so the first
    # instance would keep the stock method; overriding forward() makes the
    # lookup happen at call time for every instance.
    cls.forward = forward
    cls._xielu_shim_patched = True


_patch_vllm_xielu()
