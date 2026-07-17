from collections.abc import Iterable

import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential


def set_grad_checkpoint(model, use_fp32_attention=False, gc_step=1, use_reentrant=True):
    assert isinstance(model, nn.Module)

    def set_attr(module):
        module.grad_checkpointing = True
        module.fp32_attention = use_fp32_attention
        module.grad_checkpointing_step = gc_step
        module.grad_checkpointing_reentrant = use_reentrant

    model.apply(set_attr)


def auto_grad_checkpoint(module, *args, **kwargs):
    if getattr(module, "grad_checkpointing", False):
        use_reentrant = getattr(module, "grad_checkpointing_reentrant", True)
        if not isinstance(module, Iterable):
            return checkpoint(module, *args, use_reentrant=use_reentrant, **kwargs)
        gc_step = module[0].grad_checkpointing_step
        return checkpoint_sequential(module, gc_step, *args, use_reentrant=use_reentrant, **kwargs)
    return module(*args, **kwargs)
