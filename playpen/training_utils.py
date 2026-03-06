from __future__ import annotations


def resolve_training_device(preferred_device=None):
    import torch

    if preferred_device is not None:
        return torch.device(preferred_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def has_meta_tensors(model) -> bool:
    try:
        if any(getattr(p, "is_meta", False) for p in model.parameters()):
            return True
        if any(getattr(b, "is_meta", False) for b in model.buffers()):
            return True
    except Exception:
        return False
    return False


def materialize_meta_tensors(model, device=None):
    import torch

    target = resolve_training_device(device)

    for name, param in list(model.named_parameters(recurse=True)):
        if not getattr(param, "is_meta", False):
            continue
        if "." in name:
            module_name, param_name = name.rsplit(".", 1)
            module = model.get_submodule(module_name)
        else:
            module = model
            param_name = name
        new_param = torch.nn.Parameter(
            torch.empty(param.shape, dtype=param.dtype, device=target),
            requires_grad=param.requires_grad,
        )
        setattr(module, param_name, new_param)

    for name, buf in list(model.named_buffers(recurse=True)):
        if not getattr(buf, "is_meta", False):
            continue
        if "." in name:
            module_name, buf_name = name.rsplit(".", 1)
            module = model.get_submodule(module_name)
        else:
            module = model
            buf_name = name
        new_buf = torch.empty(buf.shape, dtype=buf.dtype, device=target)
        setattr(module, buf_name, new_buf)

    return model


def prepare_model_for_trainer(model, device=None):
    if has_meta_tensors(model):
        return materialize_meta_tensors(model, device=device)
    return model
