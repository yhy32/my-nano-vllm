import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
from nanovllm.utils.fp8 import FP8Config


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)

def load_model(model: nn.Module, path: str, *, model_type: str | None = None, fp8Config: FP8Config):
    suffix_mapping = getattr(model, "weight_suffix_mapping", {})
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    is_FP8 = fp8Config.is_FP8
    files = glob(os.path.join(path, "*.safetensors"))
    local_ids = getattr(model, "local_expert_ids", None)
    # Scale and weight may live in different shards. Index names without reading tensors.
    scale_files = {}
    if is_FP8:
        for file in files:
            with safe_open(file, "pt", "cpu") as f:
                for key in f.keys():
                    if key.endswith(".weight_scale_inv"):
                        scale_files[key] = file

    for file in files:
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                if weight_name.endswith(".weight_scale_inv"):
                    continue
                if local_ids is not None and ".experts." in weight_name:
                    expert_id = int(weight_name.split(".experts.", 1)[1].split(".", 1)[0])
                    if expert_id not in local_ids:
                        continue
                scale_key = weight_name + "_scale_inv"
                scale_file = scale_files.get(scale_key)
                scale_inv = None
                if scale_file == file:
                    scale_inv = f.get_tensor(scale_key)
                elif scale_file is not None:
                    with safe_open(scale_file, "pt", "cpu") as scales:
                        scale_inv = scales.get_tensor(scale_key)
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        if scale_inv is None:
                            weight_loader(param, f.get_tensor(weight_name), shard_id)
                        else:
                            weight_loader(param, f.get_tensor(weight_name), shard_id, scale_inv, fp8Config.weight_block_size)
                        break
                else:
                    param_name = weight_name
                    for suffix, replacement in suffix_mapping.items():
                        if weight_name.endswith(suffix):
                            param_name = weight_name.removesuffix(suffix) + replacement
                            break
                    param = model.get_parameter(param_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    if scale_inv is None:
                        weight_loader(param, f.get_tensor(weight_name))
                    else:
                        weight_loader(param, f.get_tensor(weight_name), scale_inv, fp8Config.weight_block_size)
