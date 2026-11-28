from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Literal

import torch
from transformers import AutoConfig

ACTIVATION_SCHEMES = ["static", "dynamic"]

class FP8Config:

    def __init__(
        self,
        config: AutoConfig
    ):
        qtz_config = getattr(config, "quantization_config", None)
        self.is_FP8 = False
        if qtz_config is None:
            return
        
        self.is_FP8 = True
        self.quant_method = qtz_config["quant_method"]
        self.activation_scheme = qtz_config["activation_scheme"]
        if self.activation_scheme != "dynamic":
            raise ValueError("activation_scheme is not dynamic")
        self.fmt = qtz_config["fmt"]
        self.weight_block_size = qtz_config.get("weight_block_size", None)
        if self.weight_block_size is not None:
            self.weight_block_size = tuple(self.weight_block_size)