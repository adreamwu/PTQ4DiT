import logging
import torch.nn as nn
from quant.quant_layer import QuantModule, UniformAffineQuantizer, StraightThrough
from models import DiTBlock, FinalLayer


logger = logging.getLogger(__name__)


class BaseQuantBlock(nn.Module):
    """
    Base implementation of block structures for all networks.
    """
    def __init__(self, act_quant_params: dict = {}):
        super().__init__()
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        self.activation_function = StraightThrough()

        self.ignore_reconstruction = False

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class QuantDiTBlock(BaseQuantBlock):
    def __init__(self, block, act_quant_params, sm_abit=8):
        super().__init__(act_quant_params)
        self.norm1 = block.norm1
        self.attn = block.attn
        self.norm2 = block.norm2
        self.mlp = block.mlp
        self.adaLN_modulation = block.adaLN_modulation
        
        self.attn.act_quantizer_q = UniformAffineQuantizer(**act_quant_params)
        self.attn.act_quantizer_k = UniformAffineQuantizer(**act_quant_params)
        self.attn.act_quantizer_v = UniformAffineQuantizer(**act_quant_params)
        act_quant_params_w = act_quant_params.copy()
        act_quant_params_w['n_bits'] = sm_abit
        act_quant_params_w['always_zero'] = True
        self.attn.act_quantizer_w = UniformAffineQuantizer(**act_quant_params_w)

        self.attn.use_act_quant = False
        self.time_step = 0

    def forward(self, x, c, calib=False):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)  # Each of size [16, 1152]
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), calib=calib)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp), calib=calib)
        return x

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.attn.use_act_quant = act_quant
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)


class QuantFinalLayer(BaseQuantBlock):
    def __init__(self, layer, act_quant_params):
        super().__init__(act_quant_params)
        self.norm_final = layer.norm_final
        self.linear = layer.linear
        self.adaLN_modulation = layer.adaLN_modulation

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)  # Each of size [16, 1152]
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)


def get_specials():
    specials = {
        DiTBlock: QuantDiTBlock,
        FinalLayer: QuantFinalLayer
    }
    return specials
