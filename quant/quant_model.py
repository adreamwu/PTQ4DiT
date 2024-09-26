import logging
import torch
import torch.nn as nn
from quant.quant_block import get_specials, BaseQuantBlock, QuantDiTBlock, QuantFinalLayer
from quant.quant_layer import QuantModule, StraightThrough
from models import DiTBlock, FinalLayer


logger = logging.getLogger(__name__)


class QuantModel(nn.Module):
    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, **kwargs):
        super().__init__()
        self.model = model
        self.sm_abit = kwargs.get('sm_abit', 8)
        self.specials = get_specials()
        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)
        self.quant_block_refactor(self.model, weight_quant_params, act_quant_params)

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        """
        Recursively replace the normal layers (conv2D, conv1D, Linear etc.) to QuantModule
        :param module: nn.Module with nn.Conv2d, nn.Conv1d, or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        for name, child_module in module.named_children():
            if isinstance(child_module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                setattr(module, name, QuantModule(
                    child_module, weight_quant_params, act_quant_params))
            elif isinstance(child_module, StraightThrough):
                continue
            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params)

    def quant_block_refactor(self, module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        for name, child_module in module.named_children():
            if isinstance(child_module, DiTBlock):        
                setattr(module, name, QuantDiTBlock(child_module, act_quant_params, sm_abit=self.sm_abit))
            elif isinstance(child_module, FinalLayer):
                setattr(module, name, QuantFinalLayer(child_module, act_quant_params))
            else:
                self.quant_block_refactor(child_module, weight_quant_params, act_quant_params)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock, QuantFinalLayer)):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, x, t, y, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.model(combined, t, y)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
    
    def set_running_stat(self, running_stat: bool, sm_only=False):
        for m in self.model.modules():
            if isinstance(m, QuantDiTBlock):
                if sm_only:
                    m.attn.act_quantizer_w.running_stat = running_stat
                else:
                    m.attn.act_quantizer_q.running_stat = running_stat
                    m.attn.act_quantizer_k.running_stat = running_stat
                    m.attn.act_quantizer_v.running_stat = running_stat
                    m.attn.act_quantizer_w.running_stat = running_stat
            if isinstance(m, QuantModule) and not sm_only:
                m.set_running_stat(running_stat)

    def set_grad_ckpt(self, grad_ckpt: bool):
        for m in self.model.modules():
            if isinstance(m, (QuantDiTBlock)):
                m.checkpoint = grad_ckpt
