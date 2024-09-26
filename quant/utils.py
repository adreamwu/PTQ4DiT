import logging
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from quant.quant_layer import QuantModule, UniformAffineQuantizer
from quant.quant_block import BaseQuantBlock
from quant.quant_model import QuantModel
from quant.adaptive_rounding import AdaRoundQuantizer
from quant.quant_layer import UniformAffineQuantizer


logger = logging.getLogger(__name__)


def save_inp_oup_data(model: QuantModel, layer: Union[QuantModule, BaseQuantBlock], cali_data: torch.Tensor,
                      asym: bool = False, act_quant: bool = False, batch_size: int = 32, keep_gpu: bool = True):
    """
    Save input data and output data of a particular layer/block over calibration dataset.

    :param model: QuantModel
    :param layer: QuantModule or QuantBlock
    :param cali_data: calibration data set
    :param asym: if Ture, save quantized input and full precision output
    :param act_quant: use activation quantization
    :param batch_size: mini-batch size for calibration
    :param keep_gpu: put saved data on GPU for faster optimization
    :param cond: conditional generation or not
    :return: input and output data
    """
    device = next(model.parameters()).device
    get_inp_out = GetLayerInpOut(model, layer, device=device, asym=asym, act_quant=act_quant)
    cached_batches = []
    cached_inps, cached_outs = None, None
    torch.cuda.empty_cache()

    cali_xs, cali_ts, cali_ys, cfg_scale = cali_data
    
    num = int(cali_xs.size(0) / batch_size)
    for i in range(num):
        cur_inp, cur_out = get_inp_out(
            cali_xs[i * batch_size:(i + 1) * batch_size].to(device), 
            cali_ts[i * batch_size:(i + 1) * batch_size].to(device),
            cali_ys[i * batch_size:(i + 1) * batch_size].to(device), 
            cfg_scale
        )
        if isinstance(cur_inp, tuple):
            cur_x, cur_t = cur_inp
            cached_batches.append(((cur_x.cpu(), cur_t.cpu()), cur_out.cpu()))
        else:
            cached_batches.append((cur_inp.cpu(), cur_out.cpu()))

    if isinstance(cached_batches[0][0], tuple):
        cached_inps = [
            torch.cat([x[0][0] for x in cached_batches]), 
            torch.cat([x[0][1] for x in cached_batches])
        ]
    else:
        cached_inps = torch.cat([x[0] for x in cached_batches])
    cached_outs = torch.cat([x[1] for x in cached_batches])
    
    if isinstance(cached_inps, list):
        logger.info(f"in 1 shape: {cached_inps[0].shape}, in 2 shape: {cached_inps[1].shape}")
    else:
        logger.info(f"in shape: {cached_inps.shape}")
    logger.info(f"out shape: {cached_outs.shape}")
    torch.cuda.empty_cache()
    if keep_gpu:
        if isinstance(cached_inps, list):
            cached_inps[0] = cached_inps[0].to(device)
            cached_inps[1] = cached_inps[1].to(device)
        else:
            cached_inps = cached_inps.to(device)
        cached_outs = cached_outs.to(device)
    return cached_inps, cached_outs


def save_grad_data(model: QuantModel, layer: Union[QuantModule, BaseQuantBlock], cali_data: torch.Tensor,
                   damping: float = 1., act_quant: bool = False, batch_size: int = 32,
                   keep_gpu: bool = True):
    """
    Save gradient data of a particular layer/block over calibration dataset.

    :param model: QuantModel
    :param layer: QuantModule or QuantBlock
    :param cali_data: calibration data set
    :param damping: damping the second-order gradient by adding some constant in the FIM diagonal
    :param act_quant: use activation quantization
    :param batch_size: mini-batch size for calibration
    :param keep_gpu: put saved data on GPU for faster optimization
    :return: gradient data
    """
    device = next(model.parameters()).device
    get_grad = GetLayerGrad(model, layer, device, act_quant=act_quant)
    cached_batches = []
    torch.cuda.empty_cache()

    for i in range(int(cali_data.size(0) / batch_size)):
        cur_grad = get_grad(cali_data[i * batch_size:(i + 1) * batch_size])
        cached_batches.append(cur_grad.cpu())

    cached_grads = torch.cat([x for x in cached_batches])
    cached_grads = cached_grads.abs() + 1.0
    # scaling to make sure its mean is 1
    # cached_grads = cached_grads * torch.sqrt(cached_grads.numel() / cached_grads.pow(2).sum())
    torch.cuda.empty_cache()
    if keep_gpu:
        cached_grads = cached_grads.to(device)
    return cached_grads


class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """
    pass


class DataSaverHook:
    """
    Forward hook that stores the input and output of a block
    """
    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException


class GetLayerInpOut:
    def __init__(self, model: QuantModel, layer: Union[QuantModule, BaseQuantBlock],
                 device: torch.device, asym: bool = False, act_quant: bool = False):
        self.model = model
        self.layer = layer
        self.asym = asym
        self.device = device
        self.act_quant = act_quant
        self.data_saver = DataSaverHook(store_input=True, store_output=True, stop_forward=True)

    def __call__(self, x, t, y, cfg_scale):
        self.model.eval()
        self.model.set_quant_state(False, False)

        handle = self.layer.register_forward_hook(self.data_saver)
        with torch.no_grad():
            try:
                _ = self.model(x, t, y, cfg_scale)
            except StopForwardException:
                pass

            if self.asym:
                # Recalculate input with network quantized
                self.data_saver.store_output = False
                self.model.set_quant_state(weight_quant=True, act_quant=self.act_quant)
                try:
                    _ = self.model(x, t, y, cfg_scale)
                except StopForwardException:
                    pass
                self.data_saver.store_output = True

        handle.remove()

        self.model.set_quant_state(False, False)
        self.layer.set_quant_state(True, self.act_quant)
        self.model.train()

        if len(self.data_saver.input_store) > 1 and torch.is_tensor(self.data_saver.input_store[1]):
            return (self.data_saver.input_store[0].detach(),  
                self.data_saver.input_store[1].detach()), self.data_saver.output_store.detach()
        else:
            return self.data_saver.input_store[0].detach(), self.data_saver.output_store.detach()


class GradSaverHook:
    def __init__(self, store_grad=True):
        self.store_grad = store_grad
        self.stop_backward = False
        self.grad_out = None

    def __call__(self, module, grad_input, grad_output):
        if self.store_grad:
            self.grad_out = grad_output[0]
        if self.stop_backward:
            raise StopForwardException


class GetLayerGrad:
    def __init__(self, model: QuantModel, layer: Union[QuantModule, BaseQuantBlock],
                 device: torch.device, act_quant: bool = False):
        self.model = model
        self.layer = layer
        self.device = device
        self.act_quant = act_quant
        self.data_saver = GradSaverHook(True)

    def __call__(self, model_input):
        """
        Compute the gradients of block output, note that we compute the
        gradient by calculating the KL loss between fp model and quant model

        :param model_input: calibration data samples
        :return: gradients
        """
        self.model.eval()

        handle = self.layer.register_backward_hook(self.data_saver)
        with torch.enable_grad():
            try:
                self.model.zero_grad()
                inputs = model_input.to(self.device)
                self.model.set_quant_state(False, False)
                out_fp = self.model(inputs)
                quantize_model_till(self.model, self.layer, self.act_quant)
                out_q = self.model(inputs)
                loss = F.kl_div(F.log_softmax(out_q, dim=1), F.softmax(out_fp, dim=1), reduction='batchmean')
                loss.backward()
            except StopForwardException:
                pass

        handle.remove()
        self.model.set_quant_state(False, False)
        self.layer.set_quant_state(True, self.act_quant)
        self.model.train()
        return self.data_saver.grad_out.data


def quantize_model_till(model: QuantModule, layer: Union[QuantModule, BaseQuantBlock], act_quant: bool = False):
    """
    We assumes modules are correctly ordered, holds for all models considered
    :param model: quantized_model
    :param layer: a block or a single layer.
    """
    model.set_quant_state(False, False)
    for name, module in model.named_modules():
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            module.set_quant_state(True, act_quant)
        if module == layer:
            break


def get_train_samples(args, sample_data):
    num_samples, num_st = args.cali_n, args.cali_st
    if num_st == 1:
        xs = sample_data[:num_samples]
        ts = (torch.ones(num_samples) * 800)
    else:
        # get the real number of timesteps (especially for DDIM)
        nsteps = len(sample_data["ts"])
        timesteps = list(range(0, nsteps, nsteps//num_st))
        logger.info(f'Selected {len(timesteps)} steps from {nsteps} sampling steps')
        xs_lst = [sample_data["xs"][i][:num_samples] for i in timesteps]
        ts_lst = [sample_data["ts"][i][:num_samples] for i in timesteps]
        ys_lst = [sample_data["y"][i][:num_samples] for i in timesteps]
        xs = torch.cat(xs_lst, dim=0)
        ts = torch.cat(ts_lst, dim=0)
        ys = torch.cat(ys_lst, dim=0)
        return xs, ts, ys


def convert_adaround(model):
    for name, module in model.named_children():
        if isinstance(module, QuantModule):
            if module.ignore_reconstruction is True:
                continue
            else:
                module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode='learned_hard_sigmoid',
                                                   weight_tensor=module.weight)
        elif isinstance(module, BaseQuantBlock):
            if module.ignore_reconstruction is True:
                continue
            else:
                for name, sub_module in module.named_modules():
                    if isinstance(sub_module, QuantModule):
                        sub_module.weight_quantizer = AdaRoundQuantizer(uaq=sub_module.weight_quantizer, round_mode='learned_hard_sigmoid',
                                                                weight_tensor=sub_module.weight)
        else:
            convert_adaround(module)


def resume_cali_model(qnn, ckpt_path, cali_data):
    print("Loading quantized model checkpoint: ", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    print("Initializing weight quantization parameters")
    qnn.set_quant_state(True, False)
    
    cali_xs, cali_ts, cali_ys, cfg_scale = cali_data
    _ = qnn(cali_xs.cuda(), cali_ts.cuda(), cali_ys.cuda(), cfg_scale)
    # change weight quantizer from uniform to adaround
    convert_adaround(qnn)
    
    for n, m in qnn.model.named_modules():
        # if isinstance(m, AdaRoundQuantizer):
        if isinstance(m, (UniformAffineQuantizer, AdaRoundQuantizer)) and "act" not in n:
            m.zero_point = nn.Parameter(m.zero_point)
            m.delta = nn.Parameter(m.delta)

    # remove act_quantizer states for now
    keys = [key for key in ckpt.keys() if "act" in key]
    for key in keys:
        del ckpt[key]
    qnn.load_state_dict(ckpt, strict=True)
    qnn.set_quant_state(weight_quant=True, act_quant=False)
    
    for m in qnn.model.modules():
        if isinstance(m, AdaRoundQuantizer):
            zero_data = m.zero_point.data
            delattr(m, "zero_point")
            m.zero_point = zero_data

            delta_data = m.delta.data
            delattr(m, "delta")
            m.delta = delta_data

    print("Initializing act quantization parameters")
    qnn.set_quant_state(True, True)
    _ = qnn(cali_xs.cuda(), cali_ts.cuda(), cali_ys.cuda(), cfg_scale)
    print("Loading quantized model checkpoint again")
    
    for m in qnn.model.modules():
        if isinstance(m, AdaRoundQuantizer):
            m.zero_point = nn.Parameter(m.zero_point)
            m.delta = nn.Parameter(m.delta)
        elif isinstance(m, UniformAffineQuantizer):
            m.delta = nn.Parameter(m.delta)
            if m.zero_point is not None:
                if not torch.is_tensor(m.zero_point):
                    m.zero_point = nn.Parameter(torch.tensor(float(m.zero_point)))
                else:
                    m.zero_point = nn.Parameter(m.zero_point)
                
    ckpt = torch.load(ckpt_path, map_location='cpu')
    qnn.load_state_dict(ckpt)
    qnn.set_quant_state(weight_quant=True, act_quant=True)