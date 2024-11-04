# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import logging
import argparse, os, datetime, gc
from quant.utils import resume_cali_model, get_train_samples
from quant.quant_model import QuantModel
from quant.quant_block import QuantDiTBlock, QuantFinalLayer
from quant.quant_layer import QuantModule, UniformAffineQuantizer
from quant.adaptive_rounding import AdaRoundQuantizer
import numpy as np
from quant.layer_recon import layer_reconstruction
from quant.block_recon import block_reconstruction


logger = logging.getLogger(__name__)

def main(args):
    # Setup save path:
    os.makedirs(args.outdir, exist_ok=True)
    outpath = os.path.join(args.outdir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(outpath)
    log_path = os.path.join(outpath, "run.log")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Arguments: {args}")
    logger.info(f"Saving to {outpath}")

    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    logger.info(f"Loaded {args.model} with {sum(p.numel() for p in model.parameters()):,} parameters.")

    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Setup quantization:
    a_scale_method = 'mse'
    wq_params = {'n_bits': args.weight_bit, 'channel_wise': True, 'scale_method': 'mse'}
    aq_params = {'n_bits': args.act_bit, 'symmetric': False, 'channel_wise': False, 'scale_method': a_scale_method, 'leaf_param': True}
    if args.resume:
        logger.info('Load with min-max quick initialization')
        wq_params['scale_method'] = 'max'
        aq_params['scale_method'] = 'max'
    qnn = QuantModel(
            model=model, weight_quant_params=wq_params, act_quant_params=aq_params, sm_abit=args.sm_abit)
    qnn.cuda()
    qnn.eval()
    logger.info(qnn)

    # Quantize the model:
    if not args.ptq:
        qnn.set_quant_state(False, False)
    else:
        logger.info(f"Sampling data from {args.cali_st} timesteps for calibration")
        sample_data = torch.load(args.cali_data_path)
        cali_data = get_train_samples(args, sample_data)
        del(sample_data)
        gc.collect()
        logger.info(f"Calibration data shape: {cali_data[0].shape} {cali_data[1].shape} {cali_data[2].shape}")
        cali_xs, cali_ts, cali_ys = cali_data
        timesteps = [cali_ts[args.cali_n*i] for i in range(args.cali_st)]

        inds = []
        n_per_t = 4
        for t in timesteps:
            idx = torch.where(cali_ts == t)[0][:n_per_t]
            inds.extend(idx.tolist())
        # rearrange data
        cali_xs_init = cali_xs[inds].cpu()
        cali_ts_init = cali_ts[inds].cpu()
        cali_ys_init = cali_ys[inds].cpu()
        normal_index = torch.where(cali_ys_init != 1000)[0].cpu()
        null_index = torch.where(cali_ys_init == 1000)[0].cpu()
        cali_xs_init = torch.cat([cali_xs_init[normal_index], cali_xs_init[null_index]], 0)
        cali_ts_init = torch.cat([cali_ts_init[normal_index], cali_ts_init[null_index]], 0)
        cali_ys_init = torch.cat([cali_ys_init[normal_index], cali_ys_init[null_index]], 0)
        logger.info('cali_init shape: {}, {}, {}'.format(cali_xs_init.shape, cali_ts_init.shape, cali_ys_init.shape))
        logger.info('cali_init_ts: {}'.format(cali_ts_init))
        logger.info('cali_init_ys: {}'.format(cali_ys_init))

        logger.info("Initializing scaling factors")
        qnn.set_quant_state(False, False)
        with torch.no_grad():
            _ = qnn(cali_xs_init.cuda(), cali_ts_init.cuda(), cali_ys_init.cuda(), args.cfg_scale)
        logger.info("Scaling factor initialization has done!")

        # Reconstruction
        if args.resume:
            cali_data = (torch.randn(4, 4, latent_size, latent_size), torch.randint(0, 1000, (4,)), torch.randint(0, 1000, (4,)), args.cfg_scale)
            resume_cali_model(qnn, args.cali_ckpt, cali_data)
        else:
            logger.info("Initializing weight quantization parameters")
            qnn.set_quant_state(True, False)
            with torch.no_grad():
                _ = qnn(cali_xs_init.cuda(), cali_ts_init.cuda(), cali_ys_init.cuda(), args.cfg_scale)
            logger.info("Weight initialization has done!")

            logger.info("Doing activation quantization")
            qnn.set_quant_state(True, True)
            with torch.no_grad():
                _ = qnn(cali_xs_init.cuda(), cali_ts_init.cuda(), cali_ys_init.cuda(), args.cfg_scale)
            logger.info("Activation quantization has done!")

            for m in qnn.model.modules():
                if isinstance(m, UniformAffineQuantizer):
                    m.delta = nn.Parameter(m.delta)
                    if m.zero_point is not None:
                        if not torch.is_tensor(m.zero_point):
                            m.zero_point = nn.Parameter(torch.tensor(float(m.zero_point)))
                        else:
                            m.zero_point = nn.Parameter(m.zero_point.float())

        if args.recon:
            # Kwargs for calibration
            cali_data = (cali_xs, cali_ts, cali_ys, args.cfg_scale)
            kwargs = dict(cali_data=cali_data, batch_size=args.cali_batch_size, 
                        iters=args.cali_iters, weight=0.01, asym=True, b_range=(20, 2),
                        warmup=0.2, act_quant=True, opt_mode=args.opt_mode, outpath=outpath)

            def recon_model(model):
                for name, module in model.named_children():
                    if isinstance(module, QuantModule):
                        logger.info('Reconstruction for layer {}'.format(name))
                        layer_reconstruction(qnn, module, **kwargs)
                    elif isinstance(module, QuantDiTBlock) or isinstance(module, QuantFinalLayer):
                        logger.info('Reconstruction for block {}'.format(name))
                        block_reconstruction(qnn, module, **kwargs)
                    else:
                        recon_model(module)
            torch.set_grad_enabled(True)
            recon_model(qnn)
            
            logger.info("Saving calibrated quantized UNet model")
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
                            m.zero_point = nn.Parameter(m.zero_point.float())
            torch.save(qnn.state_dict(), os.path.join(outpath, "ckpt.pth"))
            logger.info('Calibrated model saved to {}'.format(os.path.join(outpath, "ckpt.pth")))
        qnn.set_quant_state(True, True)

    if args.inference:
        # Labels to condition the model with (feel free to change):
        all_labels = np.arange(1000)
        outdir = os.path.join(outpath, "inference/")
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        count = args.c_begin * args.n_c

        # Create sampling noise:
        for c in all_labels:
            if c < args.c_begin or c > args.c_end:
                continue
            if c % 10 == 0:
                print(f"Generating data for class {c}")
            class_labels = [c] * args.n_c

            # Create sampling noise:
            n = len(class_labels)
            z = torch.randn(n, 4, latent_size, latent_size, device=device)
            y = torch.tensor(class_labels, device=device)

            # Setup classifier-free guidance:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

            # Sample images:
            samples = diffusion.p_sample_loop(
                qnn.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            samples = vae.decode(samples / 0.18215).sample

            # Save and display images:
            for i, sample in enumerate(samples):
                save_image(sample, os.path.join(outdir, f"{count}.png"), normalize=True, value_range=(-1, 1))
                count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # General arguments:
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--outdir", type=str, default="output/")

    # Quantization arguments:
    parser.add_argument("--ptq", action="store_true", help="Perform post-training quantization")
    parser.add_argument("--weight_bit", type=int, default=8, help="int bit for weight quantization")
    parser.add_argument("--act_bit", type=int, default=8, help="int bit for activation quantization")
    parser.add_argument("--cali_ckpt", type=str, help="path for calibrated model ckpt")
    parser.add_argument("--cali_data_path", type=str, default="sd_coco_sample1024_allst.pt", help="calibration dataset name")
    parser.add_argument("--resume", action="store_true", help="resume the calibrated model")
    parser.add_argument("--cali_st", type=int, default=1, help="number of timesteps used for calibration")
    parser.add_argument("--cali_n", type=int, default=1024, help="number of samples for each timestep for reconstruction")
    parser.add_argument("--cali_iters", type=int, default=20000, help="number of iterations for each reconstruction")
    parser.add_argument("--cali_batch_size", type=int, default=32, help="batch size for reconstruction")
    parser.add_argument('--cali_lr', default=4e-4, type=float, help='learning rate for reconstruction')
    parser.add_argument('--cali_p', default=2.4, type=float, help='L_p norm minimization for reconstruction')
    parser.add_argument("--sm_abit",type=int, default=8, help="attn softmax activation bit")
    parser.add_argument("--recon", action="store_true", help="reconstruct the model")
    parser.add_argument("--opt-mode", type=str, default="mse", help="optimization loss for reconstruction")
    parser.add_argument("--inference", action="store_true", help="inference for all classes")
    parser.add_argument("--n_c", type=int, default=10, help="number of samples for each class for inference")
    parser.add_argument("--c_begin", type=int, default=0, help="begining class index for inference")
    parser.add_argument("--c_end", type=int, default=999, help="ending class index for inference")
    args = parser.parse_args()
    main(args)
