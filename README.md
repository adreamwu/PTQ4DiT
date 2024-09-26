# Official Repository of [PTQ4DiT: Post-training Quantization for Diffusion Transformers](https://arxiv.org/abs/2405.16005) [NeurIPS 2024]

![PTQ4DiT](figures/PTQ4DiT.png)

## Usage

### Environment Setup

```bash
conda env create -f environment.yml
conda activate DiT
```

### Calibration Data



### Quantization
```bash
python quant_sample.py --model DiT-XL/2 --image-size 256 \
--ckpt pretrained_models/DiT-XL-2-256x256.pt \
--num-sampling-steps 50 --quant_act \
--weight_bit 8 --act_bit 8 --cali_st 25 --cali_n 64 --cali_batch_size 32 --sm_abit 8 \
--cali_data_path imagenet_DiT-256_sample4000_50steps_allst.pt --outdir output/ \
--cfg-scale 1.5 --seed 1 --recon --ptq --sample
```





### Evaluation





## TODO
Please stay tuned!

- [] Release core code of PTQ4DiT
- [] Release implementation of Re-Parameterization for fast inference


## Citation
If you find PTQ4DiT useful or relevant to your project and research, please kindly cite our paper:)

```bibtex
@article{wu2024ptq4dit,
  title={PTQ4DiT: Post-training Quantization for Diffusion Transformers},
  author={Wu, Junyi and Wang, Haoxuan and Shang, Yuzhang and Shah, Mubarak and Yan, Yan},
  journal={arXiv preprint arXiv:2405.16005},
  year={2024}
}
```
