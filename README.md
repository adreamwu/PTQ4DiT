# Official Repository of [PTQ4DiT: Post-training Quantization for Diffusion Transformers](https://arxiv.org/abs/2405.16005) [NeurIPS 2024]

![PTQ4DiT](figures/PTQ4DiT.png)

## Usage

### Environment Setup

```bash
conda env create -f environment.yml
conda activate DiT
```

### Calibration Data
We will release the generated calibration datasets. You can also customize the calibration data using *get_calibration_set.py*.






## TODO
Please stay tuned!

- [] Release core code of PTQ4DiT
- [] Release calibration datasets
- [] Release quantized DiT checkpoints
- [] Release implementation of Re-Parameterization for fast inference


## Citation
If you find PTQ4DiT useful or relevant to your project and research, please kindly cite our paper:)
@article{wu2024ptq4dit,
  title={PTQ4DiT: Post-training Quantization for Diffusion Transformers},
  author={Wu, Junyi and Wang, Haoxuan and Shang, Yuzhang and Shah, Mubarak and Yan, Yan},
  journal={arXiv preprint arXiv:2405.16005},
  year={2024}
}
