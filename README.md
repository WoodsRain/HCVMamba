<div align="center">
  <h1>HCVMamba</h1>
</div> 

<div align="center">
  <h4>Hybrid Cross-Scale Visual State-Space Model for Structural Crack Segmentation and Visual Integrity Assessment</h4>
</div> 

This code is directly related to the manuscript we are currently submitting to The Visual Computer.


## Abstract

Visual crack detection inherently serves as a typical research carrier for image analysis and feature representation, which aims to extract structural semantics and realize intelligent visual monitoring of surface defects. Yet slender and low-contrast defects challenge existing segmentation methods. Traditional approaches struggle with cross-scale structural consistency and region saliency under complex backgrounds. This work introduces a hybrid visual state-space framework that unifies long-range dependency modeling, cross-scale geometric consistency constraints, and detail-aware feature fusion. We design a cross-scale structure modulation module to stabilize topological propagation across scales and a mask-aware fusion block to enhance crack responses while suppressing noise. Experiments on three public datasets show clear gains over state-of-the-art models, with mIoU improvements up to 2.12%. The method improves structural stability and fine-detail recovery for real-world visual inspection.

## Getting Started

### Environment Setup

You can create the conda environment using the following commands:

```shell
conda create -n SCSegamba python=3.10 -y
conda activate SCSegamba

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html

pip install -U openmim
mim install mmcv-full

pip install mamba-ssm==1.2.0
pip install timm lmdb mmengine numpy
```

### Dataset Preparation

The experiments in this repository are conducted on the following publicly available datasets:

- DeepCrack: https://github.com/qinnzou/DeepCrack
- CrackMap: https://github.com/ikatsamenis/CrackMap
- Crack500: https://github.com/fyangneil/pavement-crack-detection

Please download the datasets from the corresponding official repositories and organize them according to your local training configuration.

### Training

You can modify the training configuration in `main.py` and start training with:

```shell
python main.py
```

### Evaluation

After generating prediction results, evaluation metrics can be calculated using:

```shell
python eval_compute.py

cd eval
python evaluate.py
```

Please modify the dataset paths and prediction result paths according to your local environment before evaluation.

### Checkpoints

The pretrained checkpoints and inference resources are currently not publicly available during the review and organization stage.

They will be released in future updates of this repository.

## Citation

If you find this repository useful for your research, please cite:

```bibtex
@article{sun2026hybrid,
  title={Hybrid Cross-Scale Visual State-Space Model for Structural Crack Segmentation and Visual Integrity Assessment},
  author={Sun, Mingsi and Yan, Lelei and Song, Pinyi and Zhao, Hongwei and Shao, Xue},
  journal={The Visual Computer},
  year={2026}
}
```

## TODO

- [ ] Release pretrained checkpoints

## License

This project is released under the Apache 2.0 license.

## Acknowledgment
This work stands on the shoulders of the following **open-source projects**:
<div style="display: flex; justify-content: center; gap: 30px; flex-wrap: wrap; margin: 20px 0;">
  <div>
    <a href="https://github.com/Karl1109/SCSegamba" target="_blank">SCSegamba</a> 
    <a href="https://doi.org/10.48550/arXiv.2503.01113">[Paper]</a>
  </div>
  <div>
    <a href="https://github.com/yhlleo/DeepCrack" target="_blank">DeepCrack</a> 
    <a href="https://doi.org/10.1016/j.neucom.2019.01.036">[Paper]</a>
  </div>
  <div>
    <a href="https://github.com/fyangneil/pavement-crack-detection" target="_blank">Crack500</a> 
    <a href="https://doi.org/10.1109/TITS.2019.2910595">[Paper]</a>
  </div>
     <div>
    <a href="https://github.com/ikatsamenis/CrackMap" target="_blank">CrackMap</a> 
    <a href="https://doi.org/10.1007/978-3-031-47969-4_16">[Paper]</a>
  </div>
</div>

