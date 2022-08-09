# DnSwin 

## DnSwin: Toward Real-World Denoising via Continuous Wavelet Sliding-Transformer [[arxiv](https://arxiv.org/abs/2207.13861)]
### Hao Li, Zhijing Yang, Xiaobin Hong, Ziying Zhao, Junyang Chen, Yukai Shi, Jinshan Pan

### *Guangdong University of Technology, South China University of Technology, Sun Yat-sen University, Nanjing University of Science and Technology*
---
## Update - Aug, 2022
- Coming Soon.

## Abstract

Real-world image denoising is a practical image restoration problem that aims to obtain clean images from in-the-wild noisy input. Recently, Vision Transformer (ViT) exhibits a strong ability to capture long-range dependencies and many researchers attempt to apply ViT to image denoising tasks. However, real-world image is an isolated frame that makes the ViT build the long-range dependencies on the internal patches, which divides images into patches and disarranges the noise pattern and gradient continuity. In this article, we propose to resolve this issue by using a continuous Wavelet Sliding-Transformer that builds frequency correspondence under real-world scenes, called DnSwin. Specifically, we first extract the bottom features from noisy input images by using a CNN encoder. The key to DnSwin is to separate high-frequency and low-frequency information from the features and build frequency dependencies. To this end, we propose Wavelet Sliding-Window Transformer that utilizes discrete wavelet transform, self-attention and inverse discrete wavelet transform to extract deep features. Finally, we reconstruct the deep features into denoised images using a CNN decoder. Both quantitative and qualitative evaluations on real-world denoising benchmarks demonstrate that the proposed DnSwin performs favorably against the state-of-the-art methods.


## Citation:
If you find this work useful for your research, please cite:

```
@artical{Li2022dnswin,
  title={DnSwin: Toward Real-World Denoising via Continuous Wavelet Sliding-Transformer},
  author={Li, Hao and Yang, Zhijing and Hong, Xiaobin and Zhao, Ziying and Chen, Junyang and Shi, Yukai and Pan, Jinshan},
  journal={Knowledge-Based Systems},
  year={2022}
}
```

## Data

Download the dataset from [GoogleDrive](https://drive.google.com/drive/folders/1n2NKB7z2r13HAqFUNe4UDjq7d1JoGhU0?usp=sharing).

Extract the files to `data` folder as follow:

```
~/
  data/
    SIDD_train/
      ... (scene id)
    SIDD_valid/
      ... (id)
    Syn_train/
      ... (id)
    DND/
      images_srgb/
        ... (mat files)
      ... (mat files)
```

### Synthesize

The code to generate a synthetic dataset is provided by [GMSNet](https://github.com/IDKiro/GMSNet) .

The code you can find in `utils/syn`.

## Train

Copy the template code to build your own model:

```
~/
  model/
    DnSwin.py
    template.py
    ... (your model)
```

Train your own model:

```
python train.py --model ... (model name)
```

## Submit

Evaluate the trained model (`--ensemble` for higher score):

```
python submit_dnd.py --model ... (model name) --ensemble
```

The results are in `result/submit_dnd/bundled`

## Contact:
Please contact me if there is any question (Hao Li: haoli@njust.edu.cn).