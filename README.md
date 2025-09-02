# Optimal Image Compression through Integration of Deep Learning Architectures
### [Project Page](http://matthewtancik.com/nerf) | [Paper](StandardCitations_Optimal_Image_Compression_through_Integration_of_Deep_Learning_Architectures.pdf)
[![Open CAE in Colab]](https://colab.research.google.com/drive/1cGdR4h0VHmoqLsL20he8giuW51dkLBAD?usp=sharing)
[![Open CRB in Colab]](https://colab.research.google.com/drive/1pm47RUYwW_jZHydkPRAGeOcq106I3CKa?usp=sharing)
[![Open GCRB in Colab]](https://colab.research.google.com/drive/1xIDywdkdVjHGxKZz2Bu4X2hJptWCamyK?usp=sharing)

### PAPER | Official implementation of the paper:  
**"Optimal Image Compression through Integration of Deep Learning Architectures"**  
by Krish Siddhiwala (2025)

<p align="center">
  <img src="imgs/Banner.svg">
</p>

---

## Overview
This work presents a **lightweight, end-to-end deep learning architecture for image compression**, combining:

- **Convolutional Autoencoders (CAEs)** – for compact latent representation.  
- **Residual Blocks (ResBlocks)** – for improved feature retention.  
- **Generative Adversarial Networks (GANs)** – for perceptual fidelity.  

The hybrid approach balances **compression ratio, pixel-level fidelity, and perceptual quality** while remaining deployable on **resource-constrained devices**.


## Features
- End-to-end training with **adaptive loss balancing**.  
- Ablation study on **CAE, CAE+ResBlocks (CRB), and CAE+ResBlocks+GAN (GCRB)**.  
- Benchmarked on **Microsoft Kaggle Cats vs Dogs dataset** (25k images).  
- Evaluated with **CR, PSNR, and SSIM** metrics.  


## Method Overview
<img src="imgs/Model Diagram (1).png">

The model integrates:
- **Encoder–Decoder CAE** for compact latent representations.  
- **Residual Blocks** for deeper feature extraction and stability.  
- **GAN Discriminator** for perceptual realism.  
- **Loss Functions**: MSE, LPIPS, Hinge Loss with adaptive weighting.  

---

## Results

### Qualitative Results
  <img src="imgs/Final Output Results.svg" width="600">

### Quantitative Metrics

|   Model   | Compression Ratio (CR) |    PSNR   |  SSIM  |
|-----------|------------------------|-----------|--------|
| **CAE**   | 48:1                   | 25.15     | 0.7264 |
| **CRB**   | 12:1                   | 31.11     | 0.9151 |
| **GCRB**  | 12:1                   | 30.29     | 0.9365 |

### Abalation Study
<img src="imgs/Palette.svg" width="600">

---
## Getting Started

### Prerequisites
- Python 3.9+
- PyTorch >= 1.12
- torchvision
- numpy, matplotlib, tqdm
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity) for perceptual loss

Install dependencies:
```bash
git clone https://github.com/Krishsidd8/deep-image-compression.git
cd deep-image-compression
pip install -r requirements.txt
```
---

### Dataset

This project uses the Microsoft Kaggle Cats and Dogs Dataset (25k images).
Download via:

```bash
kaggle datasets download -d tongpython/cat-and-dog
```
Preprocess to 128×128 resolution before training.


### Citation

```
@article{siddhiwala2025compression,
  title={Optimal Image Compression through Integration of Deep Learning Architectures},
  author={Siddhiwala, Krish},
  year={2025}
}
```