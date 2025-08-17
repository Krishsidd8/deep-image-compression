# Optimal Image Compression through Integration of Deep Learning Architectures

Official implementation of the paper:  
**"Optimal Image Compression through Integration of Deep Learning Architectures"**  
by Krish Siddhiwala (2025)

<p align="center">
  <img src="results/figures/model_architecture.png" width="600">
</p>

---

## 📖 Overview
This work presents a **lightweight, end-to-end deep learning architecture for image compression**, combining:

- **Convolutional Autoencoders (CAEs)** – for compact latent representation.  
- **Residual Blocks (ResBlocks)** – for improved feature retention.  
- **Generative Adversarial Networks (GANs)** – for perceptual fidelity.  

The hybrid approach balances **compression ratio, pixel-level fidelity, and perceptual quality** while remaining deployable on **resource-constrained devices** (mobile, drones, IoT).

---

## 🚀 Features
- End-to-end training with **adaptive loss balancing**.  
- Ablation study on **CAE, CAE+ResBlocks, and CAE+ResBlocks+GAN**.  
- Benchmarked on **Microsoft Kaggle Cats vs Dogs dataset** (25k images).  
- Evaluated with **CR, PSNR, and SSIM** metrics.  

---

## 📊 Results
| Model | Compression Ratio (CR) | PSNR (dB) | SSIM |
|-------|-------------------------|-----------|------|
| CAE   | 48:1                   | 23.00     | 0.6483 |
| CRB   | 12:1                   | 28.86     | 0.8911 |
| GCRB  | 12:1                   | 22.95     | 0.7433 |

<p align="center">
  <img src="results/figures/reconstructions.png" width="700">
</p>

---

## 📦 Installation
```bash
git clone https://github.com/<your-username>/Optimal-Image-Compression.git
cd Optimal-Image-Compression
pip install -r requirements.txt
